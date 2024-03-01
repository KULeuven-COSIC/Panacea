use crate::{
    context::{Context, FftBuffer},
    decision_tree::{bit_decomposed_rgsw, demux_with},
    naive_hash::NaiveHash,
    num_types::{AlignedScalarContainer, ComplexBox, One, Scalar, ScalarContainer, Zero},
    params::TFHEParameters,
    rlwe::{convert_standard_ggsw_to_fourier, decomposed_rlwe_to_rgsw, neg_gsw_std},
    utils::transpose,
    utils::{flatten_fourier_ggsw, pt_to_lossy_u64},
};

use aligned_vec::avec;
use dyn_stack::ReborrowMut;
use rayon::prelude::*;
use std::{
    sync::{Arc, Mutex, RwLock},
    time::{Duration, Instant},
};

use tfhe::core_crypto::{
    entities::FourierGgswCiphertext,
    prelude::{
        add_external_product_assign_mem_optimized, cmux_assign_mem_optimized,
        decrypt_glwe_ciphertext, encrypt_glwe_ciphertext, glwe_ciphertext_add_assign,
        glwe_ciphertext_sub_assign, par_encrypt_constant_ggsw_ciphertext, GgswCiphertext,
        GlweCiphertext, GlweSecretKey, Plaintext, PlaintextList,
    },
};

struct FftBufferPool {
    pool: Vec<Arc<Mutex<FftBuffer>>>,
}

impl FftBufferPool {
    fn new(max_size: usize, ctx: &Context<Scalar>) -> Self {
        Self {
            pool: (0..max_size)
                .map(|_| Arc::new(Mutex::new(ctx.gen_fft_ctx())))
                .collect(),
        }
    }
}

struct EncOP(GgswCiphertext<ScalarContainer>);

enum OP {
    READ,
    WRITE,
}

/// The client's encrypted query.
pub struct ClientQuery {
    a: Vec<GgswCiphertext<ScalarContainer>>,
    op: EncOP,
    data: GlweCiphertext<AlignedScalarContainer>,
}

impl ClientQuery {
    /// Turn the `op` field into a single RGSW ciphertext that indicates if
    /// the operation is a read (0) or a write (1).
    fn into_rw_flag(
        self,
        ctx: &Context<Scalar>,
        buf: &mut FftBuffer,
    ) -> FourierGgswCiphertext<ComplexBox> {
        convert_standard_ggsw_to_fourier(self.op.0, ctx, buf)
    }
}

/// ORAM client state.
pub struct Client {
    pub sk: GlweSecretKey<ScalarContainer>,
    pub ctx: Context<Scalar>,
    pub rows: usize,
    pub cols: usize,
}

impl Client {
    /// Setup a new client. Set rows = 1 if no batching is required.
    /// The exact number of elements in the database depends on the
    /// number of hash functions.
    pub fn new(rows: usize, cols: usize, params: TFHEParameters) -> Self {
        let mut ctx = Context::new(params);

        let sk = GlweSecretKey::generate_new_binary(
            ctx.glwe_dimension,
            ctx.poly_size,
            &mut ctx.secret_generator,
        );

        Self {
            sk,
            ctx,
            rows,
            cols,
        }
    }

    /// Generate a dummy database for performance testing.
    pub fn gen_dummy_database(
        &mut self,
        hash: &NaiveHash,
    ) -> (
        Vec<PlaintextList<ScalarContainer>>,
        Vec<Vec<GlweCiphertext<AlignedScalarContainer>>>,
    ) {
        // TODO put NaiveHash in Client struct
        let item_count = self.rows * self.cols / hash.h_count();
        let mut db = vec![vec![self.ctx.empty_glwe_ciphertext(); self.cols]; self.rows];

        // to make sure it doesn't take a long time to create a dummy database
        // we encrypt at most UNIQUE_LIMIT elements and put them in the first UNIQUE_LIMIT indices
        // and then replicate these elements to all the remaining indices too
        const UNIQUE_LIMIT: usize = 128;
        let unique_count = if item_count < UNIQUE_LIMIT {
            item_count
        } else {
            UNIQUE_LIMIT
        };

        let mut pts: Vec<PlaintextList<ScalarContainer>> = (0..unique_count)
            .map(|i| {
                let mut pt_i = self.ctx.gen_pt();
                let pt_i_clone = pt_i.clone();
                // encrypt the plaintext at index i and place
                // in the location computed by the first hash function
                let (first_r, first_c) = hash.hash_to_tuple(0, i, self.cols);
                assert!(first_r < self.rows && first_c < self.cols);

                self.ctx.codec.poly_encode(&mut pt_i.as_mut_polynomial());

                encrypt_glwe_ciphertext(
                    &self.sk,
                    &mut db[first_r][first_c],
                    &pt_i,
                    self.ctx.std,
                    &mut self.ctx.encryption_generator,
                );
                pt_i_clone
            })
            .collect();
        let mut rest: Vec<PlaintextList<ScalarContainer>> = (unique_count..item_count)
            .map(|i| {
                let (orig_first_r, orig_first_c) =
                    hash.hash_to_tuple(0, i % UNIQUE_LIMIT, self.cols);
                let (first_r, first_c) = hash.hash_to_tuple(0, i, self.cols);
                db[first_r][first_c] = db[orig_first_r][orig_first_c].clone();
                pts[i % UNIQUE_LIMIT].clone()
            })
            .collect();
        pts.append(&mut rest);

        // apply the rest of the hash function and copy remaining elements
        for i in 0..item_count {
            let h_count = hash.h_count();
            for h in 1..h_count {
                let (first_r, first_c) = hash.hash_to_tuple(0, i, self.cols);
                let (r, c) = hash.hash_to_tuple(h, i, self.cols);
                assert!(r < self.rows && c < self.cols);
                db[r][c] = db[first_r][first_c].clone();
            }
        }

        (pts, db)
    }

    fn gen_enc_op(&mut self, op: &OP) -> EncOP {
        let mut enc = GgswCiphertext::<ScalarContainer>::new(
            Scalar::zero(),
            self.ctx.glwe_size,
            self.ctx.poly_size,
            self.ctx.base_log,
            self.ctx.level_count,
            self.ctx.ciphertext_modulus,
        );
        let pt = match op {
            OP::READ => Plaintext(Scalar::zero()),
            OP::WRITE => Plaintext(Scalar::one()),
        };
        par_encrypt_constant_ggsw_ciphertext(
            &self.sk,
            &mut enc,
            pt,
            self.ctx.std,
            &mut self.ctx.encryption_generator,
        );

        EncOP(enc)
    }

    fn enc_data_rlwe(
        &mut self,
        mut alpha: PlaintextList<ScalarContainer>,
    ) -> GlweCiphertext<AlignedScalarContainer> {
        let mut out: GlweCiphertext<AlignedScalarContainer> = self.ctx.empty_glwe_ciphertext();

        self.ctx.codec.poly_encode(&mut alpha.as_mut_polynomial());

        encrypt_glwe_ciphertext(
            &self.sk,
            &mut out,
            &alpha,
            self.ctx.std,
            &mut self.ctx.encryption_generator,
        );
        out
    }

    fn make_read_query(&mut self, i: usize) -> ClientQuery {
        // let item_count = self.rows * self.cols / hash.h_count();
        let a = bit_decomposed_rgsw(i, self.cols, &self.sk, &mut self.ctx);
        let dummy_data = self.ctx.gen_binary_pt();

        ClientQuery {
            a,
            op: self.gen_enc_op(&OP::READ),
            data: self.enc_data_rlwe(dummy_data),
        }
    }

    /// Generate a read query for a single element at index `i`.
    pub fn gen_read_query_one(&mut self, i: usize, hash: &NaiveHash) -> ClientQuery {
        self.gen_read_query_batch(&[i], hash).remove(0)
    }

    /// Generate a read query for a batch of elements.
    pub fn gen_read_query_batch(
        &mut self,
        indices: &[usize],
        hash: &NaiveHash,
    ) -> Vec<ClientQuery> {
        let mapping = hash.hash_to_mapping(indices, self.cols);
        (0..self.rows)
            .map(|r| match mapping.get(&r) {
                Some(c) => self.make_read_query(c.0),
                None => self.make_read_query(0),
            })
            .collect()
    }

    fn make_write_query(&mut self, i: usize, alpha: PlaintextList<ScalarContainer>) -> ClientQuery {
        // let item_count = self.rows * self.cols / hash.h_count();
        let a = bit_decomposed_rgsw(i, self.cols, &self.sk, &mut self.ctx);
        ClientQuery {
            a,
            op: self.gen_enc_op(&OP::WRITE),
            data: self.enc_data_rlwe(alpha),
        }
    }

    /// Generate a write query for a single element.
    pub fn gen_write_query_one(
        &mut self,
        i: usize,
        alpha: PlaintextList<ScalarContainer>,
        hash: &NaiveHash,
    ) -> ClientQuery {
        self.gen_write_query_batch(&[i], alpha, hash).remove(0)
    }

    /// Generate a write query for a batch of elements.
    pub fn gen_write_query_batch(
        &mut self,
        indices: &[usize],
        alpha: PlaintextList<ScalarContainer>,
        hash: &NaiveHash,
    ) -> Vec<ClientQuery> {
        // TODO use multiple alphas
        let mapping = hash.hash_to_mapping(indices, self.cols);
        (0..self.rows)
            .map(|r| match mapping.get(&r) {
                Some(c) => self.make_write_query(c.0, alpha.clone()),
                None => self.make_read_query(0),
            })
            .collect()
    }
}

fn rw(
    op: &EncOP,
    a: &GlweCiphertext<AlignedScalarContainer>,
    b: &GlweCiphertext<AlignedScalarContainer>,
    ctx: &Context<Scalar>,
    buf: &mut FftBuffer,
) -> GlweCiphertext<AlignedScalarContainer> {
    let mut c0 = b.clone();
    let fourier_op = convert_standard_ggsw_to_fourier(op.0.clone(), ctx, buf);

    cmux_assign_mem_optimized(
        &mut c0,
        &mut a.clone(),
        &fourier_op,
        ctx.fft.as_view(),
        buf.mem.stack().rb_mut(),
    );

    c0
}

// process one query
fn process_one_mt(
    query: &ClientQuery,
    data: Arc<RwLock<Vec<GlweCiphertext<AlignedScalarContainer>>>>,
    neg_s: FourierGgswCiphertext<ComplexBox>,
    buf: &FftBufferPool,
    ctx: &Context<Scalar>,
) -> (
    GlweCiphertext<AlignedScalarContainer>,
    Vec<FourierGgswCiphertext<ComplexBox>>,
) {
    let f = |level: usize| {
        let tid = rayon::current_thread_index().unwrap();
        let mut c = ctx.empty_glwe_ciphertext();
        let shift: usize = (Scalar::BITS as usize) - ctx.base_log.0 * level;
        // *c.get_mut_body().as_mut_tensor().first_mut() = Scalar::one() << shift;
        (*c.get_mut_body().as_mut())[0] = Scalar::one() << shift;
        let query_a_fourier = query
            .a
            .iter()
            .map(|ct| {
                convert_standard_ggsw_to_fourier(
                    ct.clone(),
                    &ctx,
                    &mut buf.pool[tid].clone().lock().unwrap(),
                )
            })
            .collect();
        let tmp = demux_with(
            c,
            &query_a_fourier,
            ctx,
            &mut buf.pool[tid].clone().lock().unwrap(),
        );
        assert_eq!(tmp.len(), 1 << query.a.len());
        tmp
    };

    let levels = (1..=ctx.level_count.0).collect::<Vec<usize>>();
    let demux_res: Vec<Vec<GlweCiphertext<AlignedScalarContainer>>> =
        levels.par_iter().map(|level| f(*level)).collect();

    // NOTE: can we parallelize transpose?
    let decomposed_l = transpose(demux_res);
    let cols = data.read().unwrap().len();
    assert_eq!(decomposed_l.len(), cols);

    let g = |j: usize| {
        let tid = rayon::current_thread_index().unwrap();
        let l = decomposed_rlwe_to_rgsw(
            &decomposed_l[j],
            &neg_s,
            ctx,
            &mut buf.pool[tid].clone().lock().unwrap(),
        );

        // compute the partial response
        let mut tmp = ctx.empty_glwe_ciphertext();

        add_external_product_assign_mem_optimized(
            &mut tmp,
            &l,
            &data.clone().read().unwrap()[j],
            ctx.fft.as_view(),
            buf.pool[tid].clone().lock().unwrap().mem.stack().rb_mut(),
        );

        (tmp, l)
    };

    let indices = (0..cols).collect::<Vec<usize>>();
    let (tmps, ls): (
        Vec<GlweCiphertext<AlignedScalarContainer>>,
        Vec<FourierGgswCiphertext<ComplexBox>>,
    ) = indices.par_iter().map(|j| g(*j)).unzip();

    let y = tmps.into_par_iter().reduce(
        || ctx.empty_glwe_ciphertext(),
        |mut a, b| {
            glwe_ciphertext_add_assign(&mut a, &b);
            a
        },
    );
    (y, ls)
}

// multi threaded with a single query
fn update_db_mt(
    query: &ClientQuery,
    data: Arc<RwLock<Vec<GlweCiphertext<AlignedScalarContainer>>>>,
    ls: Vec<FourierGgswCiphertext<ComplexBox>>,
    buf: &FftBufferPool,
    ctx: &Context<Scalar>,
) {
    let g = |j: usize, l: &FourierGgswCiphertext<ComplexBox>| {
        let tid = rayon::current_thread_index().unwrap();
        let mut tmp: GlweCiphertext<AlignedScalarContainer> = rw(
            &query.op,
            &query.data,
            &data.clone().read().unwrap()[j],
            ctx,
            &mut buf.pool[tid].clone().lock().unwrap(),
        );

        // update the database at index j
        cmux_assign_mem_optimized(
            &mut data.clone().write().unwrap()[j],
            &mut tmp,
            &l,
            ctx.fft.as_view(),
            buf.pool[tid].clone().lock().unwrap().mem.stack().rb_mut(),
        );
    };

    let cols = data.read().unwrap().len();
    (0..cols).into_par_iter().zip(ls).for_each(|(j, l)| {
        g(j, &l);
    })
}

// single threaded
// note that just setting rayon to use a single thread
// is not enough since it allocates a FftBuffer for every closure g
// we avoid this in the single threaded implementation
fn process_one_st(
    query: &ClientQuery,
    data: Arc<RwLock<Vec<GlweCiphertext<AlignedScalarContainer>>>>,
    neg_s: &FourierGgswCiphertext<ComplexBox>,
    buf: &FftBufferPool,
    ctx: &Context<Scalar>,
) -> (
    GlweCiphertext<AlignedScalarContainer>,
    Vec<FourierGgswCiphertext<ComplexBox>>,
) {
    let mut demux_res: Vec<Vec<GlweCiphertext<AlignedScalarContainer>>> = vec![];
    let tid = rayon::current_thread_index().unwrap();
    for level in 1..=ctx.level_count.0 {
        let mut c = ctx.empty_glwe_ciphertext();
        let shift: usize = (Scalar::BITS as usize) - ctx.base_log.0 * level;
        (*c.get_mut_body().as_mut())[0] = Scalar::one() << shift;

        let tmp = demux_with(
            c,
            &query
                .a
                .iter()
                .map(|ct| {
                    convert_standard_ggsw_to_fourier(
                        ct.clone(),
                        ctx,
                        &mut buf.pool[tid].clone().lock().unwrap(),
                    )
                })
                .collect(),
            ctx,
            &mut buf.pool[tid].clone().lock().unwrap(),
        );
        assert_eq!(tmp.len(), 1 << query.a.len());
        demux_res.push(tmp);
    }

    let cols = data.read().unwrap().len();
    let decomposed_l = transpose(demux_res);
    assert_eq!(decomposed_l.len(), cols);

    let mut y: GlweCiphertext<AlignedScalarContainer> = ctx.empty_glwe_ciphertext();
    let mut ls: Vec<FourierGgswCiphertext<ComplexBox>> = Vec::new();
    ls.reserve_exact(cols);
    for j in 0..cols {
        // compute the output y
        // note that external_product adds to the output buffer
        ls.push(decomposed_rlwe_to_rgsw(
            &decomposed_l[j],
            neg_s,
            ctx,
            &mut buf.pool[tid].clone().lock().unwrap(),
        ));

        add_external_product_assign_mem_optimized(
            &mut y,
            &ls[j],
            &data.clone().read().unwrap()[j],
            ctx.fft.as_view(),
            buf.pool[tid].clone().lock().unwrap().mem.stack().rb_mut(),
        );
    }
    (y, ls)
}

fn update_db_st(
    query: &ClientQuery,
    data: Arc<RwLock<Vec<GlweCiphertext<AlignedScalarContainer>>>>,
    ls: &Vec<FourierGgswCiphertext<ComplexBox>>,
    buf: &FftBufferPool,
    ctx: &Context<Scalar>,
) {
    let tid = rayon::current_thread_index().unwrap();
    let cols = data.read().unwrap().len();
    for j in 0..cols {
        // update the database at index j
        let mut tmp = rw(
            &query.op,
            &query.data,
            &data.clone().read().unwrap()[j],
            ctx,
            &mut buf.pool[tid].clone().lock().unwrap(),
        );

        cmux_assign_mem_optimized(
            &mut data.clone().write().unwrap()[j],
            &mut tmp,
            &ls[j],
            ctx.fft.as_view(),
            buf.pool[tid].clone().lock().unwrap().mem.stack().rb_mut(),
        );
    }
}

pub struct Server {
    data: Vec<Arc<RwLock<Vec<GlweCiphertext<AlignedScalarContainer>>>>>,
    neg_s: FourierGgswCiphertext<ComplexBox>,
    ctx: Context<Scalar>,
}

impl Server {
    /// Create a new server that stores ciphertexts `data`.
    pub fn new(
        data: Vec<Vec<GlweCiphertext<AlignedScalarContainer>>>,
        neg_s: FourierGgswCiphertext<ComplexBox>,
        params: TFHEParameters,
    ) -> Self {
        Self {
            data: data
                .into_iter()
                .map(|row| Arc::new(RwLock::new(row)))
                .collect(),
            neg_s,
            ctx: Context::new(params),
        }
    }

    fn rows(&self) -> usize {
        self.data.len()
    }

    fn cols(&self) -> usize {
        self.data[0].clone().read().unwrap().len()
    }

    /// Process a single access query.
    pub fn process_one(
        &mut self,
        query: ClientQuery,
    ) -> (GlweCiphertext<AlignedScalarContainer>, Duration) {
        let buf = FftBufferPool::new(rayon::max_num_threads(), &self.ctx);
        let start = Instant::now();
        let (y, ls) = process_one_mt(
            &query,
            self.data[0].clone(),
            self.neg_s.clone(),
            &buf,
            &self.ctx,
        );
        let dur = start.elapsed();
        update_db_mt(&query, self.data[0].clone(), ls, &buf, &self.ctx);
        (y, dur)
    }

    // Process multiple access queries in parallel (but without batching).
    pub fn process_multi(
        &mut self,
        queries: Vec<ClientQuery>,
    ) -> (Vec<GlweCiphertext<AlignedScalarContainer>>, Duration) {
        let buf = FftBufferPool::new(rayon::max_num_threads(), &self.ctx);
        let cols = self.cols();
        // first phase we execute the single query procedure and produce the output
        // (over multiple queries)
        let start = Instant::now();
        let (ys, ls): (
            Vec<GlweCiphertext<AlignedScalarContainer>>,
            Vec<Vec<FourierGgswCiphertext<ComplexBox>>>,
        ) = queries
            .par_iter()
            .map(|query| process_one_st(query, self.data[0].clone(), &self.neg_s, &buf, &self.ctx))
            .unzip();
        let dur = start.elapsed();

        // sum the output of rw*t in parallel (over multiple queries)
        let tmps = queries.into_par_iter().zip(&ls).map(|(query, l)| {
            let tid = rayon::current_thread_index().unwrap();
            let mut out = vec![self.ctx.empty_glwe_ciphertext(); cols];
            for j in 0..cols {
                let tmp = rw(
                    &query.op,
                    &query.data,
                    &self.data[0].clone().read().unwrap()[j],
                    &self.ctx,
                    &mut buf.pool[tid].clone().lock().unwrap(),
                );
                // L * t

                add_external_product_assign_mem_optimized(
                    &mut out[j],
                    &l[j],
                    &tmp,
                    self.ctx.fft.as_view(),
                    buf.pool[tid].clone().lock().unwrap().mem.stack().rb_mut(),
                );
            }
            out
        });

        let sumed_tmps = tmps.reduce(
            || vec![self.ctx.empty_glwe_ciphertext(); cols],
            |mut xs, ys| {
                for i in 0..xs.len() {
                    glwe_ciphertext_add_assign(&mut xs[i], &ys[i])
                }
                xs
            },
        );

        let sumed_ls = ls.into_par_iter().reduce(
            || {
                vec![
                    FourierGgswCiphertext::<ComplexBox>::new(
                        self.ctx.glwe_size,
                        self.ctx.poly_size,
                        self.ctx.base_log,
                        self.ctx.level_count
                    );
                    cols
                ]
            },
            |mut xs, ys| {
                for i in 0..xs.len() {
                    let mut xs_i = flatten_fourier_ggsw(&xs[i]);
                    let ys_i = flatten_fourier_ggsw(&ys[i]);

                    xs_i.iter_mut()
                        .zip(ys_i.iter())
                        .for_each(|(xx, yy)| *xx += yy);
                    (*xs)[i] = FourierGgswCiphertext::from_container(
                        xs_i,
                        self.ctx.glwe_size,
                        self.ctx.poly_size,
                        self.ctx.base_log,
                        self.ctx.level_count,
                    );
                }
                xs
            },
        );

        // finally update the db in parallel (over multiple indices)
        (0..cols)
            .collect::<Vec<usize>>()
            .into_par_iter()
            .for_each(|j| {
                let tid = rayon::current_thread_index().unwrap();

                cmux_assign_mem_optimized(
                    &mut self.data[0].clone().write().unwrap()[j],
                    &mut sumed_tmps[j].clone(),
                    &sumed_ls[j],
                    self.ctx.fft.as_view(),
                    buf.pool[tid].clone().lock().unwrap().mem.stack().rb_mut(),
                );
            });

        (ys, dur)
    }

    /// Process multiple access queries in parallel using the batching technique.
    pub fn process_batch(
        &mut self,
        queries: Vec<ClientQuery>,
        hash: &NaiveHash,
    ) -> (Vec<GlweCiphertext<AlignedScalarContainer>>, Duration) {
        assert_eq!(queries.len(), self.rows());
        let buf = FftBufferPool::new(rayon::max_num_threads(), &self.ctx);

        // first phase we execute the single query procedure and produce the output
        let start = Instant::now();
        let (ys, ls): (
            Vec<GlweCiphertext<AlignedScalarContainer>>,
            Vec<Vec<FourierGgswCiphertext<ComplexBox>>>,
        ) = queries
            .par_iter()
            .zip(0..self.rows())
            .map(|(query, row)| {
                process_one_st(query, self.data[row].clone(), &self.neg_s, &buf, &self.ctx)
            })
            .unzip();
        let dur = start.elapsed();

        // then we update the database
        ls.par_iter()
            .zip(&queries)
            .zip(0..self.rows())
            .for_each(|((l, query), row)| {
                update_db_st(query, self.data[row].clone(), l, &buf, &self.ctx);
            });

        // then we iterate over all indices and perform consistency correction
        let mut flag_buf = self.ctx.gen_fft_ctx();
        let rw_flags: Vec<FourierGgswCiphertext<ComplexBox>> = queries
            .into_iter()
            .map(|q| q.into_rw_flag(&self.ctx, &mut flag_buf))
            .collect();
        let item_count = self.rows() * self.cols() / hash.h_count();
        (0..item_count).into_par_iter().for_each(|i| {
            let positions: Vec<(usize, usize)> = (0..hash.h_count())
                .map(|h| hash.hash_to_tuple(h, i, self.cols()))
                .collect();
            let new_data =
                correct_consistency(&positions, &self.data, &ls, &rw_flags, &self.ctx, &buf);
            for (r, c) in positions {
                self.data[r].clone().write().unwrap()[c] = new_data.clone();
            }
        });
        (ys, dur)
    }

    /// Decrypt the database and turn the items into u64.
    /// Note that the conversion is lossy if one PlaintestList is larger
    /// than 64 bits, which it usually is.
    pub fn decrypt_db(&self, sk: &GlweSecretKey<AlignedScalarContainer>) -> Vec<Vec<u64>> {
        self.data
            .iter()
            .map(|row| {
                row.clone()
                    .read()
                    .unwrap()
                    .iter()
                    .map(|v| {
                        let mut pt = self.ctx.gen_zero_pt();
                        decrypt_glwe_ciphertext(&sk, &v, &mut pt);
                        self.ctx.codec.poly_decode(&mut pt.as_mut_polynomial());

                        pt_to_lossy_u64(&pt)
                    })
                    .collect::<Vec<u64>>()
            })
            .collect()
    }
}

/// Setup the ORAM client and server for experimentation.
pub fn setup_random_oram(
    rows: usize,
    cols: usize,
    hash: &NaiveHash,
    params: TFHEParameters,
) -> (Client, Server, Vec<PlaintextList<ScalarContainer>>) {
    assert_eq!(rows * cols, hash.input_domain() * hash.h_count());
    // rows = 1, cols = n if we're not batching
    let mut client = Client::new(rows, cols, params.clone());

    let mut buf = client.ctx.gen_fft_ctx();
    // Generate the -s keyswitching key.
    let neg_sk_ct = convert_standard_ggsw_to_fourier(
        neg_gsw_std(&client.sk, &mut client.ctx),
        &client.ctx,
        &mut buf,
    );

    let (plain, data) = client.gen_dummy_database(hash);
    let server = Server::new(data, neg_sk_ct, params);

    (client, server, plain)
}

// (1 - (b_j * c_i1 + b_j * c_i2 + b_j * c_i3)) * v_i1 +
// (b_j * c_i1 * v_i1) + (b_j * c_i2 * v_i2) + (b_j * c_i3 * v_i3)
fn correct_consistency(
    positions: &[(usize, usize)],
    data: &[Arc<RwLock<Vec<GlweCiphertext<AlignedScalarContainer>>>>],
    ls: &[Vec<FourierGgswCiphertext<ComplexBox>>],
    rw_flags: &[FourierGgswCiphertext<ComplexBox>],
    ctx: &Context<Scalar>,
    buf: &FftBufferPool,
) -> GlweCiphertext<AlignedScalarContainer> {
    let tid = rayon::current_thread_index().unwrap();
    let first_r = positions[0].0;
    let first_c = positions[0].1;
    let v1s: Vec<GlweCiphertext<AlignedScalarContainer>> = positions
        .iter()
        .map(|(r, c)| {
            let mut tmp = GlweCiphertext::from_container(
                avec![Scalar::zero(); ctx.poly_size.0 * 2],
                ctx.poly_size,
                ctx.ciphertext_modulus,
            );
            let mut out = GlweCiphertext::from_container(
                avec![Scalar::zero(); ctx.poly_size.0 * 2],
                ctx.poly_size,
                ctx.ciphertext_modulus,
            );

            add_external_product_assign_mem_optimized(
                &mut tmp,
                &rw_flags[*r],
                &data[first_r].clone().read().unwrap()[first_c],
                ctx.fft.as_view(),
                buf.pool[tid].clone().lock().unwrap().mem.stack().rb_mut(),
            );

            add_external_product_assign_mem_optimized(
                &mut out,
                &ls[*r][*c],
                &tmp,
                ctx.fft.as_view(),
                buf.pool[tid].clone().lock().unwrap().mem.stack().rb_mut(),
            );
            out
        })
        .collect();

    // compute (1 - (b_j * c_i1 + b_j * c_i2 + b_j * c_i3 + ...)) * v_i1
    let mut first_term = data[first_r].clone().read().unwrap()[first_c].clone();
    for vs in &v1s {
        glwe_ciphertext_sub_assign(&mut first_term, vs);
    }

    // compute the remaining terms
    // (b_j * c_i1 * v_i1) + (b_j * c_i2 * v_i2) + (b_j * c_i3 * v_i3) + ...
    let second_terms = positions.iter().skip(1).map(|(r, c)| {
        let mut tmp = GlweCiphertext::from_container(
            avec![Scalar::zero(); ctx.poly_size.0 * 2],
            ctx.poly_size,
            ctx.ciphertext_modulus,
        );
        let mut out = GlweCiphertext::from_container(
            avec![Scalar::zero(); ctx.poly_size.0 * 2],
            ctx.poly_size,
            ctx.ciphertext_modulus,
        );

        add_external_product_assign_mem_optimized(
            &mut tmp,
            &rw_flags[*r],
            &data[*r].clone().read().unwrap()[*c],
            ctx.fft.as_view(),
            buf.pool[tid].clone().lock().unwrap().mem.stack().rb_mut(),
        );

        add_external_product_assign_mem_optimized(
            &mut out,
            &ls[*r][*c],
            &tmp,
            ctx.fft.as_view(),
            buf.pool[tid].clone().lock().unwrap().mem.stack().rb_mut(),
        );

        out
    });

    let mut second_term = v1s[0].clone();
    for term in second_terms {
        glwe_ciphertext_add_assign(&mut second_term, &term);
    }

    glwe_ciphertext_add_assign(&mut second_term, &first_term);

    second_term
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::utils::compute_noise_encoded;

    #[test]
    fn test_oram() {
        let n = 16usize;
        let hash = NaiveHash::new(1, n);
        let (mut client, mut server, pts) =
            setup_random_oram(1, n, &hash, TFHEParameters::default());
        assert_eq!(n, pts.len());

        {
            // read
            let idx = 1usize;
            let query = client.gen_read_query_one(idx, &hash);
            let y = server.process_one(query).0;
            let mut pt = client.ctx.gen_zero_pt();

            decrypt_glwe_ciphertext(&client.sk, &y, &mut pt);
            client.ctx.codec.poly_decode(&mut pt.as_mut_polynomial());

            assert_eq!(pts[idx], pt);

            // check noise
            println!(
                "noise for read: {}",
                compute_noise_encoded(&client.sk, &y, &pt, &client.ctx.codec)
            );
        }

        {
            // write
            let idx = 1usize;
            let new_data = client.ctx.gen_binary_pt();
            let write_query = client.gen_write_query_one(idx, new_data.clone(), &hash);
            server.process_one(write_query);

            let read_query = client.gen_read_query_one(idx, &hash);
            let y = server.process_one(read_query).0;
            let mut pt = client.ctx.gen_zero_pt();
            decrypt_glwe_ciphertext(&client.sk, &y, &mut pt);
            client.ctx.codec.poly_decode(&mut pt.as_mut_polynomial());

            assert_eq!(new_data, pt);
        }
    }

    #[test]
    #[ignore]
    fn test_oram_more() {
        let n = 2048usize;
        let hash = NaiveHash::new(1, n);
        let (mut client, mut server, pts) =
            setup_random_oram(1, n, &hash, TFHEParameters::default());
        assert_eq!(n, pts.len());

        for i in 0..20 {
            // read
            let idx = 1usize;
            let query = client.gen_read_query_one(idx, &hash);
            let y = server.process_one(query).0;
            let mut pt = client.ctx.gen_zero_pt();
            decrypt_glwe_ciphertext(&client.sk, &y, &mut pt);
            client.ctx.codec.poly_decode(&mut pt.as_mut_polynomial());

            assert_eq!(pts[idx], pt);

            // check noise
            println!(
                "noise for read at iter {}: {}",
                i,
                compute_noise_encoded(&client.sk, &y, &pt, &client.ctx.codec)
            );
        }
    }

    #[test]
    fn test_oram_multi() {
        let n = 16usize;
        let hash = NaiveHash::new(1, n);
        let (mut client, mut server, pts) =
            setup_random_oram(1, n, &hash, TFHEParameters::default());
        assert_eq!(n, pts.len());

        {
            // read
            let idx1 = 1usize;
            let idx2 = 2usize;
            let queries = vec![
                client.gen_read_query_one(idx1, &hash),
                client.gen_read_query_one(idx2, &hash),
            ];
            let ys = server.process_multi(queries).0;

            {
                let mut pt = client.ctx.gen_zero_pt();
                decrypt_glwe_ciphertext(&client.sk, &ys[0], &mut pt);
                client.ctx.codec.poly_decode(&mut pt.as_mut_polynomial());
                assert_eq!(pts[idx1], pt);
                // check noise
                println!(
                    "noise for read: {}",
                    compute_noise_encoded(&client.sk, &ys[0], &pt, &client.ctx.codec)
                );
            }

            {
                let mut pt = client.ctx.gen_zero_pt();
                decrypt_glwe_ciphertext(&client.sk, &ys[1], &mut pt);
                client.ctx.codec.poly_decode(&mut pt.as_mut_polynomial());
                assert_eq!(pts[idx2], pt);
            }
        }

        {
            // write
            let idx1 = 1usize;
            let idx2 = 2usize;
            let new_data = client.ctx.gen_binary_pt();
            let write_queries = vec![
                client.gen_write_query_one(idx1, new_data.clone(), &hash),
                client.gen_write_query_one(idx2, new_data.clone(), &hash),
            ];
            server.process_multi(write_queries);

            let read_queries = vec![
                client.gen_read_query_one(idx1, &hash),
                client.gen_read_query_one(idx2, &hash),
            ];
            let ys = server.process_multi(read_queries).0;

            {
                let mut pt = client.ctx.gen_zero_pt();
                decrypt_glwe_ciphertext(&client.sk, &ys[0], &mut pt);
                client.ctx.codec.poly_decode(&mut pt.as_mut_polynomial());
                assert_eq!(new_data, pt);
                // check noise
                println!(
                    "noise for write: {}",
                    compute_noise_encoded(&client.sk, &ys[0], &pt, &client.ctx.codec)
                );
            }

            {
                let mut pt = client.ctx.gen_zero_pt();
                decrypt_glwe_ciphertext(&client.sk, &ys[1], &mut pt);
                client.ctx.codec.poly_decode(&mut pt.as_mut_polynomial());
                assert_eq!(new_data, pt);
            }
        }
    }

    #[test]
    fn test_oram_batch() {
        let n = 16usize;
        let h_count = 4usize;
        let rows = 8usize;
        let cols = h_count * n / rows;
        let hash = NaiveHash::new(h_count, n);
        let (mut client, mut server, pts) =
            setup_random_oram(rows, cols, &hash, TFHEParameters::default());
        assert_eq!(n, pts.len());

        {
            // read
            let indices = vec![0usize, 1usize];
            let mapping = hash.hash_to_mapping(&indices, cols);
            let query = client.gen_read_query_batch(&indices, &hash);
            let ys = server.process_batch(query, &hash).0;

            let mut pt = client.ctx.gen_zero_pt();
            let mut noise_checked = false;
            for (r, (_, i)) in mapping {
                decrypt_glwe_ciphertext(&client.sk, &ys[r], &mut pt);
                client.ctx.codec.poly_decode(&mut pt.as_mut_polynomial());
                assert_eq!(pts[indices[i]], pt);

                // check noise
                if !noise_checked {
                    println!(
                        "noise for read: {}",
                        compute_noise_encoded(&client.sk, &ys[r], &pt, &client.ctx.codec)
                    );
                    noise_checked = true;
                }
            }
        }

        {
            // write
            let indices = vec![1usize, 2usize];
            let mapping = hash.hash_to_mapping(&indices, cols);
            let new_data = client.ctx.gen_binary_pt();
            let write_query = client.gen_write_query_batch(&indices, new_data.clone(), &hash);
            server.process_batch(write_query, &hash);

            let read_query = client.gen_read_query_batch(&indices, &hash);
            let ys = server.process_batch(read_query, &hash).0;
            let mut pt = client.ctx.gen_zero_pt();
            for (r, _) in mapping {
                decrypt_glwe_ciphertext(&client.sk, &ys[r], &mut pt);
                client.ctx.codec.poly_decode(&mut pt.as_mut_polynomial());
                assert_eq!(new_data, pt);
            }

            // additionally check the database is consistent
            for i in indices {
                for h in 0..h_count {
                    let (r, c) = hash.hash_to_tuple(h, i, cols);

                    decrypt_glwe_ciphertext(
                        &client.sk,
                        &server.data[r].clone().read().unwrap()[c],
                        &mut pt,
                    );
                    client.ctx.codec.poly_decode(&mut pt.as_mut_polynomial());

                    assert_eq!(new_data, pt);
                }
            }
        }
    }

    #[test]
    #[ignore]
    fn test_oram_batch_more() {
        let h_count = 3usize;
        let rows = 3;
        let cols = 1024;
        let n = rows * cols / h_count;
        let hash = NaiveHash::new(h_count, n);

        let setup_time = Instant::now();
        let (mut client, mut server, pts) =
            setup_random_oram(rows, cols, &hash, TFHEParameters::default());
        println!("DB Setup Time: {}s\t", setup_time.elapsed().as_secs_f64());

        assert_eq!(n, pts.len());
        for iter in 0..20 {
            // read
            let indices = vec![0usize, 1usize];
            let mapping = hash.hash_to_mapping(&indices, cols);
            let query = client.gen_read_query_batch(&indices, &hash);

            let server_process = Instant::now();
            let ys = server.process_batch(query, &hash).0;
            print!(
                "server processing time: {}s\t",
                server_process.elapsed().as_secs_f64()
            );

            let mut pt = client.ctx.gen_zero_pt();
            let mut noise_checked = false;
            for (r, (_, i)) in mapping {
                decrypt_glwe_ciphertext(&client.sk, &ys[r], &mut pt);
                client.ctx.codec.poly_decode(&mut pt.as_mut_polynomial());
                assert_eq!(pts[indices[i]], pt);

                // check noise
                if !noise_checked {
                    println!(
                        "noise for read at iteration {}: {}",
                        iter,
                        compute_noise_encoded(&client.sk, &ys[r], &pt, &client.ctx.codec)
                    );
                    noise_checked = true;
                }
            }
        }
    }
}
