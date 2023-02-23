use crate::{
    context::{Context, FftBuffer},
    decision_tree::{bit_decomposed_rgsw, demux_with},
    naive_hash::NaiveHash,
    num_types::{One, Scalar, Zero},
    params::TFHEParameters,
    rgsw::RGSWCiphertext,
    rlwe::{decomposed_rlwe_to_rgsw, RLWECiphertext, RLWESecretKey},
    utils::pt_to_lossy_u64,
    utils::transpose,
};
use concrete_core::commons::{
    crypto::encoding::{Plaintext, PlaintextList},
    math::tensor::AsMutTensor,
};
use rayon::prelude::*;
use std::{
    sync::{Arc, Mutex, RwLock},
    time::{Duration, Instant},
};

struct FftBufferPool {
    pool: Vec<Arc<Mutex<FftBuffer>>>,
}

impl FftBufferPool {
    fn new(max_size: usize, ctx: &Context) -> Self {
        Self {
            pool: (0..max_size)
                .map(|_| Arc::new(Mutex::new(ctx.gen_fft_ctx())))
                .collect(),
        }
    }
}

struct EncOP {
    cb0: RGSWCiphertext,
    cb1: RGSWCiphertext,
}

enum OP {
    READ,
    WRITE,
}

/// The client's encrypted query.
pub struct ClientQuery {
    a: Vec<RGSWCiphertext>,
    op: EncOP,
    data: RLWECiphertext,
}

impl ClientQuery {
    /// Turn the `op` field into a single RGSW ciphertext that indicates if
    /// the operation is a read (0) or a write (1).
    fn into_rw_flag(self) -> RGSWCiphertext {
        self.op.cb0
    }
}

/// ORAM client state.
pub struct Client {
    pub sk: RLWESecretKey,
    pub ctx: Context,
    pub rows: usize,
    pub cols: usize,
}

impl Client {
    /// Setup a new client. Set rows = 1 if no batching is required.
    /// The exact number of elements in the database depends on the
    /// number of hash functions.
    pub fn new(rows: usize, cols: usize, params: TFHEParameters) -> Self {
        let mut ctx = Context::new(params);
        let sk = ctx.gen_rlwe_sk();
        Self {
            sk,
            ctx,
            rows,
            cols,
        }
    }

    /// Generate the -s keyswitching key.
    pub fn gen_neg_sk(&mut self) -> RGSWCiphertext {
        self.sk.neg_gsw(&mut self.ctx)
    }

    /// Generate a dummy database for performance testing.
    pub fn gen_dummy_database(
        &mut self,
        hash: &NaiveHash,
    ) -> (Vec<PlaintextList<Vec<Scalar>>>, Vec<Vec<RLWECiphertext>>) {
        // TODO put NaiveHash in Client struct
        let item_count = self.rows * self.cols / hash.h_count();
        let mut db = vec![vec![RLWECiphertext::allocate(self.ctx.poly_size); self.cols]; self.rows];

        // to make sure it doesn't take a long time to create a dummy database
        // we encrypt at most UNIQUE_LIMIT elements and put them in the first UNIQUE_LIMIT indices
        // and then replicate these elements to all the remaining indices too
        const UNIQUE_LIMIT: usize = 128;
        let unique_count = if item_count < UNIQUE_LIMIT {
            item_count
        } else {
            UNIQUE_LIMIT
        };

        let mut pts: Vec<PlaintextList<Vec<Scalar>>> = (0..unique_count)
            .map(|i| {
                let pt_i = self.ctx.gen_pt();
                // encrypt the plaintext at index i and place
                // in the location computed by the first hash function
                let (first_r, first_c) = hash.hash_to_tuple(0, i, self.cols);
                assert!(first_r < self.rows && first_c < self.cols);
                self.sk
                    .encode_encrypt_rlwe(&mut db[first_r][first_c], &pt_i, &mut self.ctx);
                pt_i
            })
            .collect();
        let mut rest: Vec<PlaintextList<Vec<Scalar>>> = (unique_count..item_count)
            .map(|i| {
                let item = pts[i % UNIQUE_LIMIT].clone();
                let (orig_first_r, orig_first_c) =
                    hash.hash_to_tuple(0, i % UNIQUE_LIMIT, self.cols);
                let (first_r, first_c) = hash.hash_to_tuple(0, i, self.cols);
                db[first_r][first_c] = db[orig_first_r][orig_first_c].clone();
                item
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
        let mut enc_0: RGSWCiphertext =
            RGSWCiphertext::allocate(self.ctx.poly_size, self.ctx.base_log, self.ctx.level_count);
        let mut enc_1: RGSWCiphertext =
            RGSWCiphertext::allocate(self.ctx.poly_size, self.ctx.base_log, self.ctx.level_count);
        self.sk
            .encrypt_constant_rgsw(&mut enc_0, &Plaintext(Scalar::zero()), &mut self.ctx);
        self.sk
            .encrypt_constant_rgsw(&mut enc_1, &Plaintext(Scalar::one()), &mut self.ctx);
        match op {
            OP::READ => EncOP {
                cb0: enc_0,
                cb1: enc_1,
            },
            OP::WRITE => EncOP {
                cb0: enc_1,
                cb1: enc_0,
            },
        }
    }

    fn enc_data_rlwe(&mut self, alpha: &PlaintextList<Vec<Scalar>>) -> RLWECiphertext {
        let mut out: RLWECiphertext = RLWECiphertext::allocate(self.ctx.poly_size);
        self.sk.encode_encrypt_rlwe(&mut out, alpha, &mut self.ctx);
        out
    }

    fn make_read_query(&mut self, i: usize) -> ClientQuery {
        // let item_count = self.rows * self.cols / hash.h_count();
        let a = bit_decomposed_rgsw(i, self.cols, &self.sk, &mut self.ctx);
        let dummy_data = self.ctx.gen_binary_pt();

        ClientQuery {
            a,
            op: self.gen_enc_op(&OP::READ),
            data: self.enc_data_rlwe(&dummy_data),
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

    fn make_write_query(&mut self, i: usize, alpha: &PlaintextList<Vec<Scalar>>) -> ClientQuery {
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
        alpha: &PlaintextList<Vec<Scalar>>,
        hash: &NaiveHash,
    ) -> ClientQuery {
        self.gen_write_query_batch(&[i], alpha, hash).remove(0)
    }

    /// Generate a write query for a batch of elements.
    pub fn gen_write_query_batch(
        &mut self,
        indices: &[usize],
        alpha: &PlaintextList<Vec<Scalar>>,
        hash: &NaiveHash,
    ) -> Vec<ClientQuery> {
        // TODO use multiple alphas
        let mapping = hash.hash_to_mapping(indices, self.cols);
        (0..self.rows)
            .map(|r| match mapping.get(&r) {
                Some(c) => self.make_write_query(c.0, alpha),
                None => self.make_read_query(0),
            })
            .collect()
    }
}

fn rw(
    op: &EncOP,
    a: &RLWECiphertext,
    b: &RLWECiphertext,
    ctx: &Context,
    buf: &mut FftBuffer,
) -> RLWECiphertext {
    let mut c0 = RLWECiphertext::allocate(ctx.poly_size);
    let mut c1 = RLWECiphertext::allocate(ctx.poly_size);
    op.cb0.external_product_with_buf(&mut c0, a, buf);
    op.cb1.external_product_with_buf(&mut c1, b, buf);
    c0.update_with_add(&c1);
    c0
}

// process one query
fn process_one_mt(
    query: &ClientQuery,
    data: Arc<RwLock<Vec<RLWECiphertext>>>,
    neg_s: &RGSWCiphertext,
    buf: &FftBufferPool,
    ctx: &Context,
) -> (RLWECiphertext, Vec<RGSWCiphertext>) {
    let f = |level: usize| {
        let tid = rayon::current_thread_index().unwrap();
        let mut c = RLWECiphertext::allocate(ctx.poly_size);
        let shift: usize = (Scalar::BITS as usize) - ctx.base_log.0 * level;
        *c.get_mut_body().as_mut_tensor().first_mut() = Scalar::one() << shift;

        let tmp = demux_with(c, &query.a, ctx, &mut buf.pool[tid].clone().lock().unwrap());
        assert_eq!(tmp.len(), 1 << query.a.len());
        tmp
    };

    let levels = (1..=ctx.level_count.0).collect::<Vec<usize>>();
    let demux_res: Vec<Vec<RLWECiphertext>> = levels.par_iter().map(|level| f(*level)).collect();

    // NOTE: can we parallelize transpose?
    let decomposed_l = transpose(demux_res);
    let cols = data.read().unwrap().len();
    assert_eq!(decomposed_l.len(), cols);

    let g = |j: usize| {
        let tid = rayon::current_thread_index().unwrap();
        let l = decomposed_rlwe_to_rgsw(
            &decomposed_l[j],
            neg_s,
            ctx,
            &mut buf.pool[tid].clone().lock().unwrap(),
        );

        // compute the partial response
        let mut tmp = RLWECiphertext::allocate(ctx.poly_size);
        l.external_product_with_buf(
            &mut tmp,
            &data.clone().read().unwrap()[j],
            &mut buf.pool[tid].clone().lock().unwrap(),
        );

        (tmp, l)
    };

    let indices = (0..cols).collect::<Vec<usize>>();
    let (tmps, ls): (Vec<RLWECiphertext>, Vec<RGSWCiphertext>) =
        indices.par_iter().map(|j| g(*j)).unzip();

    let y = tmps.into_par_iter().reduce(
        || RLWECiphertext::allocate(ctx.poly_size),
        |mut a, b| {
            a.update_with_add(&b);
            a
        },
    );
    (y, ls)
}

// multi threaded with a single query
fn update_db_mt(
    query: &ClientQuery,
    data: Arc<RwLock<Vec<RLWECiphertext>>>,
    ls: Vec<RGSWCiphertext>,
    buf: &FftBufferPool,
    ctx: &Context,
) {
    let g = |j: usize, l: &RGSWCiphertext| {
        let tid = rayon::current_thread_index().unwrap();
        let tmp: RLWECiphertext = rw(
            &query.op,
            &query.data,
            &data.clone().read().unwrap()[j],
            ctx,
            &mut buf.pool[tid].clone().lock().unwrap(),
        );

        // update the database at index j
        let mut new_data: RLWECiphertext = RLWECiphertext::allocate(ctx.poly_size);
        l.cmux_with_buf(
            &mut new_data,
            &data.clone().read().unwrap()[j],
            &tmp,
            &mut buf.pool[tid].clone().lock().unwrap(),
        );
        data.clone().write().unwrap()[j].fill_with_copy(&new_data);
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
    data: Arc<RwLock<Vec<RLWECiphertext>>>,
    neg_s: &RGSWCiphertext,
    buf: &FftBufferPool,
    ctx: &Context,
) -> (RLWECiphertext, Vec<RGSWCiphertext>) {
    let mut demux_res: Vec<Vec<RLWECiphertext>> = vec![];
    let tid = rayon::current_thread_index().unwrap();
    for level in 1..=ctx.level_count.0 {
        let mut c = RLWECiphertext::allocate(ctx.poly_size);
        let shift: usize = (Scalar::BITS as usize) - ctx.base_log.0 * level;
        *c.get_mut_body().as_mut_tensor().first_mut() = Scalar::one() << shift;

        let tmp = demux_with(c, &query.a, ctx, &mut buf.pool[tid].clone().lock().unwrap());
        assert_eq!(tmp.len(), 1 << query.a.len());
        demux_res.push(tmp);
    }

    let cols = data.read().unwrap().len();
    let decomposed_l = transpose(demux_res);
    assert_eq!(decomposed_l.len(), cols);

    let mut y: RLWECiphertext = RLWECiphertext::allocate(ctx.poly_size);
    let mut ls: Vec<RGSWCiphertext> =
        vec![RGSWCiphertext::allocate(ctx.poly_size, ctx.base_log, ctx.level_count); cols];
    for j in 0..cols {
        // compute the output y
        // note that external_product adds to the output buffer
        ls[j] = decomposed_rlwe_to_rgsw(
            &decomposed_l[j],
            neg_s,
            ctx,
            &mut buf.pool[tid].clone().lock().unwrap(),
        );
        ls[j].external_product_with_buf(
            &mut y,
            &data.clone().read().unwrap()[j],
            &mut buf.pool[tid].clone().lock().unwrap(),
        );
    }
    (y, ls)
}

fn update_db_st(
    query: &ClientQuery,
    data: Arc<RwLock<Vec<RLWECiphertext>>>,
    ls: &Vec<RGSWCiphertext>,
    buf: &FftBufferPool,
    ctx: &Context,
) {
    let tid = rayon::current_thread_index().unwrap();
    let cols = data.read().unwrap().len();
    for j in 0..cols {
        // update the database at index j
        let tmp = rw(
            &query.op,
            &query.data,
            &data.clone().read().unwrap()[j],
            ctx,
            &mut buf.pool[tid].clone().lock().unwrap(),
        );

        let mut new_data = RLWECiphertext::allocate(ctx.poly_size);
        ls[j].cmux_with_buf(
            &mut new_data,
            &data.clone().read().unwrap()[j],
            &tmp,
            &mut buf.pool[tid].clone().lock().unwrap(),
        ); // 0 -> self.data, 1 -> tmp
        data.clone().write().unwrap()[j].fill_with_copy(&new_data);
    }
}

pub struct Server {
    data: Vec<Arc<RwLock<Vec<RLWECiphertext>>>>,
    neg_s: RGSWCiphertext,
    ctx: Context,
}

impl Server {
    /// Create a new server that stores ciphertexts `data`.
    pub fn new(
        data: Vec<Vec<RLWECiphertext>>,
        neg_s: RGSWCiphertext,
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
    pub fn process_one(&mut self, query: ClientQuery) -> (RLWECiphertext, Duration) {
        let buf = FftBufferPool::new(rayon::max_num_threads(), &self.ctx);
        let start = Instant::now();
        let (y, ls) = process_one_mt(&query, self.data[0].clone(), &self.neg_s, &buf, &self.ctx);
        let dur = start.elapsed();
        update_db_mt(&query, self.data[0].clone(), ls, &buf, &self.ctx);
        (y, dur)
    }

    /// Process multiple access queries in parallel (but without batching).
    pub fn process_multi(&mut self, queries: Vec<ClientQuery>) -> (Vec<RLWECiphertext>, Duration) {
        let buf = FftBufferPool::new(rayon::max_num_threads(), &self.ctx);
        let cols = self.cols();
        // first phase we execute the single query procedure and produce the output
        // (over multiple queries)
        let start = Instant::now();
        let (ys, ls): (Vec<RLWECiphertext>, Vec<Vec<RGSWCiphertext>>) = queries
            .par_iter()
            .map(|query| process_one_st(query, self.data[0].clone(), &self.neg_s, &buf, &self.ctx))
            .unzip();
        let dur = start.elapsed();

        // sum the output of rw*t in parallel (over multiple queries)
        let tmps = queries.into_par_iter().zip(&ls).map(|(query, l)| {
            let tid = rayon::current_thread_index().unwrap();
            let mut out = vec![RLWECiphertext::allocate(self.ctx.poly_size); cols];
            for j in 0..cols {
                let tmp = rw(
                    &query.op,
                    &query.data,
                    &self.data[0].clone().read().unwrap()[j],
                    &self.ctx,
                    &mut buf.pool[tid].clone().lock().unwrap(),
                );
                // L * t
                l[j].external_product_with_buf(
                    &mut out[j],
                    &tmp,
                    &mut buf.pool[tid].clone().lock().unwrap(),
                );
            }
            out
        });

        let sumed_tmps = tmps.reduce(
            || vec![RLWECiphertext::allocate(self.ctx.poly_size); cols],
            |mut xs, ys| {
                for i in 0..xs.len() {
                    xs[i].update_with_add(&ys[i]);
                }
                xs
            },
        );

        let sumed_ls = ls.into_par_iter().reduce(
            || {
                vec![
                    RGSWCiphertext::allocate(
                        self.ctx.poly_size,
                        self.ctx.base_log,
                        self.ctx.level_count
                    );
                    cols
                ]
            },
            |mut xs, ys| {
                for i in 0..xs.len() {
                    xs[i].update_with_add(&ys[i]);
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
                let mut new_data = RLWECiphertext::allocate(self.ctx.poly_size);
                sumed_ls[j].cmux_with_buf(
                    &mut new_data,
                    &self.data[0].clone().read().unwrap()[j],
                    &sumed_tmps[j],
                    &mut buf.pool[tid].clone().lock().unwrap(),
                );
                self.data[0].clone().write().unwrap()[j].fill_with_copy(&new_data);
            });

        (ys, dur)
    }

    /// Process multiple access queries in parallel using the batching technique.
    pub fn process_batch(
        &mut self,
        queries: Vec<ClientQuery>,
        hash: &NaiveHash,
    ) -> (Vec<RLWECiphertext>, Duration) {
        assert_eq!(queries.len(), self.rows());
        let buf = FftBufferPool::new(rayon::max_num_threads(), &self.ctx);
        // first phase we execute the single query procedure and produce the output
        let start = Instant::now();
        let (ys, ls): (Vec<RLWECiphertext>, Vec<Vec<RGSWCiphertext>>) = queries
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
                update_db_st(query, self.data[row].clone(), &l, &buf, &self.ctx);
            });

        // then we iterate over all indices and perform consistency correction
        let rw_flags: Vec<RGSWCiphertext> = queries.into_iter().map(|q| q.into_rw_flag()).collect();
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
    pub fn decrypt_db(&self, sk: &RLWESecretKey) -> Vec<Vec<u64>> {
        self.data
            .iter()
            .map(|row| {
                row.clone()
                    .read()
                    .unwrap()
                    .iter()
                    .map(|v| {
                        let mut pt = self.ctx.gen_zero_pt();
                        sk.decrypt_decode_rlwe(&mut pt, v, &self.ctx);
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
) -> (Client, Server, Vec<PlaintextList<Vec<Scalar>>>) {
    assert_eq!(rows * cols, hash.input_domain() * hash.h_count());
    // rows = 1, cols = n if we're not batching
    let mut client = Client::new(rows, cols, params.clone());
    let neg_sk_ct = client.gen_neg_sk();

    let (plain, data) = client.gen_dummy_database(hash);
    let server = Server::new(data, neg_sk_ct, params);

    (client, server, plain)
}

// (1 - (b_j * c_i1 + b_j * c_i2 + b_j * c_i3)) * v_i1 +
// (b_j * c_i1 * v_i1) + (b_j * c_i2 * v_i2) + (b_j * c_i3 * v_i3)
fn correct_consistency(
    positions: &[(usize, usize)],
    data: &[Arc<RwLock<Vec<RLWECiphertext>>>],
    ls: &[Vec<RGSWCiphertext>],
    rw_flags: &[RGSWCiphertext],
    ctx: &Context,
    buf: &FftBufferPool,
) -> RLWECiphertext {
    let tid = rayon::current_thread_index().unwrap();
    let first_r = positions[0].0;
    let first_c = positions[0].1;
    let v1s: Vec<RLWECiphertext> = positions
        .iter()
        .map(|(r, c)| {
            let mut tmp = RLWECiphertext::allocate(ctx.poly_size);
            let mut out = RLWECiphertext::allocate(ctx.poly_size);
            rw_flags[*r].external_product_with_buf(
                &mut tmp,
                &data[first_r].clone().read().unwrap()[first_c],
                &mut buf.pool[tid].clone().lock().unwrap(),
            );
            ls[*r][*c].external_product_with_buf(
                &mut out,
                &tmp,
                &mut buf.pool[tid].clone().lock().unwrap(),
            );
            out
        })
        .collect();

    // compute (1 - (b_j * c_i1 + b_j * c_i2 + b_j * c_i3 + ...)) * v_i1
    let mut first_term = data[first_r].clone().read().unwrap()[first_c].clone();
    for vs in &v1s {
        first_term.update_with_sub(vs);
    }

    // compute the remaining terms
    // (b_j * c_i1 * v_i1) + (b_j * c_i2 * v_i2) + (b_j * c_i3 * v_i3) + ...
    let second_terms = positions.iter().skip(1).map(|(r, c)| {
        let mut tmp = RLWECiphertext::allocate(ctx.poly_size);
        let mut out = RLWECiphertext::allocate(ctx.poly_size);
        rw_flags[*r].external_product_with_buf(
            &mut tmp,
            &data[*r].clone().read().unwrap()[*c],
            &mut buf.pool[tid].clone().lock().unwrap(),
        );
        ls[*r][*c].external_product_with_buf(
            &mut out,
            &tmp,
            &mut buf.pool[tid].clone().lock().unwrap(),
        );
        out
    });

    let mut second_term = v1s[0].clone();
    for term in second_terms {
        second_term.update_with_add(&term);
    }

    second_term.update_with_add(&first_term);
    second_term
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::rlwe::compute_noise_encoded;

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
            client.sk.decrypt_decode_rlwe(&mut pt, &y, &client.ctx);

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
            let write_query = client.gen_write_query_one(idx, &new_data, &hash);
            server.process_one(write_query);

            let read_query = client.gen_read_query_one(idx, &hash);
            let y = server.process_one(read_query).0;
            let mut pt = client.ctx.gen_zero_pt();
            client.sk.decrypt_decode_rlwe(&mut pt, &y, &client.ctx);

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
            client.sk.decrypt_decode_rlwe(&mut pt, &y, &client.ctx);

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
                client.sk.decrypt_decode_rlwe(&mut pt, &ys[0], &client.ctx);
                assert_eq!(pts[idx1], pt);
                // check noise
                println!(
                    "noise for read: {}",
                    compute_noise_encoded(&client.sk, &ys[0], &pt, &client.ctx.codec)
                );
            }

            {
                let mut pt = client.ctx.gen_zero_pt();
                client.sk.decrypt_decode_rlwe(&mut pt, &ys[1], &client.ctx);
                assert_eq!(pts[idx2], pt);
            }
        }

        {
            // write
            let idx1 = 1usize;
            let idx2 = 2usize;
            let new_data = client.ctx.gen_binary_pt();
            let write_queries = vec![
                client.gen_write_query_one(idx1, &new_data, &hash),
                client.gen_write_query_one(idx2, &new_data, &hash),
            ];
            server.process_multi(write_queries);

            let read_queries = vec![
                client.gen_read_query_one(idx1, &hash),
                client.gen_read_query_one(idx2, &hash),
            ];
            let ys = server.process_multi(read_queries).0;

            {
                let mut pt = client.ctx.gen_zero_pt();
                client.sk.decrypt_decode_rlwe(&mut pt, &ys[0], &client.ctx);
                assert_eq!(new_data, pt);
                // check noise
                println!(
                    "noise for write: {}",
                    compute_noise_encoded(&client.sk, &ys[0], &pt, &client.ctx.codec)
                );
            }

            {
                let mut pt = client.ctx.gen_zero_pt();
                client.sk.decrypt_decode_rlwe(&mut pt, &ys[1], &client.ctx);
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
                client.sk.decrypt_decode_rlwe(&mut pt, &ys[r], &client.ctx);
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
            let write_query = client.gen_write_query_batch(&indices, &new_data, &hash);
            server.process_batch(write_query, &hash);

            let read_query = client.gen_read_query_batch(&indices, &hash);
            let ys = server.process_batch(read_query, &hash).0;
            let mut pt = client.ctx.gen_zero_pt();
            for (r, _) in mapping {
                client.sk.decrypt_decode_rlwe(&mut pt, &ys[r], &client.ctx);
                assert_eq!(new_data, pt);
            }

            // additionally check the database is consistent
            for i in indices {
                for h in 0..h_count {
                    let (r, c) = hash.hash_to_tuple(h, i, cols);
                    client.sk.decrypt_decode_rlwe(
                        &mut pt,
                        &server.data[r].clone().read().unwrap()[c],
                        &client.ctx,
                    );
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
                client.sk.decrypt_decode_rlwe(&mut pt, &ys[r], &client.ctx);
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
