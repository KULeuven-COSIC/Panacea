use crate::{
    context::{Context, FftBuffer},
    decomposer::{collect_next_term, new_signed_decomp_tensor},
    num_types::{
        AlignedScalarContainer, Complex, ComplexBox, ComplexContainer, One, Scalar,
        ScalarContainer, Zero,
    },
    utils::{eval_x_k, log2},
};
use aligned_vec::{avec, CACHELINE_ALIGN};
use concrete_fft::c64;
use rayon::prelude::*;
use tfhe::core_crypto::{
    fft_impl::fft64::math::polynomial::FourierPolynomialMutView,
    prelude::{
        add_external_product_assign_mem_optimized,
        add_external_product_assign_mem_optimized_requirement,
        convert_standard_ggsw_ciphertext_to_fourier_mem_optimized,
        convert_standard_ggsw_ciphertext_to_fourier_mem_optimized_requirement,
        encrypt_glwe_ciphertext, encrypt_glwe_ciphertext_assign,
        par_encrypt_constant_ggsw_ciphertext,
        polynomial_algorithms::{polynomial_wrapping_add_assign, polynomial_wrapping_sub_assign},
        slice_algorithms::slice_wrapping_scalar_mul_assign,
        ComputationBuffers, ContiguousEntityContainer, ContiguousEntityContainerMut, Fft,
        FourierGgswCiphertext, FourierPolynomial, GgswCiphertext, GlweCiphertext, GlweSecretKey,
        Plaintext, PlaintextList, Polynomial, PolynomialSize, PolynomialView, SignedDecomposer,
    },
};

use dyn_stack::{GlobalPodBuffer, PodStack, ReborrowMut};

use std::collections::HashMap;

pub fn external_product(
    rgsw: &GgswCiphertext<ScalarContainer>,
    d: &GlweCiphertext<AlignedScalarContainer>,
    ctx: &Context<Scalar>,
) -> GlweCiphertext<AlignedScalarContainer> {
    let mut out = ctx.empty_glwe_ciphertext();
    external_product_add(&mut out, rgsw, d, ctx);
    out
}

pub fn external_product_add(
    out: &mut GlweCiphertext<AlignedScalarContainer>,
    rgsw: &GgswCiphertext<ScalarContainer>,
    d: &GlweCiphertext<AlignedScalarContainer>,
    ctx: &Context<Scalar>,
) {
    let mut mem = ComputationBuffers::new();
    mem.resize(
        add_external_product_assign_mem_optimized_requirement::<Scalar>(
            ctx.glwe_size,
            ctx.poly_size,
            ctx.fft.as_view(),
        )
        .unwrap()
        .unaligned_bytes_required()
        .max(
            convert_standard_ggsw_ciphertext_to_fourier_mem_optimized_requirement(
                ctx.fft.as_view(),
            )
            .unwrap()
            .unaligned_bytes_required(),
        ),
    );

    let mut stack = mem.stack();
    let mut fourier_ggsw = FourierGgswCiphertext::from_container(
        avec![
            Complex::zero();
            ctx.poly_size.0 / 2
                * rgsw.decomposition_level_count().0
                * ctx.glwe_size.0
                * ctx.glwe_size.0
        ],
        ctx.glwe_size,
        ctx.poly_size,
        rgsw.decomposition_base_log(),
        rgsw.decomposition_level_count(),
    );
    convert_standard_ggsw_ciphertext_to_fourier_mem_optimized(
        rgsw,
        &mut fourier_ggsw,
        ctx.fft.as_view(),
        stack.rb_mut(),
    );

    add_external_product_assign_mem_optimized(
        out,
        &fourier_ggsw,
        d,
        ctx.fft.as_view(),
        stack.rb_mut(),
    );
}

/// Compare a ciphertext ct, which encrypts m on the exponent
/// against a plaintext scalar value d
/// the resulting ciphertext encrypts a polynomial m(X) such that
/// m0 = 1 if m <= d, otherwise m0 = 0, where m0 is the constant term of m(X).
/// Note that encrypting on the exponent means m -> X^m.
pub fn less_eq_than(ct: &mut GlweCiphertext<AlignedScalarContainer>, d: Scalar) {
    let d = d as usize;
    let n = ct.polynomial_size().0;

    assert!(d < n);

    // Todo: Encoding Issue here
    let t_poly = {
        let mut t = vec![Scalar::zero(); n];
        t[0] = Scalar::one();
        for x in t.iter_mut().take(n).skip(n - d) {
            *x = Scalar::MAX; // -1
        }
        Polynomial::from_container(t)
    };
    fourier_update_with_mul(
        &mut ct.get_mut_body().as_mut_polynomial(),
        &t_poly.as_view(),
    );

    fourier_update_with_mul(
        &mut ct.get_mut_mask().as_mut_polynomial_list().get_mut(0),
        &t_poly.as_view(),
    );
}

/// Checks whether this ciphertext ct,
/// which encrypts a value m on the exponent,
/// equals to d.
pub fn eq_to(ct: &mut GlweCiphertext<AlignedScalarContainer>, d: Scalar) {
    let d = d as usize;
    let n = ct.polynomial_size().0;

    assert!(d < n);
    let t_poly = {
        let mut t = vec![Scalar::zero(); n];
        if d == 0 {
            t[0] = Scalar::one();
        } else {
            t[n - d] = Scalar::MAX;
        }
        Polynomial::from_container(t)
    };

    fourier_update_with_mul(
        &mut ct.get_mut_mask().as_mut_polynomial_list().get_mut(0),
        &t_poly.as_view(),
    );
    fourier_update_with_mul(
        &mut ct.get_mut_body().as_mut_polynomial(),
        &t_poly.as_view(),
    );
}

/// Run the not gate on the input ciphertext,
///  the ciphertext must encrypt a binary scalar.
/// If c = (a, b = a s + e + q/2 b), then negating it becomes
/// (-a, q/2 - b) = (-a, -a s - e + q/2 NOT(b))
pub fn not_in_place(ct: &mut GlweCiphertext<AlignedScalarContainer>) {
    let delta = Scalar::one() << (Scalar::BITS - 1);

    for x in ct.as_mut().iter_mut() {
        *x = Scalar::zero().wrapping_sub(*x);
    }

    ct.get_mut_body().as_mut()[0] = ct.get_body().as_ref()[0].wrapping_add(delta);
}

/// Return NOT(ct) where self must encrypt a binary scalar.
pub fn not(ct: &GlweCiphertext<AlignedScalarContainer>) -> GlweCiphertext<AlignedScalarContainer> {
    let delta = Scalar::one() << (Scalar::BITS - 1);
    let mut out = GlweCiphertext::from_container(
        avec![Scalar::zero(); ct.polynomial_size().0 * 2],
        ct.polynomial_size(),
        ct.ciphertext_modulus(),
    );

    polynomial_wrapping_sub_assign(
        &mut out.get_mut_mask().as_mut_polynomial_list().get_mut(0),
        &ct.get_mask().as_polynomial_list().get(0),
    );
    polynomial_wrapping_sub_assign(
        &mut out.get_mut_body().as_mut_polynomial(),
        &ct.get_body().as_polynomial(),
    );

    out.get_mut_body().as_mut()[0] = (out.get_body().as_ref()[0]).wrapping_add(delta);
    out
}

/// Run the `trace1(RLWE(\sum_i` `a_i` X^i)) = `RLWE((1/N)*a_0`) operation on this ciphertext
/// using key switching keys in the fourier domain.
pub fn trace1(
    ct: &GlweCiphertext<AlignedScalarContainer>,
    ksk_map: &HashMap<usize, FourierRLWEKeyswitchKey>,
    ctx: &Context<Scalar>,
) -> GlweCiphertext<AlignedScalarContainer> {
    let n = ct.polynomial_size().0;

    let fft = Fft::new(PolynomialSize(n));
    let fft = fft.as_view();

    let mut mem = GlobalPodBuffer::new(
        fft.forward_scratch()
            .unwrap()
            .and(fft.backward_scratch().unwrap()),
    );
    let mut stack = PodStack::new(&mut mem);

    let mut out = ct.clone();

    for i in 1..=log2(n) {
        let k = n / (1 << (i - 1)) + 1;
        let ksk = ksk_map.get(&k).unwrap();

        assert_eq!(ksk.get_subs_k(), k);
        let buf_fourier = ksk.subs(out.clone(), ctx);

        fft.add_backward_as_torus(
            out.get_mut_mask().as_mut_polynomial_list().get_mut(0),
            buf_fourier.mask.as_view(),
            stack.rb_mut(),
        );

        fft.add_backward_as_torus(
            out.get_mut_body().as_mut_polynomial(),
            buf_fourier.body.as_view(),
            stack.rb_mut(),
        );
    }
    out
}

#[derive(Debug, Clone, PartialEq)]
/// An RLWE ciphertext in the Fourier domain.
pub struct FourierRLWECiphertext {
    pub mask: FourierPolynomial<ComplexContainer>,
    pub body: FourierPolynomial<ComplexContainer>,
}

impl FourierRLWECiphertext {
    /// `poly_size` refers to the number of coefficients required per polynomial in the standard domain
    pub fn new(poly_size: PolynomialSize) -> Self {
        Self {
            mask: FourierPolynomial {
                data: avec![Complex::zero(); poly_size.0 / 2],
            },
            body: FourierPolynomial {
                data: avec![Complex::zero(); poly_size.0 / 2],
            },
        }
    }

    /// Convert the ciphertext back to standard domain.
    pub fn backward_as_torus(
        &mut self,
        ctx: &Context<Scalar>,
    ) -> GlweCiphertext<AlignedScalarContainer> {
        let p = PolynomialSize(self.body.data.len() * 2);
        assert_eq!(p, ctx.poly_size);
        let mut out = ctx.empty_glwe_ciphertext();

        let fft = Fft::new(p);
        let fft = fft.as_view();

        let mut mem = GlobalPodBuffer::new(
            fft.forward_scratch()
                .unwrap()
                .and(fft.backward_scratch().unwrap()),
        );
        let mut stack = PodStack::new(&mut mem);

        fft.add_backward_as_torus(
            out.get_mut_mask().as_mut_polynomial_list().get_mut(0),
            self.mask.as_view(),
            stack.rb_mut(),
        );
        fft.add_backward_as_torus(
            out.get_mut_body().as_mut_polynomial(),
            self.body.as_view(),
            stack.rb_mut(),
        );

        out
    }
}

#[derive(Debug, Clone, PartialEq)]
/// An RLWE key switching key in the Fourier domain.
pub struct FourierRLWEKeyswitchKey {
    ksks: Vec<FourierRLWECiphertext>,
    subs_k: usize,
}

impl FourierRLWEKeyswitchKey {
    pub fn new(
        before_key: GlweSecretKey<ScalarContainer>,
        after_key: &GlweSecretKey<ScalarContainer>,
        ctx: &mut Context<Scalar>,
    ) -> Self {
        let decomp_level_count = ctx.ks_level_count.0;
        let decomp_base_log = ctx.ks_base_log.0;

        let fft = Fft::new(ctx.poly_size);
        let fft = fft.as_view();
        let mut mem = GlobalPodBuffer::new(
            fft.forward_scratch()
                .unwrap()
                .and(fft.backward_scratch().unwrap()),
        );

        let mut stack = PodStack::new(&mut mem);

        let mut prep = ctx.empty_glwe_ciphertext();

        FourierRLWEKeyswitchKey {
            ksks: (1..=decomp_level_count)
                .map(|level| {
                    prep.get_mut_body()
                        .as_mut()
                        .iter_mut()
                        .zip(before_key.as_ref().iter())
                        .for_each(|(prp, bk)| {
                            *prp = *bk << ((Scalar::BITS as usize) - decomp_base_log * level)
                        });

                    encrypt_glwe_ciphertext_assign(
                        after_key,
                        &mut prep,
                        ctx.std,
                        &mut ctx.encryption_generator,
                    );

                    let mut fp_mask = FourierPolynomial {
                        data: avec![Complex::zero(); ctx.poly_size.0 / 2],
                    };
                    let mut fp_body = FourierPolynomial {
                        data: avec![Complex::zero(); ctx.poly_size.0 / 2],
                    };

                    fft.forward_as_torus(
                        fp_mask.as_mut_view(),
                        prep.get_mask().as_polynomial_list().get(0),
                        stack.rb_mut(),
                    );
                    fft.forward_as_torus(
                        fp_body.as_mut_view(),
                        prep.get_body().as_polynomial(),
                        stack.rb_mut(),
                    );

                    FourierRLWECiphertext {
                        mask: fp_mask,
                        body: fp_body,
                    }
                })
                .collect(),

            subs_k: 0,
        }
    }

    pub fn new_subs(
        after_key: &GlweSecretKey<ScalarContainer>,
        k: usize,
        ctx: &mut Context<Scalar>,
    ) -> Self {
        let before_key =
            GlweSecretKey::from_container(eval_x_k(after_key.as_ref(), k).to_vec(), ctx.poly_size);

        let mut fourier_ksk = Self::new(before_key, after_key, ctx);
        fourier_ksk.subs_k = k;
        fourier_ksk
    }

    // }
    /// Perform key switching but don't convert the new ciphertext to the standard domain.
    pub fn keyswitch_ciphertext(
        &self,
        before: GlweCiphertext<AlignedScalarContainer>,
        ctx: &Context<Scalar>,
    ) -> FourierRLWECiphertext {
        let fft = Fft::new(ctx.poly_size);
        let fft = fft.as_view();

        let fourier_size = ctx.poly_size.to_fourier_polynomial_size().0;
        let mut mem = GlobalPodBuffer::new(
            fft.forward_scratch()
                .unwrap()
                .and(fft.forward_scratch().unwrap())
                .and(fft.forward_scratch().unwrap())
                .and(fft.forward_scratch().unwrap())
                .and(fft.backward_scratch().unwrap()),
        );

        let mut stack = PodStack::new(&mut mem);

        let mut after = FourierRLWECiphertext::new(ctx.poly_size);

        let mut first_fourier = FourierPolynomial {
            data: avec![Complex::zero(); fourier_size],
        };
        fft.forward_as_torus(
            first_fourier.as_mut_view(),
            before.get_body().as_polynomial(),
            stack.rb_mut(),
        );
        // clean the output ctxt and add c_1
        for (c, b) in izip!(&mut *after.body.data, &*first_fourier.data) {
            *c = *b;
        }

        let decomposer = SignedDecomposer::<Scalar>::new(ctx.ks_base_log, ctx.ks_level_count);

        let (mut decomposition, mut substack1) = new_signed_decomp_tensor(
            before
                .get_mask()
                .as_ref()
                .iter()
                .map(|s| decomposer.closest_representable(*s)),
            decomposer.base_log(),
            decomposer.level_count(),
            stack.rb_mut(),
        );

        let mut ksk_iter = self.ksks.iter().rev();

        loop {
            match (ksk_iter.next(), ksk_iter.next()) {
                (Some(first), Some(second)) => {
                    let (_, mut term1, mut substack2) =
                        collect_next_term(&mut decomposition, &mut substack1, CACHELINE_ALIGN);

                    term1
                        .par_iter_mut()
                        .for_each(|x| *x = Scalar::zero().wrapping_sub(*x));

                    let (_, mut term2, mut substack3) =
                        collect_next_term(&mut decomposition, &mut substack2, CACHELINE_ALIGN);

                    term2
                        .par_iter_mut()
                        .for_each(|x| *x = Scalar::zero().wrapping_sub(*x));

                    let (mut first_fourier, mut substack4) = substack3
                        .rb_mut()
                        .make_aligned_raw::<c64>(fourier_size, CACHELINE_ALIGN);

                    let (mut second_fourier, mut substack5) = substack4
                        .rb_mut()
                        .make_aligned_raw::<c64>(fourier_size, CACHELINE_ALIGN);

                    let first_fourier = fft.forward_as_integer(
                        FourierPolynomialMutView {
                            data: &mut first_fourier,
                        },
                        PolynomialView::from_container(&term1),
                        substack5.rb_mut(),
                    );

                    let second_fourier = fft.forward_as_integer(
                        FourierPolynomialMutView {
                            data: &mut second_fourier,
                        },
                        PolynomialView::from_container(&term2),
                        substack5.rb_mut(),
                    );

                    pre_fourier_update_with_two_multiply_accumulate(
                        &mut after.mask.as_mut_view(),
                        &first.mask.as_view(),
                        &first_fourier.as_view(),
                        &second.mask.as_view(),
                        &second_fourier.as_view(),
                    );

                    pre_fourier_update_with_two_multiply_accumulate(
                        &mut after.body.as_mut_view(),
                        &first.body.as_view(),
                        &first_fourier.as_view(),
                        &second.body.as_view(),
                        &second_fourier.as_view(),
                    );
                }
                (Some(first), None) => {
                    let (_, mut term1, mut substack2) =
                        collect_next_term(&mut decomposition, &mut substack1, CACHELINE_ALIGN);

                    term1
                        .par_iter_mut()
                        .for_each(|x| *x = Scalar::zero().wrapping_sub(*x));

                    let (mut first_fourier, mut substack3) = substack2
                        .rb_mut()
                        .make_aligned_raw::<c64>(fourier_size, CACHELINE_ALIGN);

                    let first_fourier = fft.forward_as_integer(
                        FourierPolynomialMutView {
                            data: &mut first_fourier,
                        },
                        PolynomialView::from_container(&term1),
                        substack3.rb_mut(),
                    );

                    pre_fourier_update_with_multiply_accumulate(
                        &mut after.mask.as_mut_view(),
                        &first.mask.as_view(),
                        &first_fourier.as_view(),
                    );

                    pre_fourier_update_with_multiply_accumulate(
                        &mut after.body.as_mut_view(),
                        &first.body.as_view(),
                        &first_fourier.as_view(),
                    );
                }
                _ => break,
            }
        }
        after
    }

    /// Perform the substitution operation that converts RLWE(p(X)) to RLWE(p(X^k)).
    /// The key switching key must be of the form s(X^k) to s(X).
    pub fn subs(
        &self,
        before: GlweCiphertext<AlignedScalarContainer>,
        ctx: &Context<Scalar>,
    ) -> FourierRLWECiphertext {
        let k = self.subs_k;

        let mask_cont: AlignedScalarContainer =
            eval_x_k(before.get_mask().as_polynomial_list().get(0).as_ref(), k);
        let body_cont: AlignedScalarContainer = eval_x_k(before.get_body().as_ref(), k);

        self.keyswitch_ciphertext(ctx.glwe_ciphertext_from(mask_cont, body_cont), ctx)
    }

    pub const fn get_subs_k(&self) -> usize {
        self.subs_k
    }
}

/// Generate all the key switching keys needed for the substitution operation
/// in the Fourier domain.
pub fn gen_all_subs_ksk(
    after_key: &GlweSecretKey<ScalarContainer>,
    ctx: &mut Context<Scalar>,
) -> HashMap<usize, FourierRLWEKeyswitchKey> {
    let poly_size = ctx.poly_size;
    let mut hm = HashMap::new();

    for i in 1..=log2(poly_size.0) {
        let k = poly_size.0 / (1 << (i - 1)) + 1;
        let ksk = FourierRLWEKeyswitchKey::new_subs(after_key, k, ctx);
        hm.insert(k, ksk);
    }
    hm
}

/// Expand/convert RLWE ciphertexts to an RGSW ciphertext.
#[allow(clippy::ptr_arg)]
pub fn expand(
    cs: &Vec<GlweCiphertext<AlignedScalarContainer>>,
    ksk_map: &HashMap<usize, FourierRLWEKeyswitchKey>,
    neg_s: &FourierGgswCiphertext<ComplexBox>,
    ctx: &Context<Scalar>,
    buf: &mut FftBuffer,
) -> FourierGgswCiphertext<ComplexBox> {
    let mut out = GgswCiphertext::<ScalarContainer>::new(
        Scalar::zero(),
        ctx.glwe_size,
        ctx.poly_size,
        ctx.base_log,
        ctx.level_count,
        ctx.ciphertext_modulus,
    );
    let mut c_prime = ctx.empty_glwe_ciphertext();

    for (i, mut c) in out.as_mut_glwe_list().iter_mut().enumerate() {
        let k = i / 2;
        if i % 2 == 0 {
            c_prime = trace1(&cs[k], ksk_map, ctx);

            add_external_product_assign_mem_optimized(
                &mut c,
                neg_s,
                &c_prime,
                ctx.fft.as_view(),
                buf.mem.stack().rb_mut(),
            );
        } else {
            c.as_mut()
                .iter_mut()
                .zip(c_prime.as_ref().iter())
                .for_each(|(c, cp)| *c = *cp);
        }
    }
    convert_standard_ggsw_to_fourier(out, ctx, buf)
}

pub fn decomposed_rlwe_to_rgsw(
    cs: &[GlweCiphertext<AlignedScalarContainer>],
    neg_s: &FourierGgswCiphertext<ComplexBox>,
    ctx: &Context<Scalar>,
    buf: &mut FftBuffer,
) -> FourierGgswCiphertext<ComplexBox> {
    let mut out = GgswCiphertext::new(
        Scalar::zero(),
        ctx.glwe_size,
        ctx.poly_size,
        ctx.base_log,
        ctx.level_count,
        ctx.ciphertext_modulus,
    );

    for (i, mut c) in out.as_mut_glwe_list().iter_mut().enumerate() {
        let k = i / 2;
        if i % 2 == 0 {
            add_external_product_assign_mem_optimized(
                &mut c,
                neg_s,
                &cs[k],
                ctx.fft.as_view(),
                buf.mem.stack().rb_mut(),
            );
        } else {
            c.as_mut()
                .iter_mut()
                .zip(cs[k].as_ref().iter())
                .for_each(|(c, cp)| *c = *cp);
        }
    }
    convert_standard_ggsw_to_fourier(out, ctx, buf)
}

fn fourier_update_with_mul(p1: &mut Polynomial<&mut [Scalar]>, p2: &Polynomial<&[Scalar]>) {
    let fft = Fft::new(p1.polynomial_size());
    let fft = fft.as_view();

    let mut mem = ComputationBuffers::new();
    mem.resize(
        fft.forward_scratch()
            .unwrap()
            .and(fft.backward_scratch().unwrap())
            .unaligned_bytes_required(),
    );

    let mut stack = mem.stack();

    let mut fp1 = FourierPolynomial {
        data: vec![Complex::zero(); p1.polynomial_size().0 / 2],
    };
    let mut fp2 = FourierPolynomial {
        data: vec![Complex::zero(); p2.polynomial_size().0 / 2],
    };

    fft.forward_as_torus(fp1.as_mut_view(), p1.as_view(), stack.rb_mut());
    fft.forward_as_integer(fp2.as_mut_view(), p2.as_view(), stack.rb_mut());

    for (f0, f1) in izip!(&mut *fp1.data, &*fp2.data) {
        *f0 *= *f1;
    }

    p1.as_mut_view().iter_mut().for_each(|x| *x = 0);

    fft.backward_as_torus(p1.as_mut_view(), fp1.as_view(), stack.rb_mut());
}

fn pre_fourier_update_with_multiply_accumulate(
    out: &mut FourierPolynomial<&mut [Complex]>,
    fp1: &FourierPolynomial<&[Complex]>,
    fp2: &FourierPolynomial<&[Complex]>,
) {
    for (o, f1, f2) in izip!(&mut *out.data, fp1.data, fp2.data) {
        *o += *f1 * *f2;
    }
}

fn pre_fourier_update_with_two_multiply_accumulate(
    out: &mut FourierPolynomial<&mut [Complex]>,
    fp1: &FourierPolynomial<&[Complex]>,
    fp2: &FourierPolynomial<&[Complex]>,
    fp3: &FourierPolynomial<&[Complex]>,
    fp4: &FourierPolynomial<&[Complex]>,
) {
    for (o, f1, f2, f3, f4) in izip!(&mut *out.data, fp1.data, fp2.data, fp3.data, fp4.data) {
        *o += (*f1 * *f2) + (*f3 * *f4);
    }
}

/// Create RLWE ciphertexts that are suitable to be used by expand.
pub fn make_decomposed_rlwe_ct(
    sk: &GlweSecretKey<ScalarContainer>,
    bit: Scalar,
    ctx: &mut Context<Scalar>,
) -> Vec<GlweCiphertext<AlignedScalarContainer>> {
    assert!(bit == Scalar::one() || bit == Scalar::zero());
    let logn = log2(ctx.poly_size.0);
    let out = (1..=ctx.level_count.0).map(|level| {
        assert!(ctx.base_log.0 * level + logn <= Scalar::BITS as usize);
        let shift: usize = (Scalar::BITS as usize) - ctx.base_log.0 * level - logn;
        let ptxt = {
            let mut p = ctx.gen_ternary_ptxt();
            (*p.as_mut_polynomial().as_mut())[0] = bit << shift;
            p
        };
        let mut ct = ctx.empty_glwe_ciphertext();
        encrypt_glwe_ciphertext(sk, &mut ct, &ptxt, ctx.std, &mut ctx.encryption_generator);
        ct
    });
    out.collect()
}

/// Compute RGSW(-sk), where sk is a GlweSecretKey
pub fn convert_standard_ggsw_to_fourier(
    ggsw: GgswCiphertext<ScalarContainer>,
    ctx: &Context<Scalar>,
    buf: &mut FftBuffer,
) -> FourierGgswCiphertext<ComplexBox> {
    let mut fourier_ggsw = FourierGgswCiphertext::new(
        ctx.glwe_size,
        ctx.poly_size,
        ctx.negs_base_log,
        ctx.negs_level_count,
    );

    convert_standard_ggsw_ciphertext_to_fourier_mem_optimized(
        &ggsw,
        &mut fourier_ggsw,
        ctx.fft.as_view(),
        buf.mem.stack().rb_mut(),
    );

    fourier_ggsw
}

/// Compute RGSW(-sk) in the standard domain, where sk is a `GlweSecretKey`
pub fn neg_gsw_std(
    sk: &GlweSecretKey<ScalarContainer>,
    ctx: &mut Context<Scalar>,
) -> GgswCiphertext<ScalarContainer> {
    let neg_sk = PlaintextList::<ScalarContainer>::from_container(
        sk.as_ref()
            .into_par_iter()
            .map(|x| x.wrapping_mul(Scalar::MAX))
            .collect(),
    );

    let mut neg_sk_ct = GgswCiphertext::new(
        Scalar::zero(),
        ctx.glwe_size,
        ctx.poly_size,
        ctx.negs_base_log,
        ctx.negs_level_count,
        ctx.ciphertext_modulus,
    );

    encrypt_as_ggsw(sk, &mut neg_sk_ct, &neg_sk, ctx);

    neg_sk_ct
}

pub fn encrypt_as_ggsw(
    sk: &GlweSecretKey<ScalarContainer>,
    out: &mut GgswCiphertext<ScalarContainer>,
    pt: &PlaintextList<ScalarContainer>,
    ctx: &mut Context<Scalar>,
) {
    par_encrypt_constant_ggsw_ciphertext(
        sk,
        out,
        Plaintext(Scalar::zero()),
        ctx.std,
        &mut ctx.encryption_generator,
    );

    let mut buffer = PlaintextList::new(Scalar::zero(), ctx.plaintext_count());
    for (i, mut m) in out.as_mut_glwe_list().iter_mut().enumerate() {
        let level = (i / 2) + 1;
        let shift: usize = (Scalar::BITS as usize) - ctx.base_log.0 * level;
        buffer.clone_from(pt);
        slice_wrapping_scalar_mul_assign(buffer.as_mut(), 1 << shift);

        if i % 2 == 0 {
            polynomial_wrapping_add_assign(
                &mut m
                    .get_mut_mask()
                    .as_mut_polynomial_list()
                    .get_mut(0)
                    .as_mut_view(),
                &buffer.as_polynomial().as_view(),
            );
        } else {
            polynomial_wrapping_add_assign(
                &mut m.get_mut_body().as_mut_polynomial().as_mut_view(),
                &buffer.as_polynomial().as_view(),
            );
        }
    }
}
#[cfg(test)]
mod test {

    use crate::{
        codec::Codec,
        params::TFHEParameters,
        utils::{compute_noise, compute_noise_encoded, compute_noise_ternary},
    };

    use super::*;
    use tfhe::core_crypto::{
        algorithms::slice_algorithms::slice_wrapping_add_scalar_mul_assign,
        prelude::{
            decrypt_glwe_ciphertext, glwe_ciphertext_add_assign, DispersionParameter,
            LogStandardDev,
        },
    };

    #[test]
    fn test_keyswitching() {
        let mut ctx = Context::new(TFHEParameters::default());
        let mut messages = ctx.gen_ternary_ptxt();
        let messages_clone = messages.clone();

        let sk_after = GlweSecretKey::generate_new_binary(
            ctx.glwe_dimension,
            ctx.poly_size,
            &mut ctx.secret_generator,
        );
        let sk_before = GlweSecretKey::generate_new_binary(
            ctx.glwe_dimension,
            ctx.poly_size,
            &mut ctx.secret_generator,
        );

        let mut ct_before = ctx.empty_glwe_ciphertext();

        let ksk_fourier = FourierRLWEKeyswitchKey::new(sk_before.clone(), &sk_after, &mut ctx);

        // encrypts with the before key our messages
        ctx.codec
            .poly_ternary_encode(&mut messages.as_mut_polynomial());
        encrypt_glwe_ciphertext(
            &sk_before,
            &mut ct_before,
            &messages,
            ctx.std,
            &mut ctx.encryption_generator,
        );
        // println!("msg before: {:?}", messages.as_tensor());
        let mut dec_messages_1 = PlaintextList::new(Scalar::zero(), ctx.plaintext_count());
        decrypt_glwe_ciphertext(&sk_before, &ct_before, &mut dec_messages_1);
        ctx.codec
            .poly_ternary_decode(&mut dec_messages_1.as_mut_polynomial());
        // println!("msg after dec: {:?}", dec_messages_1.as_tensor());
        println!(
            "initial noise: {:?}",
            compute_noise_ternary(&sk_before, &ct_before, &messages.clone(), &ctx)
        );

        let mut ct_after_fourier = ksk_fourier.keyswitch_ciphertext(ct_before, &ctx);
        let ct_after = ct_after_fourier.backward_as_torus(&ctx);

        let mut dec_messages_2 = PlaintextList::new(Scalar::zero(), ctx.plaintext_count());

        decrypt_glwe_ciphertext(&sk_after, &ct_after, &mut dec_messages_2);
        ctx.codec
            .poly_ternary_decode(&mut dec_messages_2.as_mut_polynomial());

        // println!("msg after ks: {:?}", dec_messages_2.as_tensor());

        assert_eq!(dec_messages_1, dec_messages_2);
        assert_eq!(dec_messages_1, messages_clone);
        println!(
            "final noise: {:?}",
            compute_noise_ternary(&sk_after, &ct_after, &messages, &ctx)
        );
    }

    #[test]
    fn test_subs() {
        let mut ctx = Context::new(TFHEParameters::default());
        let k = ctx.poly_size.0 + 1;
        let mut messages = ctx.gen_ternary_ptxt();
        let expected = eval_x_k(messages.as_ref(), k);

        let sk_after = GlweSecretKey::generate_new_binary(
            ctx.glwe_dimension,
            ctx.poly_size,
            &mut ctx.secret_generator,
        );

        let mut ct_before = ctx.empty_glwe_ciphertext();

        let ksk_fourier = FourierRLWEKeyswitchKey::new_subs(&sk_after, k, &mut ctx);

        // encrypt the message using the after key, put it in ct_before
        ctx.codec
            .poly_ternary_encode(&mut messages.as_mut_polynomial());
        encrypt_glwe_ciphertext(
            &sk_after,
            &mut ct_before,
            &messages,
            ctx.std,
            &mut ctx.encryption_generator,
        );
        let mut ct_after_fourier = ksk_fourier.subs(ct_before, &ctx);
        let ct_after = ct_after_fourier.backward_as_torus(&ctx);

        let mut decrypted = PlaintextList::new(Scalar::zero(), ctx.plaintext_count());

        decrypt_glwe_ciphertext(&sk_after, &ct_after, &mut decrypted);
        ctx.codec
            .poly_ternary_decode(&mut decrypted.as_mut_polynomial());

        println!("msg after ks: {:?}", decrypted.as_view());
        println!("expected msg: {:?}", expected);
        assert_eq!(decrypted.into_container(), expected.to_vec());
    }

    #[test]
    fn test_eval_poly() {
        let neg_one = Scalar::MAX;
        let neg_two = neg_one - 1;
        let neg_three = neg_one - 2;
        {
            let poly = eval_x_k(&[0, 1, 2, 3], 3);
            let expected = avec![0, 3, neg_two, 1];
            assert_eq!(poly, expected);
        }
        {
            let poly = eval_x_k(&[0, 1, 2, 3], 5);
            let expected = avec![0, neg_one, 2, neg_three];
            assert_eq!(poly, expected);
        }
    }

    #[test]
    fn test_trace1() {
        let mut ctx = Context::new(TFHEParameters::default());

        let orig_msg = ctx.gen_binary_pt();
        // println!("ptxt before: {:?}", orig_msg);
        let mut encoded_msg = orig_msg.clone();
        ctx.codec.poly_encode(&mut encoded_msg.as_mut_polynomial());
        // we need to divide the encoded message by n, because n is multiplied into the trace output
        for coeff in encoded_msg.as_mut_polynomial().iter_mut() {
            *coeff /= ctx.poly_size.0 as Scalar;
        }

        let sk = GlweSecretKey::generate_new_binary(
            ctx.glwe_dimension,
            ctx.poly_size,
            &mut ctx.secret_generator,
        );

        let mut ct = ctx.empty_glwe_ciphertext();

        let all_ksk = gen_all_subs_ksk(&sk, &mut ctx);
        encrypt_glwe_ciphertext(
            &sk,
            &mut ct,
            &encoded_msg,
            ctx.std,
            &mut ctx.encryption_generator,
        );

        println!(
            "initial noise: {:?}",
            compute_noise(&sk, &ct.as_view(), &encoded_msg)
        );

        let out = trace1(&ct, &all_ksk, &ctx);

        let mut decrypted = PlaintextList::new(Scalar::zero(), ctx.plaintext_count());

        decrypt_glwe_ciphertext(&sk, &out, &mut decrypted);
        ctx.codec.poly_decode(&mut decrypted.as_mut_polynomial());

        // println!("ptxt after: {:?}", decrypted);

        let expected = {
            let mut tmp = PlaintextList::new(Scalar::zero(), ctx.plaintext_count());
            (*tmp.as_mut_polynomial().as_mut())[0] = (*orig_msg.as_polynomial().as_ref())[0];
            tmp
        };
        println!(
            "final noise: {:?}",
            compute_noise_encoded(&sk, &out, &expected, &ctx.codec)
        );
        assert_eq!(decrypted, expected);
    }

    #[test]
    fn test_binary_enc() {
        let mut ctx = Context::new(TFHEParameters::default());
        let ptxt_expected = ctx.gen_binary_pt();
        let mut ptxt_expected_encoded = ptxt_expected.clone();

        let sk = GlweSecretKey::generate_new_binary(
            ctx.glwe_dimension,
            ctx.poly_size,
            &mut ctx.secret_generator,
        );
        let mut ct = ctx.empty_glwe_ciphertext();

        ctx.codec
            .poly_encode(&mut ptxt_expected_encoded.as_mut_polynomial());
        encrypt_glwe_ciphertext(
            &sk,
            &mut ct,
            &ptxt_expected_encoded,
            ctx.std,
            &mut ctx.encryption_generator,
        );

        let mut ptxt_actual = PlaintextList::new(Scalar::zero(), ctx.plaintext_count());

        decrypt_glwe_ciphertext(&sk, &ct, &mut ptxt_actual);
        ctx.codec.poly_decode(&mut ptxt_actual.as_mut_polynomial());

        assert_eq!(ptxt_actual, ptxt_expected);
    }

    #[test]
    fn test_ternary_enc() {
        let mut ctx = Context::new(TFHEParameters::default());
        let mut ptxt_expected = ctx.gen_ternary_ptxt();

        println!("{:?}", ptxt_expected.clone());

        let sk = GlweSecretKey::generate_new_binary(
            ctx.glwe_dimension,
            ctx.poly_size,
            &mut ctx.secret_generator,
        );

        let mut ct = GlweCiphertext::new(
            Scalar::zero(),
            ctx.glwe_size,
            ctx.poly_size,
            ctx.ciphertext_modulus,
        );
        encrypt_glwe_ciphertext(
            &sk,
            &mut ct,
            &ptxt_expected,
            ctx.std,
            &mut ctx.encryption_generator,
        );

        let mut ptxt_actual = PlaintextList::new(Scalar::zero(), ctx.plaintext_count());

        decrypt_glwe_ciphertext(&sk, &ct, &mut ptxt_actual);
        ctx.codec
            .poly_ternary_decode(&mut ptxt_actual.as_mut_polynomial());
        ctx.codec
            .poly_ternary_decode(&mut ptxt_expected.as_mut_polynomial());

        assert_eq!(ptxt_actual, ptxt_expected);
    }

    #[test]
    fn test_encrypt_rgsw() {
        let mut ctx = Context::new(TFHEParameters::default());
        let mut one_pt = ctx.gen_unit_pt();

        let sk = GlweSecretKey::generate_new_binary(
            ctx.glwe_dimension,
            ctx.poly_size,
            &mut ctx.secret_generator,
        );

        let mut lwe_ct = ctx.empty_glwe_ciphertext();

        ctx.codec.poly_encode(&mut one_pt.as_mut_polynomial());
        encrypt_glwe_ciphertext(
            &sk,
            &mut lwe_ct,
            &one_pt,
            ctx.std,
            &mut ctx.encryption_generator,
        );
        ctx.codec.poly_decode(&mut one_pt.as_mut_polynomial());

        println!(
            "initial noise: {:?}",
            compute_noise_encoded(&sk, &lwe_ct, &one_pt, &ctx.codec)
        );

        let gsw_pt = ctx.gen_binary_pt();
        let mut gsw_ct = GgswCiphertext::new(
            Scalar::zero(),
            ctx.glwe_size,
            ctx.poly_size,
            ctx.base_log,
            ctx.level_count,
            ctx.ciphertext_modulus,
        );
        encrypt_as_ggsw(&sk, &mut gsw_ct, &gsw_pt, &mut ctx);

        {
            // check the first row of the RGSW ciphertext
            // the first row should have the form (a + m*(q/B), a*s + e),
            // so we subtract m*(q/B) and then check the noise
            let mut pt = gsw_pt.clone();
            let shift: usize = (Scalar::BITS as usize) - ctx.base_log.0;
            slice_wrapping_scalar_mul_assign(pt.as_mut(), 1 << shift);
            let mut gsw_clone = gsw_ct.clone();

            polynomial_wrapping_sub_assign(
                &mut gsw_clone
                    .as_mut_glwe_list()
                    .iter_mut()
                    .next()
                    .unwrap()
                    .get_mut_mask()
                    .as_mut_polynomial_list()
                    .get_mut(0),
                &pt.as_polynomial(),
            );

            println!(
                "first row noise: {:?}",
                compute_noise(
                    &sk,
                    &gsw_clone
                        .as_mut_glwe_list()
                        .iter()
                        .next()
                        .unwrap()
                        .as_view(),
                    &ctx.gen_zero_pt()
                )
            );
        }

        {
            // check the second row of the RGSW ciphertext

            let mut pt = gsw_pt.clone();

            let shift: usize = (Scalar::BITS as usize) - ctx.base_log.0;

            slice_wrapping_scalar_mul_assign(pt.as_mut(), 1 << shift);
            println!(
                "second row noise: {:?}",
                compute_noise(
                    &sk,
                    &gsw_ct
                        .as_mut_glwe_list()
                        .iter_mut()
                        .nth(1)
                        .unwrap()
                        .as_view(),
                    &pt
                )
            );
        }
        {
            // check the last row of the RGSW ciphertext
            let mut pt = gsw_pt.clone();
            let shift: usize = (Scalar::BITS as usize) - ctx.base_log.0 * ctx.level_count.0;
            slice_wrapping_scalar_mul_assign(pt.as_mut(), 1 << shift);
            println!(
                "last row noise: {:?}",
                compute_noise(
                    &sk,
                    &gsw_ct
                        .as_glwe_list()
                        .iter()
                        .nth(gsw_ct.decomposition_level_count().0 * 2 - 1)
                        .unwrap()
                        .as_view(),
                    &pt
                )
            );
        }

        let prod_ct = external_product(&gsw_ct, &lwe_ct, &ctx);

        let mut actual_pt = PlaintextList::new(Scalar::zero(), ctx.plaintext_count());
        decrypt_glwe_ciphertext(&sk, &prod_ct, &mut actual_pt);
        ctx.codec.poly_decode(&mut actual_pt.as_mut_polynomial());

        println!(
            "final noise: {:?}",
            compute_noise_encoded(&sk, &prod_ct, &gsw_pt, &ctx.codec)
        );
        assert_eq!(actual_pt, gsw_pt);
    }

    #[test]
    fn test_negs() {
        let mut ctx = Context::new(TFHEParameters::default());
        let mut buf = ctx.gen_fft_ctx();
        // we use another noise
        // so that the initial rlwe ciphertext has noise of ~28 bits,
        // which is the final noise of running trace
        let mut ctx_noisy = Context {
            std: LogStandardDev(-37.5),
            ..Context::new(TFHEParameters::default())
        };

        let sk = GlweSecretKey::generate_new_binary(
            ctx.glwe_dimension,
            ctx.poly_size,
            &mut ctx.secret_generator,
        );
        let mut neg_sk0 = PlaintextList::new(Scalar::zero(), ctx.plaintext_count());
        slice_wrapping_add_scalar_mul_assign(neg_sk0.as_mut(), sk.as_ref(), Scalar::MAX);

        let binding = neg_gsw_std(&sk, &mut ctx);
        let neg_sk = binding.as_glwe_list();

        // check noise of some rows
        {
            let row_ct = neg_sk.iter().last().unwrap();
            let mut row_pt = neg_sk0.clone();
            let shift: usize =
                (Scalar::BITS as usize) - ctx.negs_base_log.0 * ctx.negs_level_count.0;
            slice_wrapping_scalar_mul_assign(row_pt.as_mut(), 1 << shift);
            println!("last row noise: {:?}", compute_noise(&sk, &row_ct, &row_pt));
        }
        {
            let row_ct = neg_sk.iter().nth(1).unwrap();
            let mut row_pt = neg_sk0.clone();
            let shift: usize = (Scalar::BITS as usize) - ctx.negs_base_log.0;
            slice_wrapping_scalar_mul_assign(row_pt.as_mut(), 1 << shift);

            println!(
                "second row noise: {:?}",
                compute_noise(&sk, &row_ct, &row_pt)
            );
        }

        let mut one_pt = ctx.gen_unit_pt();
        let mut ct_lwe = ctx.empty_glwe_ciphertext();
        ctx_noisy
            .codec
            .poly_ternary_encode(&mut one_pt.as_mut_polynomial());
        encrypt_glwe_ciphertext(
            &sk,
            &mut ct_lwe,
            &one_pt,
            ctx_noisy.std,
            &mut ctx_noisy.encryption_generator,
        );
        println!(
            "initial noise: {:?}",
            compute_noise_ternary(&sk, &ct_lwe, &one_pt, &ctx)
        );

        let mut ct_prod = ctx.empty_glwe_ciphertext();

        let neg_sk_gsw =
            convert_standard_ggsw_to_fourier(neg_gsw_std(&sk, &mut ctx), &ctx, &mut buf);

        add_external_product_assign_mem_optimized(
            &mut ct_prod,
            &neg_sk_gsw,
            &ct_lwe,
            ctx.fft.as_view(),
            buf.mem.stack().rb_mut(),
        );

        let mut actual = PlaintextList::new(Scalar::zero(), ctx.plaintext_count());

        decrypt_glwe_ciphertext(&sk, &ct_prod, &mut actual);
        ctx.codec
            .poly_ternary_decode(&mut actual.as_mut_polynomial());

        assert_eq!(actual, neg_sk0);
        println!(
            "final noise: {:?}",
            compute_noise_ternary(&sk, &ct_prod, &neg_sk0, &ctx)
        );
    }

    #[test]
    fn test_expand() {
        let mut ctx = Context::new(TFHEParameters::default());
        let mut buf = ctx.gen_fft_ctx();

        let sk = GlweSecretKey::generate_new_binary(
            ctx.glwe_dimension,
            ctx.poly_size,
            &mut ctx.secret_generator,
        );

        let neg_sk_ct =
            convert_standard_ggsw_to_fourier(neg_gsw_std(&sk, &mut ctx), &ctx, &mut buf);
        let ksk_map = gen_all_subs_ksk(&sk, &mut ctx);

        let mut test_pt = ctx.gen_binary_pt();
        let test_pt_clone = test_pt.clone();
        let mut test_ct = GlweCiphertext::from_container(
            avec![Scalar::zero(); ctx.poly_size.0 * 2],
            ctx.poly_size,
            ctx.ciphertext_modulus,
        );

        ctx.codec.poly_encode(&mut test_pt.as_mut_polynomial());
        encrypt_glwe_ciphertext(
            &sk,
            &mut test_ct,
            &test_pt,
            ctx.std,
            &mut ctx.encryption_generator,
        );

        {
            let zero_cts = make_decomposed_rlwe_ct(&sk, Scalar::one(), &mut ctx);
            let gsw_ct = expand(&zero_cts, &ksk_map, &neg_sk_ct, &ctx, &mut buf); // this should be 1

            // println!(
            //     "average row noise: {:?}",
            //     compute_noise_rgsw1(&sk, &gsw_ct, &ctx)
            // );

            // decrypt and compare
            let mut lwe_ct = ctx.empty_glwe_ciphertext();
            add_external_product_assign_mem_optimized(
                &mut lwe_ct,
                &gsw_ct,
                &test_ct,
                ctx.fft.as_view(),
                buf.mem.stack().rb_mut(),
            );

            let mut pt = PlaintextList::new(Scalar::zero(), ctx.plaintext_count());

            decrypt_glwe_ciphertext(&sk, &lwe_ct, &mut pt);
            ctx.codec.poly_decode(&mut pt.as_mut_polynomial());
            assert_eq!(test_pt_clone, pt);
            println!(
                "final noise: {:?}",
                compute_noise_encoded(&sk, &lwe_ct, &test_pt_clone, &ctx.codec)
            );
        }
        {
            let zero_cts = make_decomposed_rlwe_ct(&sk, Scalar::zero(), &mut ctx);
            let gsw_ct = expand(&zero_cts, &ksk_map, &neg_sk_ct, &ctx, &mut buf);

            // decrypt and compare
            // let lwe_ct = external_product(&gsw_ct, &test_ct, &ctx);
            let mut lwe_ct = ctx.empty_glwe_ciphertext();
            add_external_product_assign_mem_optimized(
                &mut lwe_ct,
                &gsw_ct,
                &test_ct,
                ctx.fft.as_view(),
                buf.mem.stack().rb_mut(),
            );

            let mut pt = PlaintextList::new(Scalar::zero(), ctx.plaintext_count());

            decrypt_glwe_ciphertext(&sk, &lwe_ct, &mut pt);
            ctx.codec.poly_decode(&mut pt.as_mut_polynomial());
            let zero_pt = PlaintextList::new(Scalar::zero(), ctx.plaintext_count());
            assert_eq!(zero_pt, pt);
            println!(
                "final noise: {:?}",
                compute_noise_encoded(&sk, &lwe_ct, &zero_pt, &ctx.codec)
            );
        }
    }

    #[test]
    fn test_less_eq() {
        let mut ctx = Context::new(TFHEParameters::default());
        let sk = GlweSecretKey::generate_new_binary(
            ctx.glwe_dimension,
            ctx.poly_size,
            &mut ctx.secret_generator,
        );

        let m = ctx.poly_size.0 / 2;
        let mut encoded_ptxt = PlaintextList::new(Scalar::zero(), ctx.plaintext_count());
        (*encoded_ptxt.as_mut_polynomial().as_mut())[m] = Scalar::one();
        ctx.codec.poly_encode(&mut encoded_ptxt.as_mut_polynomial());

        for i in 1..(ctx.poly_size.0 - m) {
            let mut ct = ctx.empty_glwe_ciphertext();
            encrypt_glwe_ciphertext(
                &sk,
                &mut ct,
                &encoded_ptxt,
                ctx.std,
                &mut ctx.encryption_generator,
            );

            less_eq_than(&mut ct, (m + i) as u64);

            let mut out = PlaintextList::new(Scalar::zero(), ctx.plaintext_count());

            decrypt_glwe_ciphertext(&sk, &ct, &mut out);
            ctx.codec.poly_decode(&mut out.as_mut_polynomial());

            assert_eq!((*out.as_polynomial().as_ref())[0], Scalar::one());
        }

        for i in 1..(ctx.poly_size.0 - m) {
            let mut ct = ctx.empty_glwe_ciphertext();
            encrypt_glwe_ciphertext(
                &sk,
                &mut ct,
                &encoded_ptxt,
                ctx.std,
                &mut ctx.encryption_generator,
            );

            less_eq_than(&mut ct, (m - i) as u64);

            let mut out = PlaintextList::new(Scalar::zero(), ctx.plaintext_count());

            decrypt_glwe_ciphertext(&sk, &ct, &mut out);
            ctx.codec.poly_decode(&mut out.as_mut_polynomial());

            assert_eq!((*out.as_polynomial().as_ref())[0], Scalar::zero());
        }
    }

    #[test]
    fn test_eq_to() {
        let mut ctx = Context::new(TFHEParameters::default());
        let sk = GlweSecretKey::generate_new_binary(
            ctx.glwe_dimension,
            ctx.poly_size,
            &mut ctx.secret_generator,
        );

        let m = ctx.poly_size.0 / 2;
        let mut encoded_ptxt = PlaintextList::new(Scalar::zero(), ctx.plaintext_count());
        (*encoded_ptxt.as_mut_polynomial().as_mut())[m] = Scalar::one();
        ctx.codec.poly_encode(&mut encoded_ptxt.as_mut_polynomial());

        for i in 0..ctx.poly_size.0 {
            let mut ct = ctx.empty_glwe_ciphertext();

            encrypt_glwe_ciphertext(
                &sk,
                &mut ct,
                &encoded_ptxt,
                ctx.std,
                &mut ctx.encryption_generator,
            );

            eq_to(&mut ct, i as u64);

            let mut out = PlaintextList::new(Scalar::zero(), ctx.plaintext_count());

            decrypt_glwe_ciphertext(&sk, &ct, &mut out);
            ctx.codec.poly_decode(&mut out.as_mut_polynomial());
            let res = (*out.as_polynomial().as_ref())[0];
            if i == m {
                assert_eq!(res, 1);
            } else {
                assert_eq!(res, 0);
            }
        }
    }

    #[test]
    fn test_compute_noise() {
        let mut ctx = Context::new(TFHEParameters::default());
        let sk = GlweSecretKey::generate_new_binary(
            ctx.glwe_dimension,
            ctx.poly_size,
            &mut ctx.secret_generator,
        );

        let zero_msg = PlaintextList::new(Scalar::zero(), ctx.plaintext_count());
        let mut binary_msg = ctx.gen_binary_pt();
        ctx.codec.poly_encode(&mut binary_msg.as_mut_polynomial());

        let mut ct = GlweCiphertext::new(
            Scalar::zero(),
            ctx.glwe_size,
            ctx.poly_size,
            ctx.ciphertext_modulus,
        );
        let mut ct_zero = GlweCiphertext::new(
            Scalar::zero(),
            ctx.glwe_size,
            ctx.poly_size,
            ctx.ciphertext_modulus,
        );

        // sk.encrypt_rlwe(&mut ct, &binary_msg, ctx.std, &mut ctx.encryption_generator);
        encrypt_glwe_ciphertext(
            &sk,
            &mut ct,
            &binary_msg,
            ctx.std,
            &mut ctx.encryption_generator,
        );

        encrypt_glwe_ciphertext(
            &sk,
            &mut ct_zero,
            &zero_msg,
            ctx.std,
            &mut ctx.encryption_generator,
        );

        // the real support in all of the reals, but we need to approximate it
        // the log support is about log2(6*sigma), and sigma = Scalar::MAX * error_std
        let max_log_support = 3 + i64::from(Scalar::BITS) + ctx.std.get_log_standard_dev() as i64;
        println!("support: {max_log_support:?}");

        let noise_0 = compute_noise(&sk, &ct.as_view(), &binary_msg);
        println!("noise_0: {noise_0:?}");
        assert!(noise_0 < max_log_support as f64);

        // now if we add another ciphertext then the noise should increase

        glwe_ciphertext_add_assign(&mut ct, &ct_zero);
        let noise_1 = compute_noise(&sk, &ct.as_view(), &binary_msg);
        println!("noise_1: {noise_1:?}");
        assert!(noise_0 < noise_1);
        // assert!(noise_1 < max_log_support as f64);
    }

    #[test]
    fn test_not_in_place() {
        let mut ctx = Context {
            codec: Codec::new(2),
            ..Context::new(TFHEParameters::default())
        };
        let sk = GlweSecretKey::generate_new_binary(
            ctx.glwe_dimension,
            ctx.poly_size,
            &mut ctx.secret_generator,
        );

        let mut one = ctx.gen_unit_pt();
        let mut one_ct = ctx.empty_glwe_ciphertext();

        ctx.codec.poly_encode(&mut one.as_mut_polynomial());

        encrypt_glwe_ciphertext(
            &sk,
            &mut one_ct,
            &one,
            ctx.std,
            &mut ctx.encryption_generator,
        );

        not_in_place(&mut one_ct);
        let mut actual = PlaintextList::new(Scalar::zero(), ctx.plaintext_count());

        decrypt_glwe_ciphertext(&sk, &one_ct, &mut actual);
        ctx.codec.poly_decode(&mut actual.as_mut_polynomial());
        let expected = ctx.gen_zero_pt();
        assert_eq!(expected, actual);

        {
            not_in_place(&mut one_ct);
            let mut actual = PlaintextList::new(Scalar::zero(), ctx.plaintext_count());

            decrypt_glwe_ciphertext(&sk, &one_ct, &mut actual);
            ctx.codec.poly_decode(&mut actual.as_mut_polynomial());
            let expected = ctx.gen_unit_pt();
            assert_eq!(expected, actual);
        }
    }

    #[test]
    fn test_not() {
        let mut ctx = Context {
            codec: Codec::new(2),
            ..Context::new(TFHEParameters::default())
        };
        let sk = GlweSecretKey::generate_new_binary(
            ctx.glwe_dimension,
            ctx.poly_size,
            &mut ctx.secret_generator,
        );

        let mut one = ctx.gen_unit_pt();
        let mut one_ct = ctx.empty_glwe_ciphertext();

        ctx.codec.poly_encode(&mut one.as_mut_polynomial());
        encrypt_glwe_ciphertext(
            &sk,
            &mut one_ct,
            &one,
            ctx.std,
            &mut ctx.encryption_generator,
        );

        {
            let mut actual = PlaintextList::new(Scalar::zero(), ctx.plaintext_count());

            decrypt_glwe_ciphertext(&sk, &not(&one_ct), &mut actual);
            ctx.codec.poly_decode(&mut actual.as_mut_polynomial());
            assert_eq!(ctx.gen_zero_pt(), actual);
        }

        {
            let mut actual = PlaintextList::new(Scalar::zero(), ctx.plaintext_count());

            decrypt_glwe_ciphertext(&sk, &not(&not(&one_ct)), &mut actual);
            ctx.codec.poly_decode(&mut actual.as_mut_polynomial());
            assert_eq!(ctx.gen_unit_pt(), actual);
        }
    }
}
