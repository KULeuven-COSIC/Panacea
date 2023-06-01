use crate::{
    context::{Context, FftBuffer},
    num_types::{Complex, One, Scalar, ScalarContainer, Zero},
    rlwe::{compute_noise, RLWECiphertext, RLWEKeyswitchKey, RLWESecretKey},
    utils::mul_const,
};

use concrete_core::{
    backends::fft::private::crypto::ggsw::{external_product, FourierGgswCiphertext},
    commons::{
        crypto::{ggsw::StandardGgswCiphertext, glwe::GlweCiphertext},
        math::tensor::{AsMutSlice, AsMutTensor, AsRefSlice, AsRefTensor},
    },
    prelude::{
        CiphertextCount, DecompositionBaseLog, DecompositionLevelCount, GlweSize, MonomialDegree,
        PolynomialSize,
    },
};
use dyn_stack::{DynStack, ReborrowMut};

#[derive(Debug, Clone)]
/// An RGSW ciphertext.
/// It is a wrapper around `StandardGgswCiphertext` from concrete.
pub struct RGSWCiphertext(pub StandardGgswCiphertext<ScalarContainer>);

impl RGSWCiphertext {
    pub fn allocate(
        poly_size: PolynomialSize,
        decomp_base_log: DecompositionBaseLog,
        decomp_level: DecompositionLevelCount,
    ) -> Self {
        // TODO consider using Fourier version
        Self(StandardGgswCiphertext::allocate(
            Scalar::zero(),
            poly_size,
            GlweSize(2),
            decomp_level,
            decomp_base_log,
        ))
    }

    pub fn polynomial_size(&self) -> PolynomialSize {
        self.0.polynomial_size()
    }

    pub fn decomposition_level_count(&self) -> DecompositionLevelCount {
        self.0.decomposition_level_count()
    }

    pub fn decomposition_base_log(&self) -> DecompositionBaseLog {
        self.0.decomposition_base_log()
    }

    pub fn ciphertext_count(&self) -> CiphertextCount {
        self.0.as_glwe_list().ciphertext_count()
    }

    pub fn update_with_add(&mut self, other: &RGSWCiphertext) {
        self.0
            .as_mut_tensor()
            .update_with_wrapping_add(other.0.as_tensor())
    }

    pub fn external_product_with_buf_glwe(
        &self,
        out: &mut GlweCiphertext<&mut [Scalar]>,
        d: &RLWECiphertext,
        buf: &mut FftBuffer,
    ) {
        let glwe_size = GlweSize(2);
        let mut transformed = FourierGgswCiphertext::new(
            vec![
                Complex::zero();
                self.polynomial_size().0 / 2
                    * self.decomposition_level_count().0
                    * glwe_size.0
                    * glwe_size.0
            ]
            .into_boxed_slice(),
            self.polynomial_size(),
            glwe_size,
            self.decomposition_base_log(),
            self.decomposition_level_count(),
        );

        // this is just a wrapper around buf.mem
        let mut stack = DynStack::new(&mut buf.mem);

        transformed.as_mut_view().fill_with_forward_fourier(
            self.0.as_view(),
            buf.fft.as_view(),
            stack.rb_mut(),
        );

        external_product(
            out.as_mut_view(),
            transformed.as_view(),
            d.0.as_view(),
            buf.fft.as_view(),
            stack.rb_mut(),
        );
    }

    pub fn external_product_with_buf(
        &self,
        out: &mut RLWECiphertext,
        d: &RLWECiphertext,
        buf: &mut FftBuffer,
    ) {
        self.external_product_with_buf_glwe(&mut out.0.as_mut_view(), d, buf);
    }

    pub fn external_product(&self, out: &mut RLWECiphertext, d: &RLWECiphertext) {
        let mut buf = FftBuffer::new(d.polynomial_size());
        self.external_product_with_buf(out, d, &mut buf);
    }

    pub fn cmux(&self, out: &mut RLWECiphertext, ct0: &RLWECiphertext, ct1: &RLWECiphertext) {
        let mut buf = FftBuffer::new(ct0.polynomial_size());
        self.cmux_with_buf(out, ct0, ct1, &mut buf);
    }

    pub fn cmux_with_buf(
        &self,
        out: &mut RLWECiphertext,
        ct0: &RLWECiphertext,
        ct1: &RLWECiphertext,
        buf: &mut FftBuffer,
    ) {
        assert_eq!(ct0.polynomial_size(), ct1.polynomial_size());
        // TODO: consider removing tmp
        let mut tmp = RLWECiphertext::allocate(ct1.polynomial_size());
        tmp.0
            .as_mut_tensor()
            .as_mut_slice()
            .clone_from_slice(ct1.0.as_tensor().as_slice());
        out.0
            .as_mut_tensor()
            .as_mut_slice()
            .clone_from_slice(ct0.0.as_tensor().as_slice());
        tmp.update_with_sub(ct0);
        self.external_product_with_buf(out, &tmp, buf);
    }

    /*
    pub fn cmux_with_buf2(&self, ct0: &mut RLWECiphertext, ct1: &mut RLWECiphertext, buf: &mut FftBuffer) {
        let glwe_size = GlweSize(2);
        let mut transformed = FourierGgswCiphertext::new(
            vec![
                Complex::zero();
                self.polynomial_size().0 / 2
                    * self.decomposition_level_count().0
                    * glwe_size.0
                    * glwe_size.0
            ]
            .into_boxed_slice(),
            self.polynomial_size(),
            glwe_size,
            self.decomposition_base_log(),
            self.decomposition_level_count(),
        );

        // this is just a wrapper around buf.mem
        let mut stack = DynStack::new(&mut buf.mem);

        transformed.as_mut_view().fill_with_forward_fourier(
            self.0.as_view(),
            buf.fft.as_view(),
            stack.rb_mut(),
        );

        cmux(ct0.0.as_mut_view(), ct1.0.as_mut_view(), transformed.as_view(), buf.fft.as_view(), stack.rb_mut());
    }
    */

    pub fn get_last_row(&self) -> RLWECiphertext {
        self.get_nth_row(self.decomposition_level_count().0 * 2 - 1)
    }

    pub fn get_nth_row(&self, n: usize) -> RLWECiphertext {
        let mut glwe_ct =
            GlweCiphertext::allocate(Scalar::zero(), self.polynomial_size(), GlweSize(2));
        glwe_ct.as_mut_tensor().fill_with_copy(
            self.0
                .as_glwe_list()
                .ciphertext_iter()
                .nth(n)
                .unwrap()
                .as_tensor(),
        );
        RLWECiphertext(glwe_ct)
    }

    /// Convert the RLWE key switching key to a RGSW ciphertext.
    /// Not recommended to use.
    pub fn from_keyswitch_key(ksk: &RLWEKeyswitchKey) -> Self {
        let ell = ksk.decomposition_level_count();
        let base_log = ksk.decomposition_base_log();
        let mut rgsw = Self::allocate(ksk.polynomial_size(), base_log, ell);
        let ks = ksk.get_keyswitch_key();
        assert_eq!(rgsw.0.as_glwe_list().ciphertext_count().0, 2 * ell.0);
        assert_eq!(ks.len(), ell.0);

        for (i, mut ct) in rgsw.0.as_mut_glwe_list().ciphertext_iter_mut().enumerate() {
            let level = i / 2;
            if i % 2 == 0 {
                ct.get_mut_mask()
                    .as_mut_polynomial_list()
                    .get_mut_polynomial(0)
                    .update_with_wrapping_sub(
                        &ks[level].get_mask().as_polynomial_list().get_polynomial(0),
                    );
                ct.get_mut_body()
                    .as_mut_polynomial()
                    .update_with_wrapping_sub(&ks[level].get_body().as_polynomial());
            } else {
                let shift: usize = (Scalar::BITS as usize) - base_log.0 * (level + 1);
                ct.get_mut_body()
                    .as_mut_polynomial()
                    .get_mut_monomial(MonomialDegree(0))
                    .set_coefficient(Scalar::one() << shift);
            }
        }
        rgsw
    }

    /// Execute the key switching operation with a buffer when self is a key switching key.
    pub fn keyswitch_ciphertext_with_buf(
        &self,
        after: &mut RLWECiphertext,
        before: &RLWECiphertext,
        buf: &mut FftBuffer,
    ) {
        self.external_product_with_buf(after, before, buf);
    }

    /// Execute the key switching operation when self is a key switching key.
    pub fn keyswitch_ciphertext(&self, after: &mut RLWECiphertext, before: &RLWECiphertext) {
        self.external_product(after, before);
    }
}

/// Compute the average noise in the RGSW ciphertext
/// by computing the noise on the RLWE rows and taking the average.
pub fn compute_noise_rgsw1(sk: &RLWESecretKey, ct: &RGSWCiphertext, ctx: &Context) -> f64 {
    let mut total_noise = 0f64;
    for level in 0..ctx.level_count.0 {
        let shift = (Scalar::BITS as usize) - ctx.base_log.0 * (level + 1);
        let mut pt = ctx.gen_unit_pt();
        mul_const(&mut pt.as_mut_tensor(), 1 << shift);
        let noise = compute_noise(sk, &ct.get_nth_row(level * 2 + 1), &pt);
        total_noise += noise;
    }
    total_noise / ctx.level_count.0 as f64
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::context::Context;
    use crate::params::TFHEParameters;
    use crate::rlwe::{compute_noise_encoded, compute_noise_ternary, RLWESecretKey};
    use concrete_core::commons::crypto::encoding::{Plaintext, PlaintextList};

    #[test]
    fn test_external_product() {
        let mut ctx = Context::new(TFHEParameters::default());
        let orig_pt = ctx.gen_binary_pt();

        let sk = RLWESecretKey::generate_binary(ctx.poly_size, &mut ctx.secret_generator);

        let mut rgsw_ct_0 = RGSWCiphertext::allocate(ctx.poly_size, ctx.base_log, ctx.level_count);
        let mut rgsw_ct_1 = RGSWCiphertext::allocate(ctx.poly_size, ctx.base_log, ctx.level_count);
        sk.encrypt_constant_rgsw(&mut rgsw_ct_0, &Plaintext(Scalar::zero()), &mut ctx);
        sk.encrypt_constant_rgsw(&mut rgsw_ct_1, &Plaintext(Scalar::one()), &mut ctx);

        let mut rlwe_ct = RLWECiphertext::allocate(ctx.poly_size);
        sk.encode_encrypt_rlwe(&mut rlwe_ct, &orig_pt, &mut ctx);
        println!(
            "initial noise: {:?}",
            compute_noise_encoded(&sk, &rlwe_ct, &orig_pt, &ctx.codec)
        );

        let zero_ptxt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());

        {
            let mut out_0 = RLWECiphertext::allocate(ctx.poly_size);
            let mut out_1 = RLWECiphertext::allocate(ctx.poly_size);
            rgsw_ct_0.external_product(&mut out_0, &rlwe_ct);
            rgsw_ct_1.external_product(&mut out_1, &rlwe_ct);

            let mut decrypted = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
            sk.decrypt_decode_rlwe(&mut decrypted, &out_0, &ctx);
            assert_eq!(decrypted, zero_ptxt);
            sk.decrypt_decode_rlwe(&mut decrypted, &out_1, &ctx);
            assert_eq!(decrypted, orig_pt);
            println!(
                "final noise: {:?}",
                compute_noise_encoded(&sk, &out_1, &orig_pt, &ctx.codec)
            );
        }
    }

    #[test]
    fn test_rgsw_shape() {
        let mut ctx = Context {
            poly_size: PolynomialSize(8),
            ..Context::new(TFHEParameters::default())
        };

        let sk = RLWESecretKey::generate_binary(ctx.poly_size, &mut ctx.secret_generator);

        let mut rgsw_ct = RGSWCiphertext::allocate(ctx.poly_size, ctx.base_log, ctx.level_count);
        sk.0.trivial_encrypt_constant_ggsw(
            &mut rgsw_ct.0,
            &Plaintext(Scalar::one()),
            ctx.std,
            &mut ctx.encryption_generator,
        );

        // the way RGSW ciphertext arranged is different from some literature
        // usually it's Z + m*G, where Z is the RLWE encryption of zeros and G is the gadget matrix
        //      g_1 0
        // G =  g_2 0
        //      0   g_1
        //      0   g_2
        // but concrete has it arragned like this:
        //      g_1 0
        // G =  0   g_1
        //      g_2 0
        //      0   g_2
        let mut level_count = 0;
        for m in rgsw_ct.0.level_matrix_iter() {
            let mut row_count = 0;
            for row in m.row_iter() {
                let ct = row.into_glwe();
                println!(
                    "mask : {:?}",
                    ct.get_mask().as_polynomial_list().get_polynomial(0)
                );
                println!("body: {:?}", ct.get_body().as_polynomial());
                row_count += 1;
            }
            assert_eq!(row_count, 2);
            level_count += 1;
        }
        assert_eq!(level_count, ctx.level_count.0);
    }

    #[test]
    fn test_keyswitching() {
        let mut ctx = Context::new(TFHEParameters::default());
        let messages = ctx.gen_ternary_ptxt();

        let sk_after = ctx.gen_rlwe_sk();
        let sk_before = ctx.gen_rlwe_sk();

        let mut ct_after = RLWECiphertext::allocate(ctx.poly_size);
        let mut ct_before = RLWECiphertext::allocate(ctx.poly_size);

        let mut ksk_slow =
            RLWEKeyswitchKey::allocate(ctx.ks_base_log, ctx.ks_level_count, ctx.poly_size);
        ksk_slow.fill_with_keyswitch_key(
            &sk_before,
            &sk_after,
            ctx.std,
            &mut ctx.encryption_generator,
        );
        let ksk = RGSWCiphertext::from_keyswitch_key(&ksk_slow);

        // encrypts with the before key our messages
        sk_before.ternary_encrypt_rlwe(&mut ct_before, &messages, &mut ctx);
        println!("msg before: {:?}", messages.as_tensor());
        let mut dec_messages_1 = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
        sk_before.ternary_decrypt_rlwe(&mut dec_messages_1, &ct_before);
        println!("msg after dec: {:?}", dec_messages_1.as_tensor());
        println!(
            "initial noise: {:?}",
            compute_noise_ternary(&sk_before, &ct_before, &messages)
        );

        ksk.keyswitch_ciphertext(&mut ct_after, &ct_before);

        let mut dec_messages_2 = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
        sk_after.ternary_decrypt_rlwe(&mut dec_messages_2, &ct_after);
        println!("msg after ks: {:?}", dec_messages_2.as_tensor());

        assert_eq!(dec_messages_1, dec_messages_2);
        assert_eq!(dec_messages_1, messages);
        println!(
            "final noise: {:?}",
            compute_noise_ternary(&sk_after, &ct_after, &messages)
        );
    }

    #[test]
    fn test_cmux() {
        let mut ctx = Context::new(TFHEParameters::default());
        let pt_0 = ctx.gen_ternary_ptxt();
        let pt_1 = ctx.gen_ternary_ptxt();
        let sk = ctx.gen_rlwe_sk();

        let mut ct_0 = RLWECiphertext::allocate(ctx.poly_size);
        let mut ct_1 = RLWECiphertext::allocate(ctx.poly_size);
        sk.ternary_encrypt_rlwe(&mut ct_0, &pt_0, &mut ctx);
        sk.ternary_encrypt_rlwe(&mut ct_1, &pt_1, &mut ctx);

        let mut ct_gsw = RGSWCiphertext::allocate(ctx.poly_size, ctx.base_log, ctx.level_count);

        {
            // set choice bit to 0
            sk.encrypt_constant_rgsw(&mut ct_gsw, &Plaintext(Scalar::zero()), &mut ctx);
            let mut ct_result = RLWECiphertext::allocate(ctx.poly_size);
            let mut pt_result = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());

            ct_gsw.cmux(&mut ct_result, &ct_0, &ct_1);
            sk.ternary_decrypt_rlwe(&mut pt_result, &ct_result);
            assert_eq!(pt_result, pt_0);
        }

        {
            // set choice bit to 1
            sk.encrypt_constant_rgsw(&mut ct_gsw, &Plaintext(Scalar::one()), &mut ctx);
            let mut ct_result = RLWECiphertext::allocate(ctx.poly_size);
            let mut pt_result = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());

            ct_gsw.cmux(&mut ct_result, &ct_0, &ct_1);
            sk.ternary_decrypt_rlwe(&mut pt_result, &ct_result);
            assert_eq!(pt_result, pt_1);
        }
    }
}
