use aligned_vec::{AVec, CACHELINE_ALIGN};

use tfhe::core_crypto::prelude::{
    par_encrypt_constant_ggsw_ciphertext, ContiguousEntityContainer, GgswCiphertext,
    GlweCiphertext, GlweSecretKey, LweCiphertext, LweSecretKey, Plaintext, SignedDecomposer,
};

use crate::{
    context::Context,
    num_types::{AlignedScalarContainer, Scalar, ScalarContainer, Zero},
};

pub fn lwe_to_rlwe_ksks(
    lwe_sk: &LweSecretKey<ScalarContainer>,
    ctx: &mut Context<Scalar>,
) -> Vec<GgswCiphertext<ScalarContainer>> {
    assert_eq!(ctx.poly_size.0, lwe_sk.lwe_dimension().0);
    let rlwe_sk = lwe_sk_to_rlwe_sk(lwe_sk, ctx);
    lwe_sk
        .as_ref()
        .iter()
        .map(|s| {
            let mut tmp = GgswCiphertext::new(
                Scalar::zero(),
                ctx.glwe_size,
                ctx.poly_size,
                ctx.ks_base_log,
                ctx.ks_level_count,
                ctx.ciphertext_modulus,
            );
            par_encrypt_constant_ggsw_ciphertext(
                &rlwe_sk,
                &mut tmp,
                Plaintext(*s),
                ctx.std,
                &mut ctx.encryption_generator,
            );
            tmp
        })
        .collect()
}

pub fn lwe_sk_to_rlwe_sk(
    sk: &LweSecretKey<ScalarContainer>,
    ctx: &Context<Scalar>,
) -> GlweSecretKey<AlignedScalarContainer> {
    assert!(ctx.poly_size.0 == sk.lwe_dimension().0);
    GlweSecretKey::from_container(
        AVec::<Scalar>::from_iter(CACHELINE_ALIGN, sk.as_ref().iter().copied()),
        ctx.poly_size,
    )
}

pub fn conv_lwe_to_rlwe(
    ksks: &[GgswCiphertext<ScalarContainer>],
    lwe: &LweCiphertext<ScalarContainer>,
    ctx: &Context<Scalar>,
) -> GlweCiphertext<AlignedScalarContainer> {
    let mut out: GlweCiphertext<AlignedScalarContainer> = ctx.empty_glwe_ciphertext();

    for (ksk, a) in ksks.iter().zip(lwe.get_mask().as_ref().iter()) {
        // Setup decomposition stuff
        let decomposer = SignedDecomposer::new(ctx.ks_base_log, ctx.ks_level_count);
        let closest = decomposer.closest_representable(*a);
        let decomposer_iter = decomposer.decompose(closest);

        // Get an iterator of every second row
        // we only need every second ciphertext since that is
        // a valid RLWE ciphertext in a RGSW

        ksk.iter()
            .rev()
            .zip(decomposer_iter)
            .for_each(|(m, decomposed_a)| {
                out.as_mut()
                    .iter_mut()
                    .zip(m.as_glwe_list().iter().nth(1).unwrap().as_ref().iter())
                    .for_each(|(o, c)| *o = o.wrapping_sub(c.wrapping_mul(decomposed_a.value())));
            });
    }

    out.get_mut_body().as_mut_polynomial().as_mut()[0] =
        out.get_mut_body().as_mut_polynomial().as_mut()[0].wrapping_add(*lwe.get_body().data);
    out
}

#[cfg(test)]
mod test {
    use tfhe::core_crypto::{
        algorithms::{
            allocate_and_encrypt_new_lwe_ciphertext, decrypt_glwe_ciphertext,
            decrypt_lwe_ciphertext, encrypt_glwe_ciphertext,
            extract_lwe_sample_from_glwe_ciphertext,
        },
        prelude::{MonomialDegree, PolynomialSize},
    };

    use crate::params::TFHEParameters;

    use super::*;

    #[test]
    fn test_lwe() {
        let mut ctx = Context::new(TFHEParameters::default());
        let sk = LweSecretKey::generate_new_binary(ctx.lwe_dim(), &mut ctx.secret_generator);
        for _ in 0..10 {
            let expected = ctx.gen_scalar_binary_pt();

            let mut encoded_pt = expected;
            ctx.codec.encode(&mut encoded_pt.0);

            let ct = allocate_and_encrypt_new_lwe_ciphertext(
                &sk,
                encoded_pt,
                ctx.std,
                ctx.ciphertext_modulus,
                &mut ctx.encryption_generator,
            );

            let mut actual = decrypt_lwe_ciphertext(&sk, &ct);
            ctx.codec.decode(&mut actual.0);

            assert_eq!(expected, actual);
        }
    }

    #[test]
    fn test_lwe_fill() {
        let mut ctx = Context::new(TFHEParameters::default());
        let sk = LweSecretKey::generate_new_binary(ctx.lwe_dim(), &mut ctx.secret_generator);

        let expected = ctx.gen_scalar_binary_pt();

        let mut encoded_pt = expected;
        ctx.codec.encode(&mut encoded_pt.0);

        let ct = allocate_and_encrypt_new_lwe_ciphertext(
            &sk,
            encoded_pt,
            ctx.std,
            ctx.ciphertext_modulus,
            &mut ctx.encryption_generator,
        );

        let mut actual = decrypt_lwe_ciphertext(&sk, &ct);
        ctx.codec.decode(&mut actual.0);

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_lwe_to_rlwe() {
        let mut ctx = Context {
            poly_size: PolynomialSize(1024),
            ..Context::new(TFHEParameters::default())
        };

        let lwe_sk = LweSecretKey::generate_new_binary(ctx.lwe_dim(), &mut ctx.secret_generator);
        let rlwe_sk = lwe_sk_to_rlwe_sk(&lwe_sk, &ctx);

        let mut expected = ctx.gen_scalar_binary_pt();

        // create ciphertext
        let encoded_pt = expected;

        let lwe_ct = allocate_and_encrypt_new_lwe_ciphertext(
            &lwe_sk,
            encoded_pt,
            ctx.std,
            ctx.ciphertext_modulus,
            &mut ctx.encryption_generator,
        );

        let ksks = lwe_to_rlwe_ksks(&lwe_sk, &mut ctx);

        // switch it to a rlwe ciphertext and decrypt
        let rlwe_ct = conv_lwe_to_rlwe(&ksks, &lwe_ct, &ctx);
        let mut actual = ctx.gen_zero_pt();
        decrypt_glwe_ciphertext(&rlwe_sk, &rlwe_ct, &mut actual);
        ctx.codec
            .poly_decode(&mut actual.as_mut_polynomial().as_mut_view());

        ctx.codec.decode(&mut expected.0);
        // find the constant term and compare
        assert_eq!(*actual.as_ref().iter().next().unwrap(), expected.0);
    }

    #[test]
    fn test_sample_extract() {
        let mut ctx = Context::new(TFHEParameters::default());
        let lwe_sk = LweSecretKey::generate_new_binary(ctx.lwe_dim(), &mut ctx.secret_generator);
        let rlwe_sk = lwe_sk_to_rlwe_sk(&lwe_sk, &ctx);

        // create a ciphertext
        let pt = ctx.gen_binary_pt();

        let mut rlwe_ct = GlweCiphertext::new(
            Scalar::zero(),
            ctx.glwe_size,
            ctx.poly_size,
            ctx.ciphertext_modulus,
        );

        let mut encoded_pt = pt.clone();
        ctx.codec.poly_encode(&mut encoded_pt.as_mut_polynomial());

        encrypt_glwe_ciphertext(
            &rlwe_sk,
            &mut rlwe_ct,
            &encoded_pt,
            ctx.std,
            &mut ctx.encryption_generator,
        );

        // make sample extract
        let mut lwe_ct = LweCiphertext::new(0, ctx.lwe_size(), ctx.ciphertext_modulus);

        extract_lwe_sample_from_glwe_ciphertext(&rlwe_ct, &mut lwe_ct, MonomialDegree(0));

        // decrypt and compare
        let mut actual_pt = decrypt_lwe_ciphertext(&lwe_sk, &lwe_ct);
        ctx.codec.decode(&mut actual_pt.0);

        assert_eq!((*pt.as_ref())[0], actual_pt.0);
    }
}
