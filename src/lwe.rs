use concrete_core::commons::math::tensor::{AsRefSlice, Tensor};
use concrete_core::{
    commons::{
        crypto::{
            encoding::Plaintext,
            lwe::{LweBody, LweCiphertext, LweMask},
            secret::generators::SecretRandomGenerator,
            secret::{generators::EncryptionRandomGenerator, LweSecretKey},
        },
        math::{
            decomposition::SignedDecomposer,
            polynomial::Polynomial,
            tensor::{AsMutTensor, AsRefTensor},
        },
    },
    prelude::{
        BinaryKeyKind, DispersionParameter, LweDimension, LweSize, MonomialDegree, PolynomialSize,
    },
};
use concrete_csprng::generators::SoftwareRandomGenerator;

use crate::{
    context::Context,
    num_types::{Scalar, Zero},
    rgsw::RGSWCiphertext,
    rlwe::{RLWECiphertext, RLWESecretKey},
    utils::mul_const,
};

#[derive(Debug, Clone)]
/// An LWE ciphertext.
pub struct LWECiphertext(pub LweCiphertext<Vec<Scalar>>);

impl LWECiphertext {
    pub fn allocate(size: LweSize) -> Self {
        Self(LweCiphertext::allocate(Scalar::zero(), size))
    }

    /// Return the length of the mask + 1 for the body.
    pub fn lwe_size(&self) -> LweSize {
        self.0.lwe_size()
    }

    pub fn get_body(&self) -> &LweBody<Scalar> {
        self.0.get_body()
    }

    pub fn get_mask(&self) -> LweMask<&[Scalar]> {
        self.0.get_mask()
    }

    pub fn get_mut_mask(&mut self) -> LweMask<&mut [Scalar]> {
        self.0.get_mut_mask()
    }

    pub fn get_mut_body(&mut self) -> &mut LweBody<Scalar> {
        self.0.get_mut_body()
    }

    pub fn clear(&mut self) {
        self.0.as_mut_tensor().fill_with(Scalar::zero);
    }

    pub fn fill_with_sample_extract(&mut self, c: &RLWECiphertext, n_th: MonomialDegree) {
        self.0.fill_with_glwe_sample_extraction(&c.0, n_th);
    }

    pub fn fill_with_const_sample_extract(&mut self, c: &RLWECiphertext) {
        self.0
            .fill_with_glwe_sample_extraction(&c.0, MonomialDegree(0));
    }

    pub fn fill_with_tensor<C>(&mut self, t: &Tensor<C>)
    where
        Tensor<C>: AsRefSlice<Element = Scalar>,
    {
        self.0.as_mut_tensor().fill_with_copy(t);
    }

    pub fn as_tensor(&self) -> &Tensor<Vec<Scalar>> {
        self.0.as_tensor()
    }
}

#[derive(Debug, Clone)]
/// An LWE secret key.
pub struct LWESecretKey(pub(crate) LweSecretKey<BinaryKeyKind, Vec<Scalar>>);

impl LWESecretKey {
    /// Generate a secret key where the coefficients are binary.
    pub fn generate_binary(
        lwe_dimension: LweDimension,
        generator: &mut SecretRandomGenerator<SoftwareRandomGenerator>,
    ) -> Self {
        Self(LweSecretKey::generate_binary(lwe_dimension, generator))
    }

    pub fn encrypt_lwe(
        &self,
        output: &mut LWECiphertext,
        pt: &Plaintext<Scalar>,
        noise_parameters: impl DispersionParameter,
        generator: &mut EncryptionRandomGenerator<SoftwareRandomGenerator>,
    ) {
        self.0
            .encrypt_lwe(&mut output.0, pt, noise_parameters, generator);
    }

    pub fn encode_encrypt_lwe(
        &self,
        output: &mut LWECiphertext,
        pt: &Plaintext<Scalar>,
        ctx: &mut Context,
    ) {
        let mut encoded_pt = *pt;
        ctx.codec.encode(&mut encoded_pt.0);
        self.encrypt_lwe(output, &encoded_pt, ctx.std, &mut ctx.encryption_generator);
    }

    pub fn decrypt_lwe(&self, output: &mut Plaintext<Scalar>, ct: &LWECiphertext) {
        self.0.decrypt_lwe(output, &ct.0);
    }

    /// Decrypt a LWE ciphertext and then decode.
    pub fn decode_decrypt_lwe(
        &self,
        pt: &mut Plaintext<Scalar>,
        encrypted: &LWECiphertext,
        ctx: &Context,
    ) {
        self.decrypt_lwe(pt, encrypted);
        ctx.codec.decode(&mut pt.0);
    }

    pub fn to_rlwe_sk(&self) -> RLWESecretKey {
        let mut sk = RLWESecretKey::zero(PolynomialSize(self.0.key_size().0));
        sk.fill_with_tensor(self.0.as_tensor());
        sk
    }

    pub fn key_size(&self) -> LweDimension {
        self.0.key_size()
    }
}

#[derive(Debug, Clone)]
/// An LWE to RLWE key switching key.
pub struct LWEtoRLWEKeyswitchKey {
    // TODO At the moment it's a list of full RGSW ciphertexts,
    // we should remove half of the rows.
    pub(crate) inner: Vec<RGSWCiphertext>,
}

impl LWEtoRLWEKeyswitchKey {
    pub fn allocate(ctx: &Context) -> Self {
        Self {
            inner: vec![
                RGSWCiphertext::allocate(ctx.poly_size, ctx.base_log, ctx.level_count);
                ctx.poly_size.0
            ],
        }
    }

    pub fn fill_with_keyswitching_key(&mut self, sk: &LWESecretKey, ctx: &mut Context) {
        assert_eq!(ctx.poly_size.0, sk.key_size().0);
        let rlwe_sk = sk.to_rlwe_sk();
        self.inner = vec![];
        for s in sk.0.as_tensor().iter() {
            // TODO what is the decomposition parameters?
            let mut rgsw_ct =
                RGSWCiphertext::allocate(ctx.poly_size, ctx.ks_base_log, ctx.ks_level_count);
            rlwe_sk.encrypt_constant_rgsw(&mut rgsw_ct, &Plaintext(*s), ctx);
            self.inner.push(rgsw_ct);
        }
    }
}

pub fn conv_lwe_to_rlwe(
    ksks: &LWEtoRLWEKeyswitchKey,
    lwe: &LWECiphertext,
    ctx: &Context,
) -> RLWECiphertext {
    let mut out = RLWECiphertext::allocate(ctx.poly_size);

    for (ksk, a) in ksks.inner.iter().zip(lwe.get_mask().as_tensor().iter()) {
        // Setup decomposition stuff
        // TODO what parameters for decomposition?
        let decomposer = SignedDecomposer::new(ctx.ks_base_log, ctx.ks_level_count);
        let closest = decomposer.closest_representable(*a);
        let decomposer_iter = decomposer.decompose(closest);

        // Get an iterator of every second row
        // we only need every second ciphertext since that is
        // a valid RLWE ciphertext in a RGSW
        let ksk_iter = ksk.0.level_matrix_iter().rev().map(|m| {
            let ct = m.row_iter().nth(1).unwrap().into_glwe();
            // TODO avoid copying
            let mut out = RLWECiphertext::allocate(ctx.poly_size);
            out.update_mask_with_add(&ct.get_mask().as_polynomial_list().get_polynomial(0));
            out.update_body_with_add(&ct.get_body().as_polynomial());
            out
        });

        for (mut ct, decomposed_a) in ksk_iter.zip(decomposer_iter) {
            mul_const(
                ct.get_mut_mask()
                    .as_mut_polynomial_list()
                    .get_mut_polynomial(0)
                    .as_mut_tensor(),
                decomposed_a.value(),
            );
            mul_const(
                ct.get_mut_body().as_mut_polynomial().as_mut_tensor(),
                decomposed_a.value(),
            );
            out.get_mut_mask()
                .as_mut_polynomial_list()
                .get_mut_polynomial(0)
                .update_with_wrapping_sub(&ct.get_mask().as_polynomial_list().get_polynomial(0));
            out.get_mut_body()
                .as_mut_polynomial()
                .update_with_wrapping_sub(&ct.get_body().as_polynomial());
        }
    }

    let b_poly = {
        let mut v = vec![Scalar::zero(); ctx.poly_size.0];
        v[0] = lwe.get_body().0;
        Polynomial::from_container(v)
    };

    out.get_mut_body()
        .as_mut_polynomial()
        .update_with_wrapping_add(&b_poly);
    out
}

#[cfg(test)]
mod test {
    use concrete_core::prelude::PolynomialSize;

    use crate::params::TFHEParameters;

    use super::*;

    #[test]
    fn test_lwe() {
        let mut ctx = Context::new(TFHEParameters::default());
        let sk = LWESecretKey::generate_binary(ctx.lwe_dim(), &mut ctx.secret_generator);
        for _ in 0..10 {
            let expected = ctx.gen_scalar_binary_pt();
            let mut ct = LWECiphertext::allocate(ctx.lwe_size());
            sk.encode_encrypt_lwe(&mut ct, &expected, &mut ctx);

            let mut actual = ctx.gen_scalar_zero_pt();
            sk.decode_decrypt_lwe(&mut actual, &ct, &ctx);
            assert_eq!(expected, actual);
        }
    }

    #[test]
    fn test_lwe_fill() {
        let mut ctx = Context::new(TFHEParameters::default());
        let sk = LWESecretKey::generate_binary(ctx.lwe_dim(), &mut ctx.secret_generator);

        let expected = ctx.gen_scalar_binary_pt();
        let mut ct = LWECiphertext::allocate(ctx.lwe_size());
        sk.encode_encrypt_lwe(&mut ct, &expected, &mut ctx);

        let mut ct2 = LWECiphertext::allocate(ctx.lwe_size());
        ct2.fill_with_tensor(ct.as_tensor());

        let mut actual = ctx.gen_scalar_zero_pt();
        sk.decode_decrypt_lwe(&mut actual, &ct2, &ctx);
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_lwe_to_rlwe() {
        let mut ctx = Context {
            poly_size: PolynomialSize(1024),
            ..Context::new(TFHEParameters::default())
        };

        let lwe_sk = LWESecretKey::generate_binary(ctx.lwe_dim(), &mut ctx.secret_generator);
        let rlwe_sk = lwe_sk.to_rlwe_sk();
        let expected = ctx.gen_scalar_binary_pt();

        // create ciphertext
        let mut lwe_ct = LWECiphertext::allocate(ctx.lwe_size());
        lwe_sk.encode_encrypt_lwe(&mut lwe_ct, &expected, &mut ctx);

        // create ksk
        let mut ksks = LWEtoRLWEKeyswitchKey::allocate(&ctx);
        ksks.fill_with_keyswitching_key(&lwe_sk, &mut ctx);

        // switch it to a rlwe ciphertext and decrypt
        let rlwe_ct = conv_lwe_to_rlwe(&ksks, &lwe_ct, &ctx);
        let mut actual = ctx.gen_zero_pt();
        rlwe_sk.decrypt_decode_rlwe(&mut actual, &rlwe_ct, &ctx);

        // find the constant term and compare
        assert_eq!(*actual.plaintext_iter().next().unwrap(), expected);
    }

    #[test]
    fn test_sample_extract() {
        let mut ctx = Context::new(TFHEParameters::default());
        let lwe_sk = LWESecretKey::generate_binary(ctx.lwe_dim(), &mut ctx.secret_generator);
        let rlwe_sk = lwe_sk.to_rlwe_sk();

        // create a ciphertext
        let pt = ctx.gen_binary_pt();
        let mut rlwe_ct = RLWECiphertext::allocate(ctx.poly_size);
        rlwe_sk.encode_encrypt_rlwe(&mut rlwe_ct, &pt, &mut ctx);

        // make sample extract
        let mut lwe_ct = LWECiphertext::allocate(ctx.lwe_size());
        lwe_ct.fill_with_const_sample_extract(&rlwe_ct);

        // decrypt and compare
        let mut actual_pt = Plaintext(Scalar::zero());
        lwe_sk.decode_decrypt_lwe(&mut actual_pt, &lwe_ct, &ctx);
        assert_eq!(*pt.as_tensor().get_element(0), actual_pt.0);
    }
}
