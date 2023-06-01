use crate::{
    codec::Codec,
    context::{Context, FftBuffer},
    lwe::{conv_lwe_to_rlwe, LWECiphertext, LWEtoRLWEKeyswitchKey},
    num_types::{Complex, ComplexContaier, One, Scalar, ScalarContainer, SignedScalar, Zero},
    rgsw::RGSWCiphertext,
    utils::{eval_x_k, log2, mul_const},
};

use concrete_core::{
    backends::fft::private::math::{fft::Fft, polynomial::FourierPolynomial},
    commons::{
        crypto::{
            encoding::{Plaintext, PlaintextList},
            glwe::{GlweBody, GlweCiphertext, GlweMask},
            secret::{
                generators::{EncryptionRandomGenerator, SecretRandomGenerator},
                GlweSecretKey,
            },
        },
        math::{
            decomposition::SignedDecomposer,
            polynomial::{MonomialDegree, Polynomial},
            tensor::{AsMutSlice, AsMutTensor, AsRefSlice, AsRefTensor, Tensor},
        },
    },
    prelude::{
        BinaryKeyKind, DecompositionBaseLog, DecompositionLevelCount, DispersionParameter,
        GlweDimension, PlaintextCount, PolynomialSize,
    },
};

use concrete_csprng::generators::SoftwareRandomGenerator;
use dyn_stack::{DynStack, GlobalMemBuffer, ReborrowMut};

use std::collections::HashMap;

#[derive(Debug, Clone)]
/// An RLWE ciphertext.
/// It is a wrapper around `GlweCiphertext` from concrete.
pub struct RLWECiphertext(pub GlweCiphertext<ScalarContainer>);

impl RLWECiphertext {
    pub fn allocate(poly_size: PolynomialSize) -> Self {
        Self(GlweCiphertext::from_container(
            vec![Scalar::zero(); poly_size.0 * 2],
            poly_size,
        ))
    }

    pub fn polynomial_size(&self) -> PolynomialSize {
        self.0.polynomial_size()
    }

    pub fn get_body(&self) -> GlweBody<&[Scalar]> {
        self.0.get_body()
    }

    pub fn get_mask(&self) -> GlweMask<&[Scalar]> {
        self.0.get_mask()
    }

    pub fn get_mut_mask(&mut self) -> GlweMask<&mut [Scalar]> {
        self.0.get_mut_mask()
    }

    pub fn get_mut_body(&mut self) -> GlweBody<&mut [Scalar]> {
        self.0.get_mut_body()
    }

    pub fn clear(&mut self) {
        self.0.as_mut_tensor().fill_with(Scalar::zero);
    }

    pub fn fill_with_copy(&mut self, other: &RLWECiphertext) {
        self.0.as_mut_tensor().fill_with_copy(other.0.as_tensor());
    }

    pub fn fill_with_tensor<C>(&mut self, t: &Tensor<C>)
    where
        Tensor<C>: AsRefSlice<Element = Scalar>,
    {
        self.0.as_mut_tensor().fill_with_copy(t);
    }

    pub fn update_mask_with_add<C>(&mut self, other: &Polynomial<C>)
    where
        C: AsRefSlice<Element = Scalar>,
    {
        self.get_mut_mask()
            .as_mut_polynomial_list()
            .get_mut_polynomial(0)
            .update_with_wrapping_add(other);
    }

    pub fn update_mask_with_sub(&mut self, other: &Polynomial<&[Scalar]>) {
        self.get_mut_mask()
            .as_mut_polynomial_list()
            .get_mut_polynomial(0)
            .update_with_wrapping_sub(other);
    }

    pub fn update_mask_with_mul(&mut self, other: &Polynomial<&mut [Scalar]>) {
        let mut poly_buffer = Polynomial::allocate(Scalar::zero(), self.polynomial_size());
        poly_buffer
            .as_mut_tensor()
            .fill_with_copy(self.get_mask().as_tensor());
        self.get_mut_mask().as_mut_tensor().fill_with(Scalar::zero);

        naive_update_with_mul_acc(
            &mut self
                .get_mut_mask()
                .as_mut_polynomial_list()
                .get_mut_polynomial(0),
            &poly_buffer.as_mut_view(),
            other,
        );
    }

    pub fn update_mask_with_mul_with_buf(&mut self, other: &Polynomial<&[Scalar]>) {
        fourier_update_with_mul(
            &mut self
                .get_mut_mask()
                .as_mut_polynomial_list()
                .get_mut_polynomial(0),
            other,
        );
    }

    pub fn update_body_with_add<C>(&mut self, other: &Polynomial<C>)
    where
        C: AsRefSlice<Element = Scalar>,
    {
        self.get_mut_body()
            .as_mut_polynomial()
            .update_with_wrapping_add(other);
    }

    pub fn update_body_with_sub(&mut self, other: &Polynomial<&[Scalar]>) {
        self.get_mut_body()
            .as_mut_polynomial()
            .update_with_wrapping_sub(other);
    }

    pub fn update_body_with_mul(&mut self, other: &Polynomial<&mut [Scalar]>) {
        let mut poly_buffer = Polynomial::allocate(Scalar::zero(), self.polynomial_size());
        poly_buffer
            .as_mut_tensor()
            .fill_with_copy(self.get_body().as_tensor());
        self.get_mut_body().as_mut_tensor().fill_with(Scalar::zero);
        naive_update_with_mul(&mut self.get_mut_body().as_mut_polynomial(), other);
    }

    pub fn update_body_with_mul_with_buf(&mut self, other: &Polynomial<&[Scalar]>) {
        fourier_update_with_mul(&mut self.get_mut_body().as_mut_polynomial(), other);
    }

    pub fn update_with_add(&mut self, other: &RLWECiphertext) {
        self.update_mask_with_add(&other.get_mask().as_polynomial_list().get_polynomial(0));
        self.update_body_with_add(&other.get_body().as_polynomial());
    }

    pub fn update_with_sub(&mut self, other: &RLWECiphertext) {
        self.update_mask_with_sub(&other.get_mask().as_polynomial_list().get_polynomial(0));
        self.update_body_with_sub(&other.get_body().as_polynomial());
    }

    pub fn update_with_monomial_div(&mut self, m: MonomialDegree) {
        self.get_mut_body()
            .as_mut_polynomial()
            .update_with_wrapping_unit_monomial_div(m);
        self.get_mut_mask()
            .as_mut_polynomial_list()
            .get_mut_polynomial(0)
            .update_with_wrapping_unit_monomial_div(m);
    }

    /// Run the `trace1(RLWE(\sum_i` `a_i` X^i)) = `RLWE((1/N)*a_0`) operation on this ciphertext.
    pub fn trace1(&self, ksk_map: &HashMap<usize, RLWEKeyswitchKey>) -> Self {
        let n = self.0.polynomial_size().0;
        let mut buf = Self::allocate(PolynomialSize(n));
        let mut out = Self(self.0.clone());
        for i in 1..=log2(n) {
            let k = n / (1 << (i - 1)) + 1;
            let ksk = ksk_map.get(&k).unwrap();
            assert_eq!(ksk.get_subs_k(), k);
            ksk.subs(&mut buf, &out);
            out.update_with_add(&buf);
        }
        out
    }

    /// Run the `trace1(RLWE(\sum_i` `a_i` X^i)) = `RLWE((1/N)*a_0`) operation on this ciphertext
    /// using key switching keys in the fourier domain.
    pub fn trace1_fourier(
        &self,
        out: &mut RLWECiphertext,
        ksk_map: &HashMap<usize, FourierRLWEKeyswitchKey>,
    ) {
        let n = self.0.polynomial_size().0;

        let fft = Fft::new(PolynomialSize(n));
        let fft = fft.as_view();

        let mut mem = GlobalMemBuffer::new(
            fft.forward_scratch()
                .unwrap()
                .and(fft.backward_scratch().unwrap()),
        );
        let mut stack = DynStack::new(&mut mem);

        out.0.as_mut_tensor().fill_with_copy(self.0.as_tensor());

        // TODO remove allocation
        let mut buf_fourier = FourierRLWECiphertext::new(n);

        for i in 1..=log2(n) {
            let k = n / (1 << (i - 1)) + 1;
            let ksk = ksk_map.get(&k).unwrap();

            assert_eq!(ksk.get_subs_k(), k);
            ksk.subs(&mut buf_fourier, out);

            fft.add_backward_as_torus(
                out.get_mut_mask()
                    .as_mut_polynomial_list()
                    .get_mut_polynomial(0),
                buf_fourier.mask.as_view(),
                stack.rb_mut(),
            );

            fft.add_backward_as_torus(
                out.get_mut_body().as_mut_polynomial(),
                buf_fourier.body.as_view(),
                stack.rb_mut(),
            );
        }
    }

    /// Compare this ciphertext c, which encrypts m on the exponent against a value d
    /// the resulting ciphertext encrypts a polynomial m(X) such that
    /// m0 = 1 if m <= d, otherwise m0 = 0, where m0 is the constant term of m(X).
    /// Note that encrypting on the exponent means m -> X^m.
    pub fn less_eq_than(&mut self, d: usize) {
        let n = self.polynomial_size().0;
        assert!(d < n);
        let t_poly = {
            let mut t = vec![Scalar::zero(); n];
            t[0] = Scalar::one();
            for x in t.iter_mut().take(n).skip(n - d) {
                *x = Scalar::MAX; // -1
            }
            Polynomial::from_container(t)
        };

        self.update_body_with_mul_with_buf(&t_poly.as_view());
        self.update_mask_with_mul_with_buf(&t_poly.as_view());
    }

    /// Checks whether this ciphertext c, which encrypts a value m on the exponent
    /// equals to d.
    pub fn eq_to(&mut self, d: usize) {
        let n = self.polynomial_size().0;
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

        self.update_body_with_mul_with_buf(&t_poly.as_view());
        self.update_mask_with_mul_with_buf(&t_poly.as_view());
    }

    /// Run the not gate on this ciphertext, the ciphertext must encrypt a binary scalar.
    /// If c = (a, b = a s + e + q/2 b), then negating it becomes
    /// (-a, q/2 - b) = (-a, -a s - e + q/2 NOT(b))
    pub fn not_in_place(&mut self) {
        let delta = Scalar::one() << (Scalar::BITS - 1);
        for x in self.0.as_mut_tensor().iter_mut() {
            *x = Scalar::zero().wrapping_sub(*x);
        }
        *self.get_mut_body().as_mut_tensor().first_mut() =
            (*self.get_body().as_tensor().first()).wrapping_add(delta);
    }

    /// Return NOT(self) where self must encrypt a binary scalar.
    pub fn not(&self) -> Self {
        let delta = Scalar::one() << (Scalar::BITS - 1);
        let mut out = Self::allocate(self.polynomial_size());
        out.0
            .as_mut_tensor()
            .update_with_wrapping_sub(self.0.as_tensor());
        *out.get_mut_body().as_mut_tensor().first_mut() =
            (*out.get_body().as_tensor().first()).wrapping_add(delta);
        out
    }
}

#[derive(Debug, Clone)]
/// An RLWE secret key.
pub struct RLWESecretKey(pub GlweSecretKey<BinaryKeyKind, Vec<Scalar>>);

impl RLWESecretKey {
    /// Generate a secret key where the coefficients are binary.
    pub fn generate_binary(
        poly_size: PolynomialSize,
        generator: &mut SecretRandomGenerator<SoftwareRandomGenerator>,
    ) -> Self {
        Self(GlweSecretKey::generate_binary(
            GlweDimension(1),
            poly_size,
            generator,
        ))
    }

    /// Generate a trivial secret key where the coefficients are all zero.
    pub fn zero(poly_size: PolynomialSize) -> Self {
        Self(GlweSecretKey::binary_from_container(
            vec![Scalar::zero(); poly_size.0],
            poly_size,
        ))
    }

    pub fn fill_with_tensor<C>(&mut self, t: &Tensor<C>)
    where
        Tensor<C>: AsRefSlice<Element = Scalar>,
    {
        self.0.as_mut_tensor().fill_with_copy(t);
    }

    /// Encode and then encrypt the plaintext pt.
    pub fn encode_encrypt_rlwe(
        &self,
        encrypted: &mut RLWECiphertext,
        pt: &PlaintextList<Vec<Scalar>>,
        ctx: &mut Context,
    ) {
        let mut binary_encoded = pt.clone();
        ctx.codec
            .poly_encode(&mut binary_encoded.as_mut_polynomial());
        self.encrypt_rlwe(
            encrypted,
            &binary_encoded,
            ctx.std,
            &mut ctx.encryption_generator,
        );
    }

    /// Encode and then encrypt the plaintext pt.
    pub fn ternary_encrypt_rlwe(
        &self,
        encrypted: &mut RLWECiphertext,
        pt: &PlaintextList<Vec<Scalar>>,
        ctx: &mut Context,
    ) {
        let mut ternary_encoded = pt.clone();
        Codec::poly_ternary_encode(&mut ternary_encoded.as_mut_polynomial());
        self.encrypt_rlwe(
            encrypted,
            &ternary_encoded,
            ctx.std,
            &mut ctx.encryption_generator,
        );
    }

    /// Encrypt a plaintext pt.
    // TODO change API to use Context
    pub fn encrypt_rlwe(
        &self,
        encrypted: &mut RLWECiphertext,
        pt: &PlaintextList<Vec<Scalar>>,
        noise_parameter: impl DispersionParameter,
        generator: &mut EncryptionRandomGenerator<SoftwareRandomGenerator>,
    ) {
        self.0
            .encrypt_glwe(&mut encrypted.0, pt, noise_parameter, generator);
    }

    /// Encrypt a scalar.
    pub fn encrypt_constant_rlwe(
        &self,
        encrypted: &mut RLWECiphertext,
        pt: &Plaintext<Scalar>,
        ctx: &mut Context,
    ) {
        let mut encoded = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
        *encoded
            .as_mut_polynomial()
            .get_mut_monomial(MonomialDegree(0))
            .get_mut_coefficient() = pt.0;
        self.0.encrypt_glwe(
            &mut encrypted.0,
            &encoded,
            ctx.std,
            &mut ctx.encryption_generator,
        );
    }

    /// Decrypt a RLWE ciphertext.
    pub fn decrypt_rlwe(&self, pt: &mut PlaintextList<Vec<Scalar>>, encrypted: &RLWECiphertext) {
        self.0.decrypt_glwe(pt, &encrypted.0);
    }

    /// Decrypt a RLWE ciphertext and then decode.
    pub fn decrypt_decode_rlwe(
        &self,
        pt: &mut PlaintextList<Vec<Scalar>>,
        encrypted: &RLWECiphertext,
        ctx: &Context,
    ) {
        self.decrypt_rlwe(pt, encrypted);
        ctx.codec.poly_decode(&mut pt.as_mut_polynomial());
    }

    /// Decrypt a RLWE ciphertext and then decode.
    pub fn ternary_decrypt_rlwe(
        &self,
        pt: &mut PlaintextList<Vec<Scalar>>,
        encrypted: &RLWECiphertext,
    ) {
        self.decrypt_rlwe(pt, encrypted);
        Codec::poly_ternary_decode(&mut pt.as_mut_polynomial());
    }

    /// Create an RGSW ciphertext of a constant.
    pub fn encrypt_constant_rgsw(
        &self,
        out: &mut RGSWCiphertext,
        pt: &Plaintext<Scalar>,
        ctx: &mut Context,
    ) {
        self.0
            .encrypt_constant_ggsw(&mut out.0, pt, ctx.std, &mut ctx.encryption_generator);
        // NOTE:for debugging we can use
        // self.0.trivial_encrypt_constant_ggsw(&mut out.0, encoded, ctx.std, &mut ctx.encryption_generator)
    }

    /// Create an RGSW ciphertext of a polynomial.
    pub fn encrypt_rgsw(
        &self,
        out: &mut RGSWCiphertext,
        encoded: &PlaintextList<Vec<Scalar>>,
        ctx: &mut Context,
    ) {
        // first create a constant encryption of 0, then add the decomposed encoded value to it
        self.encrypt_constant_rgsw(out, &Plaintext(Scalar::zero()), ctx);
        let mut buf = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
        for (i, mut m) in out.0.as_mut_glwe_list().ciphertext_iter_mut().enumerate() {
            let level = (i / 2) + 1;
            let shift: usize = (Scalar::BITS as usize) - ctx.base_log.0 * level;
            buf.as_mut_tensor().fill_with_copy(encoded.as_tensor());
            mul_const(buf.as_mut_tensor(), 1 << shift);
            if i % 2 == 0 {
                // in this case we're in the "top half" of the ciphertext
                m.get_mut_mask()
                    .as_mut_polynomial_list()
                    .get_mut_polynomial(0)
                    .update_with_wrapping_add(&buf.as_polynomial());
            } else {
                // this is the "bottom half"
                m.get_mut_body()
                    .as_mut_polynomial()
                    .update_with_wrapping_add(&buf.as_polynomial());
            }
        }
    }

    /// Create a vector of RGSW ciphertexts of a polynomial.
    pub fn encrypt_constant_rgsw_vec(
        &self,
        v: &[Plaintext<Scalar>],
        ctx: &mut Context,
    ) -> Vec<RGSWCiphertext> {
        v.iter()
            .map(|pt| {
                let mut rgsw_ct =
                    RGSWCiphertext::allocate(ctx.poly_size, ctx.base_log, ctx.level_count);
                self.encrypt_constant_rgsw(&mut rgsw_ct, pt, ctx);
                rgsw_ct
            })
            .collect()
    }

    pub fn polynomial_size(&self) -> PolynomialSize {
        self.0.polynomial_size()
    }

    /// Compute RGSW(-s), where s is self
    pub fn neg_gsw(&self, ctx: &mut Context) -> RGSWCiphertext {
        let neg_sk = {
            let mut pt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
            for (x, y) in pt.as_mut_tensor().iter_mut().zip(self.0.as_tensor().iter()) {
                *x = y * Scalar::MAX;
            }
            pt
        };
        let mut neg_sk_ct =
            RGSWCiphertext::allocate(ctx.poly_size, ctx.negs_base_log, ctx.negs_level_count);
        self.encrypt_rgsw(&mut neg_sk_ct, &neg_sk, ctx);
        neg_sk_ct
    }
}

#[derive(Debug, Clone)]
/// An RLWE key switching key.
pub struct RLWEKeyswitchKey {
    ksks: Vec<RLWECiphertext>,
    decomp_base_log: DecompositionBaseLog,
    decomp_level_count: DecompositionLevelCount,
    polynomial_size: PolynomialSize,
    subs_k: usize,
}

impl RLWEKeyswitchKey {
    pub fn allocate(
        decomp_base_log: DecompositionBaseLog,
        decomp_level_count: DecompositionLevelCount,
        polynomial_size: PolynomialSize,
    ) -> Self {
        Self {
            ksks: vec![RLWECiphertext::allocate(polynomial_size); decomp_level_count.0],
            decomp_base_log,
            decomp_level_count,
            polynomial_size,
            subs_k: 0,
        }
    }

    pub const fn decomposition_base_log(&self) -> DecompositionBaseLog {
        self.decomp_base_log
    }

    pub const fn decomposition_level_count(&self) -> DecompositionLevelCount {
        self.decomp_level_count
    }

    pub const fn polynomial_size(&self) -> PolynomialSize {
        self.polynomial_size
    }

    /// Fill this object with the appropriate key switching key
    /// that is used for the substitution (subs) operation
    /// where `after_key` is s(X) and `before_key` is computed as s(X^k).
    pub fn fill_with_subs_keyswitch_key(
        &mut self,
        before_key: &mut RLWESecretKey,
        after_key: &RLWESecretKey,
        k: usize,
        noise_parameters: impl DispersionParameter,
        generator: &mut EncryptionRandomGenerator<SoftwareRandomGenerator>,
    ) {
        // TODO reduce copy
        let before_poly = eval_x_k(&after_key.0.as_polynomial_list().get_polynomial(0), k);
        before_key.fill_with_tensor(before_poly.as_tensor());
        self.fill_with_keyswitch_key(before_key, after_key, noise_parameters, generator);
        self.subs_k = k;
    }

    /// Fill this object with the appropriate key switching key
    /// that transforms ciphertexts under `before_key` to ciphertexts under `after_key`.
    pub fn fill_with_keyswitch_key(
        &mut self,
        before_key: &RLWESecretKey,
        after_key: &RLWESecretKey,
        noise_parameters: impl DispersionParameter,
        generator: &mut EncryptionRandomGenerator<SoftwareRandomGenerator>,
    ) {
        assert_eq!(before_key.0.as_polynomial_list().polynomial_count().0, 1);
        assert_eq!(after_key.0.as_polynomial_list().polynomial_count().0, 1);

        let mut buf =
            PlaintextList::allocate(Scalar::zero(), PlaintextCount(self.polynomial_size.0));

        // We retrieve decomposition arguments
        let decomp_level_count = self.decomp_level_count.0;
        let decomp_base_log = self.decomp_base_log.0;

        for (level, ksk) in (1..=decomp_level_count).zip(&mut self.ksks) {
            buf.as_mut_tensor().fill_with(Scalar::zero);
            buf.as_mut_tensor().fill_with_copy(before_key.0.as_tensor());
            let shift: usize = (Scalar::BITS as usize) - decomp_base_log * level;
            mul_const(buf.as_mut_tensor(), 1 << shift);

            after_key.encrypt_rlwe(ksk, &buf, noise_parameters, generator);
        }
        self.subs_k = 0;
    }

    /// Convert the key switching key into Fourier domain.
    pub fn into_fourier(self) -> FourierRLWEKeyswitchKey {
        let fft = Fft::new(self.polynomial_size);
        let fft = fft.as_view();

        let mut mem = GlobalMemBuffer::new(
            fft.forward_scratch()
                .unwrap()
                .and(fft.backward_scratch().unwrap()),
        );

        let mut stack = DynStack::new(&mut mem);

        let fourier_vec = self
            .ksks
            .iter()
            .map(|ksk| {
                let mut fp_mask = FourierPolynomial {
                    data: vec![Complex::zero(); self.polynomial_size.0 / 2],
                };
                let mut fp_body = FourierPolynomial {
                    data: vec![Complex::zero(); self.polynomial_size.0 / 2],
                };
                fft.forward_as_torus(
                    unsafe { fp_mask.as_mut_view().into_uninit() },
                    ksk.get_mask().as_polynomial_list().get_polynomial(0),
                    stack.rb_mut(),
                );
                fft.forward_as_torus(
                    unsafe { fp_body.as_mut_view().into_uninit() },
                    ksk.get_body().as_polynomial(),
                    stack.rb_mut(),
                );
                FourierRLWECiphertext {
                    mask: fp_mask,
                    body: fp_body,
                }
            })
            .collect();
        FourierRLWEKeyswitchKey {
            ksks: fourier_vec,
            decomp_base_log: self.decomp_base_log,
            decomp_level_count: self.decomp_level_count,
            polynomial_size: self.polynomial_size,
            subs_k: self.subs_k,
        }
    }

    /// Run key switching.
    pub fn keyswitch_ciphertext(&self, after: &mut RLWECiphertext, before: &RLWECiphertext) {
        // clean the output ctxt and add c_1
        after.clear();
        after
            .get_mut_body()
            .as_mut_tensor()
            .update_with_wrapping_add(before.get_body().as_tensor());

        let decomposer = SignedDecomposer::new(self.decomp_base_log, self.decomp_level_count);
        let mut rounded_mask = Tensor::allocate(Scalar::zero(), self.polynomial_size.0);
        decomposer.fill_tensor_with_closest_representable(
            &mut rounded_mask,
            before.get_mask().as_tensor(),
        );
        let mut decomposed_mask = decomposer.decompose_tensor(&rounded_mask);

        // TODO reduce the temporary allocation
        let mut poly_mask = Polynomial::allocate(Scalar::zero(), self.polynomial_size);
        let mut poly_body = Polynomial::allocate(Scalar::zero(), self.polynomial_size);

        // Every chunk is a key switching key
        for ksk in self.ksks.iter().rev() {
            decomposed_mask.next_term().map_or_else(
                || {
                    panic!("no more next_term");
                },
                |term| {
                    assert_eq!(ksk.get_mask().as_polynomial_list().polynomial_count().0, 1);
                    poly_mask.as_mut_tensor().fill_with(Scalar::zero);
                    poly_body.as_mut_tensor().fill_with(Scalar::zero);

                    fourier_update_with_mul_acc(
                        &mut poly_mask.as_mut_view(),
                        &ksk.get_mask().as_polynomial_list().get_polynomial(0),
                        &Polynomial::from_container(term.as_tensor().as_slice()),
                    );
                    fourier_update_with_mul_acc(
                        &mut poly_body.as_mut_view(),
                        &ksk.get_body().as_polynomial(),
                        &Polynomial::from_container(term.as_tensor().as_slice()),
                    );
                    after.update_mask_with_sub(&poly_mask.as_view());
                    after.update_body_with_sub(&poly_body.as_view());
                },
            );
        }
    }

    /// The key switching key must be of the form s(X^k) to s(X),
    /// i.e., `fill_with_subs_keyswitch_key` must be called.
    pub fn subs(&self, after: &mut RLWECiphertext, before: &RLWECiphertext) {
        let k = self.subs_k;
        let mut c = RLWECiphertext::allocate(self.polynomial_size);
        c.0.as_mut_tensor().fill_with_copy(before.0.as_tensor());

        // TODO reduce copying
        let c_mask_k = eval_x_k(&c.get_mask().as_polynomial_list().get_polynomial(0), k);
        let c_body_k = eval_x_k(&c.get_body().as_polynomial(), k);
        c.get_mut_mask()
            .as_mut_tensor()
            .fill_with_copy(c_mask_k.as_tensor());
        c.get_mut_body()
            .as_mut_tensor()
            .fill_with_copy(c_body_k.as_tensor());

        self.keyswitch_ciphertext(after, &c);
    }

    pub const fn get_keyswitch_key(&self) -> &Vec<RLWECiphertext> {
        &self.ksks
    }

    pub const fn get_subs_k(&self) -> usize {
        self.subs_k
    }
}

#[derive(Debug, Clone)]
/// An RLWE ciphertext in the Fourier domain.
pub struct FourierRLWECiphertext {
    pub mask: FourierPolynomial<ComplexContaier>,
    pub body: FourierPolynomial<ComplexContaier>,
}

impl FourierRLWECiphertext {
    pub fn new(poly_size: usize) -> Self {
        Self {
            mask: FourierPolynomial {
                data: vec![Complex::zero(); poly_size / 2],
            },
            body: FourierPolynomial {
                data: vec![Complex::zero(); poly_size / 2],
            },
        }
    }

    /// Convert the ciphertext back to standard domain.
    pub fn backward_as_torus(&mut self) -> RLWECiphertext {
        let p = PolynomialSize(self.body.data.len() * 2);
        let fft = Fft::new(p);
        let fft = fft.as_view();
        let mut out = RLWECiphertext::allocate(p);

        let mut mem = GlobalMemBuffer::new(
            fft.forward_scratch()
                .unwrap()
                .and(fft.backward_scratch().unwrap()),
        );

        let mut stack = DynStack::new(&mut mem);

        fft.add_backward_as_torus(
            out.get_mut_mask()
                .as_mut_polynomial_list()
                .get_mut_polynomial(0),
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

#[derive(Debug, Clone)]
/// An RLWE key switching key in the Fourier domain.
pub struct FourierRLWEKeyswitchKey {
    ksks: Vec<FourierRLWECiphertext>,
    decomp_base_log: DecompositionBaseLog,
    decomp_level_count: DecompositionLevelCount,
    polynomial_size: PolynomialSize,
    subs_k: usize,
}

impl FourierRLWEKeyswitchKey {
    /// Perform key switching but don't convert the new ciphertext to the standard domain.
    pub fn keyswitch_ciphertext(&self, after: &mut FourierRLWECiphertext, before: &RLWECiphertext) {
        let fft = Fft::new(self.polynomial_size);
        let fft = fft.as_view();

        let mut mem = GlobalMemBuffer::new(
            fft.forward_scratch()
                .unwrap()
                .and(fft.forward_scratch().unwrap())
                .and(fft.forward_scratch().unwrap())
                .and(fft.forward_scratch().unwrap())
                .and(fft.forward_scratch().unwrap())
                .and(fft.backward_scratch().unwrap()),
        );

        let mut stack = DynStack::new(&mut mem);

        let mut first_fourier = FourierPolynomial {
            data: vec![Complex::zero(); self.polynomial_size.0 / 2].into_boxed_slice(),
        };
        let mut second_fourier = FourierPolynomial {
            data: vec![Complex::zero(); self.polynomial_size.0 / 2].into_boxed_slice(),
        };

        fft.forward_as_torus(
            unsafe { first_fourier.as_mut_view().into_uninit() },
            before.get_mask().as_polynomial_list().get_polynomial(0),
            stack.rb_mut(),
        );
        fft.forward_as_torus(
            unsafe { second_fourier.as_mut_view().into_uninit() },
            before.get_body().as_polynomial(),
            stack.rb_mut(),
        );

        // clean the output ctxt and add c_1
        for c in after.mask.data.as_mut_slice().iter_mut() {
            *c = Complex::zero();
        }

        // TODO body.data is 2048 but fourier is 1024
        for (c, b) in izip!(&mut *after.body.data, &*second_fourier.data) {
            *c = *b;
        }

        // TODO the decomposer isn't an iterator so we need to make extra allocation
        let decomposer = SignedDecomposer::new(self.decomp_base_log, self.decomp_level_count);
        let mut rounded_mask = Tensor::allocate(Scalar::zero(), self.polynomial_size.0);
        decomposer.fill_tensor_with_closest_representable(
            &mut rounded_mask,
            before.get_mask().as_tensor(),
        );
        let mut decomposed_mask = decomposer.decompose_tensor(&rounded_mask);
        let mut terms = vec![];
        terms.reserve(self.decomp_level_count.0);
        for _ in 0..self.decomp_level_count.0 {
            decomposed_mask.next_term().map_or_else(
                || {
                    panic!("not enough terms");
                },
                |term| {
                    let term = Polynomial::from_container(
                        term.as_tensor()
                            .iter()
                            .map(|x| (0 as Scalar).wrapping_sub(*x))
                            .collect::<Vec<Scalar>>(),
                    );
                    terms.push(term);
                },
            );
        }

        let mut terms_iter = terms.iter();
        let mut ksk_iter = self.ksks.iter().rev();

        loop {
            match (ksk_iter.next(), ksk_iter.next()) {
                (Some(first), Some(second)) => {
                    let term1 = terms_iter.next().unwrap();
                    let term2 = terms_iter.next().unwrap();

                    fft.forward_as_integer(
                        unsafe { first_fourier.as_mut_view().into_uninit() },
                        term1.as_view(),
                        stack.rb_mut(),
                    );
                    fft.forward_as_integer(
                        unsafe { second_fourier.as_mut_view().into_uninit() },
                        term2.as_view(),
                        stack.rb_mut(),
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
                    let term1 = terms_iter.next().unwrap();

                    fft.forward_as_integer(
                        unsafe { first_fourier.as_mut_view().into_uninit() },
                        term1.as_view(),
                        stack.rb_mut(),
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
    }

    /// Perform the substitution operation that converts RLWE(p(X)) to RLWE(p(X^k)).
    /// The key switching key must be of the form s(X^k) to s(X).
    pub fn subs(&self, after: &mut FourierRLWECiphertext, before: &RLWECiphertext) {
        let k = self.subs_k;
        let mut c = RLWECiphertext::allocate(self.polynomial_size);
        c.0.as_mut_tensor().fill_with_copy(before.0.as_tensor());

        // TODO reduce copying
        let c_mask_k = eval_x_k(&c.get_mask().as_polynomial_list().get_polynomial(0), k);
        let c_body_k = eval_x_k(&c.get_body().as_polynomial(), k);
        c.get_mut_mask()
            .as_mut_tensor()
            .fill_with_copy(c_mask_k.as_tensor());
        c.get_mut_body()
            .as_mut_tensor()
            .fill_with_copy(c_body_k.as_tensor());

        self.keyswitch_ciphertext(after, &c);
    }

    pub const fn get_subs_k(&self) -> usize {
        self.subs_k
    }
}

/// Generate all the key switching keys needed for the substitution operation.
pub fn gen_all_subs_ksk(
    after_key: &RLWESecretKey,
    ctx: &mut Context,
) -> HashMap<usize, RLWEKeyswitchKey> {
    let poly_size = ctx.poly_size;
    let mut hm = HashMap::new();
    let mut dummy_sk = RLWESecretKey::zero(poly_size);
    for i in 1..=log2(poly_size.0) {
        let k = poly_size.0 / (1 << (i - 1)) + 1;
        let mut ksk = RLWEKeyswitchKey::allocate(ctx.ks_base_log, ctx.ks_level_count, poly_size);
        ksk.fill_with_subs_keyswitch_key(
            &mut dummy_sk,
            after_key,
            k,
            ctx.std,
            &mut ctx.encryption_generator,
        );
        hm.insert(k, ksk);
    }
    hm
}

/// Generate all the key switching keys needed for the substitution operation
/// in the Fourier domain.
pub fn gen_all_subs_ksk_fourier(
    after_key: &RLWESecretKey,
    ctx: &mut Context,
) -> HashMap<usize, FourierRLWEKeyswitchKey> {
    let poly_size = ctx.poly_size;
    let mut hm = HashMap::new();
    let mut dummy_sk = RLWESecretKey::zero(poly_size);
    for i in 1..=log2(poly_size.0) {
        let k = poly_size.0 / (1 << (i - 1)) + 1;
        let mut ksk = RLWEKeyswitchKey::allocate(ctx.ks_base_log, ctx.ks_level_count, poly_size);
        ksk.fill_with_subs_keyswitch_key(
            &mut dummy_sk,
            after_key,
            k,
            ctx.std,
            &mut ctx.encryption_generator,
        );
        hm.insert(k, ksk.into_fourier());
    }
    hm
}

/// Expand/convert RLWE ciphertexts to an RGSW ciphertext.
/// The number of RLWE ciphertexts is defined by the decomposition level.
pub fn expand(
    cs: &[RLWECiphertext],
    ksk_map: &HashMap<usize, RLWEKeyswitchKey>,
    neg_s: &RGSWCiphertext,
    ctx: &Context,
) -> RGSWCiphertext {
    let mut buf = FftBuffer::new(cs[0].polynomial_size());
    let cs_prime: Vec<RLWECiphertext> = cs.iter().map(|c| c.trace1(ksk_map)).collect();
    let mut out = RGSWCiphertext::allocate(ctx.poly_size, ctx.base_log, ctx.level_count);
    for (i, mut c) in out.0.as_mut_glwe_list().ciphertext_iter_mut().enumerate() {
        let k = i / 2;
        if i % 2 == 0 {
            neg_s.external_product_with_buf_glwe(&mut c, &cs_prime[k], &mut buf);
        } else {
            c.as_mut_tensor().fill_with_copy(cs_prime[k].0.as_tensor());
        }
    }
    out
}

/// Same as expand but using key switching keys in the Fourier domain.
#[allow(clippy::ptr_arg)]
pub fn expand_fourier(
    cs: &Vec<RLWECiphertext>,
    ksk_map: &HashMap<usize, FourierRLWEKeyswitchKey>,
    neg_s: &RGSWCiphertext,
    ctx: &Context,
) -> RGSWCiphertext {
    let mut buf = FftBuffer::new(cs[0].polynomial_size());
    let mut out = RGSWCiphertext::allocate(ctx.poly_size, ctx.base_log, ctx.level_count);
    let mut c_prime = RLWECiphertext::allocate(ctx.poly_size);
    for (i, mut c) in out.0.as_mut_glwe_list().ciphertext_iter_mut().enumerate() {
        let k = i / 2;
        if i % 2 == 0 {
            cs[k].trace1_fourier(&mut c_prime, ksk_map);
            neg_s.external_product_with_buf_glwe(&mut c, &c_prime, &mut buf);
        } else {
            c.as_mut_tensor().fill_with_copy(c_prime.0.as_tensor());
        }
    }
    out
}

/// Use the slower method to convert a set of scaled RLWE ciphertext
/// into a RGSW ciphertext, this operation takes O(N^2) where
/// N is the degree of the polynomial.
pub fn expand_slow(
    cs: &Vec<RLWECiphertext>,
    ksks: &LWEtoRLWEKeyswitchKey,
    neg_s: &RGSWCiphertext,
    ctx: &Context,
) -> RGSWCiphertext {
    assert_eq!(ctx.poly_size.0, ksks.inner.len());
    assert_eq!(cs.len(), ctx.level_count.0);

    let mut buf = FftBuffer::new(cs[0].polynomial_size());
    let rlwe_cts: Vec<RLWECiphertext> = cs
        .iter()
        .map(|c| {
            // TODO reduce allocation
            let mut lwe = LWECiphertext::allocate(ctx.lwe_size());
            lwe.fill_with_const_sample_extract(c);
            // new rlwe below
            conv_lwe_to_rlwe(ksks, &lwe, ctx)
        })
        .collect();

    decomposed_rlwe_to_rgsw(&rlwe_cts, neg_s, ctx, &mut buf)
}

pub fn decomposed_rlwe_to_rgsw(
    cs: &[RLWECiphertext],
    neg_s: &RGSWCiphertext,
    ctx: &Context,
    buf: &mut FftBuffer,
) -> RGSWCiphertext {
    let mut out = RGSWCiphertext::allocate(ctx.poly_size, ctx.base_log, ctx.level_count);
    for (i, mut c) in out.0.as_mut_glwe_list().ciphertext_iter_mut().enumerate() {
        let k = i / 2;
        if i % 2 == 0 {
            neg_s.external_product_with_buf_glwe(&mut c, &cs[k], buf);
        } else {
            c.as_mut_tensor().fill_with_copy(cs[k].0.as_tensor());
        }
    }
    out
}

fn fourier_update_with_mul(p1: &mut Polynomial<&mut [Scalar]>, p2: &Polynomial<&[Scalar]>) {
    let fft = Fft::new(p1.polynomial_size());
    let fft = fft.as_view();

    let mut mem = GlobalMemBuffer::new(
        fft.forward_scratch()
            .unwrap()
            .and(fft.backward_scratch().unwrap()),
    );

    let mut stack = DynStack::new(&mut mem);

    let mut fp1 = FourierPolynomial {
        data: vec![Complex::zero(); p1.polynomial_size().0 / 2].into_boxed_slice(),
    };
    let mut fp2 = FourierPolynomial {
        data: vec![Complex::zero(); p2.polynomial_size().0 / 2].into_boxed_slice(),
    };

    fft.forward_as_torus(
        unsafe { fp1.as_mut_view().into_uninit() },
        p1.as_view(),
        stack.rb_mut(),
    );
    fft.forward_as_integer(
        unsafe { fp2.as_mut_view().into_uninit() },
        p2.as_view(),
        stack.rb_mut(),
    );

    for (f0, f1) in izip!(&mut *fp1.data, &*fp2.data) {
        *f0 *= *f1;
    }

    fft.backward_as_torus(
        unsafe { p1.as_mut_view().into_uninit() },
        fp1.as_view(),
        stack,
    );
}

fn fourier_update_with_mul_acc(
    out: &mut Polynomial<&mut [Scalar]>,
    p1: &Polynomial<&[Scalar]>,
    p2: &Polynomial<&[Scalar]>,
) {
    let fft = Fft::new(p1.polynomial_size());
    let fft = fft.as_view();

    let mut mem = GlobalMemBuffer::new(
        fft.forward_scratch()
            .unwrap()
            .and(fft.backward_scratch().unwrap()),
    );

    let mut stack = DynStack::new(&mut mem);

    let mut fp1 = FourierPolynomial {
        data: vec![Complex::zero(); p1.polynomial_size().0 / 2].into_boxed_slice(),
    };
    let mut fp2 = FourierPolynomial {
        data: vec![Complex::zero(); p2.polynomial_size().0 / 2].into_boxed_slice(),
    };

    fft.forward_as_torus(
        unsafe { fp1.as_mut_view().into_uninit() },
        p1.as_view(),
        stack.rb_mut(),
    );
    fft.forward_as_integer(
        unsafe { fp2.as_mut_view().into_uninit() },
        p2.as_view(),
        stack.rb_mut(),
    );

    for (f0, f1) in izip!(&mut *fp1.data, &*fp2.data) {
        *f0 *= *f1;
    }

    fft.add_backward_as_torus(out.as_mut_view(), fp1.as_view(), stack);
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

pub fn naive_update_with_mul_acc<M, C>(
    out: &mut Polynomial<M>,
    p1: &Polynomial<C>,
    p2: &Polynomial<C>,
) where
    C: AsRefSlice<Element = Scalar>,
    M: AsMutSlice<Element = Scalar>,
{
    let mut tmp = Polynomial::allocate(Scalar::zero(), out.polynomial_size());
    tmp.fill_with_wrapping_mul(p1, p2);
    out.update_with_wrapping_add(&tmp);
}

pub fn naive_update_with_mul<M, C>(p1: &mut Polynomial<M>, p2: &Polynomial<C>)
where
    C: AsRefSlice<Element = Scalar>,
    M: AsMutSlice<Element = Scalar>,
{
    let mut tmp = Polynomial::allocate(Scalar::zero(), p1.polynomial_size());
    tmp.fill_with_wrapping_mul(p1, p2);
    p1.as_mut_tensor().fill_with_copy(tmp.as_tensor());
}

/// Create RLWE ciphertexts that are suitable to be used by expand.
pub fn make_decomposed_rlwe_ct(
    sk: &RLWESecretKey,
    bit: Scalar,
    ctx: &mut Context,
) -> Vec<RLWECiphertext> {
    assert!(bit == Scalar::one() || bit == Scalar::zero());
    let logn = log2(ctx.poly_size.0);
    let out = (1..=ctx.level_count.0).map(|level| {
        assert!(ctx.base_log.0 * level + logn <= Scalar::BITS as usize);
        let shift: usize = (Scalar::BITS as usize) - ctx.base_log.0 * level - logn;
        let ptxt = {
            let mut p = ctx.gen_ternary_ptxt();
            *p.as_mut_polynomial()
                .get_mut_monomial(MonomialDegree(0))
                .get_mut_coefficient() = bit << shift;
            p
        };
        let mut ct = RLWECiphertext::allocate(ctx.poly_size);
        sk.encrypt_rlwe(&mut ct, &ptxt, ctx.std, &mut ctx.encryption_generator);
        ct
    });
    out.collect()
}

/// Create RLWE ciphertexts that are suitable to be used by `expand_slow`.
/// It does not have the - logn term in the shift.
pub fn make_decomposed_rlwe_ct2(
    sk: &RLWESecretKey,
    bit: Scalar,
    ctx: &mut Context,
) -> Vec<RLWECiphertext> {
    assert!(bit == Scalar::one() || bit == Scalar::zero());
    let out = (1..=ctx.level_count.0).map(|level| {
        let shift: usize = (Scalar::BITS as usize) - ctx.base_log.0 * level;
        let ptxt = {
            let mut p = ctx.gen_ternary_ptxt();
            *p.as_mut_polynomial()
                .get_mut_monomial(MonomialDegree(0))
                .get_mut_coefficient() = bit << shift;
            p
        };
        let mut ct = RLWECiphertext::allocate(ctx.poly_size);
        sk.encrypt_rlwe(&mut ct, &ptxt, ctx.std, &mut ctx.encryption_generator);
        ct
    });
    out.collect()
}

/// Compute the noise for ciphertext `ct`
/// given the (possibly encoded) plaintext `ptxt`.
pub fn compute_noise<C>(
    sk: &RLWESecretKey,
    ct: &RLWECiphertext,
    encoded_ptxt: &PlaintextList<C>,
) -> f64
where
    C: AsRefSlice<Element = Scalar>,
{
    // pt = b - a*s = Delta*m + e
    let mut pt = PlaintextList::allocate(Scalar::zero(), encoded_ptxt.count());
    sk.decrypt_rlwe(&mut pt, ct);

    // pt = pt - Delta*m = e (encoded_ptxt is Delta*m)
    pt.as_mut_polynomial()
        .update_with_wrapping_sub(&encoded_ptxt.as_polynomial());

    let mut max_e: SignedScalar = 0;
    for x in pt.as_tensor().iter() {
        // convert x to signed
        let z = (*x as SignedScalar).abs();
        if z > max_e {
            max_e = z;
        }
    }
    (max_e as f64).log2()
}

pub fn compute_noise_ternary<C>(
    sk: &RLWESecretKey,
    ct: &RLWECiphertext,
    ptxt: &PlaintextList<C>,
) -> f64
where
    C: AsRefSlice<Element = Scalar>,
{
    let mut tmp = PlaintextList::allocate(Scalar::zero(), ptxt.count());
    tmp.as_mut_tensor().fill_with_copy(ptxt.as_tensor());
    Codec::poly_ternary_decode(&mut tmp.as_mut_polynomial());
    compute_noise(sk, ct, &tmp)
}

/// Compute the noise for ciphertext `ct`
/// given the unencoded plaintext `ptxt`.
/// So the codec must be given.
pub fn compute_noise_encoded<C>(
    sk: &RLWESecretKey,
    ct: &RLWECiphertext,
    ptxt: &PlaintextList<C>,
    codec: &Codec,
) -> f64
where
    C: AsRefSlice<Element = Scalar>,
{
    let mut tmp = PlaintextList::allocate(Scalar::zero(), ptxt.count());
    tmp.as_mut_tensor().fill_with_copy(ptxt.as_tensor());
    codec.poly_encode(&mut tmp.as_mut_polynomial());
    compute_noise(sk, ct, &tmp)
}

#[cfg(test)]
mod test {

    use crate::params::TFHEParameters;
    use crate::rgsw::compute_noise_rgsw1;

    use super::*;
    use concrete_core::{commons::math::tensor::AsRefTensor, prelude::LogStandardDev};

    #[test]
    fn test_keyswitching() {
        let mut ctx = Context::new(TFHEParameters::default());
        let messages = ctx.gen_ternary_ptxt();

        let sk_after = ctx.gen_rlwe_sk();
        let sk_before = ctx.gen_rlwe_sk();

        let mut ct_after = RLWECiphertext::allocate(ctx.poly_size);
        let mut ct_before = RLWECiphertext::allocate(ctx.poly_size);

        let mut ksk =
            RLWEKeyswitchKey::allocate(ctx.ks_base_log, ctx.ks_level_count, ctx.poly_size);
        ksk.fill_with_keyswitch_key(
            &sk_before,
            &sk_after,
            ctx.std,
            &mut ctx.encryption_generator,
        );

        // encrypts with the before key our messages
        sk_before.ternary_encrypt_rlwe(&mut ct_before, &messages, &mut ctx);
        // println!("msg before: {:?}", messages.as_tensor());
        let mut dec_messages_1 = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
        sk_before.ternary_decrypt_rlwe(&mut dec_messages_1, &ct_before);
        // println!("msg after dec: {:?}", dec_messages_1.as_tensor());
        println!(
            "initial noise: {:?}",
            compute_noise_ternary(&sk_before, &ct_before, &messages)
        );

        ksk.keyswitch_ciphertext(&mut ct_after, &ct_before);

        let mut dec_messages_2 = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
        sk_after.ternary_decrypt_rlwe(&mut dec_messages_2, &ct_after);
        // println!("msg after ks: {:?}", dec_messages_2.as_tensor());

        assert_eq!(dec_messages_1, dec_messages_2);
        assert_eq!(dec_messages_1, messages);
        println!(
            "final noise: {:?}",
            compute_noise_ternary(&sk_after, &ct_after, &messages)
        );
    }

    #[test]
    fn test_keyswitching_fourier() {
        let mut ctx = Context::new(TFHEParameters::default());
        let messages = ctx.gen_ternary_ptxt();

        let sk_after = ctx.gen_rlwe_sk();
        let sk_before = ctx.gen_rlwe_sk();

        let mut ct_after_fourier = FourierRLWECiphertext::new(ctx.poly_size.0);
        let mut ct_before = RLWECiphertext::allocate(ctx.poly_size);

        let mut ksk =
            RLWEKeyswitchKey::allocate(ctx.ks_base_log, ctx.ks_level_count, ctx.poly_size);
        ksk.fill_with_keyswitch_key(
            &sk_before,
            &sk_after,
            ctx.std,
            &mut ctx.encryption_generator,
        );
        // let mut buffers = FourierBuffers::new(ctx.poly_size, GlweSize(2));
        let ksk_fourier = ksk.into_fourier();

        // encrypts with the before key our messages
        sk_before.ternary_encrypt_rlwe(&mut ct_before, &messages, &mut ctx);
        // println!("msg before: {:?}", messages.as_tensor());
        let mut dec_messages_1 = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
        sk_before.ternary_decrypt_rlwe(&mut dec_messages_1, &ct_before);
        // println!("msg after dec: {:?}", dec_messages_1.as_tensor());
        println!(
            "initial noise: {:?}",
            compute_noise_ternary(&sk_before, &ct_before, &messages)
        );

        ksk_fourier.keyswitch_ciphertext(&mut ct_after_fourier, &ct_before);
        let ct_after = ct_after_fourier.backward_as_torus();

        let mut dec_messages_2 = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
        sk_after.ternary_decrypt_rlwe(&mut dec_messages_2, &ct_after);
        // println!("msg after ks: {:?}", dec_messages_2.as_tensor());

        assert_eq!(dec_messages_1, dec_messages_2);
        assert_eq!(dec_messages_1, messages);
        println!(
            "final noise: {:?}",
            compute_noise_ternary(&sk_after, &ct_after, &messages)
        );
    }

    #[test]
    fn test_subs() {
        let mut ctx = Context::new(TFHEParameters::default());
        let messages = ctx.gen_ternary_ptxt();
        let k = ctx.poly_size.0 + 1;

        let sk_after = ctx.gen_rlwe_sk();
        let mut sk_before = ctx.gen_rlwe_sk();

        let mut ct_after = RLWECiphertext::allocate(ctx.poly_size);
        let mut ct_before = RLWECiphertext::allocate(ctx.poly_size);

        let mut ksk =
            RLWEKeyswitchKey::allocate(ctx.ks_base_log, ctx.ks_level_count, ctx.poly_size);
        ksk.fill_with_subs_keyswitch_key(
            &mut sk_before,
            &sk_after,
            k,
            ctx.std,
            &mut ctx.encryption_generator,
        );

        // encrypt the message using the after key, put it in ct_before
        sk_after.ternary_encrypt_rlwe(&mut ct_before, &messages, &mut ctx);
        ksk.subs(&mut ct_after, &ct_before);

        let mut decrypted = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
        sk_after.ternary_decrypt_rlwe(&mut decrypted, &ct_after);

        let mut expected = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
        expected
            .as_mut_tensor()
            .fill_with_copy(eval_x_k(&messages.as_polynomial(), k).as_tensor());
        println!("msg after ks: {:?}", decrypted.as_tensor());
        println!("expected msg: {:?}", expected.as_tensor());
        assert_eq!(decrypted, expected);
    }

    #[test]
    fn test_subs_fourier() {
        let mut ctx = Context::new(TFHEParameters::default());
        let messages = ctx.gen_ternary_ptxt();
        let k = ctx.poly_size.0 + 1;

        let sk_after = ctx.gen_rlwe_sk();
        let mut sk_before = ctx.gen_rlwe_sk();

        let mut ct_after_fourier = FourierRLWECiphertext::new(ctx.poly_size.0);
        let mut ct_before = RLWECiphertext::allocate(ctx.poly_size);

        let mut ksk =
            RLWEKeyswitchKey::allocate(ctx.ks_base_log, ctx.ks_level_count, ctx.poly_size);
        ksk.fill_with_subs_keyswitch_key(
            &mut sk_before,
            &sk_after,
            k,
            ctx.std,
            &mut ctx.encryption_generator,
        );
        // let mut buffers = FourierBuffers::new(ctx.poly_size, GlweSize(2));
        let ksk_fourier = ksk.into_fourier();

        // encrypt the message using the after key, put it in ct_before
        sk_after.ternary_encrypt_rlwe(&mut ct_before, &messages, &mut ctx);
        ksk_fourier.subs(&mut ct_after_fourier, &ct_before);
        let ct_after = ct_after_fourier.backward_as_torus();

        let mut decrypted = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
        sk_after.ternary_decrypt_rlwe(&mut decrypted, &ct_after);

        let mut expected = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
        expected
            .as_mut_tensor()
            .fill_with_copy(eval_x_k(&messages.as_polynomial(), k).as_tensor());
        println!("msg after ks: {:?}", decrypted.as_tensor());
        println!("expected msg: {:?}", expected.as_tensor());
        assert_eq!(decrypted, expected);
    }

    #[test]
    fn test_eval_poly() {
        let neg_one = Scalar::MAX;
        let neg_two = neg_one - 1;
        let neg_three = neg_one - 2;
        let poly = Polynomial::from_container(vec![0, 1, 2, 3]);
        {
            let out = eval_x_k(&poly, 3);
            let expected = Polynomial::from_container(vec![0, 3, neg_two, 1]);
            assert_eq!(out, expected);
        }
        {
            let out = eval_x_k(&poly, 5);
            let expected = Polynomial::from_container(vec![0, neg_one, 2, neg_three]);
            assert_eq!(out, expected);
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
        for coeff in encoded_msg.as_mut_polynomial().coefficient_iter_mut() {
            *coeff /= ctx.poly_size.0 as Scalar;
        }

        let sk = RLWESecretKey::generate_binary(ctx.poly_size, &mut ctx.secret_generator);
        let mut ct = RLWECiphertext::allocate(ctx.poly_size);

        let all_ksk = gen_all_subs_ksk(&sk, &mut ctx);

        sk.encrypt_rlwe(
            &mut ct,
            &encoded_msg,
            ctx.std,
            &mut ctx.encryption_generator,
        );
        println!("initial noise: {:?}", compute_noise(&sk, &ct, &encoded_msg));

        let out = ct.trace1(&all_ksk);

        let mut decrypted = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
        sk.decrypt_decode_rlwe(&mut decrypted, &out, &ctx);

        // println!("ptxt after: {:?}", decrypted);

        let expected = {
            let mut tmp = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
            *tmp.as_mut_polynomial()
                .get_mut_monomial(MonomialDegree(0))
                .get_mut_coefficient() = *orig_msg
                .as_polynomial()
                .get_monomial(MonomialDegree(0))
                .get_coefficient();
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

        let sk = ctx.gen_rlwe_sk();
        let mut ct = RLWECiphertext::allocate(ctx.poly_size);
        sk.encode_encrypt_rlwe(&mut ct, &ptxt_expected, &mut ctx);

        let mut ptxt_actual = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
        sk.decrypt_decode_rlwe(&mut ptxt_actual, &ct, &ctx);

        assert_eq!(ptxt_actual, ptxt_expected);
    }

    #[test]
    fn test_ternary_enc() {
        let mut ctx = Context::new(TFHEParameters::default());
        let ptxt_expected = ctx.gen_ternary_ptxt();

        let sk = ctx.gen_rlwe_sk();
        let mut ct = RLWECiphertext::allocate(ctx.poly_size);
        sk.ternary_encrypt_rlwe(&mut ct, &ptxt_expected, &mut ctx);

        let mut ptxt_actual = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
        sk.ternary_decrypt_rlwe(&mut ptxt_actual, &ct);

        assert_eq!(ptxt_actual, ptxt_expected);
    }

    #[test]
    fn test_encrypt_rgsw() {
        let mut ctx = Context::new(TFHEParameters::default());
        let gsw_pt = ctx.gen_binary_pt();
        let one_pt = ctx.gen_unit_pt();

        let sk = ctx.gen_rlwe_sk();
        let mut gsw_ct = RGSWCiphertext::allocate(ctx.poly_size, ctx.base_log, ctx.level_count);
        let mut lwe_ct = RLWECiphertext::allocate(ctx.poly_size);

        sk.encrypt_rgsw(&mut gsw_ct, &gsw_pt, &mut ctx);
        sk.encode_encrypt_rlwe(&mut lwe_ct, &one_pt, &mut ctx);
        println!(
            "initial noise: {:?}",
            compute_noise_encoded(&sk, &lwe_ct, &one_pt, &ctx.codec)
        );

        {
            // check the first row of the RGSW ciphertext
            // the first row should have the form (a + m*(q/B), a*s + e),
            // so we subtract m*(q/B) and then check the noise
            let mut pt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
            pt.as_mut_tensor().fill_with_copy(gsw_pt.as_tensor());
            let shift: usize = (Scalar::BITS as usize) - ctx.base_log.0;
            mul_const(pt.as_mut_tensor(), 1 << shift);

            let mut row_ct = gsw_ct.get_nth_row(0);
            row_ct
                .get_mut_mask()
                .as_mut_polynomial_list()
                .get_mut_polynomial(0)
                .update_with_wrapping_sub(&pt.as_polynomial());
            println!(
                "first row noise: {:?}",
                compute_noise(&sk, &row_ct, &ctx.gen_zero_pt())
            );
        }
        {
            // check the second row of the RGSW ciphertext
            let row_ct = gsw_ct.get_nth_row(1);
            let mut pt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
            pt.as_mut_tensor().fill_with_copy(gsw_pt.as_tensor());
            let shift: usize = (Scalar::BITS as usize) - ctx.base_log.0;
            mul_const(pt.as_mut_tensor(), 1 << shift);
            println!("second row noise: {:?}", compute_noise(&sk, &row_ct, &pt));
        }
        {
            // check the last row of the RGSW ciphertext
            let row_ct = gsw_ct.get_last_row();
            let mut pt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
            pt.as_mut_tensor().fill_with_copy(gsw_pt.as_tensor());
            let shift: usize = (Scalar::BITS as usize) - ctx.base_log.0 * ctx.level_count.0;
            mul_const(pt.as_mut_tensor(), 1 << shift);
            println!("last row noise: {:?}", compute_noise(&sk, &row_ct, &pt));
        }

        let mut prod_ct = RLWECiphertext::allocate(ctx.poly_size);
        gsw_ct.external_product(&mut prod_ct, &lwe_ct);
        let mut actual_pt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
        sk.decrypt_decode_rlwe(&mut actual_pt, &prod_ct, &ctx);

        assert_eq!(actual_pt, gsw_pt);
        println!(
            "final noise: {:?}",
            compute_noise_encoded(&sk, &prod_ct, &gsw_pt, &ctx.codec)
        );
    }

    #[test]
    fn test_negs() {
        let mut ctx = Context::new(TFHEParameters::default());

        // we use another noise
        // so that the initial rlwe ciphertext has noise of ~28 bits,
        // which is the final noise of running trace
        let mut ctx_noisy = Context {
            std: LogStandardDev(-37.5),
            ..Context::new(TFHEParameters::default())
        };

        let sk = ctx.gen_rlwe_sk();
        let neg_sk = {
            let mut pt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
            for (x, y) in pt.as_mut_tensor().iter_mut().zip(sk.0.as_tensor().iter()) {
                *x = y * Scalar::MAX;
            }
            pt
        };
        let neg_gsw_sk = sk.neg_gsw(&mut ctx);
        // check noise of some rows
        {
            let row_ct = neg_gsw_sk.get_last_row();
            let mut row_pt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
            row_pt.as_mut_tensor().fill_with_copy(neg_sk.as_tensor());
            let shift: usize =
                (Scalar::BITS as usize) - ctx.negs_base_log.0 * ctx.negs_level_count.0;
            mul_const(row_pt.as_mut_tensor(), 1 << shift);
            println!("last row noise: {:?}", compute_noise(&sk, &row_ct, &row_pt));
        }
        {
            let row_ct = neg_gsw_sk.get_nth_row(1);
            let mut row_pt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
            row_pt.as_mut_tensor().fill_with_copy(neg_sk.as_tensor());
            let shift: usize = (Scalar::BITS as usize) - ctx.negs_base_log.0;
            mul_const(row_pt.as_mut_tensor(), 1 << shift);
            println!(
                "second row noise: {:?}",
                compute_noise(&sk, &row_ct, &row_pt)
            );
        }

        let one_pt = ctx.gen_unit_pt();
        let mut ct_lwe = RLWECiphertext::allocate(ctx.poly_size);
        sk.ternary_encrypt_rlwe(&mut ct_lwe, &one_pt, &mut ctx_noisy);
        println!(
            "initial noise: {:?}",
            compute_noise_ternary(&sk, &ct_lwe, &one_pt)
        );

        let mut ct_prod = RLWECiphertext::allocate(ctx.poly_size);
        neg_gsw_sk.external_product(&mut ct_prod, &ct_lwe);

        let mut actual = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
        sk.ternary_decrypt_rlwe(&mut actual, &ct_prod);

        assert_eq!(actual, neg_sk);
        println!(
            "final noise: {:?}",
            compute_noise_ternary(&sk, &ct_prod, &neg_sk)
        );
    }

    #[test]
    fn test_expand() {
        let mut ctx = Context::new(TFHEParameters::default());

        let sk = ctx.gen_rlwe_sk();
        let neg_sk_ct = sk.neg_gsw(&mut ctx);
        let ksk_map = gen_all_subs_ksk(&sk, &mut ctx);

        let test_pt = ctx.gen_binary_pt();
        let mut test_ct = RLWECiphertext::allocate(ctx.poly_size);
        sk.encode_encrypt_rlwe(&mut test_ct, &test_pt, &mut ctx);

        {
            let zero_cts = make_decomposed_rlwe_ct(&sk, Scalar::one(), &mut ctx);
            let gsw_ct = expand(&zero_cts, &ksk_map, &neg_sk_ct, &ctx); // this should be 1

            // check noise of some rows
            {
                let neg_sk = {
                    let mut pt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
                    for (x, y) in pt.as_mut_tensor().iter_mut().zip(sk.0.as_tensor().iter()) {
                        *x = y * Scalar::MAX;
                    }
                    pt
                };
                let row_ct = gsw_ct.get_nth_row(0);
                let mut row_pt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
                row_pt
                    .as_mut_tensor()
                    .fill_with_copy(ctx.gen_unit_pt().as_tensor());
                let shift: usize = (Scalar::BITS as usize) - ctx.negs_base_log.0;
                mul_const(row_pt.as_mut_tensor(), 1 << shift);
                naive_update_with_mul(&mut row_pt.as_mut_polynomial(), &neg_sk.as_polynomial());
                println!(
                    "first row noise: {:?}",
                    compute_noise(&sk, &row_ct, &row_pt)
                );
            }
            {
                let row_ct = gsw_ct.get_nth_row(1);
                let mut row_pt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
                row_pt
                    .as_mut_tensor()
                    .fill_with_copy(ctx.gen_unit_pt().as_tensor());
                let shift: usize = (Scalar::BITS as usize) - ctx.negs_base_log.0;
                mul_const(row_pt.as_mut_tensor(), 1 << shift);
                println!(
                    "second row noise: {:?}",
                    compute_noise(&sk, &row_ct, &row_pt)
                );
            }
            {
                let row_ct = gsw_ct.get_last_row();
                let mut row_pt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
                row_pt
                    .as_mut_tensor()
                    .fill_with_copy(ctx.gen_unit_pt().as_tensor());
                let shift: usize =
                    (Scalar::BITS as usize) - ctx.negs_base_log.0 * ctx.negs_level_count.0;
                mul_const(row_pt.as_mut_tensor(), 1 << shift);
                println!("last row noise: {:?}", compute_noise(&sk, &row_ct, &row_pt));
            }

            println!(
                "average row nosie: {:?}",
                compute_noise_rgsw1(&sk, &gsw_ct, &ctx)
            );

            // decrypt and compare
            let mut lwe_ct = RLWECiphertext::allocate(ctx.poly_size);
            gsw_ct.external_product(&mut lwe_ct, &test_ct);
            let mut pt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
            sk.decrypt_decode_rlwe(&mut pt, &lwe_ct, &ctx);
            assert_eq!(test_pt, pt);
            println!(
                "final noise: {:?}",
                compute_noise_encoded(&sk, &lwe_ct, &test_pt, &ctx.codec)
            );
        }
        {
            let zero_cts = make_decomposed_rlwe_ct(&sk, Scalar::zero(), &mut ctx);
            let gsw_ct = expand(&zero_cts, &ksk_map, &neg_sk_ct, &ctx);

            // decrypt and compare
            let mut lwe_ct = RLWECiphertext::allocate(ctx.poly_size);
            gsw_ct.external_product(&mut lwe_ct, &test_ct);
            let mut pt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
            sk.decrypt_decode_rlwe(&mut pt, &lwe_ct, &ctx);
            let zero_pt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
            assert_eq!(zero_pt, pt);
            println!(
                "final noise: {:?}",
                compute_noise_encoded(&sk, &lwe_ct, &zero_pt, &ctx.codec)
            );
        }
    }

    #[test]
    fn test_fourier_mul() {
        let mut ctx = Context::new(TFHEParameters::default());
        let n = ctx.poly_size;
        let mut out_fourier = Polynomial::allocate(Scalar::zero(), n);
        let mut out_naive = Polynomial::allocate(Scalar::zero(), n);

        let reps = 10;
        for _ in 0..reps {
            let mut a = Polynomial::allocate(Scalar::zero(), n);
            let mut b = Polynomial::allocate(Scalar::zero(), n);
            ctx.random_generator
                .fill_tensor_with_random_uniform_ternary(&mut a);
            ctx.random_generator.fill_tensor_with_random_uniform(&mut b);

            fourier_update_with_mul_acc(&mut out_fourier.as_mut_view(), &a.as_view(), &b.as_view());
            naive_update_with_mul_acc(&mut out_naive, &a, &b);

            for (actual, expected) in out_fourier
                .coefficient_iter()
                .zip(out_naive.coefficient_iter())
            {
                assert!((*actual as f64 - *expected as f64).abs() < 1e-9 * Scalar::MAX as f64);
            }
        }
    }

    #[test]
    fn test_less_eq() {
        let mut ctx = Context::new(TFHEParameters::default());
        let sk = ctx.gen_rlwe_sk();

        let m = ctx.poly_size.0 / 2;
        let mut ptxt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
        *ptxt
            .as_mut_polynomial()
            .get_mut_monomial(MonomialDegree(m))
            .get_mut_coefficient() = Scalar::one();

        for i in 1..(ctx.poly_size.0 - m) {
            let mut ct = RLWECiphertext::allocate(ctx.poly_size);
            sk.encode_encrypt_rlwe(&mut ct, &ptxt, &mut ctx);

            ct.less_eq_than(m + i);
            let mut out = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
            sk.decrypt_decode_rlwe(&mut out, &ct, &ctx);
            assert_eq!(
                *out.as_polynomial()
                    .get_monomial(MonomialDegree(0))
                    .get_coefficient(),
                Scalar::one()
            );
        }

        for i in 1..(ctx.poly_size.0 - m) {
            let mut ct = RLWECiphertext::allocate(ctx.poly_size);
            sk.encode_encrypt_rlwe(&mut ct, &ptxt, &mut ctx);

            ct.less_eq_than(m - i);
            let mut out = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
            sk.decrypt_decode_rlwe(&mut out, &ct, &ctx);
            assert_eq!(
                *out.as_polynomial()
                    .get_monomial(MonomialDegree(0))
                    .get_coefficient(),
                Scalar::zero()
            );
        }
    }

    #[test]
    fn test_eq_to() {
        let mut ctx = Context::new(TFHEParameters::default());
        let sk = ctx.gen_rlwe_sk();

        let m = ctx.poly_size.0 / 2;
        let mut ptxt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
        *ptxt
            .as_mut_polynomial()
            .get_mut_monomial(MonomialDegree(m))
            .get_mut_coefficient() = Scalar::one();

        for i in 0..ctx.poly_size.0 {
            let mut ct = RLWECiphertext::allocate(ctx.poly_size);
            sk.encode_encrypt_rlwe(&mut ct, &ptxt, &mut ctx);

            ct.eq_to(i);
            let mut out = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
            sk.decrypt_decode_rlwe(&mut out, &ct, &ctx);
            let res = *out
                .as_polynomial()
                .get_monomial(MonomialDegree(0))
                .get_coefficient();
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
        let sk = ctx.gen_rlwe_sk();

        let zero_msg = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
        let mut binary_msg = ctx.gen_binary_pt();
        ctx.codec.poly_encode(&mut binary_msg.as_mut_polynomial());

        let mut ct = RLWECiphertext::allocate(ctx.poly_size);
        let mut ct_zero = RLWECiphertext::allocate(ctx.poly_size);
        sk.encrypt_rlwe(&mut ct, &binary_msg, ctx.std, &mut ctx.encryption_generator);
        sk.encrypt_rlwe(
            &mut ct_zero,
            &zero_msg,
            ctx.std,
            &mut ctx.encryption_generator,
        );

        // the real support in all of the reals, but we need to approximate it
        // the log support is about log2(6*sigma), and sigma = Scalar::MAX * error_std
        let max_log_support = 3 + i64::from(Scalar::BITS) + ctx.std.get_log_standard_dev() as i64;
        println!("support: {max_log_support:?}");

        let noise_0 = compute_noise(&sk, &ct, &binary_msg);
        println!("noise_0: {noise_0:?}");
        assert!(noise_0 < max_log_support as f64);

        // now if we add another ciphertext then the noise should increase
        ct.update_with_add(&ct_zero);
        let noise_1 = compute_noise(&sk, &ct, &binary_msg);
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
        let sk = ctx.gen_rlwe_sk();

        let one = ctx.gen_unit_pt();
        let mut one_ct = RLWECiphertext::allocate(ctx.poly_size);
        sk.encode_encrypt_rlwe(&mut one_ct, &one, &mut ctx);
        one_ct.not_in_place();
        let mut actual = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
        sk.decrypt_decode_rlwe(&mut actual, &one_ct, &ctx);
        let expected = ctx.gen_zero_pt();
        assert_eq!(expected, actual);

        {
            one_ct.not_in_place();
            let mut actual = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
            sk.decrypt_decode_rlwe(&mut actual, &one_ct, &ctx);
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
        let sk = ctx.gen_rlwe_sk();

        let one = ctx.gen_unit_pt();
        let mut one_ct = RLWECiphertext::allocate(ctx.poly_size);
        sk.encode_encrypt_rlwe(&mut one_ct, &one, &mut ctx);

        {
            let mut actual = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
            sk.decrypt_decode_rlwe(&mut actual, &one_ct.not(), &ctx);
            assert_eq!(ctx.gen_zero_pt(), actual);
        }

        {
            let mut actual = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
            sk.decrypt_decode_rlwe(&mut actual, &one_ct.not().not(), &ctx);
            assert_eq!(ctx.gen_unit_pt(), actual);
        }
    }

    #[test]
    fn test_expand_slow() {
        let mut ctx = Context {
            poly_size: PolynomialSize(256),
            ks_level_count: DecompositionLevelCount(11),
            ..Context::new(TFHEParameters::default())
        };
        // let mut ctx = Context::new(TFHEParameters::default());

        // setup keys
        let lwe_sk = ctx.gen_lwe_sk();
        let rlwe_sk = lwe_sk.to_rlwe_sk();
        let neg_sk_ct = rlwe_sk.neg_gsw(&mut ctx);
        let mut ksks = LWEtoRLWEKeyswitchKey::allocate(&ctx);
        ksks.fill_with_keyswitching_key(&lwe_sk, &mut ctx);

        // create a test plaintext and encrypt it
        // which we will multiply with an rgsw ciphertext encryption of 1 later
        // to check the correctness of the rgsw ciphertext
        let test_pt = ctx.gen_unit_pt();
        let mut test_ct = RLWECiphertext::allocate(ctx.poly_size);
        rlwe_sk.encode_encrypt_rlwe(&mut test_ct, &test_pt, &mut ctx);

        // create the decomposed ciphertexts and then do the conversion
        let cts = make_decomposed_rlwe_ct2(&rlwe_sk, Scalar::one(), &mut ctx);
        let actual_rgsw = expand_slow(&cts, &ksks, &neg_sk_ct, &ctx);

        // check correctness
        let mut actual_rlwe = RLWECiphertext::allocate(ctx.poly_size);
        actual_rgsw.external_product(&mut actual_rlwe, &test_ct);
        let mut pt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
        rlwe_sk.decrypt_decode_rlwe(&mut pt, &actual_rlwe, &ctx);
        assert_eq!(test_pt, pt);

        println!(
            "average row noise: {:?}",
            compute_noise_rgsw1(&rlwe_sk, &actual_rgsw, &ctx)
        );

        println!(
            "final noise: {:?}",
            compute_noise_encoded(&rlwe_sk, &actual_rlwe, &pt, &ctx.codec)
        );
    }
}
