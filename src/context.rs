use crate::{
    codec::Codec,
    num_types::{AlignedScalarContainer, One, Scalar, ScalarContainer, Zero},
    params::TFHEParameters,
};

use aligned_vec::{avec, AVec, CACHELINE_ALIGN};
use tfhe::core_crypto::{
    commons::{
        generators::DeterministicSeeder,
        math::random::{RandomGenerator, Seed},
    },
    prelude::{
        add_external_product_assign_mem_optimized_requirement, new_seeder,
        ActivatedRandomGenerator, CiphertextModulus, ComputationBuffers, DecompositionBaseLog,
        DecompositionLevelCount, DispersionParameter, EncryptionRandomGenerator, Fft,
        GlweCiphertext, GlweDimension, GlweSize, LogStandardDev, LweDimension, LweSize, Plaintext,
        PlaintextCount, PlaintextList, PolynomialSize, SecretRandomGenerator, UnsignedInteger,
    },
};

pub struct FftBuffer {
    pub(crate) fft: Fft,
    pub(crate) mem: ComputationBuffers,
}

impl FftBuffer {
    pub fn new(poly_size: PolynomialSize) -> Self {
        let fft = Fft::new(poly_size);

        let mut mem = ComputationBuffers::new();
        mem.resize(
            add_external_product_assign_mem_optimized_requirement::<Scalar>(
                GlweSize(2),
                poly_size,
                fft.as_view(),
            )
            .unwrap()
            .unaligned_bytes_required(),
        );

        Self { fft, mem }
    }
}

/// The context structure holds the TFHE parameters and
/// random number generators.
pub struct Context<T>
where
    T: UnsignedInteger,
{
    pub random_generator: RandomGenerator<ActivatedRandomGenerator>,
    pub secret_generator: SecretRandomGenerator<ActivatedRandomGenerator>,
    pub encryption_generator: EncryptionRandomGenerator<ActivatedRandomGenerator>,
    pub glwe_size: GlweSize,
    pub glwe_dimension: GlweDimension,
    pub std: LogStandardDev,
    pub poly_size: PolynomialSize,
    pub base_log: DecompositionBaseLog,
    pub level_count: DecompositionLevelCount,
    pub ks_base_log: DecompositionBaseLog,
    pub ks_level_count: DecompositionLevelCount,
    pub negs_base_log: DecompositionBaseLog,
    pub negs_level_count: DecompositionLevelCount,
    pub codec: Codec,
    pub ciphertext_modulus: CiphertextModulus<T>,
    pub fft: Fft,
}

impl std::fmt::Display for Context<Scalar> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "q={:?}, poly_size={:?}, log_std={:?}, default_decomp=({:?},{:?}), ks_decomp=({:?},{:?}), negs_decomp=({:?},{:?}), pt_modulus=({:?})",
                 Scalar::BITS, self.poly_size.0, self.std.get_log_standard_dev(),
                 self.base_log.0, self.level_count.0, self.ks_base_log.0, self.ks_level_count.0, self.negs_base_log.0, self.negs_level_count.0, self.codec.pt_modulus())
    }
}

impl Context<Scalar> {
    /// Create the default context that is suitable for
    /// all experiments in the repository.
    pub fn new(tfhe_params: TFHEParameters) -> Self {
        let mut seeder = new_seeder();

        let random_generator =
            RandomGenerator::<ActivatedRandomGenerator>::new(seeder.as_mut().seed());
        let secret_generator =
            SecretRandomGenerator::<ActivatedRandomGenerator>::new(seeder.as_mut().seed());
        let encryption_generator = EncryptionRandomGenerator::<ActivatedRandomGenerator>::new(
            seeder.as_mut().seed(),
            seeder.as_mut(),
        );

        let glwe_size = GlweSize(2);
        let glwe_dimension = GlweDimension(1);

        let std = LogStandardDev::from_log_standard_dev(tfhe_params.standard_deviation);
        let poly_size = PolynomialSize(tfhe_params.polynomial_size);
        let base_log = DecompositionBaseLog(tfhe_params.base_log);
        let level_count = DecompositionLevelCount(tfhe_params.level_count);
        let ks_base_log = DecompositionBaseLog(tfhe_params.key_switch_base_log);
        let ks_level_count = DecompositionLevelCount(tfhe_params.key_switch_level_count);
        let negs_base_log = DecompositionBaseLog(tfhe_params.negs_base_log);
        let negs_level_count = DecompositionLevelCount(tfhe_params.negs_level_count);
        let codec = Codec::new(tfhe_params.plaintext_modulus);
        let ciphertext_modulus = CiphertextModulus::<Scalar>::new_native();

        let fft = Fft::new(poly_size);

        Self {
            random_generator,
            secret_generator,
            encryption_generator,
            glwe_size,
            glwe_dimension,
            std,
            poly_size,
            base_log,
            level_count,
            ks_base_log,
            ks_level_count,
            negs_base_log,
            negs_level_count,
            codec,
            ciphertext_modulus,
            fft,
        }
    }
    pub fn new_debug(tfhe_params: TFHEParameters) -> Self {
        let mut seeder = DeterministicSeeder::<ActivatedRandomGenerator>::new(Seed(0));

        let random_generator = RandomGenerator::<ActivatedRandomGenerator>::new(Seed(0));
        let secret_generator = SecretRandomGenerator::<ActivatedRandomGenerator>::new(Seed(0));
        let encryption_generator =
            EncryptionRandomGenerator::<ActivatedRandomGenerator>::new(Seed(0), &mut seeder);

        let glwe_size = GlweSize(2);
        let glwe_dimension = GlweDimension(1);

        let std = LogStandardDev::from_log_standard_dev(tfhe_params.standard_deviation);
        let poly_size = PolynomialSize(tfhe_params.polynomial_size);
        let base_log = DecompositionBaseLog(tfhe_params.base_log);
        let level_count = DecompositionLevelCount(tfhe_params.level_count);
        let ks_base_log = DecompositionBaseLog(tfhe_params.key_switch_base_log);
        let ks_level_count = DecompositionLevelCount(tfhe_params.key_switch_level_count);
        let negs_base_log = DecompositionBaseLog(tfhe_params.negs_base_log);
        let negs_level_count = DecompositionLevelCount(tfhe_params.negs_level_count);
        let codec = Codec::new(tfhe_params.plaintext_modulus);
        let ciphertext_modulus = CiphertextModulus::<Scalar>::new_native();

        let fft = Fft::new(poly_size);

        Self {
            random_generator,
            secret_generator,
            encryption_generator,
            glwe_size,
            glwe_dimension,
            std,
            poly_size,
            base_log,
            level_count,
            ks_base_log,
            ks_level_count,
            negs_base_log,
            negs_level_count,
            codec,
            ciphertext_modulus,
            fft,
        }
    }
    pub const fn lwe_dim(&self) -> LweDimension {
        LweDimension(self.poly_size.0)
    }

    pub const fn lwe_size(&self) -> LweSize {
        LweSize(self.poly_size.0 + 1)
    }

    /// Output the plaintext count.
    pub const fn plaintext_count(&self) -> PlaintextCount {
        PlaintextCount(self.poly_size.0)
    }

    /// Generate a binary plaintext list.
    pub fn gen_binary_pt(&mut self) -> PlaintextList<ScalarContainer> {
        PlaintextList::from_container(
            (0..self.plaintext_count().0)
                .map(|_| self.random_generator.random_uniform_binary())
                .collect(),
        )
    }
    pub fn empty_glwe_ciphertext(&self) -> GlweCiphertext<AlignedScalarContainer> {
        GlweCiphertext::from_container(
            avec![Scalar::zero(); self.poly_size.0 * 2],
            self.poly_size,
            self.ciphertext_modulus,
        )
    }
    pub fn glwe_ciphertext_from(
        &self,
        mut mask: AlignedScalarContainer,
        mut body: AlignedScalarContainer,
    ) -> GlweCiphertext<AlignedScalarContainer> {
        GlweCiphertext::from_container(
            AVec::<Scalar>::from_iter(
                CACHELINE_ALIGN,
                mask.as_mut().iter().chain(body.as_mut().iter()).copied(),
            ),
            self.poly_size,
            self.ciphertext_modulus,
        )
    }
    /// Generate a plaintext list where the coefficients sampled from the plaintext space
    /// defined by the codec.
    pub fn gen_pt(&mut self) -> PlaintextList<ScalarContainer> {
        PlaintextList::from_container(
            (Scalar::zero()..self.plaintext_count().0 as Scalar)
                .map(|_| {
                    self.random_generator
                        .random_uniform_n_lsb(self.codec.pt_modulus_bits())
                })
                .collect(),
        )
    }

    /// Generate a binary plaintext.
    pub fn gen_scalar_binary_pt(&mut self) -> Plaintext<Scalar> {
        Plaintext(self.random_generator.random_uniform_binary())
    }

    /// Generate a plaintext that has one at index `i` and zero otherwise.
    pub fn gen_demuxed_pt(&mut self, i: usize) -> PlaintextList<ScalarContainer> {
        let mut tmp = vec![Scalar::zero(); self.plaintext_count().0];
        tmp[i] = Scalar::one();
        PlaintextList::from_container(tmp)
    }

    /// Generate a ternary plaintext.
    pub fn gen_ternary_ptxt(&mut self) -> PlaintextList<ScalarContainer> {
        PlaintextList::from_container(
            (0..self.plaintext_count().0)
                .map(|_| self.random_generator.random_uniform_ternary())
                .collect(),
        )
    }

    /// Generate a unit plaintext list (all coefficients are 0 except the constant term is 1).
    pub fn gen_unit_pt(&self) -> PlaintextList<ScalarContainer> {
        let mut tmp = vec![Scalar::zero(); self.plaintext_count().0];
        tmp[0] = Scalar::one();
        PlaintextList::from_container(tmp)
    }

    /// Generate a unit plaintext
    pub fn gen_scalar_unit_pt(&self) -> Plaintext<Scalar> {
        Plaintext(Scalar::one())
    }

    /// Generate a plaintext list where all the coefficients are 0.
    pub fn gen_zero_pt(&self) -> PlaintextList<ScalarContainer> {
        PlaintextList::from_container(vec![Scalar::zero(); self.plaintext_count().0])
    }

    /// Generate a plaintext scalar of value zero.
    pub fn gen_scalar_zero_pt(&self) -> Plaintext<Scalar> {
        Plaintext(Scalar::zero())
    }

    /// Create a FFT context suitable for external product.
    pub fn gen_fft_ctx(&self) -> FftBuffer {
        FftBuffer::new(self.poly_size)
    }
}
