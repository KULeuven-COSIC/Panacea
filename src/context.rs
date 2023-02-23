use crate::{
    codec::Codec,
    lwe::LWESecretKey,
    num_types::{One, Scalar, Zero},
    params::TFHEParameters,
    rlwe::RLWESecretKey,
};
use concrete_core::{
    backends::fft::private::crypto::ggsw::external_product_scratch,
    backends::fft::private::math::fft::Fft,
    commons::{
        crypto::{
            encoding::{Plaintext, PlaintextList},
            secret::generators::{
                DeterministicSeeder, EncryptionRandomGenerator, SecretRandomGenerator,
            },
        },
        math::{random::RandomGenerator, tensor::AsMutTensor},
    },
    prelude::{
        DecompositionBaseLog, DecompositionLevelCount, GlweSize, LweDimension, LweSize,
        MonomialDegree, PlaintextCount, PolynomialSize, Seeder, UnixSeeder,
    },
    specification::dispersion::{DispersionParameter, LogStandardDev},
};
use concrete_csprng::{generators::SoftwareRandomGenerator, seeders::Seed};
use dyn_stack::GlobalMemBuffer;
use rand::Rng;

pub struct FftBuffer {
    pub(crate) fft: Fft,
    pub(crate) mem: GlobalMemBuffer,
}

impl FftBuffer {
    pub fn new(poly_size: PolynomialSize) -> Self {
        let glwe_size = GlweSize(2);
        let fft = Fft::new(poly_size);
        let mem = GlobalMemBuffer::new(
            external_product_scratch::<Scalar>(glwe_size, poly_size, fft.as_view()).unwrap(),
        );
        Self { fft, mem }
    }
}

/// The context structure holds the TFHE parameters and
/// random number generators.
// TODO: We could replace SoftwareRandomGenerator with a Generic to use the Hardware Accelerated stuff
pub struct Context {
    pub random_generator: RandomGenerator<SoftwareRandomGenerator>,
    pub secret_generator: SecretRandomGenerator<SoftwareRandomGenerator>,
    pub encryption_generator: EncryptionRandomGenerator<SoftwareRandomGenerator>,
    pub std: LogStandardDev,
    pub poly_size: PolynomialSize,
    pub base_log: DecompositionBaseLog,
    pub level_count: DecompositionLevelCount,
    pub ks_base_log: DecompositionBaseLog,
    pub ks_level_count: DecompositionLevelCount,
    pub negs_base_log: DecompositionBaseLog,
    pub negs_level_count: DecompositionLevelCount,
    pub codec: Codec,
}

impl std::fmt::Display for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "q={:?}, poly_size={:?}, log_std={:?}, default_decomp=({:?},{:?}), ks_decomp=({:?},{:?}), negs_decomp=({:?},{:?}), pt_modulus=({:?})",
                 Scalar::BITS, self.poly_size.0, self.std.get_log_standard_dev(),
                 self.base_log.0, self.level_count.0, self.ks_base_log.0, self.ks_level_count.0, self.negs_base_log.0, self.negs_level_count.0, self.codec.pt_modulus())
    }
}

impl Context {
    /// Create the default context that is suitable for
    /// all experiments in the repository.
    pub fn new(tfhe_params: TFHEParameters) -> Self {
        let (random_generator, secret_generator, encryption_generator);
        if tfhe_params.secure_seed {
            //     if AppleSecureEnclaveSeeder::is_available() {
            //         random_generator = RandomGenerator::<SoftwareRandomGenerator>::new(
            //             AppleSecureEnclaveSeeder.seed(),
            //         );
            //         secret_generator = SecretRandomGenerator::<SoftwareRandomGenerator>::new(
            //             AppleSecureEnclaveSeeder.seed(),
            //         );
            //         encryption_generator = EncryptionRandomGenerator::<SoftwareRandomGenerator>::new(
            //             AppleSecureEnclaveSeeder.seed(),
            //             &mut AppleSecureEnclaveSeeder,
            //         );
            //     } else
            if UnixSeeder::is_available() {
                // TODO: this seed variable should be replaced by a secret which is used in case thread_rng is reading from a compromised source of entropy
                let seed = rand::thread_rng().gen::<u128>();
                let mut seeder = UnixSeeder::new(seed);
                random_generator = RandomGenerator::<SoftwareRandomGenerator>::new(seeder.seed());
                secret_generator =
                    SecretRandomGenerator::<SoftwareRandomGenerator>::new(seeder.seed());
                encryption_generator = EncryptionRandomGenerator::<SoftwareRandomGenerator>::new(
                    seeder.seed(),
                    &mut seeder,
                );
            } else {
                random_generator = RandomGenerator::<SoftwareRandomGenerator>::new(Seed(
                    rand::thread_rng().gen::<u128>(),
                ));
                secret_generator = SecretRandomGenerator::<SoftwareRandomGenerator>::new(Seed(
                    rand::thread_rng().gen::<u128>(),
                ));
                encryption_generator = EncryptionRandomGenerator::<SoftwareRandomGenerator>::new(
                    Seed(rand::thread_rng().gen::<u128>()),
                    &mut DeterministicSeeder::<SoftwareRandomGenerator>::new(Seed(
                        rand::thread_rng().gen::<u128>(),
                    )),
                );
            }
        } else {
            random_generator = RandomGenerator::<SoftwareRandomGenerator>::new(Seed(0));
            secret_generator = SecretRandomGenerator::<SoftwareRandomGenerator>::new(Seed(0));
            encryption_generator = EncryptionRandomGenerator::<SoftwareRandomGenerator>::new(
                Seed(0),
                &mut DeterministicSeeder::<SoftwareRandomGenerator>::new(Seed(0)),
            );
        }
        let std = LogStandardDev::from_log_standard_dev(tfhe_params.standard_deviation);
        let poly_size = PolynomialSize(tfhe_params.polynomial_size);
        let base_log = DecompositionBaseLog(tfhe_params.base_log);
        let level_count = DecompositionLevelCount(tfhe_params.level_count);
        let ks_base_log = DecompositionBaseLog(tfhe_params.key_switch_base_log);
        let ks_level_count = DecompositionLevelCount(tfhe_params.key_switch_level_count);
        let negs_base_log = DecompositionBaseLog(tfhe_params.negs_base_log);
        let negs_level_count = DecompositionLevelCount(tfhe_params.negs_level_count);
        let codec = Codec::new(tfhe_params.plaintext_modulus);
        Self {
            random_generator,
            secret_generator,
            encryption_generator,
            std,
            poly_size,
            base_log,
            level_count,
            ks_base_log,
            ks_level_count,
            negs_base_log,
            negs_level_count,
            codec,
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
    pub fn gen_binary_pt(&mut self) -> PlaintextList<Vec<Scalar>> {
        let cnt = self.plaintext_count();
        let mut ptxt = PlaintextList::allocate(Scalar::zero(), cnt);
        self.random_generator
            .fill_tensor_with_random_uniform_binary(ptxt.as_mut_tensor());
        ptxt
    }

    /// Generate a plaintext list where the coefficients sampled from the plaintext space
    /// defined by the codec.
    pub fn gen_pt(&mut self) -> PlaintextList<Vec<Scalar>> {
        let cnt = self.plaintext_count();
        let mut ptxt = PlaintextList::allocate(Scalar::zero(), cnt);
        self.random_generator.fill_tensor_with_random_uniform_n_lsb(
            ptxt.as_mut_tensor(),
            self.codec.pt_modulus_bits(),
        );
        ptxt
    }

    /// Generate a binary plaintext.
    pub fn gen_scalar_binary_pt(&mut self) -> Plaintext<Scalar> {
        Plaintext(self.random_generator.random_uniform_binary())
    }

    /// Generate a plaintext that has one at index `i` and zero otherwise.
    pub fn gen_demuxed_pt(&mut self, i: usize) -> PlaintextList<Vec<Scalar>> {
        let mut ptxt = PlaintextList::allocate(Scalar::zero(), self.plaintext_count());
        *ptxt
            .as_mut_polynomial()
            .get_mut_monomial(MonomialDegree(i))
            .get_mut_coefficient() = Scalar::one();
        ptxt
    }

    /// Generate a ternay plaintext.
    pub fn gen_ternary_ptxt(&mut self) -> PlaintextList<Vec<Scalar>> {
        let cnt = self.plaintext_count();
        let mut ptxt = PlaintextList::allocate(Scalar::zero(), cnt);
        self.random_generator
            .fill_tensor_with_random_uniform_ternary(ptxt.as_mut_tensor());
        ptxt
    }

    /// Generate a unit plaintext list (all coefficients are 0 except the constant term is 1).
    pub fn gen_unit_pt(&self) -> PlaintextList<Vec<Scalar>> {
        let mut ptxt = PlaintextList::allocate(Scalar::zero(), self.plaintext_count());
        *ptxt
            .as_mut_polynomial()
            .get_mut_monomial(MonomialDegree(0))
            .get_mut_coefficient() = Scalar::one();
        ptxt
    }

    /// Generate a unit plaintext
    pub fn gen_scalar_unit_pt(&self) -> Plaintext<Scalar> {
        Plaintext(Scalar::one())
    }

    /// Generate a plaintext list where all the coefficients are 0.
    pub fn gen_zero_pt(&self) -> PlaintextList<Vec<Scalar>> {
        PlaintextList::allocate(Scalar::zero(), self.plaintext_count())
    }

    /// Generate a plaintext scalar is zero.
    pub fn gen_scalar_zero_pt(&self) -> Plaintext<Scalar> {
        Plaintext(Scalar::zero())
    }

    /// Generate a RLWE secret key.
    pub fn gen_rlwe_sk(&mut self) -> RLWESecretKey {
        RLWESecretKey::generate_binary(self.poly_size, &mut self.secret_generator)
    }

    /// Generate a LWE secret key.
    pub fn gen_lwe_sk(&mut self) -> LWESecretKey {
        LWESecretKey::generate_binary(self.lwe_dim(), &mut self.secret_generator)
    }

    /// Create a FFT context suitable for external product.
    pub fn gen_fft_ctx(&self) -> FftBuffer {
        FftBuffer::new(self.poly_size)
    }
}
