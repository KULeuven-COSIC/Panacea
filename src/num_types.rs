use concrete_core::backends::fft::private::c64;

pub type Scalar = u64;
pub type Complex = c64;
pub type SignedScalar = i64;
pub type ScalarContainer = Vec<Scalar>;
pub type ComplexContaier = Vec<Complex>;

pub trait Zero<T> {
    fn zero() -> T;
}

pub trait One<T> {
    fn one() -> T;
}

impl Zero<Self> for Scalar {
    fn zero() -> Self {
        0
    }
}

impl One<Self> for Scalar {
    fn one() -> Self {
        1
    }
}

impl Zero<Self> for Complex {
    fn zero() -> Self {
        Self::default()
    }
}

impl Zero<Self> for SignedScalar {
    fn zero() -> Self {
        0
    }
}

impl One<Self> for SignedScalar {
    fn one() -> Self {
        1
    }
}
