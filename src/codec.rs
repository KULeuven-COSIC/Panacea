use crate::{num_types::Scalar, utils::log2};
use tfhe::core_crypto::prelude::{
    DecompositionBaseLog, DecompositionLevelCount, Polynomial, SignedDecomposer,
};

pub struct Codec {
    decomposer: SignedDecomposer<Scalar>,
    delta: Scalar,
}

impl Codec {
    pub fn new(t: Scalar) -> Self {
        if !t.is_power_of_two() {
            panic!("delta must be a power of 2")
        }
        let logt = log2(t as usize);
        let decomposer =
            SignedDecomposer::<Scalar>::new(DecompositionBaseLog(logt), DecompositionLevelCount(1));
        Self {
            decomposer,
            delta: 1 << (Scalar::BITS as usize - logt),
        }
    }

    pub const fn largest_error(&self) -> Scalar {
        self.delta / 2
    }

    pub fn pt_modulus_bits(&self) -> usize {
        self.decomposer.base_log().0
    }

    pub fn pt_modulus(&self) -> Scalar {
        1 << self.decomposer.base_log().0
    }

    pub fn encode(&self, x: &mut Scalar) {
        if *x >= self.pt_modulus() {
            panic!("value is too big")
        }
        *x *= self.delta
    }

    pub fn decode(&self, x: &mut Scalar) {
        let tmp = self.decomposer.closest_representable(*x);
        *x = tmp / self.delta
    }

    /// Encode ternary x as x*(q/3)
    pub fn ternary_encode(&self, x: &mut Scalar) {
        const THIRD: Scalar = (Scalar::MAX as f64 / 3.0) as Scalar;
        if *x == 0 {
            *x = 0;
        } else if *x == 1 {
            *x = THIRD;
        } else if *x == Scalar::MAX {
            *x = 2 * THIRD;
        } else {
            panic!("not a ternary scalar")
        }
    }

    pub fn ternary_decode(&self, x: &mut Scalar) {
        const SIXTH: Scalar = (Scalar::MAX as f64 / 6.0) as Scalar;
        const THIRD: Scalar = SIXTH + SIXTH;
        const HALF: Scalar = Scalar::MAX / 2;
        if *x > SIXTH && *x <= HALF {
            *x = 1;
        } else if *x > HALF && *x <= HALF + THIRD {
            *x = Scalar::MAX;
        } else {
            *x = 0;
        }
    }

    /// Encode a polynomial.
    pub fn poly_encode(&self, xs: &mut Polynomial<&mut [Scalar]>) {
        for coeff in xs.iter_mut() {
            self.encode(coeff);
        }
    }

    pub fn poly_decode(&self, xs: &mut Polynomial<&mut [Scalar]>) {
        for coeff in xs.iter_mut() {
            self.decode(coeff);
        }
    }

    /// Encode a ternary polynomial.
    pub fn poly_ternary_encode(&self, xs: &mut Polynomial<&mut [Scalar]>) {
        for coeff in xs.iter_mut() {
            self.ternary_encode(coeff);
        }
    }

    pub fn poly_ternary_decode(&self, xs: &mut Polynomial<&mut [Scalar]>) {
        for coeff in xs.iter_mut() {
            self.ternary_decode(coeff);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_binary_encoder() {
        let codec = Codec::new(1 << 1);
        {
            let mut x: Scalar = 0;
            codec.encode(&mut x);
            assert_eq!(x, 0);
        }
        {
            let mut x: Scalar = 1;
            codec.encode(&mut x);
            assert_eq!(x, 1 << (Scalar::BITS - 1));
        }
        {
            let mut x: Scalar = 10;
            codec.decode(&mut x);
            assert_eq!(x, 0);
        }
        {
            let mut x: Scalar = Scalar::MAX;
            codec.decode(&mut x);
            assert_eq!(x, 0);
        }
        {
            let mut x: Scalar = 1 << (Scalar::BITS - 1);
            codec.decode(&mut x);
            assert_eq!(x, 1);
        }
    }

    #[test]
    fn test_generic_encoder() {
        {
            let expected = 3;
            let codec = Codec::new(4);
            let mut encoded = expected;
            codec.encode(&mut encoded);

            let mut decoded1 = encoded + codec.largest_error() - 1;
            let mut decoded2 = encoded - codec.largest_error() + 1;

            codec.decode(&mut decoded1);
            codec.decode(&mut decoded2);

            assert_eq!(decoded1, expected);
            assert_eq!(decoded2, expected);
        }

        {
            let expected = 1 << 6;
            let codec = Codec::new(1 << 12);
            let mut encoded = expected;
            codec.encode(&mut encoded);

            let mut decoded1 = encoded + codec.largest_error() - 1;
            let mut decoded2 = encoded - codec.largest_error() + 1;

            codec.decode(&mut decoded1);
            codec.decode(&mut decoded2);

            assert_eq!(decoded1, expected);
            assert_eq!(decoded2, expected);
        }
    }
}
