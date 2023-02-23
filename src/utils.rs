use crate::num_types::{One, Scalar, Zero};
use concrete_core::commons::{
    crypto::encoding::PlaintextList,
    math::{
        polynomial::Polynomial,
        tensor::{AsMutSlice, AsMutTensor, AsRefSlice, AsRefTensor, Tensor},
    },
};
use std::fs::File;

pub(crate) fn mul_const<C>(poly: &mut Tensor<C>, c: Scalar)
where
    C: AsMutSlice<Element = Scalar>,
{
    for coeff in poly.iter_mut() {
        *coeff = coeff.wrapping_mul(c);
    }
}

#[inline]
pub const fn log2(input: usize) -> usize {
    core::mem::size_of::<usize>() * 8 - (input.leading_zeros() as usize) - 1
}

/// Evaluate f(x) on x^k, where k is odd
pub(crate) fn eval_x_k<C>(poly: &Polynomial<C>, k: usize) -> Polynomial<Vec<Scalar>>
where
    C: AsRefSlice<Element = Scalar>,
{
    let mut out = Polynomial::allocate(Scalar::zero(), poly.polynomial_size());
    eval_x_k_in_memory(&mut out, poly, k);
    out
}

/// Evaluate f(x) on x^k, where k is odd
pub(crate) fn eval_x_k_in_memory<C>(
    out: &mut Polynomial<Vec<Scalar>>,
    poly: &Polynomial<C>,
    k: usize,
) where
    C: AsRefSlice<Element = Scalar>,
{
    assert_eq!(k % 2, 1);
    assert!(poly.polynomial_size().0.is_power_of_two());
    *out.as_mut_tensor().get_element_mut(0) = *poly.as_tensor().get_element(0);
    for i in 1..poly.polynomial_size().0 {
        // i-th term becomes ik-th term, but reduced by n
        let j = i * k % poly.polynomial_size().0;
        let sign = if ((i * k) / poly.polynomial_size().0) % 2 == 0 {
            1
        } else {
            Scalar::MAX
        };
        let c = *poly.as_tensor().get_element(i);
        *out.as_mut_tensor().get_element_mut(j) = sign.wrapping_mul(c);
    }
}

pub fn transpose<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>> {
    assert!(!v.is_empty());
    let len = v[0].len();
    let mut iters: Vec<_> = v.into_iter().map(IntoIterator::into_iter).collect();
    (0..len)
        .map(|_| {
            iters
                .iter_mut()
                .map(|n| n.next().unwrap())
                .collect::<Vec<T>>()
        })
        .collect()
}

// TODO: Unused
pub fn parse_csv(path: &std::path::Path) -> Vec<Vec<usize>> {
    let x_test_f = File::open(path).expect("csv file not found, consider using --artificial");

    let mut x_test: Vec<Vec<usize>> = vec![];
    let mut x_train_rdr = csv::Reader::from_reader(x_test_f);
    for res in x_train_rdr.records() {
        let record = res.unwrap();
        let row = record.iter().map(|s| s.parse().unwrap()).collect();
        x_test.push(row);
    }

    x_test
}

pub fn pt_to_lossy_u64(pt: &PlaintextList<Vec<Scalar>>) -> u64 {
    let mut out = 0u64;
    for (i, x) in pt.plaintext_iter().take(64).enumerate() {
        assert!(x.0 == Scalar::zero() || x.0 == Scalar::one());
        out += x.0 * (1 << i);
    }
    out
}
