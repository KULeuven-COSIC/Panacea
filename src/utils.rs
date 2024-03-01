use crate::{
    codec::Codec,
    context::Context,
    num_types::{
        AlignedScalarContainer, ComplexBox, One, Scalar, ScalarContainer, SignedScalar, Zero,
    },
};
use aligned_vec::avec;
use std::fs::File;
use tfhe::core_crypto::prelude::{
    decrypt_glwe_ciphertext, polynomial_algorithms::polynomial_wrapping_sub_assign,
    slice_algorithms::slice_wrapping_scalar_mul_assign, ContiguousEntityContainer,
    FourierGgswCiphertext, GgswCiphertext, GlweCiphertext, GlweSecretKey, PlaintextList,
};

#[inline]
pub const fn log2(input: usize) -> usize {
    core::mem::size_of::<usize>() * 8 - (input.leading_zeros() as usize) - 1
}

/// Evaluate f(x) on x^k, where k is odd
pub(crate) fn eval_x_k(poly: &[Scalar], k: usize) -> AlignedScalarContainer {
    assert_eq!(k % 2, 1);
    let poly_size = poly.len();
    let mut out = avec![0; poly_size];
    out[0] = poly[0];

    (1..poly_size).for_each(|i| {
        // i-th term becomes ik-th term, but reduced by n
        let j = i * k % poly.len();
        let sign = if ((i * k) / poly_size) % 2 == 0 {
            1
        } else {
            Scalar::MAX
        };
        let c = poly[i];
        out[j] = sign.wrapping_mul(c);
    });

    out
}

/// # Panics
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

/// # Panics
pub fn parse_csv(path: &std::path::Path) -> Vec<Vec<Scalar>> {
    let x_test_f = File::open(path).expect("csv file not found, consider using --artificial");

    let mut x_test: Vec<Vec<Scalar>> = vec![];
    let mut x_train_rdr = csv::Reader::from_reader(x_test_f);
    for res in x_train_rdr.records() {
        let record = res.unwrap();
        let row = record.iter().map(|s| s.parse().unwrap()).collect();
        x_test.push(row);
    }

    x_test
}

/// # Panics
pub fn pt_to_lossy_u64(pt: &PlaintextList<ScalarContainer>) -> u64 {
    let mut out = 0u64;
    for (i, x) in pt.as_ref().iter().take(64).enumerate() {
        assert!(*x == Scalar::zero() || *x == Scalar::one());
        out += x * (1 << i);
    }
    out
}

/// Compute the noise for ciphertext `ct`
/// given the (possibly encoded) plaintext `ptxt`.
pub fn compute_noise(
    sk: &GlweSecretKey<ScalarContainer>,
    ct: &GlweCiphertext<&[Scalar]>,
    encoded_ptxt: &PlaintextList<ScalarContainer>,
) -> f64 {
    // pt = b - a*s = Delta*m + e
    let mut pt = PlaintextList::new(Scalar::zero(), encoded_ptxt.plaintext_count());
    decrypt_glwe_ciphertext(sk, ct, &mut pt);

    // pt = pt - Delta*m = e (encoded_ptxt is Delta*m)

    polynomial_wrapping_sub_assign(&mut pt.as_mut_polynomial(), &encoded_ptxt.as_polynomial());

    let mut max_e: SignedScalar = 0;
    for x in pt.as_ref().iter() {
        // convert x to signed
        let z = (*x as SignedScalar).abs();
        if z > max_e {
            max_e = z;
        }
    }
    (max_e as f64).log2()
}

pub fn compute_noise_ternary(
    sk: &GlweSecretKey<ScalarContainer>,
    ct: &GlweCiphertext<AlignedScalarContainer>,
    ptxt: &PlaintextList<ScalarContainer>,
    ctx: &Context<Scalar>,
) -> f64 {
    let mut tmp = ptxt.clone();
    ctx.codec.poly_ternary_decode(&mut tmp.as_mut_polynomial());
    compute_noise(sk, &ct.as_view(), &tmp)
}

/// Compute the noise for ciphertext `ct`
/// given the unencoded plaintext `ptxt`.
/// So the codec must be given.
pub fn compute_noise_encoded(
    sk: &GlweSecretKey<ScalarContainer>,
    ct: &GlweCiphertext<AlignedScalarContainer>,
    ptxt: &PlaintextList<ScalarContainer>,
    codec: &Codec,
) -> f64 {
    let mut tmp = ptxt.clone();
    codec.poly_encode(&mut tmp.as_mut_polynomial());
    compute_noise(sk, &ct.as_view(), &tmp)
}

/// Compute the average noise in the RGSW ciphertext
/// by computing the noise on the RLWE rows and taking the average.
pub fn compute_noise_rgsw1(
    sk: &GlweSecretKey<ScalarContainer>,
    ct: &GgswCiphertext<ScalarContainer>,
    ctx: &Context<Scalar>,
) -> f64 {
    let mut total_noise = 0f64;
    let rows = ct.as_glwe_list();
    for level in 0..ctx.level_count.0 {
        let shift = (Scalar::BITS as usize) - ctx.base_log.0 * (level + 1);
        let mut pt = ctx.gen_unit_pt();
        slice_wrapping_scalar_mul_assign(pt.as_mut(), 1 << shift);
        let noise = compute_noise(sk, &rows.get(level * 2 + 1), &pt);
        total_noise += noise;
    }
    total_noise / ctx.level_count.0 as f64
}

pub fn flatten_fourier_ggsw(ct: &FourierGgswCiphertext<ComplexBox>) -> ComplexBox {
    let y = ct
        .as_view()
        .into_levels()
        .fold(avec![], |mut level_acc, level| {
            level.into_rows().for_each(|row| {
                row.data()
                    .iter()
                    .for_each(|element| level_acc.push(element.to_owned()));
            });
            level_acc
        });
    y.into_boxed_slice()
}
