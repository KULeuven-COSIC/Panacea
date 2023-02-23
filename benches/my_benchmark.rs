#![allow(deprecated)]

use concrete_core::{
    commons::crypto::encoding::{Plaintext, PlaintextList},
    prelude::MonomialDegree,
};
use criterion::{criterion_group, criterion_main, Criterion};
use panacea::{
    context::{Context, FftBuffer},
    num_types::{One, Scalar, Zero},
    params::TFHEParameters,
    rgsw::RGSWCiphertext,
    rlwe::{
        expand, expand_fourier, gen_all_subs_ksk, gen_all_subs_ksk_fourier,
        make_decomposed_rlwe_ct, FourierRLWECiphertext, RLWECiphertext, RLWEKeyswitchKey,
        RLWESecretKey,
    },
};

pub fn trace1_fourier_benchmark(c: &mut Criterion) {
    let mut ctx = Context::new(TFHEParameters::default());
    let orig_msg = ctx.gen_binary_pt();

    let mut encoded_msg = orig_msg;
    ctx.codec.poly_encode(&mut encoded_msg.as_mut_polynomial());
    // we need to divide the encoded message by n, because n is multiplied into the trace output
    for coeff in encoded_msg.as_mut_polynomial().coefficient_iter_mut() {
        *coeff /= ctx.poly_size.0 as Scalar;
    }

    let sk = RLWESecretKey::generate_binary(ctx.poly_size, &mut ctx.secret_generator);
    let mut ct = RLWECiphertext::allocate(ctx.poly_size);
    sk.encrypt_rlwe(
        &mut ct,
        &encoded_msg,
        ctx.std,
        &mut ctx.encryption_generator,
    );

    let all_ksk = gen_all_subs_ksk_fourier(&sk, &mut ctx);

    let mut output = RLWECiphertext::allocate(ctx.poly_size);

    c.bench_function("trace1 fourier", |b| {
        b.iter(|| {
            ct.trace1_fourier(&mut output, &all_ksk);
        });
    });
}

pub fn less_eq_benchmark(c: &mut Criterion) {
    let mut ctx = Context::new(TFHEParameters::default());
    let sk = ctx.gen_rlwe_sk();

    let m = ctx.poly_size.0 / 2;
    let mut ptxt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
    *ptxt
        .as_mut_polynomial()
        .get_mut_monomial(MonomialDegree(m))
        .get_mut_coefficient() = Scalar::one();

    let mut ct = RLWECiphertext::allocate(ctx.poly_size);
    sk.encode_encrypt_rlwe(&mut ct, &ptxt, &mut ctx);

    c.bench_function("less_eq", |b| {
        b.iter(|| {
            ct.less_eq_than(m + 1);
        });
    });
}

pub fn expand_fourier_benchmark(c: &mut Criterion) {
    let mut ctx = Context::new(TFHEParameters::default());
    let sk = ctx.gen_rlwe_sk();
    let neg_sk_ct = sk.neg_gsw(&mut ctx);

    let ksk_map = gen_all_subs_ksk_fourier(&sk, &mut ctx);

    let test_pt = ctx.gen_binary_pt();
    let mut test_ct = RLWECiphertext::allocate(ctx.poly_size);
    sk.encode_encrypt_rlwe(&mut test_ct, &test_pt, &mut ctx);

    let cs_one = make_decomposed_rlwe_ct(&sk, Scalar::one(), &mut ctx);

    c.bench_function("expand fourier", |b| {
        b.iter(|| {
            expand_fourier(&cs_one, &ksk_map, &neg_sk_ct, &ctx);
        });
    });
}

pub fn expand_benchmark(c: &mut Criterion) {
    let mut ctx = Context::new(TFHEParameters::default());
    let sk = ctx.gen_rlwe_sk();
    let neg_sk_ct = sk.neg_gsw(&mut ctx);
    let ksk_map = gen_all_subs_ksk(&sk, &mut ctx);

    let test_pt = ctx.gen_binary_pt();
    let mut test_ct = RLWECiphertext::allocate(ctx.poly_size);
    sk.encode_encrypt_rlwe(&mut test_ct, &test_pt, &mut ctx);

    let cs_one = make_decomposed_rlwe_ct(&sk, Scalar::one(), &mut ctx);

    c.bench_function("expand", |b| {
        b.iter(|| {
            expand(&cs_one, &ksk_map, &neg_sk_ct, &ctx);
        });
    });
}

pub fn keyswitch_gsw_benchmark(c: &mut Criterion) {
    let mut ctx = Context::new(TFHEParameters::default());
    let sk_after = ctx.gen_rlwe_sk();
    let sk_before = ctx.gen_rlwe_sk();
    let mut fft_buffer = FftBuffer::new(ctx.poly_size);

    let mut ct_after = RLWECiphertext::allocate(ctx.poly_size);
    let mut ct_before = RLWECiphertext::allocate(ctx.poly_size);

    let mut ksk = RLWEKeyswitchKey::allocate(ctx.base_log, ctx.level_count, ctx.poly_size);
    ksk.fill_with_keyswitch_key(
        &sk_before,
        &sk_after,
        ctx.std,
        &mut ctx.encryption_generator,
    );
    let fast_ksk = RGSWCiphertext::from_keyswitch_key(&ksk);

    let messages = ctx.gen_ternary_ptxt();
    sk_before.ternary_encrypt_rlwe(&mut ct_before, &messages, &mut ctx);

    c.bench_function("keyswitch gsw", |b| {
        b.iter(|| {
            fast_ksk.keyswitch_ciphertext_with_buf(&mut ct_after, &ct_before, &mut fft_buffer);
        });
    });
}

pub fn keyswitch_fourier_benchmark(c: &mut Criterion) {
    let mut ctx = Context::new(TFHEParameters::default());

    let sk_after = ctx.gen_rlwe_sk();
    let sk_before = ctx.gen_rlwe_sk();

    let mut ct_after_fourier = FourierRLWECiphertext::new(ctx.poly_size.0);
    let mut ct_before = RLWECiphertext::allocate(ctx.poly_size);

    let mut ksk = RLWEKeyswitchKey::allocate(ctx.base_log, ctx.level_count, ctx.poly_size);
    ksk.fill_with_keyswitch_key(
        &sk_before,
        &sk_after,
        ctx.std,
        &mut ctx.encryption_generator,
    );
    let ksk_fourier = ksk.into_fourier();

    let messages = ctx.gen_ternary_ptxt();
    sk_before.ternary_encrypt_rlwe(&mut ct_before, &messages, &mut ctx);

    c.bench_function("keyswitch fourier", |b| {
        b.iter(|| {
            ksk_fourier.keyswitch_ciphertext(&mut ct_after_fourier, &ct_before);
        });
    });
}

pub fn keyswitch_benchmark(c: &mut Criterion) {
    let mut ctx = Context::new(TFHEParameters::default());
    let sk_after = ctx.gen_rlwe_sk();
    let sk_before = ctx.gen_rlwe_sk();

    let mut ct_after = RLWECiphertext::allocate(ctx.poly_size);
    let mut ct_before = RLWECiphertext::allocate(ctx.poly_size);

    let mut ksk = RLWEKeyswitchKey::allocate(ctx.base_log, ctx.level_count, ctx.poly_size);
    ksk.fill_with_keyswitch_key(
        &sk_before,
        &sk_after,
        ctx.std,
        &mut ctx.encryption_generator,
    );

    let messages = ctx.gen_ternary_ptxt();
    sk_before.ternary_encrypt_rlwe(&mut ct_before, &messages, &mut ctx);

    c.bench_function("keyswitch", |b| {
        b.iter(|| {
            ksk.keyswitch_ciphertext(&mut ct_after, &ct_before);
        });
    });
}

pub fn cmux_benchmark(c: &mut Criterion) {
    let mut ctx = Context::new(TFHEParameters::default());
    let sk = ctx.gen_rlwe_sk();
    let mut fft_buffer = FftBuffer::new(ctx.poly_size);

    let mut gsw_ct = RGSWCiphertext::allocate(ctx.poly_size, ctx.base_log, ctx.level_count);
    sk.encrypt_constant_rgsw(&mut gsw_ct, &Plaintext(Scalar::one()), &mut ctx);

    let mut rlwe_ct_0 = RLWECiphertext::allocate(ctx.poly_size);
    let mut rlwe_ct_1 = RLWECiphertext::allocate(ctx.poly_size);
    sk.encrypt_constant_rlwe(&mut rlwe_ct_0, &Plaintext(Scalar::zero()), &mut ctx);
    sk.encrypt_constant_rlwe(&mut rlwe_ct_1, &Plaintext(Scalar::one()), &mut ctx);

    let mut rlwe_ct_out = RLWECiphertext::allocate(ctx.poly_size);
    c.bench_function("cmux", |b| {
        b.iter(|| gsw_ct.cmux_with_buf(&mut rlwe_ct_out, &rlwe_ct_0, &rlwe_ct_1, &mut fft_buffer));
    });
}

pub fn external_product_benchmark(c: &mut Criterion) {
    let mut ctx = Context::new(TFHEParameters::default());
    let sk = ctx.gen_rlwe_sk();
    let mut fft_buffer = FftBuffer::new(ctx.poly_size);

    let mut gsw_ct = RGSWCiphertext::allocate(ctx.poly_size, ctx.base_log, ctx.level_count);
    sk.encrypt_constant_rgsw(&mut gsw_ct, &Plaintext(Scalar::one()), &mut ctx);

    let mut rlwe_ct_0 = RLWECiphertext::allocate(ctx.poly_size);
    sk.encrypt_constant_rlwe(&mut rlwe_ct_0, &Plaintext(Scalar::zero()), &mut ctx);

    let mut rlwe_ct_out = RLWECiphertext::allocate(ctx.poly_size);
    c.bench_function("external_product", |b| {
        b.iter(|| gsw_ct.external_product_with_buf(&mut rlwe_ct_out, &rlwe_ct_0, &mut fft_buffer));
    });
}

pub fn enc_benchmark(c: &mut Criterion) {
    let mut ctx = Context::new(TFHEParameters::default());
    let sk = ctx.gen_rlwe_sk();

    let mut ct = RLWECiphertext::allocate(ctx.poly_size);
    let pt = ctx.gen_binary_pt();

    c.bench_function("enc", |b| {
        b.iter(|| {
            sk.encode_encrypt_rlwe(&mut ct, &pt, &mut ctx);
        });
    });
}

pub fn dec_benchmark(c: &mut Criterion) {
    let mut ctx = Context::new(TFHEParameters::default());
    let sk = ctx.gen_rlwe_sk();

    let mut ct = RLWECiphertext::allocate(ctx.poly_size);
    let pt = ctx.gen_binary_pt();
    sk.encode_encrypt_rlwe(&mut ct, &pt, &mut ctx);

    let mut out_pt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
    c.bench_function("dec", |b| {
        b.iter(|| {
            sk.decrypt_decode_rlwe(&mut out_pt, &ct, &ctx);
        });
    });
}

criterion_group!(
    benches,
    enc_benchmark,
    dec_benchmark,
    external_product_benchmark,
    cmux_benchmark,
    keyswitch_benchmark,
    keyswitch_fourier_benchmark,
    keyswitch_gsw_benchmark,
    expand_benchmark,
    expand_fourier_benchmark,
    less_eq_benchmark,
    trace1_fourier_benchmark,
    // oram_benchmark
);
criterion_main!(benches);
