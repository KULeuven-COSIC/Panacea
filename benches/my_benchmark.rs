use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use dyn_stack::ReborrowMut;
use panacea::{
    context::Context,
    num_types::{One, Scalar, Zero},
    params::TFHEParameters,
    rlwe::{
        convert_standard_ggsw_to_fourier, expand, gen_all_subs_ksk, less_eq_than,
        make_decomposed_rlwe_ct, neg_gsw_std, trace1, FourierRLWEKeyswitchKey,
    },
};
use tfhe::core_crypto::{
    entities::{FourierGgswCiphertext, GgswCiphertext, Plaintext, PlaintextList},
    prelude::{
        add_external_product_assign_mem_optimized,
        add_external_product_assign_mem_optimized_requirement, cmux_assign_mem_optimized,
        cmux_assign_mem_optimized_requirement,
        convert_standard_ggsw_ciphertext_to_fourier_mem_optimized,
        convert_standard_ggsw_ciphertext_to_fourier_mem_optimized_requirement,
        decrypt_glwe_ciphertext, encrypt_constant_ggsw_ciphertext, encrypt_glwe_ciphertext,
        ComputationBuffers, GlweSecretKey,
    },
};

pub fn trace1_benchmark(c: &mut Criterion) {
    let mut ctx = Context::new(TFHEParameters::default());
    let orig_msg = ctx.gen_binary_pt();

    let mut encoded_msg = orig_msg;
    ctx.codec.poly_encode(&mut encoded_msg.as_mut_polynomial());
    // we need to divide the encoded message by n, because n is multiplied into the trace output
    for coeff in encoded_msg.as_mut_polynomial().iter_mut() {
        *coeff /= ctx.poly_size.0 as Scalar;
    }

    let sk = GlweSecretKey::generate_new_binary(
        ctx.glwe_dimension,
        ctx.poly_size,
        &mut ctx.secret_generator,
    );
    let mut ct = ctx.empty_glwe_ciphertext();
    encrypt_glwe_ciphertext(
        &sk,
        &mut ct,
        &encoded_msg,
        ctx.std,
        &mut ctx.encryption_generator,
    );

    let all_ksk = gen_all_subs_ksk(&sk, &mut ctx);

    c.bench_function("trace1 fourier", |b| {
        b.iter(|| {
            trace1(&ct, &all_ksk, &ctx);
        });
    });
}

pub fn less_eq_benchmark(c: &mut Criterion) {
    let mut ctx = Context::new(TFHEParameters::default());
    let sk = GlweSecretKey::generate_new_binary(
        ctx.glwe_dimension,
        ctx.poly_size,
        &mut ctx.secret_generator,
    );

    let m: Scalar = (ctx.poly_size.0 / 2) as Scalar;

    let mut ptxt = ctx.gen_unit_pt();

    let mut ct = ctx.empty_glwe_ciphertext();

    ctx.codec.poly_encode(&mut ptxt.as_mut_polynomial());
    encrypt_glwe_ciphertext(&sk, &mut ct, &ptxt, ctx.std, &mut ctx.encryption_generator);

    c.bench_function("less_eq", |b| {
        b.iter(|| {
            less_eq_than(&mut ct, m + 1);
        });
    });
}

pub fn expand_benchmark(c: &mut Criterion) {
    let mut ctx = Context::new(TFHEParameters::default());
    let sk = GlweSecretKey::generate_new_binary(
        ctx.glwe_dimension,
        ctx.poly_size,
        &mut ctx.secret_generator,
    );
    let mut buf = ctx.gen_fft_ctx();
    let neg_sk_ct = convert_standard_ggsw_to_fourier(neg_gsw_std(&sk, &mut ctx), &ctx, &mut buf);

    let ksk_map = gen_all_subs_ksk(&sk, &mut ctx);

    let test_pt = ctx.gen_binary_pt();
    let mut test_ct = ctx.empty_glwe_ciphertext();
    encrypt_glwe_ciphertext(
        &sk,
        &mut test_ct,
        &test_pt,
        ctx.std,
        &mut ctx.encryption_generator,
    );
    let cs_one = make_decomposed_rlwe_ct(&sk, Scalar::one(), &mut ctx);
    c.bench_function("expand fourier", move |b| {
        b.iter_batched(
            || cs_one.clone(),
            |ct| {
                expand(&ct, &ksk_map, &neg_sk_ct, &ctx, &mut buf);
            },
            BatchSize::PerIteration,
        );
    });
}

pub fn keyswitch_benchmark(c: &mut Criterion) {
    let mut ctx = Context::new(TFHEParameters::default());

    let sk_after = GlweSecretKey::generate_new_binary(
        ctx.glwe_dimension,
        ctx.poly_size,
        &mut ctx.secret_generator,
    );
    let sk_before = GlweSecretKey::generate_new_binary(
        ctx.glwe_dimension,
        ctx.poly_size,
        &mut ctx.secret_generator,
    );

    let mut ct_before = ctx.empty_glwe_ciphertext();

    let ksk_fourier = FourierRLWEKeyswitchKey::new(sk_before.clone(), &sk_after, &mut ctx);

    let messages = ctx.gen_ternary_ptxt();

    encrypt_glwe_ciphertext(
        &sk_before,
        &mut ct_before,
        &messages,
        ctx.std,
        &mut ctx.encryption_generator,
    );

    c.bench_function("keyswitch fourier", |b| {
        b.iter_batched(
            || ct_before.clone(),
            |ct_b| ksk_fourier.keyswitch_ciphertext(ct_b, &ctx),
            BatchSize::PerIteration,
        );
    });
}

pub fn gen_ksk_benchmark(c: &mut Criterion) {
    let mut ctx = Context::new(TFHEParameters::default());
    let sk_after = GlweSecretKey::generate_new_binary(
        ctx.glwe_dimension,
        ctx.poly_size,
        &mut ctx.secret_generator,
    );
    let sk_before = GlweSecretKey::generate_new_binary(
        ctx.glwe_dimension,
        ctx.poly_size,
        &mut ctx.secret_generator,
    );

    c.bench_function("ksk_fill_fourier", |b| {
        b.iter_batched(
            || sk_before.clone(),
            |sk_bef| FourierRLWEKeyswitchKey::new(sk_bef, &sk_after, &mut ctx),
            BatchSize::PerIteration,
        );
    });
}

pub fn cmux_benchmark(c: &mut Criterion) {
    let mut ctx = Context::new(TFHEParameters::default());
    let sk = GlweSecretKey::generate_new_binary(
        ctx.glwe_dimension,
        ctx.poly_size,
        &mut ctx.secret_generator,
    );

    let mut gsw_ct = GgswCiphertext::new(
        Scalar::zero(),
        ctx.glwe_size,
        ctx.poly_size,
        ctx.base_log,
        ctx.level_count,
        ctx.ciphertext_modulus,
    );
    encrypt_constant_ggsw_ciphertext(
        &sk,
        &mut gsw_ct,
        Plaintext(Scalar::one()),
        ctx.std,
        &mut ctx.encryption_generator,
    );

    let mut rlwe_ct_0 = ctx.empty_glwe_ciphertext();
    let mut rlwe_ct_1 = ctx.empty_glwe_ciphertext();

    let mut pt0 = PlaintextList::new(Scalar::zero(), ctx.plaintext_count());
    (*pt0.as_mut_polynomial().as_mut())[0] = Scalar::zero();

    encrypt_glwe_ciphertext(
        &sk,
        &mut rlwe_ct_0,
        &pt0,
        ctx.std,
        &mut ctx.encryption_generator,
    );
    let mut pt1 = PlaintextList::new(Scalar::zero(), ctx.plaintext_count());
    (*pt1.as_mut_polynomial().as_mut())[0] = Scalar::one();

    encrypt_glwe_ciphertext(
        &sk,
        &mut rlwe_ct_1,
        &pt1,
        ctx.std,
        &mut ctx.encryption_generator,
    );

    let mut mem = ComputationBuffers::new();
    mem.resize(
        cmux_assign_mem_optimized_requirement::<Scalar>(
            ctx.glwe_size,
            ctx.poly_size,
            ctx.fft.as_view(),
        )
        .unwrap()
        .unaligned_bytes_required()
        .max(
            convert_standard_ggsw_ciphertext_to_fourier_mem_optimized_requirement(
                ctx.fft.as_view(),
            )
            .unwrap()
            .unaligned_bytes_required(),
        ),
    );

    let mut stack = mem.stack();

    let mut fourier_ggsw =
        FourierGgswCiphertext::new(ctx.glwe_size, ctx.poly_size, ctx.base_log, ctx.level_count);
    convert_standard_ggsw_ciphertext_to_fourier_mem_optimized(
        &gsw_ct,
        &mut fourier_ggsw,
        ctx.fft.as_view(),
        stack.rb_mut(),
    );

    c.bench_function("cmux", |b| {
        b.iter(|| {
            cmux_assign_mem_optimized(
                &mut rlwe_ct_0,
                &mut rlwe_ct_1,
                &fourier_ggsw,
                ctx.fft.as_view(),
                stack.rb_mut(),
            )
        })
    });
}

pub fn external_product_benchmark(c: &mut Criterion) {
    let mut ctx = Context::new(TFHEParameters::default());
    let sk = GlweSecretKey::generate_new_binary(
        ctx.glwe_dimension,
        ctx.poly_size,
        &mut ctx.secret_generator,
    );
    let mut mem = ComputationBuffers::new();
    mem.resize(
        add_external_product_assign_mem_optimized_requirement::<Scalar>(
            ctx.glwe_size,
            ctx.poly_size,
            ctx.fft.as_view(),
        )
        .unwrap()
        .unaligned_bytes_required()
        .max(
            convert_standard_ggsw_ciphertext_to_fourier_mem_optimized_requirement(
                ctx.fft.as_view(),
            )
            .unwrap()
            .unaligned_bytes_required(),
        ),
    );

    let mut stack = mem.stack();

    let mut gsw_ct = GgswCiphertext::new(
        Scalar::zero(),
        ctx.glwe_size,
        ctx.poly_size,
        ctx.base_log,
        ctx.level_count,
        ctx.ciphertext_modulus,
    );
    encrypt_constant_ggsw_ciphertext(
        &sk,
        &mut gsw_ct,
        Plaintext(Scalar::one()),
        ctx.std,
        &mut ctx.encryption_generator,
    );

    let mut fourier_ggsw =
        FourierGgswCiphertext::new(ctx.glwe_size, ctx.poly_size, ctx.base_log, ctx.level_count);
    convert_standard_ggsw_ciphertext_to_fourier_mem_optimized(
        &gsw_ct,
        &mut fourier_ggsw,
        ctx.fft.as_view(),
        stack.rb_mut(),
    );

    let mut rlwe_ct_0 = ctx.empty_glwe_ciphertext();

    let mut out_pt = PlaintextList::new(Scalar::zero(), ctx.plaintext_count());
    (*out_pt.as_mut_polynomial().as_mut())[0] = Scalar::zero();

    encrypt_glwe_ciphertext(
        &sk,
        &mut rlwe_ct_0,
        &out_pt,
        ctx.std,
        &mut ctx.encryption_generator,
    );

    let mut rlwe_ct_out = ctx.empty_glwe_ciphertext();
    c.bench_function("external_product", |b| {
        b.iter(|| {
            add_external_product_assign_mem_optimized(
                &mut rlwe_ct_out,
                &fourier_ggsw,
                &rlwe_ct_0,
                ctx.fft.as_view(),
                stack.rb_mut(),
            )
        })
    });
}

pub fn enc_benchmark(c: &mut Criterion) {
    let mut ctx = Context::new(TFHEParameters::default());
    let sk = GlweSecretKey::generate_new_binary(
        ctx.glwe_dimension,
        ctx.poly_size,
        &mut ctx.secret_generator,
    );

    let mut ct = ctx.empty_glwe_ciphertext();
    let pt = ctx.gen_binary_pt();

    c.bench_function("enc", |b| {
        b.iter(|| {
            encrypt_glwe_ciphertext(&sk, &mut ct, &pt, ctx.std, &mut ctx.encryption_generator);
        });
    });
}

pub fn dec_benchmark(c: &mut Criterion) {
    let mut ctx = Context::new(TFHEParameters::default());
    let sk = GlweSecretKey::generate_new_binary(
        ctx.glwe_dimension,
        ctx.poly_size,
        &mut ctx.secret_generator,
    );

    let mut ct = ctx.empty_glwe_ciphertext();
    let pt = ctx.gen_binary_pt();

    encrypt_glwe_ciphertext(&sk, &mut ct, &pt, ctx.std, &mut ctx.encryption_generator);

    let mut out_pt = PlaintextList::new(Scalar::zero(), ctx.plaintext_count());
    c.bench_function("dec", |b| {
        b.iter(|| {
            decrypt_glwe_ciphertext(&sk, &ct, &mut out_pt);
            ctx.codec.poly_decode(&mut out_pt.as_mut_polynomial());
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
    expand_benchmark,
    less_eq_benchmark,
    trace1_benchmark,
    gen_ksk_benchmark,
    // oram_benchmark
);
criterion_main!(benches);
