use clap::Parser;
use heoram::{
    cli::Cli,
    naive_hash::NaiveHash,
    oram::setup_random_oram,
    params::{ORAMParameters, ServerParams, TFHEParameters},
    rlwe::compute_noise_encoded,
    utils::log2,
};
use std::{fs::File, io::BufReader, time::Instant};

fn single_query(item_count: usize, iterations: usize, tfhe_params: TFHEParameters, dryrun: bool) {
    assert!(iterations > 0);
    print!("0,{item_count},-,-,-,2^{},", log2(item_count));
    if !dryrun {
        let hash = NaiveHash::new(1, item_count);
        let setup_instant = Instant::now();
        let (mut client, mut server, pts) =
            setup_random_oram(1, item_count, &hash, tfhe_params.clone());

        // preparing for a read
        let idx = 0usize;
        let query = client.gen_read_query_one(idx, &hash);
        let setup_duration = setup_instant.elapsed().as_secs_f64();

        let server_instant = Instant::now();
        let (y, response_duration) = server.process_one(query);
        let server_duration = server_instant.elapsed().as_secs_f64();

        // check for result
        let mut pt = client.ctx.gen_zero_pt();
        client.sk.decrypt_decode_rlwe(&mut pt, &y, &client.ctx);
        assert_eq!(pts[idx], pt);

        // check the final noise
        let noise = compute_noise_encoded(&client.sk, &y, &pt, &client.ctx.codec);

        let response_duration = response_duration.as_secs_f64();
        println!("{setup_duration},{server_duration},{response_duration},{noise}");

        // run the remaining iterations
        for iter in 1..iterations {
            let query = client.gen_read_query_one(idx, &hash);
            let (y, _) = server.process_one(query);
            let mut pt = client.ctx.gen_zero_pt();
            client.sk.decrypt_decode_rlwe(&mut pt, &y, &client.ctx);
            assert_eq!(pts[idx], pt);
            let noise = compute_noise_encoded(&client.sk, &y, &pt, &client.ctx.codec);
            println!("### mode=0, iter={}, noise={}", iter, noise);
        }
    } else {
        println!("-,-,-,-");
    }
}

fn multi_query(
    item_count: usize,
    query_count: usize,
    iterations: usize,
    tfhe_params: TFHEParameters,
    dryrun: bool,
) {
    assert!(iterations > 0);
    print!("1,{item_count},{query_count},-,-,2^{},", log2(item_count));
    if !dryrun {
        let hash = NaiveHash::new(1, item_count);
        let setup_instant = Instant::now();
        let (mut client, mut server, pts) =
            setup_random_oram(1, item_count, &hash, tfhe_params.clone());
        assert_eq!(item_count, pts.len());

        let queries = (0..query_count)
            .map(|i| client.gen_read_query_one(i + 1, &hash))
            .collect();

        let setup_duration = setup_instant.elapsed().as_secs_f64();

        let server_instant = Instant::now();
        let (ys, response_duration) = server.process_multi(queries);
        let server_duration = server_instant.elapsed().as_secs_f64();

        let mut avg_noise = 0f64;
        let mut pt = client.ctx.gen_zero_pt();
        for i in 0..query_count {
            client.sk.decrypt_decode_rlwe(&mut pt, &ys[i], &client.ctx);
            avg_noise += compute_noise_encoded(&client.sk, &ys[i], &pt, &client.ctx.codec);
            assert_eq!(pts[i + 1], pt);
        }
        avg_noise /= query_count as f64;

        let response_duration = response_duration.as_secs_f64();
        println!("{setup_duration},{server_duration},{response_duration},{avg_noise}");

        // run the remaining iterations
        for iter in 1..iterations {
            let queries = (0..query_count)
                .map(|i| client.gen_read_query_one(i + 1, &hash))
                .collect();
            let (ys, _) = server.process_multi(queries);

            let mut avg_noise = 0f64;
            let mut pt = client.ctx.gen_zero_pt();
            for i in 0..query_count {
                client.sk.decrypt_decode_rlwe(&mut pt, &ys[i], &client.ctx);
                avg_noise += compute_noise_encoded(&client.sk, &ys[i], &pt, &client.ctx.codec);
                assert_eq!(pts[i + 1], pt);
            }
            avg_noise /= query_count as f64;

            println!("### mode=1, iter={}, noise={}", iter, avg_noise);
        }
    } else {
        println!("-,-,-,-");
    }
}

fn batch_query(
    rows: usize,
    cols: usize,
    iterations: usize,
    tfhe_params: TFHEParameters,
    dryrun: bool,
) {
    assert!(iterations > 0);
    let setup_instant = Instant::now();
    let h_count = 3usize;
    let item_count = (rows * cols) / h_count;

    print!("2,{item_count},-,{rows},{cols},2^{},", log2(item_count));
    if !dryrun {
        let hash = NaiveHash::new(h_count, item_count);
        let (mut client, mut server, pts) =
            setup_random_oram(rows, cols, &hash, tfhe_params.clone());

        // preparing for a read
        let indices = vec![0usize, 1usize];
        let query = client.gen_read_query_batch(&indices, &hash);
        let setup_duration = setup_instant.elapsed().as_secs_f64();

        let server_instant = Instant::now();
        let (ys, response_duration) = server.process_batch(query, &hash);
        let server_duration = server_instant.elapsed().as_secs_f64();

        let response_duration = response_duration.as_secs_f64();

        // check for result
        let mapping = hash.hash_to_mapping(&indices, cols);
        let mut pt = client.ctx.gen_zero_pt();
        let mut avg_noise = 0f64;
        for (r, (_, i)) in mapping {
            client.sk.decrypt_decode_rlwe(&mut pt, &ys[r], &client.ctx);
            assert_eq!(pts[indices[i]], pt);
            let noise = compute_noise_encoded(&client.sk, &ys[r], &pt, &client.ctx.codec);
            avg_noise += noise;
        }
        avg_noise /= indices.len() as f64;
        println!("{setup_duration},{server_duration},{response_duration},{avg_noise}");

        for iter in 1..iterations {
            let query = client.gen_read_query_batch(&indices, &hash);
            let (ys, _) = server.process_batch(query, &hash);
            let mapping = hash.hash_to_mapping(&indices, cols);
            let mut pt = client.ctx.gen_zero_pt();
            let mut avg_noise = 0f64;
            for (r, (_, i)) in mapping {
                client.sk.decrypt_decode_rlwe(&mut pt, &ys[r], &client.ctx);
                assert_eq!(pts[indices[i]], pt);
                let noise = compute_noise_encoded(&client.sk, &ys[r], &pt, &client.ctx.codec);
                avg_noise += noise;
            }
            avg_noise /= indices.len() as f64;

            println!("### mode=2, iter={}, noise={}", iter, avg_noise);
        }
    } else {
        println!("-,-,-,-");
    }
}

fn main() {
    // Parameters from SEAL PIR: https://eprint.iacr.org/2017/1142.pdf
    // n = 2^20
    // k = 256
    // b = 1.5k = 384
    // w = 3
    // every row (bucket) in the database has 3*n / b = 2^13 elements
    let cli = Cli::parse();

    // if we have a value passed to --params, all other params from cli are ignored
    //      if it is able to read the file, it gets all parameters from there
    //      otherwise, if reading the file fails (wrong path or format), then it uses default parameters
    // otherwise, it uses parameters from cli with the preset defaults

    let input_params = match File::open(&cli.params) {
        Ok(file) => ServerParams::from_input_params_list(
            serde_json::from_reader(BufReader::new(file)).unwrap_or_default(),
        ),
        _ => vec![ServerParams {
            oram: match cli.mode {
                0 => ORAMParameters::SingleQuery {
                    item_count: cli.item_count,
                    iterations: 1,
                },
                1 => ORAMParameters::MultiQuery {
                    item_count: cli.item_count,
                    query_count: cli.query_count,
                    iterations: 1,
                },
                _ => ORAMParameters::Batched {
                    rows: cli.rows,
                    cols: cli.cols,
                    iterations: 1,
                },
            },
            tfhe: TFHEParameters {
                standard_deviation: cli.standard_deviation,
                polynomial_size: cli.polynomial_size,
                base_log: cli.base_log,
                level_count: cli.level_count,
                key_switch_base_log: cli.key_switch_base_log,
                key_switch_level_count: cli.key_switch_level_count,
                negs_base_log: cli.negs_base_log,
                negs_level_count: cli.negs_level_count,
                plaintext_modulus: cli.plaintext_modulus,
                secure_seed: cli.secure_seed,
            },
        }],
    };
    println!(
        "standard_deviation,polynomial_size,base_log,level_count,key_switch_base_log,key_switch_level_count,negs_base_log,negs_level_count,plaintext_modulus,secure_seed,mode,item_count,query_count,rows,cols,n,setup_duration,server_duration,response_duration,final_noise"
    );
    for params in input_params {
        print!(
            "{},{},{},{},{},{},{},{},2^{},{},",
            params.tfhe.standard_deviation,
            params.tfhe.polynomial_size,
            params.tfhe.base_log,
            params.tfhe.level_count,
            params.tfhe.key_switch_base_log,
            params.tfhe.key_switch_level_count,
            params.tfhe.negs_base_log,
            params.tfhe.negs_level_count,
            log2(params.tfhe.plaintext_modulus as usize),
            params.tfhe.secure_seed
        );
        match params.oram {
            ORAMParameters::SingleQuery {
                item_count,
                iterations,
            } => single_query(item_count, iterations, params.tfhe, cli.dryrun),
            ORAMParameters::MultiQuery {
                item_count,
                query_count,
                iterations,
            } => multi_query(item_count, query_count, iterations, params.tfhe, cli.dryrun),
            ORAMParameters::Batched {
                rows,
                cols,
                iterations,
            } => batch_query(rows, cols, iterations, params.tfhe, cli.dryrun),
        }
    }
}
