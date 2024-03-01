use clap::Parser;

use crate::params::TFHEParameters;

#[derive(Parser, Debug, Clone)]
#[clap(author, version, about="ORAM from TFHE", long_about = None)]
pub struct Cli {
    #[clap(
        long,
        default_value_t = String::new(),
        help = "Path to JSON file with parameters"
    )]
    pub params: String,

    #[clap(
        long,
        default_value_t = 2,
        help = "0 => SingleQuery, 1 => MultiQuery, 2 => Batch"
    )]
    pub mode: usize,

    #[clap(
        long,
        default_value_t = 4096,
        help = "number of items in the database (only in SingleQuery and MultiQuery modes)"
    )]
    pub item_count: usize,

    #[clap(
        long,
        default_value_t = 1,
        help = "Number of queries to be processed (only in MultiQuery mode)"
    )]
    pub query_count: usize,

    #[clap(
        long,
        default_value_t = 96,
        help = "number of rows (buckets) in the database (only in Batch mode)"
    )]
    pub rows: usize,

    #[clap(
        long,
        default_value_t = 64,
        help = "number of columns in the database, must be a power of 2 (only in Batch mode)"
    )]
    pub cols: usize,

    #[clap(long, allow_negative_numbers = true, default_value_t = TFHEParameters::default().standard_deviation)]
    pub standard_deviation: f64,

    #[clap(long, default_value_t = TFHEParameters::default().polynomial_size)]
    pub polynomial_size: usize,

    #[clap(long, default_value_t = TFHEParameters::default().base_log)]
    pub base_log: usize,

    #[clap(long, default_value_t = TFHEParameters::default().level_count)]
    pub level_count: usize,

    #[clap(long, default_value_t = TFHEParameters::default().key_switch_base_log)]
    pub key_switch_base_log: usize,

    #[clap(long, default_value_t = TFHEParameters::default().key_switch_level_count)]
    pub key_switch_level_count: usize,

    #[clap(long, default_value_t = TFHEParameters::default().negs_base_log)]
    pub negs_base_log: usize,

    #[clap(long, default_value_t = TFHEParameters::default().negs_level_count)]
    pub negs_level_count: usize,

    #[clap(long, default_value_t = TFHEParameters::default().plaintext_modulus)]
    pub plaintext_modulus: u64,

    #[clap(long, help = "print more information")]
    pub verbose: bool,

    #[clap(long, default_value_t = false, help = "print params")]
    pub dryrun: bool,
}
