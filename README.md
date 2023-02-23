# Panacea: Non-interactive and Stateless Oblivious RAM

This is an implementation of Panacea.
The source code is based on 
[SortingHat](https://github.com/KULeuven-COSIC/SortingHat).

**WARNING**:

This is proof-of-concept implementation.
It may contain bugs and security issues.
Please do not use in production systems.

## Build and test

Our implementation is written in the Rust programming language.
Please see [rustup](https://rustup.rs/) on how to install it.
The main library we use is [concrete-core](https://github.com/zama-ai/concrete-core),
but this is installed automatically with `cargo` as shown below.

```
# to install:
cargo build --release
# to test:
cargo test --release
```

The stable toolchain can be used for Linux.
For mac OS, the nightly toolchain (1.68 or above)
must be used.
Windows is not supported.

## Running

### Getting the CLI options

```
./target/release/heoram --help
```

### Running with parameters from the paper

```
./target/release/heoram --params params.json
```

Note that larger parameters have higher memory requirement
(batched mode with rows=384 and cols=4096 needs 1 TB of memory).
Please refer to the paper to for the exact memory requirement.

### Setting the number of threads

```
# for example, 8 threads
RAYON_NUM_THREADS=8 ./target/release/heoram --params params.json
```
