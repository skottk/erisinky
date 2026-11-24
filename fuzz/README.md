# Inkling Fuzzing & Sanitizer Harnesses (Phase 4)

Coverage-guided libFuzzer targets (`cargo-fuzz`) and sanitizer commands for `inkling`.

## Targets

- `fuzz_gguf`: feeds arbitrary `&[u8]` to `Gguf::from_bytes`, then exercises `Gguf::bytes`, `try_u`/`try_f`, `audit_weights_finiteness`, and `validate_inkling_model`.
- `fuzz_tokenizer`: structured `Arbitrary` fuzzer for `Tokenizer::encode`, `decode_bytes`, and `is_special`, asserting zero control-token smuggling (`parse_special=false`), exact byte round-tripping, OOB token-ID safety, and an $O(L \log L)$ BPE wall-clock complexity bound.
- `fuzz_renderer`: structured `Arbitrary` fuzzer for `Renderer::push` and `end`, asserting valid UTF-8 output, clean `thinking` reset on `end()`, and zero leaked raw ESC/DEL/C0/C1 control sequences when `sanitize_tty` is enabled.

## Running Fuzz Targets

Requires Rust nightly + `cargo-fuzz` (`rustup toolchain install nightly --component miri,rust-src && cargo +nightly install cargo-fuzz`).

### 1. Overflow-checked + AddressSanitizer (default `cargo-fuzz` profile)
```sh
cargo +nightly fuzz run fuzz_gguf -- -max_total_time=30 -max_len=4096
cargo +nightly fuzz run fuzz_tokenizer -- -max_total_time=30 -max_len=4096
cargo +nightly fuzz run fuzz_renderer -- -max_total_time=30 -max_len=4096
```

### 2. Release-like wrapping overflow + AddressSanitizer (`-C overflow-checks=off`)
Catches silent integer wrap-arounds that could bypass bounds checks in release builds:
```sh
RUSTFLAGS="--cfg fuzzing -C passes=sancov-module -C llvm-args=-sanitizer-coverage-level=4 -C llvm-args=-sanitizer-coverage-inline-8bit-counters -C llvm-args=-sanitizer-coverage-pc-table -C llvm-args=-sanitizer-coverage-trace-compares -Z sanitizer=address -C overflow-checks=off -C debug-assertions=off" \
  cargo +nightly build --manifest-path fuzz/Cargo.toml --release --target aarch64-apple-darwin --bins
./fuzz/target/aarch64-apple-darwin/release/fuzz_gguf fuzz/corpus/fuzz_gguf -max_total_time=30 -max_len=4096
```

## Running Miri, ASan, TSan, and Clang ASan/UBSan

```sh
# Miri on pure-Rust suites (excluding mmap / GPU / FileExt tests)
PROPTEST_CASES=4 MIRIFLAGS="-Zmiri-disable-isolation" cargo +nightly miri test \
  --test phase0_smoke --test high_sev_gguf --test high_sev_prompt_smuggling \
  --test med_sev_sampler_cli --test med_sev_session_renderer \
  --test med_sev_engine_tier_bounds --test low_sev_engine_helpers \
  --test fast_math_weight_audit --test phase4_fuzz_sanitizers -- \
  --skip validate_inkling_model --skip rejects_empty_file_in_open \
  --skip phase0_gguf_from_bytes_and_open_roundtrip \
  --skip tsan_parallel_tier_read_disjoint_slices

# ASan on CPU + Metal GPU suites
RUSTFLAGS="-Z sanitizer=address" ASAN_OPTIONS=detect_leaks=0 MTL_DEBUG_LAYER=1 \
  cargo +nightly test --target aarch64-apple-darwin

# TSan on multi-threaded suites (parallel weight audit + parallel tier_read)
RUSTFLAGS="-Z sanitizer=thread" cargo +nightly test -Z build-std --target aarch64-apple-darwin \
  --test fast_math_weight_audit --test high_sev_kv_disk_security \
  --test med_sev_engine_tier_bounds --test phase4_fuzz_sanitizers

# Clang ASan + UBSan on tools/ref_logits.cpp
REF_LOGITS_EXTRA_CXXFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
  cargo test --test low_sev_ref_logits
```
