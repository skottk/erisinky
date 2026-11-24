//! Model-free CPU & synthetic Metal GPU microbenchmark suite for `inkling`.
//!
//! Run with:
//!   cargo run --release --offline --example micro_bench -- [options]
//!
//! Options:
//!   --json                 Emit compact JSON report to stdout
//!   --iters N              Measured iterations per benchmark (default: 25)
//!   --warmup N             Warmup iterations per benchmark (default: 5)
//!   --only <comma-list>    Filter: `cpu`, `gpu`, or specific benchmark names
//!   --list                 List available benchmarks and exit
//!   -h, --help             Show usage and exit

use inkling::engine::tier_math::{rec_to_rows, rows_to_rec};
use inkling::engine::{dispatch_mix_jobs, MixJob, E, HD, NKV};
use inkling::gguf::{Gguf, T_F32};
use inkling::metal::{Enc, Gpu};
use inkling::sampler::Sampler;
use inkling::tokenizer::Tokenizer;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue};
use std::collections::BTreeMap;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

const REC_BYTES: usize = 2 * NKV * HD * 2;
const USAGE: &str = "\
usage: micro_bench [options]

options:
  --json                 Emit compact JSON report to stdout
  --iters N              Measured iterations per benchmark (default: 25)
  --warmup N             Warmup iterations per benchmark (default: 5)
  --only <comma-list>    Filter: cpu, gpu, or specific benchmark names
  --list                 List available benchmarks
  -h, --help             Show this help message
";

const ALL_BENCHES: &[(&str, &str)] = &[
    ("cpu_tokenizer_long_prompt", "cpu"),
    ("cpu_tokenizer_repeat_4k", "cpu"),
    ("cpu_sampler_200k", "cpu"),
    ("cpu_tier_rows_rec_64", "cpu"),
    ("cpu_gguf_from_bytes", "cpu"),
    ("gpu_mix_t1_decode", "gpu"),
    ("gpu_mix_t1_decode_kmix", "gpu"),
    ("gpu_mix_t64_prefill", "gpu"),
    ("gpu_mix_t512_prefill", "gpu"),
    ("gpu_cmdbuf_530_dispatches", "gpu"),
];

#[derive(Clone, Debug)]
struct Config {
    json: bool,
    iters: usize,
    warmup: usize,
    selected: Vec<&'static str>,
}

#[derive(Clone, Debug)]
enum JsonVal {
    Num(f64),
    Int(u64),
    Bool(bool),
    Str(String),
}

struct BenchResult {
    name: &'static str,
    metric: &'static str,
    lower_is_better: bool,
    median: f64,
    p10: f64,
    p90: f64,
    iqr: f64,
    samples: Vec<f64>,
    extra: BTreeMap<String, JsonVal>,
    note: String,
}

fn parse_args() -> Result<Option<Config>, String> {
    let mut json = false;
    let mut iters: usize = 25;
    let mut warmup: usize = 5;
    let mut only_raw: Option<String> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--json" => json = true,
            "--iters" => {
                let v = args.next().ok_or("missing value for --iters")?;
                iters = v
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --iters '{v}': {e}"))?;
                if iters == 0 {
                    return Err("--iters must be >= 1".into());
                }
            }
            "--warmup" => {
                let v = args.next().ok_or("missing value for --warmup")?;
                warmup = v
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --warmup '{v}': {e}"))?;
            }
            "--only" => {
                let v = args.next().ok_or("missing value for --only")?;
                only_raw = Some(v);
            }
            "--list" => {
                for &(name, group) in ALL_BENCHES {
                    println!("{name:30} ({group})");
                }
                return Ok(None);
            }
            "-h" | "--help" => {
                print!("{USAGE}");
                return Ok(None);
            }
            other => return Err(format!("unknown option '{other}'\n\n{USAGE}")),
        }
    }

    let selected: Vec<&'static str> = match only_raw {
        None => ALL_BENCHES.iter().map(|&(n, _)| n).collect(),
        Some(raw) => {
            let mut picks: Vec<&'static str> = Vec::new();
            for item in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                match item {
                    "cpu" | "gpu" => {
                        for &(name, group) in ALL_BENCHES {
                            if group == item && !picks.contains(&name) {
                                picks.push(name);
                            }
                        }
                    }
                    exact => {
                        if let Some(&(name, _)) = ALL_BENCHES.iter().find(|&&(n, _)| n == exact) {
                            if !picks.contains(&name) {
                                picks.push(name);
                            }
                        } else {
                            let valid: Vec<&str> = ALL_BENCHES.iter().map(|&(n, _)| n).collect();
                            return Err(format!(
                                "unknown benchmark filter '{exact}'; valid groups: cpu, gpu; benchmarks: {}",
                                valid.join(", ")
                            ));
                        }
                    }
                }
            }
            if picks.is_empty() {
                return Err("--only filter matched no benchmarks".into());
            }
            picks
        }
    };

    Ok(Some(Config {
        json,
        iters,
        warmup,
        selected,
    }))
}

// ----------------------------------------------------------------------------- statistics & JSON

fn quantile(sorted: &[f64], q: f64) -> f64 {
    assert!(!sorted.is_empty());
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

fn summarize(samples: &[f64]) -> (f64, f64, f64, f64) {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let p10 = quantile(&sorted, 0.10);
    let p25 = quantile(&sorted, 0.25);
    let median = quantile(&sorted, 0.50);
    let p75 = quantile(&sorted, 0.75);
    let p90 = quantile(&sorted, 0.90);
    (median, p10, p90, p75 - p25)
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn fmt_f64(v: f64) -> String {
    if !v.is_finite() {
        "null".to_string()
    } else {
        format!("{v:.6}")
    }
}

fn fmt_json_val(v: &JsonVal) -> String {
    match v {
        JsonVal::Num(x) => fmt_f64(*x),
        JsonVal::Int(i) => i.to_string(),
        JsonVal::Bool(b) => if *b { "true" } else { "false" }.to_string(),
        JsonVal::Str(s) => json_escape(s),
    }
}

fn emit_json(results: &[BenchResult]) {
    let mut out = String::with_capacity(4096);
    out.push_str("{\"schema\":1,\"host\":{\"os\":");
    out.push_str(&json_escape(std::env::consts::OS));
    out.push_str(",\"arch\":");
    out.push_str(&json_escape(std::env::consts::ARCH));
    out.push_str("},\"benchmarks\":{");
    for (idx, r) in results.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        out.push_str(&json_escape(r.name));
        out.push_str(":{\"metric\":");
        out.push_str(&json_escape(r.metric));
        out.push_str(",\"lower_is_better\":");
        out.push_str(if r.lower_is_better { "true" } else { "false" });
        out.push_str(",\"median\":");
        out.push_str(&fmt_f64(r.median));
        out.push_str(",\"p10\":");
        out.push_str(&fmt_f64(r.p10));
        out.push_str(",\"p90\":");
        out.push_str(&fmt_f64(r.p90));
        out.push_str(",\"iqr\":");
        out.push_str(&fmt_f64(r.iqr));
        out.push_str(",\"samples\":[");
        for (si, &s) in r.samples.iter().enumerate() {
            if si > 0 {
                out.push(',');
            }
            out.push_str(&fmt_f64(s));
        }
        out.push_str("],\"extra\":{");
        for (ei, (k, v)) in r.extra.iter().enumerate() {
            if ei > 0 {
                out.push(',');
            }
            out.push_str(&json_escape(k));
            out.push(':');
            out.push_str(&fmt_json_val(v));
        }
        out.push_str("}}");
    }
    out.push_str("}}");
    println!("{out}");
}

fn emit_table(results: &[BenchResult], cfg: &Config) {
    println!(
        "inkling micro_bench ({}/{} | warmup={}, iters={})",
        std::env::consts::OS,
        std::env::consts::ARCH,
        cfg.warmup,
        cfg.iters
    );
    println!(
        "{:<28} {:>11} {:>11} {:>11} {:>10}  notes",
        "benchmark", "median", "p10", "p90", "iqr"
    );
    println!("{}", "-".repeat(100));
    for r in results {
        println!(
            "{:<28} {:>8.2} {:<2} {:>8.2} {:<2} {:>8.2} {:<2} {:>7.2} {:<2}  {}",
            r.name, r.median, r.metric, r.p10, r.metric, r.p90, r.metric, r.iqr, r.metric, r.note
        );
    }
}

// ----------------------------------------------------------------------------- deterministic helpers

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn next_f32_unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / ((1u32 << 24) as f32)
    }
}

// Minimal GGUF v3 writer (self-contained, zero external deps).
mod gguf_gen {
    pub const TY_U32: u32 = 4;
    pub const TY_I32: u32 = 5;
    pub const TY_F32: u32 = 6;
    pub const TY_BOOL: u32 = 7;
    pub const TY_STR: u32 = 8;
    pub const TY_ARR: u32 = 9;

    pub fn write_str(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    pub fn kv_u32(out: &mut Vec<u8>, k: &str, v: u32) {
        write_str(out, k);
        out.extend_from_slice(&TY_U32.to_le_bytes());
        out.extend_from_slice(&v.to_le_bytes());
    }

    pub fn kv_i32(out: &mut Vec<u8>, k: &str, v: i32) {
        write_str(out, k);
        out.extend_from_slice(&TY_I32.to_le_bytes());
        out.extend_from_slice(&v.to_le_bytes());
    }

    pub fn kv_f32(out: &mut Vec<u8>, k: &str, v: f32) {
        write_str(out, k);
        out.extend_from_slice(&TY_F32.to_le_bytes());
        out.extend_from_slice(&v.to_le_bytes());
    }

    pub fn kv_bool(out: &mut Vec<u8>, k: &str, v: bool) {
        write_str(out, k);
        out.extend_from_slice(&TY_BOOL.to_le_bytes());
        out.push(v as u8);
    }

    pub fn kv_str(out: &mut Vec<u8>, k: &str, v: &str) {
        write_str(out, k);
        out.extend_from_slice(&TY_STR.to_le_bytes());
        write_str(out, v);
    }

    pub fn kv_arr_str(out: &mut Vec<u8>, k: &str, xs: &[String]) {
        write_str(out, k);
        out.extend_from_slice(&TY_ARR.to_le_bytes());
        out.extend_from_slice(&TY_STR.to_le_bytes());
        out.extend_from_slice(&(xs.len() as u64).to_le_bytes());
        for s in xs {
            write_str(out, s);
        }
    }

    pub fn kv_arr_i32(out: &mut Vec<u8>, k: &str, xs: &[i32]) {
        write_str(out, k);
        out.extend_from_slice(&TY_ARR.to_le_bytes());
        out.extend_from_slice(&TY_I32.to_le_bytes());
        out.extend_from_slice(&(xs.len() as u64).to_le_bytes());
        for &x in xs {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
}

fn build_synthetic_tokenizer() -> Tokenizer {
    let mut tokens: Vec<String> = Vec::with_capacity(512);
    let mut types: Vec<i32> = Vec::with_capacity(512);

    // GPT-2 byte <-> unicode table (all 256 single-byte tokens)
    let mut n = 0u32;
    let mut byte_str = Vec::with_capacity(256);
    for b in 0..256u32 {
        let keep = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        let cp = if keep {
            b
        } else {
            n += 1;
            255 + n
        };
        let s = char::from_u32(cp).unwrap().to_string();
        byte_str.push(s.clone());
        tokens.push(s);
        types.push(1);
    }

    // Standard Inkling control tokens (tok_type = 3)
    for ctrl in [
        "<|end_message|>",
        "<|message_system|>",
        "<|message_user|>",
        "<|message_model|>",
        "<|content_text|>",
        "<|content_thinking|>",
        "<|content_model_end_sampling|>",
    ] {
        tokens.push(ctrl.to_string());
        types.push(3);
    }

    // Common English/code subword merges plus repeated-'a' merges so BPE scaling is exercised.
    let space_ch = &byte_str[b' ' as usize];
    let raw_merges: Vec<(String, String)> = vec![
        ("a".into(), "a".into()),
        ("aa".into(), "aa".into()),
        ("aaaa".into(), "aaaa".into()),
        ("t".into(), "h".into()),
        ("th".into(), "e".into()),
        ("i".into(), "n".into()),
        ("e".into(), "r".into()),
        ("a".into(), "n".into()),
        ("r".into(), "e".into()),
        ("o".into(), "n".into()),
        ("a".into(), "t".into()),
        ("e".into(), "n".into()),
        ("e".into(), "s".into()),
        ("o".into(), "r".into()),
        ("t".into(), "i".into()),
        ("i".into(), "s".into()),
        ("t".into(), "o".into()),
        ("i".into(), "t".into()),
        ("o".into(), "u".into()),
        ("a".into(), "l".into()),
        ("a".into(), "r".into()),
        ("s".into(), "t".into()),
        ("n".into(), "t".into()),
        ("n".into(), "g".into()),
        ("i".into(), "ng".into()),
        ("h".into(), "t".into()),
        ("ht".into(), "t".into()),
        ("htt".into(), "p".into()),
        ("http".into(), "s".into()),
        ("c".into(), "o".into()),
        ("co".into(), "m".into()),
        ("d".into(), "e".into()),
        ("l".into(), "l".into()),
        ("ll".into(), "a".into()),
        ("lla".into(), "m".into()),
        ("llam".into(), "a".into()),
        (space_ch.clone(), "t".into()),
        (format!("{space_ch}t"), "h".into()),
        (format!("{space_ch}th"), "e".into()),
        (space_ch.clone(), "a".into()),
        (format!("{space_ch}a"), "n".into()),
        (format!("{space_ch}an"), "d".into()),
        (space_ch.clone(), "i".into()),
        (format!("{space_ch}i"), "n".into()),
        (space_ch.clone(), "o".into()),
        (format!("{space_ch}o"), "f".into()),
        (space_ch.clone(), "to".into()),
        (space_ch.clone(), "for".into()),
        ("f".into(), "o".into()),
        ("fo".into(), "r".into()),
        ("-".into(), "-".into()),
        ("/".into(), "/".into()),
    ];

    let mut merges = Vec::with_capacity(raw_merges.len());
    for (a, b) in raw_merges {
        let merged = format!("{a}{b}");
        if !tokens.contains(&merged) {
            tokens.push(merged);
            types.push(1);
        }
        merges.push(format!("{a} {b}"));
    }

    let mut raw = Vec::with_capacity(16 * 1024);
    raw.extend_from_slice(b"GGUF");
    raw.extend_from_slice(&3u32.to_le_bytes());
    raw.extend_from_slice(&0u64.to_le_bytes()); // 0 tensors
    raw.extend_from_slice(&4u64.to_le_bytes()); // 4 kvs
    gguf_gen::kv_arr_str(&mut raw, "tokenizer.ggml.tokens", &tokens);
    gguf_gen::kv_arr_i32(&mut raw, "tokenizer.ggml.token_type", &types);
    gguf_gen::kv_arr_str(&mut raw, "tokenizer.ggml.merges", &merges);
    gguf_gen::kv_u32(&mut raw, "tokenizer.ggml.eos_token_id", 256);
    while raw.len() % 32 != 0 {
        raw.push(0);
    }

    let g = Gguf::from_bytes(&raw).expect("parse synthetic tokenizer gguf");
    Tokenizer::load(&g)
}

fn load_long_prompt() -> (String, &'static str) {
    let candidates = [
        PathBuf::from("ref/long_prompt.txt"),
        Path::new(env!("CARGO_MANIFEST_DIR")).join("ref/long_prompt.txt"),
    ];
    for p in &candidates {
        if let Ok(s) = std::fs::read_to_string(p) {
            if !s.is_empty() {
                return (s, "ref/long_prompt.txt");
            }
        }
    }
    let paragraph = "The Metal inference engine pipelines quantized matrix-vector products, \
                     sliding-window attention, and mixture-of-experts routing across 24 physical \
                     layers with 3 recurrent passes. Benchmarks measure median latency, p10, p90, \
                     and IQR to detect subtle regressions in tokenizer, sampler, and kernel paths.\n";
    let mut s = String::with_capacity(8192);
    while s.len() < 8192 {
        s.push_str(paragraph);
    }
    (s, "synthetic_8k")
}

fn build_synthetic_gguf_512kv_128t() -> Vec<u8> {
    let n_kv = 512usize;
    let n_tensors = 128usize;
    let mut out = Vec::with_capacity(256 * 1024);
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(n_tensors as u64).to_le_bytes());
    out.extend_from_slice(&(n_kv as u64).to_le_bytes());

    gguf_gen::kv_u32(&mut out, "general.alignment", 32);
    for i in 1..n_kv {
        let key = format!("bench.meta.section_{}.item_{i:04}", i / 32);
        match i % 6 {
            0 => gguf_gen::kv_u32(&mut out, &key, i as u32 * 17),
            1 => gguf_gen::kv_i32(&mut out, &key, -(i as i32) * 13),
            2 => gguf_gen::kv_f32(&mut out, &key, (i as f32) * 0.125),
            3 => gguf_gen::kv_bool(&mut out, &key, i % 2 == 0),
            4 => gguf_gen::kv_str(&mut out, &key, &format!("value_payload_{i:04}_inkling")),
            _ => {
                let arr: Vec<i32> = (0..8).map(|j| (i * 8 + j) as i32).collect();
                gguf_gen::kv_arr_i32(&mut out, &key, &arr);
            }
        }
    }

    // 128 F32 tensors of shape [32, 4] (512 bytes each -> 64 KiB total tensor payload)
    let dims = [32u64, 4u64];
    let tensor_bytes = 32usize * 4 * 4;
    for t in 0..n_tensors {
        let name = format!("blk.{}.weight_{:02}.data", t / 8, t % 8);
        gguf_gen::write_str(&mut out, &name);
        out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for &d in &dims {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&T_F32.to_le_bytes());
        let off = (t * tensor_bytes) as u64;
        out.extend_from_slice(&off.to_le_bytes());
    }
    while out.len() % 32 != 0 {
        out.push(0);
    }
    out.resize(out.len() + n_tensors * tensor_bytes, 0u8);
    out
}

// ----------------------------------------------------------------------------- CPU benchmarks

fn bench_cpu_tokenizer_long_prompt(tk: &Tokenizer, cfg: &Config) -> BenchResult {
    let (text, source) = load_long_prompt();
    let mut last_toks = 0usize;
    for _ in 0..cfg.warmup {
        let ids = tk.encode(black_box(&text), false);
        last_toks = black_box(ids.len());
    }
    let mut samples = Vec::with_capacity(cfg.iters);
    for _ in 0..cfg.iters {
        let t0 = Instant::now();
        let ids = tk.encode(black_box(&text), false);
        let us = t0.elapsed().as_secs_f64() * 1e6;
        last_toks = black_box(ids.len());
        samples.push(us);
    }
    let (median, p10, p90, iqr) = summarize(&samples);
    let tok_per_s = if median > 0.0 {
        (last_toks as f64) / (median * 1e-6)
    } else {
        0.0
    };
    let mut extra = BTreeMap::new();
    extra.insert("tokens".into(), JsonVal::Int(last_toks as u64));
    extra.insert("bytes".into(), JsonVal::Int(text.len() as u64));
    extra.insert("tok_per_s".into(), JsonVal::Num(tok_per_s));
    extra.insert("source".into(), JsonVal::Str(source.into()));
    let note = format!("{last_toks} tok, {:.0} tok/s ({source})", tok_per_s);
    BenchResult {
        name: "cpu_tokenizer_long_prompt",
        metric: "us",
        lower_is_better: true,
        median,
        p10,
        p90,
        iqr,
        samples,
        extra,
        note,
    }
}

fn bench_cpu_tokenizer_repeat_4k(tk: &Tokenizer, cfg: &Config) -> BenchResult {
    let piece = "a".repeat(4096);
    let mut last_toks = 0usize;
    for _ in 0..cfg.warmup {
        let ids = tk.encode(black_box(&piece), false);
        last_toks = black_box(ids.len());
    }
    let mut samples = Vec::with_capacity(cfg.iters);
    for _ in 0..cfg.iters {
        let t0 = Instant::now();
        let ids = tk.encode(black_box(&piece), false);
        let us = t0.elapsed().as_secs_f64() * 1e6;
        last_toks = black_box(ids.len());
        samples.push(us);
    }
    let (median, p10, p90, iqr) = summarize(&samples);
    let mut extra = BTreeMap::new();
    extra.insert("chars".into(), JsonVal::Int(piece.len() as u64));
    extra.insert("tokens".into(), JsonVal::Int(last_toks as u64));
    let note = format!("4096 chars -> {last_toks} tok");
    BenchResult {
        name: "cpu_tokenizer_repeat_4k",
        metric: "us",
        lower_is_better: true,
        median,
        p10,
        p90,
        iqr,
        samples,
        extra,
        note,
    }
}

fn bench_cpu_sampler_200k(cfg: &Config) -> BenchResult {
    const N_VOCAB: usize = 200_000;
    let mut rng = Rng::new(0x5A17_2000_0001);
    let mut logits = Vec::with_capacity(N_VOCAB);
    for i in 0..N_VOCAB {
        let base = (rng.next_f32_unit() * 2.0 - 1.0) * 6.0;
        let spike = if i % 3119 == 0 { 8.0 } else { 0.0 };
        logits.push(base + spike);
    }
    let mut sampler = Sampler::new(1.0, 64, 0.95, 30.0, 0xCAFE_BABE_1234);
    let mut acc: u64 = 0;
    for _ in 0..cfg.warmup {
        acc = acc.wrapping_add(black_box(sampler.sample(black_box(&logits))) as u64);
    }
    let mut samples = Vec::with_capacity(cfg.iters);
    for _ in 0..cfg.iters {
        let t0 = Instant::now();
        let tok = sampler.sample(black_box(&logits));
        let us = t0.elapsed().as_secs_f64() * 1e6;
        acc = acc.wrapping_add(black_box(tok) as u64);
        samples.push(us);
    }
    black_box(acc);
    let (median, p10, p90, iqr) = summarize(&samples);
    let mut extra = BTreeMap::new();
    extra.insert("vocab".into(), JsonVal::Int(N_VOCAB as u64));
    extra.insert("top_k".into(), JsonVal::Int(64));
    extra.insert("top_p".into(), JsonVal::Num(0.95));
    let note = format!("vocab={N_VOCAB}, top_k=64, top_p=0.95");
    BenchResult {
        name: "cpu_sampler_200k",
        metric: "us",
        lower_is_better: true,
        median,
        p10,
        p90,
        iqr,
        samples,
        extra,
        note,
    }
}

fn bench_cpu_tier_rows_rec_64(cfg: &Config) -> BenchResult {
    const CAP: usize = 64;
    let total_bytes = CAP * REC_BYTES;
    let mut src = vec![0u8; total_bytes];
    for (idx, b) in src.iter_mut().enumerate() {
        *b = ((idx * 37 + 13) & 0xFF) as u8;
    }
    let mut dst = vec![0u8; total_bytes];
    let mut rec = vec![0u8; REC_BYTES];

    // One-time correctness sanity check before timing.
    for idx in 0..CAP {
        rows_to_rec(&src, 0, CAP, idx, &mut rec);
        rec_to_rows(&rec, &mut dst, 0, CAP, idx);
    }
    assert_eq!(src, dst, "tier rows_to_rec/rec_to_rows round-trip mismatch");

    for _ in 0..cfg.warmup {
        for idx in 0..CAP {
            rows_to_rec(black_box(&src), 0, CAP, idx, black_box(&mut rec));
            rec_to_rows(black_box(&rec), black_box(&mut dst), 0, CAP, idx);
        }
    }
    let mut samples = Vec::with_capacity(cfg.iters);
    for _ in 0..cfg.iters {
        let t0 = Instant::now();
        for idx in 0..CAP {
            rows_to_rec(black_box(&src), 0, CAP, idx, black_box(&mut rec));
            rec_to_rows(black_box(&rec), black_box(&mut dst), 0, CAP, idx);
        }
        let us = t0.elapsed().as_secs_f64() * 1e6;
        samples.push(us);
    }
    black_box(&dst);
    let (median, p10, p90, iqr) = summarize(&samples);
    // 64 records copied out to rec + 64 records copied back to dst = 2 * 64 * REC_BYTES payload bytes.
    let bytes_moved = 2 * CAP * REC_BYTES;
    let gib_per_s = if median > 0.0 {
        (bytes_moved as f64) / (median * 1e-6) / (1024.0_f64 * 1024.0 * 1024.0)
    } else {
        0.0
    };
    let mut extra = BTreeMap::new();
    extra.insert("positions".into(), JsonVal::Int(CAP as u64));
    extra.insert("bytes_moved".into(), JsonVal::Int(bytes_moved as u64));
    extra.insert("gib_per_s".into(), JsonVal::Num(gib_per_s));
    let note = format!("{gib_per_s:.2} GiB/s ({CAP} pos x {REC_BYTES} B round-trip)");
    BenchResult {
        name: "cpu_tier_rows_rec_64",
        metric: "us",
        lower_is_better: true,
        median,
        p10,
        p90,
        iqr,
        samples,
        extra,
        note,
    }
}

fn bench_cpu_gguf_from_bytes(cfg: &Config) -> BenchResult {
    let raw = build_synthetic_gguf_512kv_128t();
    let mut n_kv = 0usize;
    let mut n_t = 0usize;
    for _ in 0..cfg.warmup {
        let g = Gguf::from_bytes(black_box(&raw)).expect("valid synthetic gguf");
        n_kv = black_box(g.kv.len());
        n_t = black_box(g.tensors.len());
    }
    let mut samples = Vec::with_capacity(cfg.iters);
    for _ in 0..cfg.iters {
        let t0 = Instant::now();
        let g = Gguf::from_bytes(black_box(&raw)).expect("valid synthetic gguf");
        let us = t0.elapsed().as_secs_f64() * 1e6;
        n_kv = black_box(g.kv.len());
        n_t = black_box(g.tensors.len());
        samples.push(us);
    }
    let (median, p10, p90, iqr) = summarize(&samples);
    let mut extra = BTreeMap::new();
    extra.insert("kv_count".into(), JsonVal::Int(n_kv as u64));
    extra.insert("tensor_count".into(), JsonVal::Int(n_t as u64));
    extra.insert("file_bytes".into(), JsonVal::Int(raw.len() as u64));
    let note = format!("{n_kv} KV, {n_t} tensors ({:.1} KiB)", raw.len() as f64 / 1024.0);
    BenchResult {
        name: "cpu_gguf_from_bytes",
        metric: "us",
        lower_is_better: true,
        median,
        p10,
        p90,
        iqr,
        samples,
        extra,
        note,
    }
}

// ----------------------------------------------------------------------------- Metal GPU benchmarks

const MICRO_CHAIN_KERNEL: &str = r#"
kernel void k_micro_chain(device const uchar * PB [[buffer(1)]],
                          device uint * A [[buffer(2)]],
                          constant uint & step [[buffer(5)]],
                          uint tid [[thread_position_in_grid]]) {
    if (tid == 0) {
        A[0] = A[0] + 1u + (uint(PB[step & 15u]) & 1u);
    }
}
"#;

fn init_gpu() -> Gpu {
    let src = format!(
        "{}\n{}\n{}",
        include_str!("../src/kernels.metal"),
        include_str!("../src/prefill.metal"),
        MICRO_CHAIN_KERNEL
    );
    Gpu::new(&src, &["k_mix", "k_mix_b_4_3", "k_micro_chain"])
}

fn fill_f32_bytes(n_floats: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(n_floats * 4);
    for _ in 0..n_floats {
        let v = (rng.next_f32_unit() * 2.0 - 1.0) * 0.5;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn build_mix_pb(gpu: &Gpu) -> inkling::metal::Buf {
    // 3 query vectors (3 * E f32) + 3 norm weights (3 * E f32) = 6 * E f32
    let mut pb = fill_f32_bytes(3 * E, 0x1111_2222);
    for _ in 0..(3 * E) {
        pb.extend_from_slice(&1.0f32.to_le_bytes());
    }
    gpu.buffer_with(&pb)
}

fn make_decode_mix_job() -> MixJob {
    let norm_base = (3 * E * 4) as i32;
    let norm_step = (E * 4) as i32;
    MixJob {
        n_s: 4,
        n_q: 3,
        s_off: [0, E as u32, (2 * E) as u32, (3 * E) as u32, 0, 0],
        q_off: 0,
        out_off: [(4 * E) as u32, (5 * E) as u32, (6 * E) as u32, 0],
        norm_off: [norm_base, norm_base + norm_step, norm_base + 2 * norm_step, -1],
        eps: 1e-6,
        tstride: 0,
        ncopy: [[-1; 3]; 4],
        ntok: 1,
        hcopy: [[-1; 3]; 4],
        hout: [-1; 4],
        hstride: (2 * E) as u32,
        nof32: 0,
    }
}

fn make_prefill_mix_job(t: usize) -> (MixJob, usize) {
    let norm_base = (3 * E * 4) as i32;
    let norm_step = (E * 4) as i32;
    let te = t * E;
    let half_te = te / 2; // f32 words occupied by [T][E] fp16
    let h0 = (4 * te) as i32;
    let h1 = (4 * te + half_te) as i32;
    let h2 = (4 * te + 2 * half_te) as i32;
    let total_f32 = 4 * te + 3 * half_te;
    let job = MixJob {
        n_s: 4,
        n_q: 3,
        s_off: [0, te as u32, (2 * te) as u32, (3 * te) as u32, 0, 0],
        q_off: 0,
        out_off: [0; 4],
        norm_off: [norm_base, norm_base + norm_step, norm_base + 2 * norm_step, -1],
        eps: 1e-6,
        tstride: E as u32,
        ncopy: [[-1; 3]; 4],
        ntok: t as u32,
        hcopy: [[-1; 3]; 4],
        hout: [h0, h1, h2, -1],
        hstride: (2 * E) as u32,
        nof32: 0xF,
    };
    (job, total_f32)
}

struct MixRunSpec<'a> {
    pb: &'a inkling::metal::Buf,
    buf_a: &'a inkling::metal::Buf,
    jobs: &'a [MixJob],
    t: usize,
    old_mixb: bool,
    reps: usize,
}

fn measure_mix_path(gpu: &Gpu, spec: &MixRunSpec<'_>, cfg: &Config) -> (Vec<f64>, Vec<f64>) {
    let run_once = || -> (f64, f64) {
        let wall_t0 = Instant::now();
        let cb = gpu.cmd();
        let e = Enc::new(&cb);
        e.buf(spec.pb, 0, 1);
        e.buf(spec.buf_a, 0, 2);
        for _ in 0..spec.reps {
            dispatch_mix_jobs(gpu, &e, spec.jobs, spec.t, spec.old_mixb);
        }
        e.end();
        cb.commit();
        cb.waitUntilCompleted();
        let wall_us = wall_t0.elapsed().as_secs_f64() * 1e6 / spec.reps as f64;
        let gpu_s = cb.GPUEndTime() - cb.GPUStartTime();
        let gpu_us = if gpu_s > 0.0 {
            gpu_s * 1e6 / spec.reps as f64
        } else {
            wall_us
        };
        (gpu_us, wall_us)
    };

    for _ in 0..cfg.warmup {
        let _ = run_once();
    }
    let mut gpu_samples = Vec::with_capacity(cfg.iters);
    let mut wall_samples = Vec::with_capacity(cfg.iters);
    for _ in 0..cfg.iters {
        let (g, w) = run_once();
        gpu_samples.push(g);
        wall_samples.push(w);
    }
    (gpu_samples, wall_samples)
}

fn bench_gpu_mix_t1(gpu: &Gpu, cfg: &Config, old_mixb: bool) -> BenchResult {
    const REPS: usize = 64;
    let pb = build_mix_pb(gpu);
    let a_bytes = fill_f32_bytes(7 * E, 0x3333_4444);
    let buf_a = gpu.buffer_with(&a_bytes);
    let jobs = [make_decode_mix_job()];

    let spec = MixRunSpec {
        pb: &pb,
        buf_a: &buf_a,
        jobs: &jobs,
        t: 1,
        old_mixb,
        reps: REPS,
    };
    let (gpu_samples, wall_samples) = measure_mix_path(gpu, &spec, cfg);
    let (median, p10, p90, iqr) = summarize(&gpu_samples);
    let (w_med, w_p10, w_p90, w_iqr) = summarize(&wall_samples);

    let mut extra = BTreeMap::new();
    extra.insert("timer".into(), JsonVal::Str("gpu".into()));
    extra.insert("t".into(), JsonVal::Int(1));
    extra.insert("reps".into(), JsonVal::Int(REPS as u64));
    extra.insert("old_mixb".into(), JsonVal::Bool(old_mixb));
    extra.insert("wall_median_us".into(), JsonVal::Num(w_med));
    extra.insert("wall_p10_us".into(), JsonVal::Num(w_p10));
    extra.insert("wall_p90_us".into(), JsonVal::Num(w_p90));
    extra.insert("wall_iqr_us".into(), JsonVal::Num(w_iqr));

    let note = if !old_mixb {
        // Directly compare against old_mixb=true (k_mix) on the same buffers.
        let kmix_spec = MixRunSpec { old_mixb: true, ..spec };
        let (kmix_gpu, kmix_wall) = measure_mix_path(gpu, &kmix_spec, cfg);
        let (km_med, km_p10, km_p90, km_iqr) = summarize(&kmix_gpu);
        let (km_wmed, _, _, _) = summarize(&kmix_wall);
        let ratio = if km_med > 0.0 { median / km_med } else { 0.0 };
        extra.insert("kmix_median_us".into(), JsonVal::Num(km_med));
        extra.insert("kmix_p10_us".into(), JsonVal::Num(km_p10));
        extra.insert("kmix_p90_us".into(), JsonVal::Num(km_p90));
        extra.insert("kmix_iqr_us".into(), JsonVal::Num(km_iqr));
        extra.insert("kmix_wall_median_us".into(), JsonVal::Num(km_wmed));
        extra.insert("ratio_mixb_over_kmix".into(), JsonVal::Num(ratio));
        format!(
            "dispatch(T=1) (wall={w_med:.2} us) vs k_mix={km_med:.2} us ({ratio:.2}x)"
        )
    } else {
        format!("k_mix T=1 (wall={w_med:.2} us, reps={REPS})")
    };

    BenchResult {
        name: if old_mixb {
            "gpu_mix_t1_decode_kmix"
        } else {
            "gpu_mix_t1_decode"
        },
        metric: "us",
        lower_is_better: true,
        median,
        p10,
        p90,
        iqr,
        samples: gpu_samples,
        extra,
        note,
    }
}

fn bench_gpu_mix_prefill(
    gpu: &Gpu,
    cfg: &Config,
    name: &'static str,
    t: usize,
    reps: usize,
) -> BenchResult {
    let pb = build_mix_pb(gpu);
    let (job, total_f32) = make_prefill_mix_job(t);
    let a_bytes = fill_f32_bytes(total_f32, 0x5555_6666 ^ t as u64);
    let buf_a = gpu.buffer_with(&a_bytes);
    let jobs = [job];

    let spec = MixRunSpec {
        pb: &pb,
        buf_a: &buf_a,
        jobs: &jobs,
        t,
        old_mixb: false,
        reps,
    };
    let (gpu_samples, wall_samples) = measure_mix_path(gpu, &spec, cfg);
    let (median, p10, p90, iqr) = summarize(&gpu_samples);
    let (w_med, w_p10, w_p90, w_iqr) = summarize(&wall_samples);

    let mut extra = BTreeMap::new();
    extra.insert("timer".into(), JsonVal::Str("gpu".into()));
    extra.insert("t".into(), JsonVal::Int(t as u64));
    extra.insert("reps".into(), JsonVal::Int(reps as u64));
    extra.insert("nof32".into(), JsonVal::Int(0xF));
    extra.insert("wall_median_us".into(), JsonVal::Num(w_med));
    extra.insert("wall_p10_us".into(), JsonVal::Num(w_p10));
    extra.insert("wall_p90_us".into(), JsonVal::Num(w_p90));
    extra.insert("wall_iqr_us".into(), JsonVal::Num(w_iqr));

    BenchResult {
        name,
        metric: "us",
        lower_is_better: true,
        median,
        p10,
        p90,
        iqr,
        samples: gpu_samples,
        extra,
        note: format!("T={t} fp16 hout/nof32=0xF (wall={w_med:.2} us, reps={reps})"),
    }
}

fn bench_gpu_cmdbuf_530_dispatches(gpu: &Gpu, cfg: &Config) -> BenchResult {
    const N_DISPATCH: u32 = 530;
    let pso = gpu.pso("k_micro_chain");
    let pb = gpu.buffer_with(&[1u8; 64]);
    let buf_a = gpu.buffer_with(&0u32.to_le_bytes());

    let run_mode = |mode: u8| -> (f64, f64, f64) {
        let wall_t0 = Instant::now();
        let cb = match mode {
            0 => gpu.cmd(),
            1 => gpu.queue.commandBuffer().expect("retained cmdbuf"),
            _ => gpu
                .queue
                .commandBufferWithUnretainedReferences()
                .expect("unretained cmdbuf"),
        };
        let e = Enc::new(&cb);
        for step in 0..N_DISPATCH {
            e.pso(&pso);
            e.buf(&pb, 0, 1);
            e.buf(&buf_a, 0, 2);
            e.bytes(&step, 5);
            e.dispatch((1, 1, 1), 1);
            e.barrier();
        }
        e.end();
        let enc_us = wall_t0.elapsed().as_secs_f64() * 1e6;
        cb.commit();
        cb.waitUntilCompleted();
        let wall_us = wall_t0.elapsed().as_secs_f64() * 1e6;
        let gpu_s = cb.GPUEndTime() - cb.GPUStartTime();
        let gpu_us = if gpu_s > 0.0 { gpu_s * 1e6 } else { wall_us };
        (wall_us, gpu_us, enc_us)
    };

    // Sanity check that dispatches actually ran and updated A[0].
    let _ = run_mode(0);
    // SAFETY: `buf_a` is a StorageModeShared buffer of at least 16 bytes and `waitUntilCompleted`
    // has finished so the GPU is no longer writing to it.
    let observed = unsafe { *(buf_a.contents().as_ptr() as *const u32) };
    assert!(observed >= N_DISPATCH, "k_micro_chain did not execute (A[0]={observed})");

    let collect = |mode: u8| -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        for _ in 0..cfg.warmup {
            let _ = run_mode(mode);
        }
        let mut wall = Vec::with_capacity(cfg.iters);
        let mut gpu_t = Vec::with_capacity(cfg.iters);
        let mut enc = Vec::with_capacity(cfg.iters);
        for _ in 0..cfg.iters {
            let (w, g, en) = run_mode(mode);
            wall.push(w);
            gpu_t.push(g);
            enc.push(en);
        }
        (wall, gpu_t, enc)
    };

    let (wall_samples, gpu_samples, enc_samples) = collect(0);
    let (ret_wall, ret_gpu, ret_enc) = collect(1);
    let (unret_wall, unret_gpu, unret_enc) = collect(2);

    let (median, p10, p90, iqr) = summarize(&wall_samples);
    let (gpu_med, _, _, _) = summarize(&gpu_samples);
    let (enc_med, _, _, _) = summarize(&enc_samples);
    let (ret_w_med, _, _, _) = summarize(&ret_wall);
    let (ret_g_med, _, _, _) = summarize(&ret_gpu);
    let (ret_e_med, _, _, _) = summarize(&ret_enc);
    let (unret_w_med, _, _, _) = summarize(&unret_wall);
    let (unret_g_med, _, _, _) = summarize(&unret_gpu);
    let (unret_e_med, _, _, _) = summarize(&unret_enc);

    let mut extra = BTreeMap::new();
    extra.insert("timer".into(), JsonVal::Str("wall".into()));
    extra.insert("dispatches".into(), JsonVal::Int(N_DISPATCH as u64));
    extra.insert("gpu_median_us".into(), JsonVal::Num(gpu_med));
    extra.insert("encode_median_us".into(), JsonVal::Num(enc_med));
    extra.insert("retained_wall_median_us".into(), JsonVal::Num(ret_w_med));
    extra.insert("retained_gpu_median_us".into(), JsonVal::Num(ret_g_med));
    extra.insert("retained_encode_median_us".into(), JsonVal::Num(ret_e_med));
    extra.insert("unretained_wall_median_us".into(), JsonVal::Num(unret_w_med));
    extra.insert("unretained_gpu_median_us".into(), JsonVal::Num(unret_g_med));
    extra.insert("unretained_encode_median_us".into(), JsonVal::Num(unret_e_med));
    extra.insert(
        "ink_unretained_cmd_env".into(),
        JsonVal::Bool(std::env::var("INK_UNRETAINED_CMD").is_ok()),
    );

    BenchResult {
        name: "gpu_cmdbuf_530_dispatches",
        metric: "us",
        lower_is_better: true,
        median,
        p10,
        p90,
        iqr,
        samples: wall_samples,
        extra,
        note: format!(
            "enc={enc_med:.1} gpu={gpu_med:.1} us | retained={ret_w_med:.1} vs unretained={unret_w_med:.1} us"
        ),
    }
}

/// Brief deterministic CPU spin warmup so Apple Silicon P-core DVFS frequency
/// settles before the first CPU microbenchmark runs.
fn cpu_dvfs_warmup(duration: std::time::Duration) {
    let t0 = Instant::now();
    let mut acc: u64 = 0x9E37_79B9_7F4A_7C15;
    while t0.elapsed() < duration {
        for _ in 0..1024 {
            acc = acc
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            acc ^= acc >> 17;
        }
        black_box(acc);
    }
    black_box(acc);
}

fn main() {
    let cfg = match parse_args() {
        Ok(Some(c)) => c,
        Ok(None) => return,
        Err(msg) => {
            eprintln!("error: {msg}");
            std::process::exit(2);
        }
    };

    let need_tok = cfg
        .selected
        .iter()
        .any(|&n| n == "cpu_tokenizer_long_prompt" || n == "cpu_tokenizer_repeat_4k");
    let need_cpu = cfg.selected.iter().any(|&n| n.starts_with("cpu_"));
    let need_gpu = cfg.selected.iter().any(|&n| n.starts_with("gpu_"));

    let tk = if need_tok {
        Some(build_synthetic_tokenizer())
    } else {
        None
    };
    let gpu = if need_gpu { Some(init_gpu()) } else { None };

    if need_cpu && cfg.iters > 3 {
        cpu_dvfs_warmup(std::time::Duration::from_millis(25));
    }

    let mut results = Vec::with_capacity(cfg.selected.len());
    for &name in &cfg.selected {
        let r = match name {
            "cpu_tokenizer_long_prompt" => {
                bench_cpu_tokenizer_long_prompt(tk.as_ref().unwrap(), &cfg)
            }
            "cpu_tokenizer_repeat_4k" => bench_cpu_tokenizer_repeat_4k(tk.as_ref().unwrap(), &cfg),
            "cpu_sampler_200k" => bench_cpu_sampler_200k(&cfg),
            "cpu_tier_rows_rec_64" => bench_cpu_tier_rows_rec_64(&cfg),
            "cpu_gguf_from_bytes" => bench_cpu_gguf_from_bytes(&cfg),
            "gpu_mix_t1_decode" => bench_gpu_mix_t1(gpu.as_ref().unwrap(), &cfg, false),
            "gpu_mix_t1_decode_kmix" => bench_gpu_mix_t1(gpu.as_ref().unwrap(), &cfg, true),
            "gpu_mix_t64_prefill" => {
                bench_gpu_mix_prefill(gpu.as_ref().unwrap(), &cfg, "gpu_mix_t64_prefill", 64, 4)
            }
            "gpu_mix_t512_prefill" => {
                bench_gpu_mix_prefill(gpu.as_ref().unwrap(), &cfg, "gpu_mix_t512_prefill", 512, 1)
            }
            "gpu_cmdbuf_530_dispatches" => {
                bench_gpu_cmdbuf_530_dispatches(gpu.as_ref().unwrap(), &cfg)
            }
            _ => unreachable!(),
        };
        results.push(r);
    }

    if cfg.json {
        emit_json(&results);
    } else {
        emit_table(&results, &cfg);
    }
}
