//! Smoke and unit tests for the model-free microbenchmark suite (`examples/micro_bench.rs`)
//! and paired interleaved A/B runner (`tools/perf_ab.py`).

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn perf_ab_path() -> PathBuf {
    manifest_dir().join("tools/perf_ab.py")
}

fn micro_bench_bin() -> &'static PathBuf {
    static BIN: OnceLock<PathBuf> = OnceLock::new();
    BIN.get_or_init(|| {
        let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
        let status = Command::new(&cargo)
            .current_dir(manifest_dir())
            .args(["build", "--offline", "--example", "micro_bench"])
            .status()
            .expect("invoke cargo build --example micro_bench");
        assert!(
            status.success(),
            "examples/micro_bench failed to compile (status: {status})"
        );
        let target_dir = std::env::var("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| manifest_dir().join("target"));
        let bin = target_dir.join("debug/examples/micro_bench");
        assert!(
            bin.exists(),
            "expected compiled micro_bench binary at {}",
            bin.display()
        );
        bin
    })
}

fn run_python(script: &str, args: &[&str]) -> std::process::Output {
    Command::new("python3")
        .arg("-c")
        .arg(script)
        .args(args)
        .output()
        .expect("run python3")
}

#[test]
fn micro_bench_cpu_sampler_json_smoke() {
    let bin = micro_bench_bin();
    let out = Command::new(bin)
        .args(["--only", "cpu_sampler_200k", "--iters", "3", "--warmup", "1", "--json"])
        .output()
        .expect("run micro_bench");
    assert!(
        out.status.success(),
        "micro_bench failed: stdout={} stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).expect("valid utf-8 json");
    let check = run_python(
        r#"
import json, math, sys
data = json.loads(sys.argv[1])
assert data["schema"] == 1, data
assert isinstance(data["host"]["os"], str) and data["host"]["os"]
assert isinstance(data["host"]["arch"], str) and data["host"]["arch"]
benches = data["benchmarks"]
assert list(benches.keys()) == ["cpu_sampler_200k"], benches.keys()
b = benches["cpu_sampler_200k"]
assert b["metric"] == "us", b
assert b["lower_is_better"] is True, b
for k in ("median", "p10", "p90", "iqr"):
    assert isinstance(b[k], (int, float)) and math.isfinite(b[k]), (k, b[k])
assert 0.0 < b["p10"] <= b["median"] <= b["p90"], b
assert b["iqr"] >= 0.0, b
assert isinstance(b["samples"], list) and len(b["samples"]) == 3, b["samples"]
assert all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0.0 for x in b["samples"])
assert isinstance(b["extra"], dict), b["extra"]
"#,
        &[&stdout],
    );
    assert!(
        check.status.success(),
        "JSON schema validation failed:\nstdout={stdout}\nstderr={}",
        String::from_utf8_lossy(&check.stderr)
    );
}

#[test]
fn micro_bench_all_cpu_and_table_mode() {
    let bin = micro_bench_bin();
    let out = Command::new(bin)
        .args(["--only", "cpu", "--iters", "2", "--warmup", "1", "--json"])
        .output()
        .expect("run micro_bench --only cpu");
    assert!(
        out.status.success(),
        "micro_bench --only cpu failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).expect("valid utf-8 json");
    let check = run_python(
        r#"
import json, math, sys
data = json.loads(sys.argv[1])
benches = data["benchmarks"]
expected = {
    "cpu_tokenizer_long_prompt",
    "cpu_tokenizer_repeat_4k",
    "cpu_sampler_200k",
    "cpu_tier_rows_rec_64",
    "cpu_gguf_from_bytes",
}
assert set(benches.keys()) == expected, set(benches.keys())
for name, b in benches.items():
    assert b["metric"] == "us", (name, b)
    assert b["lower_is_better"] is True, (name, b)
    assert len(b["samples"]) == 2, (name, b)
    assert math.isfinite(b["median"]) and b["median"] > 0.0, (name, b)
assert benches["cpu_tokenizer_long_prompt"]["extra"]["tok_per_s"] > 0.0
assert benches["cpu_tier_rows_rec_64"]["extra"]["gib_per_s"] > 0.0
"#,
        &[&stdout],
    );
    assert!(
        check.status.success(),
        "CPU suite schema check failed: {}",
        String::from_utf8_lossy(&check.stderr)
    );

    // Table mode (without --json)
    let table = Command::new(bin)
        .args(["--only", "cpu_sampler_200k", "--iters", "2", "--warmup", "1"])
        .output()
        .expect("run micro_bench table mode");
    assert!(table.status.success());
    let table_out = String::from_utf8_lossy(&table.stdout);
    assert!(
        table_out.contains("cpu_sampler_200k") && table_out.contains("us"),
        "unexpected table output: {table_out}"
    );

    // Unknown filter -> exit code 2
    let bad = Command::new(bin)
        .args(["--only", "nonexistent_bench_xyz"])
        .output()
        .expect("run micro_bench with bad filter");
    assert_eq!(bad.status.code(), Some(2));
}

#[test]
fn perf_ab_help_and_unit_tests() {
    let script = perf_ab_path();
    assert!(script.exists(), "missing {}", script.display());
    let mode = std::fs::metadata(&script).unwrap().permissions().mode();
    assert!(
        mode & 0o111 != 0,
        "tools/perf_ab.py must be executable (mode={mode:#o})"
    );

    let help = Command::new("python3")
        .arg(&script)
        .arg("--help")
        .output()
        .expect("run perf_ab.py --help");
    assert!(
        help.status.success(),
        "perf_ab.py --help failed: {}",
        String::from_utf8_lossy(&help.stderr)
    );

    let unit_test = run_python(
        r#"
import importlib.util, math, sys

spec = importlib.util.spec_from_file_location("perf_ab", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# 1. ABBA schedule check
sched = mod.build_abba_schedule(5)
assert sched == [
    ("base", "head"),
    ("head", "base"),
    ("base", "head"),
    ("head", "base"),
    ("base", "head"),
], sched

# 2. Model bench stdout parser (standard + --kv-disk)
sample_std = """
prompt  1024 tok:    412.5 ms   2482.4 tok/s | decode  128 tok:   1980.4 ms   64.63 tok/s
memory besides the weights: K/V cache 185 MiB at 1152 tokens | prefill scratch 210 MiB while prefilling, 12.5 MiB while generating
"""
parsed = mod.parse_model_bench_output(sample_std)
assert math.isclose(parsed["prefill_ms"]["value"], 412.5)
assert parsed["prefill_ms"]["lower_is_better"] is True
assert math.isclose(parsed["prefill_tok_s"]["value"], 2482.4)
assert parsed["prefill_tok_s"]["lower_is_better"] is False
assert math.isclose(parsed["decode_ms"]["value"], 1980.4)
assert parsed["decode_ms"]["lower_is_better"] is True
assert math.isclose(parsed["decode_tok_s"]["value"], 64.63)
assert parsed["decode_tok_s"]["lower_is_better"] is False
assert math.isclose(parsed["kv_mem_mib"]["value"], 185.0)
assert math.isclose(parsed["prefill_scratch_mib"]["value"], 210.0)
assert math.isclose(parsed["gen_scratch_mib"]["value"], 12.5)

sample_disk = """
prompt  2048 tok:    850.0 ms   2409.4 tok/s | decode  128 tok:   3200.0 ms   40.00 tok/s
memory besides the weights: K/V cache 140 MiB in memory + 220 MiB on disk (all in memory: 360 MiB) at 2176 tokens | prefill scratch 310 MiB while prefilling, 16.0 MiB while generating
K/V file: 180 MiB read per token, 18.5 GiB/s, 9.5 ms of each token's 25.0 ms
"""
parsed_d = mod.parse_model_bench_output(sample_disk)
assert math.isclose(parsed_d["kv_mem_mib"]["value"], 140.0)
assert math.isclose(parsed_d["kv_disk_mib"]["value"], 220.0)
assert math.isclose(parsed_d["kv_file_gib_s"]["value"], 18.5)
assert parsed_d["kv_file_gib_s"]["lower_is_better"] is False
assert math.isclose(parsed_d["kv_file_io_ms_per_tok"]["value"], 9.5)
assert parsed_d["kv_file_io_ms_per_tok"]["lower_is_better"] is True

# 3. Paired stats: lower_is_better=True (latency)
res_same = mod.analyze_paired([100.0]*5, [100.0]*5, lower_is_better=True, max_regression_pct=2.0, seed=42)
assert res_same["verdict"] == "PASS", res_same
assert math.isclose(res_same["median_delta_pct"], 0.0)

res_reg = mod.analyze_paired([100.0, 101.0, 99.0, 100.5, 99.5], [110.0, 111.1, 108.9, 110.55, 109.45],
                             lower_is_better=True, max_regression_pct=2.0, seed=42)
assert res_reg["verdict"] == "FAIL", res_reg
assert res_reg["ci_low_pct"] > 2.0, res_reg

res_imp = mod.analyze_paired([100.0]*5, [90.0]*5, lower_is_better=True, max_regression_pct=2.0, seed=42)
assert res_imp["verdict"] in ("PASS", "IMPROVED"), res_imp

# Noisy case straddling threshold -> not FAIL
res_noisy = mod.analyze_paired([100.0]*5, [103.0, 102.5, 102.2, 99.0, 98.5],
                               lower_is_better=True, max_regression_pct=2.0, seed=42)
assert res_noisy["verdict"] != "FAIL", res_noisy

# Reproducibility with fixed seed
r1 = mod.analyze_paired([100, 102, 98, 101, 99], [103, 101, 104, 102, 100], True, 2.0, seed=1337)
r2 = mod.analyze_paired([100, 102, 98, 101, 99], [103, 101, 104, 102, 100], True, 2.0, seed=1337)
assert r1["ci_low_pct"] == r2["ci_low_pct"] and r1["ci_high_pct"] == r2["ci_high_pct"]

# 4. Paired stats: lower_is_better=False (throughput tok/s)
res_tps_reg = mod.analyze_paired([100.0]*5, [90.0]*5, lower_is_better=False, max_regression_pct=2.0, seed=42)
assert res_tps_reg["verdict"] == "FAIL", res_tps_reg
res_tps_imp = mod.analyze_paired([100.0]*5, [110.0]*5, lower_is_better=False, max_regression_pct=2.0, seed=42)
assert res_tps_imp["verdict"] in ("PASS", "IMPROVED"), res_tps_imp
"#,
        &[script.to_str().unwrap()],
    );
    assert!(
        unit_test.status.success(),
        "perf_ab.py unit tests failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&unit_test.stdout),
        String::from_utf8_lossy(&unit_test.stderr)
    );
}

#[test]
fn perf_ab_micro_end_to_end_and_baseline() {
    let bin = micro_bench_bin();
    let script = perf_ab_path();
    let dir = tempfile::tempdir().expect("tempdir");
    let baseline = dir.path().join("baseline.json");

    let run = Command::new("python3")
        .arg(&script)
        .args([
            "micro",
            "--base-bin",
            bin.to_str().unwrap(),
            "--head-bin",
            bin.to_str().unwrap(),
            "--only",
            "cpu_sampler_200k",
            "--iters",
            "2",
            "--warmup",
            "1",
            "--rounds",
            "2",
            "--cooldown-s",
            "0",
            "--max-regression-pct",
            "50.0",
            "--save-baseline",
            baseline.to_str().unwrap(),
        ])
        .output()
        .expect("run perf_ab.py micro");
    assert_eq!(
        run.status.code(),
        Some(0),
        "perf_ab.py micro failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(baseline.exists(), "baseline JSON was not written");

    let cmp = Command::new("python3")
        .arg(&script)
        .args([
            "micro",
            "--head-bin",
            bin.to_str().unwrap(),
            "--compare-baseline",
            baseline.to_str().unwrap(),
            "--only",
            "cpu_sampler_200k",
            "--iters",
            "2",
            "--warmup",
            "1",
            "--rounds",
            "2",
            "--cooldown-s",
            "0",
            "--max-regression-pct",
            "50.0",
        ])
        .output()
        .expect("run perf_ab.py micro --compare-baseline");
    assert_eq!(
        cmp.status.code(),
        Some(0),
        "perf_ab.py micro --compare-baseline failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&cmp.stdout),
        String::from_utf8_lossy(&cmp.stderr)
    );
}

fn write_mock_script(dir: &Path, name: &str, body: &str) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, body).expect("write mock script");
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).expect("chmod mock");
    p
}

#[test]
fn perf_ab_model_mode_pass_fail_and_usage_error() {
    let script = perf_ab_path();
    let dir = tempfile::tempdir().expect("tempdir");
    let fake_model = dir.path().join("fake.gguf");
    std::fs::write(&fake_model, b"GGUF").unwrap();
    let fake_prompt = dir.path().join("prompt.txt");
    std::fs::write(&fake_prompt, "hello world").unwrap();

    let base_mock = write_mock_script(
        dir.path(),
        "inkling_base.sh",
        "#!/bin/sh\necho 'prompt  1024 tok:    500.0 ms   2048.0 tok/s | decode  128 tok:   2000.0 ms   64.00 tok/s'\necho 'memory besides the weights: K/V cache 180 MiB at 1152 tokens | prefill scratch 200 MiB while prefilling, 12.0 MiB while generating'\n",
    );
    let head_ok = write_mock_script(
        dir.path(),
        "inkling_head_ok.sh",
        "#!/bin/sh\necho 'prompt  1024 tok:    498.0 ms   2056.2 tok/s | decode  128 tok:   1990.0 ms   64.32 tok/s'\necho 'memory besides the weights: K/V cache 180 MiB at 1152 tokens | prefill scratch 200 MiB while prefilling, 12.0 MiB while generating'\n",
    );
    let head_slow = write_mock_script(
        dir.path(),
        "inkling_head_slow.sh",
        "#!/bin/sh\necho 'prompt  1024 tok:    500.0 ms   2048.0 tok/s | decode  128 tok:   2200.0 ms   58.18 tok/s'\necho 'memory besides the weights: K/V cache 180 MiB at 1152 tokens | prefill scratch 200 MiB while prefilling, 12.0 MiB while generating'\n",
    );

    let pass = Command::new("python3")
        .arg(&script)
        .args([
            "model",
            "--base-bin",
            base_mock.to_str().unwrap(),
            "--head-bin",
            head_ok.to_str().unwrap(),
            "-m",
            fake_model.to_str().unwrap(),
            "-f",
            fake_prompt.to_str().unwrap(),
            "-n",
            "128",
            "--rounds",
            "4",
            "--cooldown-s",
            "0",
            "--max-regression-pct",
            "2.0",
        ])
        .output()
        .expect("run perf_ab.py model pass");
    assert_eq!(
        pass.status.code(),
        Some(0),
        "expected exit 0 on pass:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&pass.stdout),
        String::from_utf8_lossy(&pass.stderr)
    );

    let fail = Command::new("python3")
        .arg(&script)
        .args([
            "model",
            "--base-bin",
            base_mock.to_str().unwrap(),
            "--head-bin",
            head_slow.to_str().unwrap(),
            "-m",
            fake_model.to_str().unwrap(),
            "-f",
            fake_prompt.to_str().unwrap(),
            "-n",
            "128",
            "--rounds",
            "4",
            "--cooldown-s",
            "0",
            "--max-regression-pct",
            "2.0",
        ])
        .output()
        .expect("run perf_ab.py model fail");
    assert_eq!(
        fail.status.code(),
        Some(1),
        "expected exit 1 on 10% decode regression:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&fail.stdout),
        String::from_utf8_lossy(&fail.stderr)
    );

    // Usage error -> exit code 2
    let usage_err = Command::new("python3")
        .arg(&script)
        .args(["micro"])
        .output()
        .expect("run perf_ab.py micro with missing args");
    assert_eq!(usage_err.status.code(), Some(2));
}
