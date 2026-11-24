#!/usr/bin/env python3
"""Paired interleaved (ABBA) A/B performance runner and baseline comparator for inkling.

Standard library only (zero pip dependencies).

Modes:
  micro  Compare two compiled `micro_bench` binaries (no GGUF model required).
  model  Compare two compiled `inkling` release binaries on `inkling bench`.

Exit codes:
  0  Pass (no statistically significant regression beyond --max-regression-pct)
  1  Fail (at least one metric's 95% bootstrap CI is strictly worse than threshold)
  2  Usage, input, subprocess, or parsing error
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple


class UsageError(Exception):
    """Raised on invalid CLI arguments, missing files, or execution/parse errors."""


def median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("median requires at least one value")
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return 0.5 * (float(s[mid - 1]) + float(s[mid]))


def percentile(sorted_vals: Sequence[float], pct: float) -> float:
    if not sorted_vals:
        raise ValueError("percentile requires at least one value")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    frac = pos - lo
    return float(sorted_vals[lo]) * (1.0 - frac) + float(sorted_vals[hi]) * frac


def pct_delta(base: float, head: float) -> float:
    if base == 0.0:
        if head == 0.0:
            return 0.0
        return math.inf if head > 0.0 else -math.inf
    return ((head - base) / base) * 100.0


def build_abba_schedule(rounds: int) -> List[Tuple[str, str]]:
    """Return balanced ABBA pair orderings: AB, BA, AB, BA, ..."""
    if rounds < 1:
        raise UsageError("--rounds must be >= 1")
    return [("base", "head") if (r % 2 == 0) else ("head", "base") for r in range(rounds)]


def _classify_verdict(
    median_delta_pct: float,
    ci_low_pct: float,
    ci_high_pct: float,
    lower_is_better: bool,
    max_regression_pct: float,
) -> Tuple[str, float, float, float]:
    """Return (verdict, worse_med, worse_ci_low, worse_ci_high)."""
    if lower_is_better:
        worse_med = median_delta_pct
        worse_ci_low = ci_low_pct
        worse_ci_high = ci_high_pct
    else:
        worse_med = -median_delta_pct
        worse_ci_low = -ci_high_pct
        worse_ci_high = -ci_low_pct

    if worse_ci_low > max_regression_pct:
        verdict = "FAIL"
    elif worse_med > max_regression_pct:
        verdict = "WARN"
    elif worse_ci_high < -max_regression_pct:
        verdict = "IMPROVED"
    else:
        verdict = "PASS"
    return verdict, worse_med, worse_ci_low, worse_ci_high


def analyze_paired(
    base_vals: Sequence[float],
    head_vals: Sequence[float],
    lower_is_better: bool,
    max_regression_pct: float = 2.0,
    bootstrap_iters: int = 2000,
    seed: int = 1337,
) -> Dict[str, Any]:
    """Compute paired per-round percentage deltas and 95% bootstrap CI of the median delta."""
    if len(base_vals) != len(head_vals) or not base_vals:
        raise ValueError("base_vals and head_vals must have equal non-zero length")
    n = len(base_vals)
    deltas = [pct_delta(b, h) for b, h in zip(base_vals, head_vals)]
    med_base = median(base_vals)
    med_head = median(head_vals)
    med_delta = median(deltas)

    rng = random.Random(seed)
    boot_meds: List[float] = []
    for _ in range(bootstrap_iters):
        draw = [deltas[rng.randrange(n)] for _ in range(n)]
        boot_meds.append(median(draw))
    boot_meds.sort()
    ci_low = percentile(boot_meds, 2.5)
    ci_high = percentile(boot_meds, 97.5)

    verdict, worse_med, worse_ci_low, worse_ci_high = _classify_verdict(
        med_delta, ci_low, ci_high, lower_is_better, max_regression_pct
    )
    return {
        "paired": True,
        "rounds": n,
        "lower_is_better": lower_is_better,
        "median_base": med_base,
        "median_head": med_head,
        "paired_deltas_pct": deltas,
        "median_delta_pct": med_delta,
        "ci_low_pct": ci_low,
        "ci_high_pct": ci_high,
        "worse_median_pct": worse_med,
        "worse_ci_low_pct": worse_ci_low,
        "worse_ci_high_pct": worse_ci_high,
        "max_regression_pct": max_regression_pct,
        "verdict": verdict,
    }


def analyze_unpaired(
    base_vals: Sequence[float],
    head_vals: Sequence[float],
    lower_is_better: bool,
    max_regression_pct: float = 2.0,
    bootstrap_iters: int = 2000,
    seed: int = 1337,
) -> Dict[str, Any]:
    """Two-sample bootstrap comparison against a saved baseline."""
    if not base_vals or not head_vals:
        raise ValueError("base_vals and head_vals must be non-empty")
    nb = len(base_vals)
    nh = len(head_vals)
    med_base = median(base_vals)
    med_head = median(head_vals)
    med_delta = pct_delta(med_base, med_head)

    rng = random.Random(seed)
    boot_deltas: List[float] = []
    for _ in range(bootstrap_iters):
        b_draw = [base_vals[rng.randrange(nb)] for _ in range(nb)]
        h_draw = [head_vals[rng.randrange(nh)] for _ in range(nh)]
        boot_deltas.append(pct_delta(median(b_draw), median(h_draw)))
    boot_deltas.sort()
    ci_low = percentile(boot_deltas, 2.5)
    ci_high = percentile(boot_deltas, 97.5)

    verdict, worse_med, worse_ci_low, worse_ci_high = _classify_verdict(
        med_delta, ci_low, ci_high, lower_is_better, max_regression_pct
    )
    return {
        "paired": False,
        "rounds": nh,
        "lower_is_better": lower_is_better,
        "median_base": med_base,
        "median_head": med_head,
        "median_delta_pct": med_delta,
        "ci_low_pct": ci_low,
        "ci_high_pct": ci_high,
        "worse_median_pct": worse_med,
        "worse_ci_low_pct": worse_ci_low,
        "worse_ci_high_pct": worse_ci_high,
        "max_regression_pct": max_regression_pct,
        "verdict": verdict,
    }


# ----------------------------------------------------------------------------- output parsers

_RE_PROMPT_DECODE = re.compile(
    r"(?:prompt|prefill)\s+\d+\s+tok:\s*([0-9]+(?:\.[0-9]+)?)\s*ms\s+([0-9]+(?:\.[0-9]+)?)\s*tok/s"
    r".*?decode\s+\d+\s+tok:\s*([0-9]+(?:\.[0-9]+)?)\s*ms\s+([0-9]+(?:\.[0-9]+)?)\s*tok/s",
    re.DOTALL,
)
_RE_PROMPT_ONLY = re.compile(
    r"(?:prompt|prefill)\s+\d+\s+tok:\s*([0-9]+(?:\.[0-9]+)?)\s*ms\s+([0-9]+(?:\.[0-9]+)?)\s*tok/s"
)
_RE_DECODE_ONLY = re.compile(
    r"decode\s+\d+\s+tok:\s*([0-9]+(?:\.[0-9]+)?)\s*ms\s+([0-9]+(?:\.[0-9]+)?)\s*tok/s"
)
_RE_KV_MEM_DISK = re.compile(
    r"K/V cache\s+([0-9]+(?:\.[0-9]+)?)\s*MiB\s+in memory\s*\+\s*([0-9]+(?:\.[0-9]+)?)\s*MiB\s+on disk"
)
_RE_KV_MEM_ONLY = re.compile(r"K/V cache\s+([0-9]+(?:\.[0-9]+)?)\s*MiB")
_RE_SCRATCH = re.compile(
    r"prefill scratch\s+([0-9]+(?:\.[0-9]+)?)\s*MiB\s+while prefilling,\s*([0-9]+(?:\.[0-9]+)?)\s*MiB\s+while generating"
)
_RE_KV_FILE = re.compile(
    r"K/V file:\s*([0-9]+(?:\.[0-9]+)?)\s*MiB read per token,\s*([0-9]+(?:\.[0-9]+)?)\s*GiB/s,\s*([0-9]+(?:\.[0-9]+)?)\s*ms"
)


def parse_micro_bench_output(stdout: str) -> Dict[str, Dict[str, Any]]:
    """Parse JSON emitted by `micro_bench --json` into metric -> {value, unit, lower_is_better}."""
    text = stdout.strip()
    if not text:
        raise UsageError("micro_bench produced empty stdout")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise UsageError(f"failed to parse micro_bench JSON: {exc}\nstdout:\n{text[:400]}") from exc
    if not isinstance(data, dict) or data.get("schema") != 1 or not isinstance(data.get("benchmarks"), dict):
        raise UsageError(f"unexpected micro_bench JSON schema: {text[:400]}")
    metrics: Dict[str, Dict[str, Any]] = {}
    for name, entry in data["benchmarks"].items():
        if not isinstance(entry, dict) or "median" not in entry:
            raise UsageError(f"malformed benchmark entry '{name}' in micro_bench JSON")
        metrics[name] = {
            "value": float(entry["median"]),
            "unit": str(entry.get("metric", "us")),
            "lower_is_better": bool(entry.get("lower_is_better", True)),
        }
    if not metrics:
        raise UsageError("micro_bench JSON contained no benchmarks")
    return metrics


def parse_model_bench_output(stdout: str) -> Dict[str, Dict[str, Any]]:
    """Parse `inkling bench` stdout (either JSON or standard human output)."""
    stripped = stdout.strip()
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
            if isinstance(data, dict) and isinstance(data.get("benchmarks"), dict):
                return parse_micro_bench_output(stripped)
            if isinstance(data, dict) and isinstance(data.get("metrics"), dict):
                out: Dict[str, Dict[str, Any]] = {}
                for k, v in data["metrics"].items():
                    if isinstance(v, dict) and "value" in v:
                        out[k] = {
                            "value": float(v["value"]),
                            "unit": str(v.get("unit", "")),
                            "lower_is_better": bool(v.get("lower_is_better", True)),
                        }
                if out:
                    return out
        except json.JSONDecodeError:
            pass

    metrics: Dict[str, Dict[str, Any]] = {}
    m = _RE_PROMPT_DECODE.search(stdout)
    if m:
        metrics["prefill_ms"] = {"value": float(m.group(1)), "unit": "ms", "lower_is_better": True}
        metrics["prefill_tok_s"] = {"value": float(m.group(2)), "unit": "tok/s", "lower_is_better": False}
        metrics["decode_ms"] = {"value": float(m.group(3)), "unit": "ms", "lower_is_better": True}
        metrics["decode_tok_s"] = {"value": float(m.group(4)), "unit": "tok/s", "lower_is_better": False}
    else:
        mp = _RE_PROMPT_ONLY.search(stdout)
        if mp:
            metrics["prefill_ms"] = {"value": float(mp.group(1)), "unit": "ms", "lower_is_better": True}
            metrics["prefill_tok_s"] = {"value": float(mp.group(2)), "unit": "tok/s", "lower_is_better": False}
        md = _RE_DECODE_ONLY.search(stdout)
        if md:
            metrics["decode_ms"] = {"value": float(md.group(1)), "unit": "ms", "lower_is_better": True}
            metrics["decode_tok_s"] = {"value": float(md.group(2)), "unit": "tok/s", "lower_is_better": False}

    mkd = _RE_KV_MEM_DISK.search(stdout)
    if mkd:
        metrics["kv_mem_mib"] = {"value": float(mkd.group(1)), "unit": "MiB", "lower_is_better": True}
        metrics["kv_disk_mib"] = {"value": float(mkd.group(2)), "unit": "MiB", "lower_is_better": True}
    else:
        mkm = _RE_KV_MEM_ONLY.search(stdout)
        if mkm:
            metrics["kv_mem_mib"] = {"value": float(mkm.group(1)), "unit": "MiB", "lower_is_better": True}

    ms = _RE_SCRATCH.search(stdout)
    if ms:
        metrics["prefill_scratch_mib"] = {"value": float(ms.group(1)), "unit": "MiB", "lower_is_better": True}
        metrics["gen_scratch_mib"] = {"value": float(ms.group(2)), "unit": "MiB", "lower_is_better": True}

    mf = _RE_KV_FILE.search(stdout)
    if mf:
        metrics["kv_file_gib_s"] = {"value": float(mf.group(2)), "unit": "GiB/s", "lower_is_better": False}
        metrics["kv_file_io_ms_per_tok"] = {"value": float(mf.group(3)), "unit": "ms", "lower_is_better": True}

    if not metrics:
        raise UsageError(f"could not parse any metrics from model bench output:\n{stdout[:500]}")
    return metrics


# ----------------------------------------------------------------------------- metadata & subprocesses

def _cmd_stdout(cmd: Sequence[str]) -> Optional[str]:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
        if res.returncode == 0:
            return res.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return None


def collect_host_metadata() -> Dict[str, Any]:
    cpu_brand = _cmd_stdout(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor() or "unknown"
    git_sha = _cmd_stdout(["git", "rev-parse", "HEAD"]) or "unknown"
    git_status = _cmd_stdout(["git", "status", "--porcelain"])
    rustc_ver = _cmd_stdout(["rustc", "--version"]) or "unknown"
    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": git_sha,
        "git_dirty": bool(git_status) if git_status is not None else None,
        "host": {
            "os": platform.system(),
            "release": platform.release(),
            "arch": platform.machine(),
            "cpu_brand": cpu_brand,
        },
        "rustc_version": rustc_ver,
    }


def run_command_checked(cmd: Sequence[str]) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise UsageError(f"failed to execute {cmd[0]!r}: {exc}") from exc
    if proc.returncode != 0:
        raise UsageError(
            f"command exited with status {proc.returncode}: {' '.join(cmd)}\n"
            f"stderr:\n{proc.stderr.strip()}"
        )
    return proc.stdout


def validate_executable(path: str, label: str) -> str:
    if not os.path.exists(path):
        raise UsageError(f"{label} does not exist: {path}")
    if not os.access(path, os.X_OK):
        raise UsageError(f"{label} is not executable: {path}")
    return path


def build_micro_cmd(bin_path: str, args: argparse.Namespace) -> List[str]:
    cmd = [bin_path, "--json", "--iters", str(args.iters), "--warmup", str(args.warmup)]
    if args.only:
        cmd.extend(["--only", args.only])
    if args.extra_arg:
        cmd.extend(args.extra_arg)
    return cmd


def build_model_cmd(bin_path: str, args: argparse.Namespace) -> List[str]:
    cmd = [bin_path, "bench", "-m", args.model, "-n", str(args.n_tokens)]
    if args.prompt_file:
        cmd.extend(["-f", args.prompt_file])
    elif args.prompt is not None:
        cmd.extend(["-p", args.prompt])
    if args.ctx is not None:
        cmd.extend(["-c", str(args.ctx)])
    if not args.no_greedy:
        cmd.append("--greedy")
    if args.extra_arg:
        cmd.extend(args.extra_arg)
    return cmd


def save_baseline(
    path: str,
    mode: str,
    rounds: int,
    series: Dict[str, List[float]],
    meta_map: Dict[str, Dict[str, Any]],
) -> None:
    payload = collect_host_metadata()
    payload["schema"] = 1
    payload["mode"] = mode
    payload["rounds"] = rounds
    metrics_out: Dict[str, Any] = {}
    for name in sorted(series.keys()):
        vals = series[name]
        metrics_out[name] = {
            "unit": meta_map[name]["unit"],
            "lower_is_better": meta_map[name]["lower_is_better"],
            "median": median(vals),
            "round_values": vals,
        }
    payload["metrics"] = metrics_out
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def load_baseline(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise UsageError(f"baseline file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise UsageError(f"failed to read baseline {path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema") != 1 or not isinstance(data.get("metrics"), dict):
        raise UsageError(f"invalid baseline JSON schema in {path}")
    return data


# ----------------------------------------------------------------------------- formatting

def format_comparison_table(
    title: str,
    results: Dict[str, Dict[str, Any]],
    units: Dict[str, str],
    max_regression_pct: float,
) -> str:
    lines = [
        f"=== {title} (max regression threshold: {max_regression_pct:.2f}%) ===",
        f"{'metric':<28} {'unit':<6} {'base':>11} {'head':>11} {'delta%':>9} {'95% CI%':>20}  {'verdict'}",
        "-" * 100,
    ]
    for name in sorted(results.keys()):
        r = results[name]
        unit = units.get(name, "")
        ci_str = f"[{r['ci_low_pct']:+6.2f}%, {r['ci_high_pct']:+6.2f}%]"
        lines.append(
            f"{name:<28} {unit:<6} {r['median_base']:>11.3f} {r['median_head']:>11.3f} "
            f"{r['median_delta_pct']:>+8.2f}% {ci_str:>20}  {r['verdict']}"
        )
    return "\n".join(lines)


# ----------------------------------------------------------------------------- CLI & orchestration

def add_common_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--base-bin", help="Path to baseline binary (A)")
    p.add_argument("--head-bin", required=True, help="Path to candidate binary (B)")
    p.add_argument("--rounds", type=int, default=5, help="Number of ABBA rounds (default: 5)")
    p.add_argument("--cooldown-s", type=float, default=0.5, help="Sleep seconds between invocations (default: 0.5)")
    p.add_argument(
        "--max-regression-pct",
        type=float,
        default=2.0,
        help="Maximum allowed regression percentage before FAIL (default: 2.0)",
    )
    p.add_argument("--bootstrap-iters", type=int, default=2000, help="Bootstrap resamples (default: 2000)")
    p.add_argument("--seed", type=int, default=1337, help="Fixed RNG seed for bootstrap CI (default: 1337)")
    p.add_argument("--save-baseline", help="Save measured head results + host metadata to JSON")
    p.add_argument("--compare-baseline", help="Compare measured head results against a saved baseline JSON")
    p.add_argument("--json", action="store_true", help="Emit structured JSON report to stdout")
    p.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra argument forwarded to the benchmark binary (repeatable)",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="perf_ab.py",
        description="Paired interleaved (ABBA) A/B performance runner for inkling.",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_micro = sub.add_parser("micro", help="Compare compiled micro_bench binaries (model-free)")
    add_common_flags(p_micro)
    p_micro.add_argument("--iters", type=int, default=25, help="micro_bench --iters (default: 25)")
    p_micro.add_argument("--warmup", type=int, default=5, help="micro_bench --warmup (default: 5)")
    p_micro.add_argument("--only", help="micro_bench --only filter (cpu, gpu, or comma-separated names)")

    p_model = sub.add_parser("model", help="Compare compiled inkling binaries on `inkling bench`")
    add_common_flags(p_model)
    p_model.add_argument("-m", "--model", required=True, help="Path to GGUF model file")
    p_model.add_argument("-f", "--prompt-file", help="Prompt file passed via -f")
    p_model.add_argument("-p", "--prompt", help="Inline prompt text passed via -p")
    p_model.add_argument("-n", "--n-tokens", type=int, default=128, help="Decode tokens passed via -n (default: 128)")
    p_model.add_argument("-c", "--ctx", type=int, help="Context size passed via -c")
    p_model.add_argument(
        "--no-greedy",
        action="store_true",
        help="Do not automatically pass --greedy (use when sampling via --extra-arg)",
    )

    return parser


def execute(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.rounds < 1:
        raise UsageError("--rounds must be >= 1")
    if args.cooldown_s < 0.0:
        raise UsageError("--cooldown-s must be >= 0")
    if args.max_regression_pct < 0.0:
        raise UsageError("--max-regression-pct must be >= 0")
    if args.bootstrap_iters < 100:
        raise UsageError("--bootstrap-iters must be >= 100")

    head_bin = validate_executable(args.head_bin, "--head-bin")
    base_bin = validate_executable(args.base_bin, "--base-bin") if args.base_bin else None

    if not base_bin and not args.save_baseline and not args.compare_baseline:
        raise UsageError("provide --base-bin for A/B comparison, or --save-baseline / --compare-baseline")

    if args.mode == "micro":
        if args.iters < 1:
            raise UsageError("--iters must be >= 1")
        if args.warmup < 0:
            raise UsageError("--warmup must be >= 0")
        parse_fn = parse_micro_bench_output
        cmd_fn = build_micro_cmd
    else:
        if not os.path.exists(args.model):
            raise UsageError(f"model file does not exist: {args.model}")
        if args.prompt_file and not os.path.exists(args.prompt_file):
            raise UsageError(f"prompt file does not exist: {args.prompt_file}")
        parse_fn = parse_model_bench_output
        cmd_fn = build_model_cmd

    base_series: Dict[str, List[float]] = {}
    head_series: Dict[str, List[float]] = {}
    metric_meta: Dict[str, Dict[str, Any]] = {}

    if base_bin:
        schedule = build_abba_schedule(args.rounds)
        total_runs = len(schedule) * 2
        run_idx = 0
        for pair in schedule:
            for side in pair:
                target_bin = base_bin if side == "base" else head_bin
                stdout = run_command_checked(cmd_fn(target_bin, args))
                parsed = parse_fn(stdout)
                dst = base_series if side == "base" else head_series
                for name, info in parsed.items():
                    dst.setdefault(name, []).append(info["value"])
                    metric_meta[name] = {
                        "unit": info["unit"],
                        "lower_is_better": info["lower_is_better"],
                    }
                run_idx += 1
                if args.cooldown_s > 0.0 and run_idx < total_runs:
                    time.sleep(args.cooldown_s)
    else:
        for r in range(args.rounds):
            stdout = run_command_checked(cmd_fn(head_bin, args))
            parsed = parse_fn(stdout)
            for name, info in parsed.items():
                head_series.setdefault(name, []).append(info["value"])
                metric_meta[name] = {
                    "unit": info["unit"],
                    "lower_is_better": info["lower_is_better"],
                }
            if args.cooldown_s > 0.0 and r + 1 < args.rounds:
                time.sleep(args.cooldown_s)

    if args.save_baseline:
        save_baseline(args.save_baseline, args.mode, args.rounds, head_series, metric_meta)

    report: Dict[str, Any] = {
        "schema": 1,
        "mode": args.mode,
        "rounds": args.rounds,
        "max_regression_pct": args.max_regression_pct,
        "meta": collect_host_metadata(),
    }
    any_fail = False
    tables: List[str] = []

    if base_bin:
        common = sorted(set(base_series.keys()) & set(head_series.keys()))
        if not common:
            raise UsageError("base and head produced no overlapping metrics")
        paired_results: Dict[str, Dict[str, Any]] = {}
        units: Dict[str, str] = {}
        for idx, name in enumerate(common):
            m_info = metric_meta[name]
            res = analyze_paired(
                base_series[name],
                head_series[name],
                lower_is_better=m_info["lower_is_better"],
                max_regression_pct=args.max_regression_pct,
                bootstrap_iters=args.bootstrap_iters,
                seed=args.seed + idx,
            )
            res["unit"] = m_info["unit"]
            paired_results[name] = res
            units[name] = m_info["unit"]
            if res["verdict"] == "FAIL":
                any_fail = True
        report["paired_ab"] = paired_results
        tables.append(
            format_comparison_table(
                f"Paired ABBA ({args.mode}, {args.rounds} rounds)",
                paired_results,
                units,
                args.max_regression_pct,
            )
        )

    if args.compare_baseline:
        base_doc = load_baseline(args.compare_baseline)
        base_metrics = base_doc["metrics"]
        common = sorted(set(base_metrics.keys()) & set(head_series.keys()))
        if not common:
            raise UsageError("baseline and head produced no overlapping metrics")
        cmp_results: Dict[str, Dict[str, Any]] = {}
        units = {}
        for idx, name in enumerate(common):
            bm = base_metrics[name]
            b_vals = bm.get("round_values") or [float(bm["median"])]
            m_info = metric_meta[name]
            res = analyze_unpaired(
                [float(x) for x in b_vals],
                head_series[name],
                lower_is_better=bool(bm.get("lower_is_better", m_info["lower_is_better"])),
                max_regression_pct=args.max_regression_pct,
                bootstrap_iters=args.bootstrap_iters,
                seed=args.seed + 1000 + idx,
            )
            res["unit"] = str(bm.get("unit", m_info["unit"]))
            cmp_results[name] = res
            units[name] = res["unit"]
            if res["verdict"] == "FAIL":
                any_fail = True
        report["baseline_comparison"] = cmp_results
        tables.append(
            format_comparison_table(
                f"Baseline comparison vs {args.compare_baseline}",
                cmp_results,
                units,
                args.max_regression_pct,
            )
        )

    if not base_bin and not args.compare_baseline:
        # Pure baseline capture summary
        summary = {
            name: {
                "unit": metric_meta[name]["unit"],
                "median": median(vals),
                "round_values": vals,
            }
            for name, vals in sorted(head_series.items())
        }
        report["captured"] = summary
        lines = [f"Saved baseline to {args.save_baseline} ({args.rounds} rounds):"]
        for name, info in summary.items():
            lines.append(f"  {name:<28} {info['median']:>11.3f} {info['unit']}")
        tables.append("\n".join(lines))

    report["overall"] = "FAIL" if any_fail else "PASS"
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print("\n\n".join(tables))
        if args.save_baseline and (base_bin or args.compare_baseline):
            print(f"\nSaved baseline to {args.save_baseline}")
        print(f"\nOverall verdict: {report['overall']}")

    return 1 if any_fail else 0


def main() -> None:
    try:
        rc = execute()
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    sys.exit(rc)


if __name__ == "__main__":
    main()
