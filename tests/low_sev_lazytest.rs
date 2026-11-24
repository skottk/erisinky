//! Regression tests for Low-Severity #4d (`lazytest` checked `mmap` + `munmap` RAII).

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn target_dir() -> PathBuf {
    std::env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest_dir().join("target"))
}

fn inkling_bin() -> &'static PathBuf {
    static BIN: OnceLock<PathBuf> = OnceLock::new();
    BIN.get_or_init(|| {
        let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
        let status = Command::new(&cargo)
            .current_dir(manifest_dir())
            .args(["build", "--offline", "--bin", "inkling"])
            .status()
            .expect("invoke cargo build --bin inkling");
        assert!(
            status.success(),
            "inkling binary failed to compile (status: {status})"
        );
        let bin = target_dir().join("debug/inkling");
        assert!(
            bin.exists(),
            "expected compiled inkling binary at {}",
            bin.display()
        );
        bin
    })
}

#[test]
fn lazytest_checks_map_failed_and_unmaps_via_raii() {
    let src = std::fs::read_to_string(manifest_dir().join("src/main.rs"))
        .expect("read src/main.rs");
    let start = src
        .find("\"lazytest\" => {")
        .expect("lazytest match arm must exist in src/main.rs");
    let rest = &src[start..];
    let end = rest
        .find("\"kvprune\" => {")
        .expect("kvprune arm following lazytest");
    let lazytest_arm = &rest[..end];

    assert!(
        !lazytest_arm.contains("libc::mmap("),
        "lazytest arm must use checked RAII AnonMap instead of raw unchecked libc::mmap"
    );
    assert!(
        lazytest_arm.contains("AnonMap::new("),
        "lazytest arm must allocate via AnonMap::new"
    );
    assert!(
        lazytest_arm.contains("gpu.drain()"),
        "lazytest arm must drain GPU work before dropping buffer/mapping"
    );
    assert!(
        src.contains("libc::MAP_FAILED"),
        "src/main.rs must check libc::MAP_FAILED"
    );
    assert!(
        src.contains("libc::munmap("),
        "src/main.rs must unmap via libc::munmap"
    );
}

#[test]
fn lazytest_binary_runs_cleanly() {
    let bin = inkling_bin();
    let out = Command::new(bin)
        .arg("lazytest")
        .output()
        .expect("run inkling lazytest");
    assert!(
        out.status.success(),
        "inkling lazytest failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("readback: 2 1 1 2 (expect 2 1 1 2)"),
        "unexpected lazytest output:\n{stdout}"
    );
}
