//! Memory safety & lifetime regression tests for High-Severity #3
//! (UAF in `Engine` drop order, unretained command buffers, `Engine::logits()`,
//! unwind safety in `Session::generate`) and `Engine::try_load_with` model validation.

mod common;

use common::gguf_builder::{GgufBuilder, KvVal};
use inkling::engine::{Engine, Logits};
use inkling::metal::{Buf, Enc, Gpu};
use inkling::session::InflightGuard;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus};
use std::cell::RefCell;
use std::io::ErrorKind;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

const BUSY_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void k_busy_sum(
    device const uint *src [[buffer(0)]],
    device uint *dst [[buffer(1)]],
    constant uint2 &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint n = params.x;
    uint iters = params.y;
    if (gid >= n) return;
    uint acc = src[gid];
    for (uint i = 0; i < iters; ++i) {
        acc = (acc * 1664525u + 1013904223u) ^ src[(gid + i) % n];
    }
    dst[gid] = acc;
}
"#;

fn cpu_busy_sum(src: &[u32], iters: u32) -> Vec<u32> {
    let n = src.len();
    (0..n)
        .map(|gid| {
            let mut acc = src[gid];
            for i in 0..iters as usize {
                acc = acc
                    .wrapping_mul(1664525u32)
                    .wrapping_add(1013904223u32)
                    ^ src[(gid + i) % n];
            }
            acc
        })
        .collect()
}

/// Compile-time check that `Engine::logits(&self)` returns `Logits<'_>`, not a detached `&[f32]`.
#[allow(dead_code)]
fn _assert_engine_logits_returns_guard(eng: &Engine) -> Logits<'_> {
    eng.logits()
}

#[test]
fn logits_guard_prevents_act_reallocation_while_borrowed() {
    let gpu = Gpu::new(BUSY_KERNEL_SRC, &["k_busy_sum"]);
    let n_f32 = 256usize;
    let buf = gpu.buffer(n_f32 * std::mem::size_of::<f32>());
    let raw_ptr = buf.contents().as_ptr() as *mut f32;
    for i in 0..n_f32 {
        // SAFETY: `buf` is `StorageModeShared`, 4-byte aligned, and has capacity `n_f32` `f32`s.
        unsafe {
            *raw_ptr.add(i) = i as f32 * 0.5;
        }
    }

    let act = RefCell::new(buf);
    let off = 64usize;
    let len = 128usize;

    {
        let guard = Logits::new(act.borrow(), off, len);
        // Zero-copy: guard slice points directly into the MTLBuffer.
        assert_eq!(guard.len(), len);
        // SAFETY: `off < n_f32`, computing pointer within the allocated buffer.
        assert_eq!(guard.as_ptr(), unsafe { (raw_ptr as *const f32).add(off) });
        assert_eq!(guard[0], 32.0);
        assert_eq!(guard[len - 1], 95.5);

        // Cannot mutably borrow / reallocate `act` while `guard` is alive.
        assert!(
            act.try_borrow_mut().is_err(),
            "RefCell<Buf> must reject mutable borrow while Logits guard is alive"
        );

        // Attempting a reallocation (as `ensure_batch` / `release_batch` does) must panic
        // cleanly instead of freeing the buffer behind `guard`.
        let realloc_attempt = catch_unwind(AssertUnwindSafe(|| {
            *act.borrow_mut() = gpu.buffer(512 * 4);
        }));
        assert!(
            realloc_attempt.is_err(),
            "reallocation of act while Logits is borrowed must fail dynamically"
        );

        // Slice data remains completely valid after the caught panic.
        assert_eq!(guard[0], 32.0);
        assert_eq!(guard[len - 1], 95.5);
    }

    // Once `guard` drops, reallocation succeeds cleanly.
    assert!(
        act.try_borrow_mut().is_ok(),
        "RefCell<Buf> must allow mutable borrow after Logits guard drops"
    );
    *act.borrow_mut() = gpu.buffer(512 * 4);

    // Out-of-bounds offset/len must be rejected.
    let oob = catch_unwind(AssertUnwindSafe(|| {
        let _ = Logits::new(act.borrow(), 500, 32);
    }));
    assert!(oob.is_err(), "out-of-bounds Logits::new must panic");
}

struct LoggedDrop<T> {
    inner: Option<T>,
    label: &'static str,
    log: Arc<Mutex<Vec<&'static str>>>,
}

impl<T> LoggedDrop<T> {
    fn new(val: T, label: &'static str, log: Arc<Mutex<Vec<&'static str>>>) -> Self {
        Self {
            inner: Some(val),
            label,
            log,
        }
    }
    fn get(&self) -> &T {
        self.inner.as_ref().unwrap()
    }
}

impl<T> Drop for LoggedDrop<T> {
    fn drop(&mut self) {
        self.inner.take();
        self.log.lock().unwrap().push(self.label);
    }
}

struct TrackedMmap {
    ptr: *mut u8,
    len: usize,
    log: Arc<Mutex<Vec<&'static str>>>,
}

impl TrackedMmap {
    fn new(len: usize, log: Arc<Mutex<Vec<&'static str>>>) -> Self {
        // SAFETY: Anonymous private mapping with page-aligned non-zero length.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANON | libc::MAP_PRIVATE,
                -1,
                0,
            )
        };
        assert_ne!(ptr, libc::MAP_FAILED, "mmap failed");
        Self {
            ptr: ptr as *mut u8,
            len,
            log,
        }
    }
}

impl Drop for TrackedMmap {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` and `self.len` came from `libc::mmap` above and are unmapped once.
        let rc = unsafe { libc::munmap(self.ptr as *mut libc::c_void, self.len) };
        assert_eq!(rc, 0, "munmap failed");
        self.log.lock().unwrap().push("mmap_unmapped");
    }
}

/// Mirrors `Engine`'s safe field declaration order:
/// buffers (`w`, `out`) -> `gpu` -> `backing` mmap LAST, with `Drop` draining GPU first.
struct EngineLike {
    w: LoggedDrop<Buf>,
    out: LoggedDrop<Buf>,
    gpu: LoggedDrop<Gpu>,
    _backing: TrackedMmap,
    log: Arc<Mutex<Vec<&'static str>>>,
}

impl Drop for EngineLike {
    fn drop(&mut self) {
        self.gpu.get().drain();
        self.log.lock().unwrap().push("gpu_drained");
    }
}

#[test]
fn gpu_retained_cmd_and_drain_survives_unwind_over_nocopy_mmap() {
    let log = Arc::new(Mutex::new(Vec::<&'static str>::new()));
    let page = 16384usize;
    let map_len = page * 4; // 64 KiB page-aligned mmap
    let n_u32 = 1024usize;
    let iters = 4096u32;

    let backing = TrackedMmap::new(map_len, log.clone());
    let mut host_src = vec![0u32; n_u32];
    for (i, slot) in host_src.iter_mut().enumerate() {
        *slot = (i as u32).wrapping_mul(2654435761u32) ^ 0xA5A5_5A5A;
        // SAFETY: `backing.ptr` has `map_len = 65536` bytes >= `n_u32 * 4` and is page-aligned.
        unsafe {
            *(backing.ptr as *mut u32).add(i) = *slot;
        }
    }
    let expected = cpu_busy_sum(&host_src, iters);

    let gpu = Gpu::new(BUSY_KERNEL_SRC, &["k_busy_sum"]);
    assert!(
        gpu.uses_retained_references(),
        "Gpu::cmd() must use retained command buffers by default for unwind safety"
    );

    let w = gpu.buffer_nocopy(backing.ptr, map_len);
    let out = gpu.buffer(n_u32 * std::mem::size_of::<u32>());
    let out_keep = out.clone(); // inspect GPU result after unwind & drain

    let eng = EngineLike {
        w: LoggedDrop::new(w, "w_dropped", log.clone()),
        out: LoggedDrop::new(out, "out_dropped", log.clone()),
        gpu: LoggedDrop::new(gpu, "gpu_dropped", log.clone()),
        _backing: backing,
        log: log.clone(),
    };

    let cb_handle: Rc<RefCell<Option<inkling::metal::CmdBuf>>> = Rc::new(RefCell::new(None));
    let cb_slot = cb_handle.clone();

    // Commit GPU work referencing the no-copy mmap buffer, then panic immediately
    // WITHOUT calling `waitUntilCompleted()`, simulating an `on_token` callback panic.
    let res = catch_unwind(AssertUnwindSafe(move || {
        let cb = eng.gpu.get().cmd();
        let enc = Enc::new(&cb);
        enc.pso(&eng.gpu.get().pso("k_busy_sum"));
        enc.buf(eng.w.get(), 0, 0);
        enc.buf(eng.out.get(), 0, 1);
        enc.bytes(&[n_u32 as u32, iters], 2);
        enc.dispatch((n_u32.div_ceil(64), 1, 1), 64);
        enc.end();
        cb.commit();
        *cb_slot.borrow_mut() = Some(cb);

        // Also allocate an uncommitted command buffer to ensure `drain()` does not hang
        // on `MTLCommandBufferStatus::NotEnqueued`.
        let _uncommitted = eng.gpu.get().cmd();

        panic!("simulated on_token callback panic while GPU is reading no-copy mmap");
    }));

    assert!(res.is_err(), "expected simulated panic to unwind");

    // Verify exact drop order: drain -> buffers -> gpu -> munmap LAST.
    let events = log.lock().unwrap().clone();
    assert_eq!(
        events,
        vec![
            "gpu_drained",
            "w_dropped",
            "out_dropped",
            "gpu_dropped",
            "mmap_unmapped"
        ],
        "drop order must drain GPU and release no-copy buffers before munmap"
    );

    // Command buffer must have finished cleanly before munmap.
    let cb = cb_handle.borrow_mut().take().unwrap();
    assert_eq!(
        cb.status(),
        MTLCommandBufferStatus::Completed,
        "in-flight command buffer must be completed by Gpu::drain() during unwind"
    );

    // Verify the GPU computed the exact expected values from the mmap before it was unmapped.
    let got = unsafe {
        // SAFETY: `out_keep` is a valid shared MTLBuffer of `n_u32` u32s and GPU work is completed.
        std::slice::from_raw_parts(out_keep.contents().as_ptr() as *const u32, n_u32)
    };
    assert_eq!(got, expected.as_slice(), "GPU result mismatch after unwind drain");
}

#[test]
fn session_inflight_guard_drains_on_unwind() {
    let gpu = Gpu::new(BUSY_KERNEL_SRC, &["k_busy_sum"]);
    let n_u32 = 512usize;
    let iters = 2048u32;
    let host_src: Vec<u32> = (0..n_u32).map(|i| (i as u32).wrapping_mul(7919) + 1).collect();
    let expected = cpu_busy_sum(&host_src, iters);

    let src_bytes: Vec<u8> = host_src.iter().flat_map(|x| x.to_le_bytes()).collect();
    let src_buf = gpu.buffer_with(&src_bytes);
    let out_buf = gpu.buffer(n_u32 * 4);
    let out_keep = out_buf.clone();

    let cb_slot: Rc<RefCell<Option<inkling::metal::CmdBuf>>> = Rc::new(RefCell::new(None));
    let cb_slot_inner = cb_slot.clone();

    let res = catch_unwind(AssertUnwindSafe(move || {
        let mut inflight = InflightGuard::new();
        let cb = gpu.cmd();
        let enc = Enc::new(&cb);
        enc.pso(&gpu.pso("k_busy_sum"));
        enc.buf(&src_buf, 0, 0);
        enc.buf(&out_buf, 0, 1);
        enc.bytes(&[n_u32 as u32, iters], 2);
        enc.dispatch((n_u32.div_ceil(64), 1, 1), 64);
        enc.end();
        cb.commit();
        *cb_slot_inner.borrow_mut() = Some(cb.clone());
        inflight.0.push_back((0, cb));
        panic!("simulated on_token panic during greedy pipelined generation");
    }));
    assert!(res.is_err());

    let cb = cb_slot.borrow_mut().take().unwrap();
    assert_eq!(
        cb.status(),
        MTLCommandBufferStatus::Completed,
        "InflightGuard must waitUntilCompleted on unwind"
    );
    let got = unsafe {
        // SAFETY: `out_keep` is a valid shared MTLBuffer of `n_u32` u32s and GPU work is completed.
        std::slice::from_raw_parts(out_keep.contents().as_ptr() as *const u32, n_u32)
    };
    assert_eq!(got, expected.as_slice());
}

#[test]
fn engine_try_load_with_rejects_malformed_gguf_cleanly() {
    // 1. Wrong architecture
    let f_arch = GgufBuilder::new()
        .kv("general.architecture", KvVal::Str("llama".into()))
        .write_temp();
    let err = Engine::try_load_with(f_arch.path().to_str().unwrap(), 64, false)
        .err()
        .expect("wrong architecture must fail");
    assert_eq!(err.kind(), ErrorKind::InvalidData);

    // 2. inkling2_loop architecture but wrong embedding_length
    let f_dim = GgufBuilder::new()
        .kv("general.architecture", KvVal::Str("inkling2_loop".into()))
        .kv("inkling2_loop.embedding_length", KvVal::U32(1024))
        .write_temp();
    let err = Engine::try_load_with(f_dim.path().to_str().unwrap(), 64, false)
        .err()
        .expect("wrong embedding_length must fail");
    assert_eq!(err.kind(), ErrorKind::InvalidData);

    // 3. Truncated / corrupt GGUF header
    let mut b = GgufBuilder::new().kv("general.architecture", KvVal::Str("inkling2_loop".into()));
    b.truncate_to = Some(12);
    let f_trunc = b.write_temp();
    let err = Engine::try_load_with(f_trunc.path().to_str().unwrap(), 64, false)
        .err()
        .expect("truncated GGUF must fail");
    assert_eq!(err.kind(), ErrorKind::InvalidData);

    // 4. All scalar metadata valid, but missing required tensors
    let f_missing_tensors = GgufBuilder::new()
        .kv("general.architecture", KvVal::Str("inkling2_loop".into()))
        .kv("inkling2_loop.embedding_length", KvVal::U32(2304))
        .kv("inkling2_loop.block_count", KvVal::U32(24))
        .kv("inkling2_loop.expert_count", KvVal::U32(128))
        .kv("inkling2_loop.expert_used_count", KvVal::U32(8))
        .kv("inkling2_loop.loop.start", KvVal::U32(6))
        .kv("inkling2_loop.loop.end", KvVal::U32(18))
        .kv("inkling2_loop.loop.count", KvVal::U32(3))
        .kv("inkling2_loop.attn_res.block_size", KvVal::U32(6))
        .kv("inkling2_loop.attention.sliding_window", KvVal::U32(1024))
        .kv("inkling2_loop.attention.relative_extent", KvVal::U32(1024))
        .kv(
            "inkling2_loop.attention.sliding_window_pattern",
            KvVal::ArrBool(vec![true; 24]),
        )
        .kv(
            "inkling2_loop.attention.layer_norm_rms_epsilon",
            KvVal::F32(1e-6),
        )
        .kv("inkling2_loop.attention.log_scale_alpha", KvVal::F32(0.1))
        .kv("inkling2_loop.attention.log_scale_floor", KvVal::F32(1.0))
        .kv("inkling2_loop.activation.situ_beta", KvVal::F32(1.0))
        .kv("inkling2_loop.activation.situ_linear_beta", KvVal::F32(1.0))
        .kv("inkling2_loop.expert_weights_scale", KvVal::F32(1.0))
        .kv("inkling2_loop.final_logit_softcapping", KvVal::F32(30.0))
        .kv("inkling2_loop.vocab_size_unpadded", KvVal::U32(1024))
        .write_temp();
    let err = Engine::try_load_with(f_missing_tensors.path().to_str().unwrap(), 64, false)
        .err()
        .expect("missing tensors must fail");
    assert_eq!(err.kind(), ErrorKind::InvalidData);
}
