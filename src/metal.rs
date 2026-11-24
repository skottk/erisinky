//! Thin, zero-overhead-ish wrapper around objc2-metal for compute work.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;

pub type Buf = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type Pso = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub type CmdBuf = Retained<ProtocolObject<dyn MTLCommandBuffer>>;

/// Private wrapper around `CmdBuf` stored in `Gpu::inflight` so `Gpu` preserves its
/// existing `Send` (and `!Sync`) contract for callers like `Mutex<Gpu>`.
struct TrackedCmdBuf(CmdBuf);

// SAFETY: `Gpu` is `!Sync` (due to `RefCell`), so a `TrackedCmdBuf` can only cross thread
// boundaries when `Gpu` is exclusively owned (e.g., locked inside a `Mutex<Gpu>`).
// Apple Metal permits calling `status()` and `waitUntilCompleted()` on a command buffer
// from any thread provided access is externally synchronized.
unsafe impl Send for TrackedCmdBuf {}

pub struct Gpu {
    pub dev: Retained<ProtocolObject<dyn MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    lib: Retained<ProtocolObject<dyn MTLLibrary>>,
    psos: std::cell::RefCell<HashMap<String, Pso>>,
    /// Tracks command buffers issued by `cmd()` so `drain()` / `Drop` can wait for
    /// all committed GPU work before backing buffers or mmaps are released.
    inflight: std::cell::RefCell<Vec<TrackedCmdBuf>>,
    /// Optional A/B benchmark override (`INK_UNRETAINED_CMD=1`); defaults to `false`
    /// (retained references) for unwind safety.
    unretained: bool,
    /// Math mode used to compile the Metal shader library (`MTLMathMode::Safe` by default
    /// so IEEE-754 `-INFINITY` sentinels and comparisons are preserved).
    math_mode: MTLMathMode,
}

impl Gpu {
    pub fn new(src: &str, kernels: &[&str]) -> Gpu {
        let math_mode = if std::env::var("INK_FAST_MATH")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            MTLMathMode::Fast
        } else {
            MTLMathMode::Safe
        };
        Self::new_with_math_mode(src, kernels, math_mode)
    }

    pub fn new_with_math_mode(src: &str, kernels: &[&str], math_mode: MTLMathMode) -> Gpu {
        let dev = MTLCreateSystemDefaultDevice().expect("no Metal device");
        let queue = dev.newCommandQueue().expect("queue");
        let opts = MTLCompileOptions::new();
        opts.setMathMode(math_mode);
        opts.setLanguageVersion(MTLLanguageVersion::Version4_0);
        let lib = dev
            .newLibraryWithSource_options_error(&NSString::from_str(src), Some(&opts))
            .unwrap_or_else(|e| panic!("Metal compile failed:\n{}", e.localizedDescription()));
        let unretained = std::env::var("INK_UNRETAINED_CMD")
            .map(|v| v == "1")
            .unwrap_or(false);
        let g = Gpu {
            dev,
            queue,
            lib,
            psos: Default::default(),
            inflight: Default::default(),
            unretained,
            math_mode,
        };
        for k in kernels { g.pso(k); }
        g
    }

    /// Returns the `MTLMathMode` used to compile this `Gpu`'s shader library.
    pub fn math_mode(&self) -> MTLMathMode {
        self.math_mode
    }

    /// Returns `true` when `cmd()` allocates command buffers with retained resource references
    /// (the default for unwind safety).
    pub fn uses_retained_references(&self) -> bool {
        !self.unretained
    }

    /// pipeline state by kernel name (compiled on first use; production kernels are pre-built at load)
    pub fn pso(&self, name: &str) -> Pso {
        if let Some(p) = self.psos.borrow().get(name) { return p.clone(); }
        let f = self.lib.newFunctionWithName(&NSString::from_str(name)).unwrap_or_else(|| panic!("kernel {name} missing"));
        let p = self.dev.newComputePipelineStateWithFunction_error(&f).unwrap_or_else(|e| panic!("pso {name}: {}", e.localizedDescription()));
        self.psos.borrow_mut().insert(name.to_string(), p.clone());
        p
    }

    pub fn buffer(&self, len: usize) -> Buf {
        self.dev.newBufferWithLength_options(len.max(16), MTLResourceOptions::StorageModeShared).expect("alloc buffer")
    }

    pub fn buffer_with(&self, data: &[u8]) -> Buf {
        let b = self.buffer(data.len());
        // SAFETY: `b` was freshly allocated as `StorageModeShared` with capacity
        // `data.len().max(16) >= data.len()`, so destination is valid, writable, and
        // non-overlapping with `data`.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), b.contents().as_ptr() as *mut u8, data.len()); }
        b
    }

    /// Wrap existing page-aligned memory (e.g. an mmapped model file) without copying.
    ///
    /// Callers must ensure `ptr..ptr+len` remains mapped and valid until the returned `Buf`
    /// is dropped and all GPU command buffers referencing it have completed (`Gpu::drain()`).
    pub fn buffer_nocopy(&self, ptr: *mut u8, len: usize) -> Buf {
        // SAFETY: Caller provides a non-null, page-aligned `ptr` and page-aligned `len`
        // whose backing allocation outlives the returned `Buf` and all GPU work using it
        // (enforced by `Engine`'s field drop order and `Drop::drop` calling `Gpu::drain()`).
        unsafe {
            self.dev
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    NonNull::new(ptr as *mut c_void).unwrap(), len, MTLResourceOptions::StorageModeShared, None)
                .expect("no-copy buffer (needs page-aligned ptr/len)")
        }
    }

    /// Create a command buffer tracked by this `Gpu`.
    ///
    /// Uses retained references (`commandBuffer()`) by default so Metal keeps all bound
    /// buffers and pipeline states alive across panics/unwinds. Set `INK_UNRETAINED_CMD=1`
    /// only for A/B benchmarking retained vs. unretained command buffer overhead.
    pub fn cmd(&self) -> CmdBuf {
        let cb = if self.unretained {
            self.queue.commandBufferWithUnretainedReferences().expect("cmdbuf")
        } else {
            self.queue.commandBuffer().expect("cmdbuf")
        };
        let mut inflight = self.inflight.borrow_mut();
        inflight.retain(|c| {
            let st = c.0.status();
            st != MTLCommandBufferStatus::Completed && st != MTLCommandBufferStatus::Error
        });
        inflight.push(TrackedCmdBuf(cb.clone()));
        cb
    }

    /// Block until all committed in-flight command buffers created from this `Gpu` complete.
    ///
    /// Skips uncommitted (`NotEnqueued`), already `Completed`, and `Error` command buffers
    /// so draining during unwind cannot block indefinitely on a never-committed buffer.
    pub fn drain(&self) {
        let pending = match self.inflight.try_borrow_mut() {
            Ok(mut v) => std::mem::take(&mut *v),
            Err(_) => return,
        };
        for TrackedCmdBuf(cb) in pending {
            let st = cb.status();
            if st != MTLCommandBufferStatus::NotEnqueued
                && st != MTLCommandBufferStatus::Completed
                && st != MTLCommandBufferStatus::Error
            {
                cb.waitUntilCompleted();
            }
        }
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        self.drain();
    }
}

pub struct Enc {
    pub e: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
}

impl Enc {
    pub fn new(cb: &CmdBuf) -> Enc {
        Enc { e: cb.computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent).expect("encoder") }
    }
    #[inline] pub fn pso(&self, p: &Pso) { self.e.setComputePipelineState(p); }
    #[inline] pub fn buf(&self, b: &Buf, off: usize, idx: usize) {
        // SAFETY: `b` is a valid live `MTLBuffer`; caller ensures `off` is within `b.length()`
        // and `idx` matches the bound Metal shader buffer slot.
        unsafe { self.e.setBuffer_offset_atIndex(Some(b), off, idx); }
    }
    #[inline] pub fn bytes<T>(&self, v: &T, idx: usize) {
        // SAFETY: `v` is a valid non-null reference to `size_of::<T>()` initialized bytes;
        // Metal copies the bytes synchronously during encoding.
        unsafe { self.e.setBytes_length_atIndex(NonNull::new(v as *const T as *mut c_void).unwrap(), std::mem::size_of::<T>(), idx); }
    }
    #[inline] pub fn bytes_slice<T>(&self, v: &[T], idx: usize) {
        // SAFETY: `v.as_ptr()` is non-null and valid for `size_of_val(v)` bytes;
        // Metal copies the bytes synchronously during encoding.
        unsafe { self.e.setBytes_length_atIndex(NonNull::new(v.as_ptr() as *mut c_void).unwrap(), std::mem::size_of_val(v), idx); }
    }
    #[inline] pub fn dispatch(&self, grid: (usize, usize, usize), tg: usize) {
        self.e.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: grid.0, height: grid.1, depth: grid.2 }, MTLSize { width: tg, height: 1, depth: 1 });
    }
    /// grid size read on the GPU from 3 uint32 at `off` bytes into `b` (written by an earlier kernel of the same encoder)
    #[inline] pub fn dispatch_indirect(&self, b: &Buf, off: usize, tg: usize) {
        // SAFETY: `b` is a valid `MTLBuffer` and caller ensures `off` is 4-byte aligned with
        // at least 12 bytes (`3 * sizeof(u32)`) remaining in `b`.
        unsafe { self.e.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(b, off, MTLSize { width: tg, height: 1, depth: 1 }); }
    }
    #[inline] pub fn barrier(&self) { self.e.memoryBarrierWithScope(MTLBarrierScope::Buffers); }
    pub fn end(self) { self.e.endEncoding(); }
}
