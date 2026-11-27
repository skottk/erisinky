# inkling

A from-scratch inference engine for **Thinking Machines' Inkling 2 Mini** (`inkling2_loop`, 26.4B total / 3.4B active MoE)
on Apple Silicon: Rust host + hand-fused Metal kernels, reading the LM Studio GGUF directly (zero-copy mmap).

## Results (Apple M5 Pro, 48 GB, `Inkling-2-Mini-Q4_K_M.gguf`, greedy)

| | LM Studio runtime (llama.cpp 2.80.1) | llama.cpp master | **inkling** | speedup vs LM Studio |
|---|---:|---:|---:|---:|
| decode, short context | 33.1 tok/s | 32.7 tok/s | **63.0 tok/s** | **1.90x** |
| decode @ 2.4K context | 31.7 tok/s | – | **58.2 tok/s** | **1.84x** |
| prefill, 2394-token prompt | 381 tok/s | 409 tok/s (pp2048) | **2200 tok/s** | **5.8x** |

LM Studio numbers are from its own bundled `llama-server` (same machine, same file). Sampling mode (`--temp > 0`)
costs ~3% vs greedy (one CPU round trip per token).

**Correctness.** Logits are checked position-by-position against llama.cpp (`tools/ref_logits.cpp` dumps the reference):
61/61 argmax agreement on the decode path (max logit diff ~0.005, worst KL 0.0026), 25/25 after a batched
2379-token prefill (worst KL 0.0069; identical results whether the prompt is one chunk or several), 41/41 after a short one
(worst KL 0.0001). The tokenizer matches `llama-tokenize` token-for-token on 74K tokens of code/markdown.

## Usage

```sh
cargo build --release
./target/release/inkling chat                       # interactive, multi-turn (KV cache reused across turns)
./target/release/inkling run -p "Explain RoPE in one paragraph" --effort low
./target/release/inkling run -f prompt.txt --greedy  # greedy = fully GPU-pipelined, fastest
./target/release/inkling bench -f prompt.txt -n 256  # prefill + decode throughput
```
Options: `--effort none|minimal|low|medium|high|xhigh`, `--temp/--top-p/--top-k/--seed`, `--system`, `-c CTX`,
`--hide-thinking`, `--raw`, `--kv-disk` (half of the exact K/V cache in memory, the rest streamed from SSD: slower
generation, bit-identical output), `--kv8` (half-size 8-bit K/V cache, not exact - both in the limits section),
`-m model.gguf` (default: LM Studio's Inkling-2-Mini Q4_K_M). Requires macOS with Metal 4
(prefill GEMMs and prefill attention use M5 neural accelerators or M4 8x8 matrix units).

## The model (what had to be implemented)

Inkling 2 Mini is not a standard transformer. Per token: 24 physical layers, of which the middle 12 run **3 times**
(48 layer executions, 48 KV/conv cache slots), and every layer has:
- **block attention residuals**: layer inputs are a learned softmax mix (over the RMS-normalized outputs of earlier blocks);
- **no RoPE**: a learned relative-position bias `r·B[d]` (16-dim query-dependent projection, clip-normalized), plus
  per-head **attention sinks** projected from the input, QK norm *clipping*, log-scaled temperature for long contexts;
- **causal short convolutions** (width 4) on K, V (input-dependent taps on full-attention layers) and on both residual
  branch outputs, with recurrent state;
- sigmoid **output gate** on attention, sliding window (1024) on 5 of every 6 layers;
- 128-expert top-8 **MoE** with sigmoid routing + selection bias and the **SiTU** activation
  (`β·tanh(g/β)·σ(g) · λ·tanh(u/λ)`);
- in loop passes 2 and 3, **low-rank loop controllers** gate the inputs with "messages" mixed from the previous pass.

## How it is fast

Decode is memory-bound: ~3.2 GB of weights per token (top-8 experts x 48 layer executions + a 380 MB lm_head), so the
floor at the measured ~290 GB/s is ~11 ms/token. The engine runs at ~15.9 ms; llama.cpp at ~30 ms.

- **Everything stays on the GPU.** One command buffer per token (~530 dependent dispatches), no CPU round trips: routing/top-k,
  attention split-combine, norms and argmax run on the GPU; greedy decoding chains tokens GPU-to-GPU with two steps in
  flight (exact: the in-flight step always consumes a real token, so stopping never corrupts the recurrent state).
- **Fusion where the model's op soup lived.** llama.cpp builds thousands of tiny graph nodes per token for the mixes,
  clips, convolutions and controllers. Here: one depth-mix kernel (all states, all queries, optional norm, normalized
  copies for the controllers), one segmented matvec for q/gate/k/v/rel/sink/conv-taps, one attention kernel that also does
  the K/V convolution + state update + clipping + norms + relative bias + sinks, matvec epilogues that apply the post
  convolutions and residuals, MoE kernels with the activation and weighted expert sum fused in.
- **Bandwidth-tuned quantized matvecs** (Q4_K/Q6_K/Q8_0/Q5_0/bf16): x read straight from cache (no threadgroup staging,
  no barriers), tuned rows-per-simdgroup: ~285 GB/s in isolation (98% of measured peak). Dense f32 router/controller
  matrices are stored as bf16 — bit-exact, since they are bf16-representable — halving their traffic.
- **Barrier-free decode attention**: each simdgroup runs its own online softmax over 8-key blocks (4 lanes per key,
  shuffle reductions), merged once per threadgroup; adaptive split-K across threadgroups.
- **GPU page pre-wiring**: a one-time GPU pass over the mmapped weights at load. Without it, the first use of every
  rarely-used expert page takes a GPU page fault mid-generation (measured: MoE kernel 122-206 GB/s cold vs 283 warm).
- **Memory sized by need.** Metal commits a buffer in full the first time it is used (and that first use costs ~20 ms per GiB),
  so every byte of capacity is real memory. Besides the mmapped weights the engine now holds:
  - *the K/V cache* (fp16, not quantized), sized to the context. The 40 sliding-window layer executions keep exactly their
    window: a ring of 1024 keys each (160 MiB once the context reaches 1024 tokens; the ring is linear until then, so it grows
    like everything else). A prefill batch attends through a scratch copy [window history | batch keys] and leaves the last
    window in the ring. The 8 full-attention executions keep every key: 32 KiB per token. Both buffers grow in small steps
    (at most 6% headroom, in steps of 64 positions): 24 MiB for a short chat (it was 36 with a 64-position minimum headroom), 240 MiB at 2.4K tokens, ~300 MiB at 4K, ~430 MiB at 8K, 672 MiB at 16K. It was
    a fixed 832 MiB. Growing in the middle of a pipelined generation is exact (the GPU is drained first).
  - *the prefill arena* (activations of a batch, ~0.4 MiB per token) only while a prefill runs; it is released when generation
    starts. It used to be 2.8 GiB allocated at load. One-shot runs allocate and map it while they still know nothing but the
    prompt length (`Engine::prepare`), so the first-use cost stays out of the prefill; in a chat, a large paste pays it (~2%).
  Process footprint beyond the weights: ~0.38 GB for a short chat (was 3.6 GB), 1.6 GB at the peak of a 2.4K-token prefill.
- **Prefill on the M5 neural accelerators** (MetalPerformancePrimitives `matmul2d`, the whole prompt as one batch up to 4096
  tokens). What made it 2x faster again (1085 -> 2200 tok/s), roughly in order of size:
  - *One tile shape for every GEMM, one tensor op per simdgroup.* A threadgroup stages a 64-row x 32-k weight tile per k-step
    (dequantized to fp16, double-buffered) and each of its simdgroups runs its own `matmul2d` on its own 16 tokens, activations
    streamed as fp16 from device memory, accumulators in registers: ~30 TFLOPs of the ~33 the hardware does with no dequant at
    all. Simdgroups whose tokens do not exist skip the matmul, so the ragged per-expert token counts of the MoE cost
    16-token granularity instead of whole padded tiles (routing on the GPU: counting sort + tile lists + indirect dispatch).
  - *Dispatch order is a cache policy.* Metal sweeps the grid's y axis fastest. Tiles that share activations must be
    neighbours in y, or every threadgroup re-reads its input from DRAM: with the axes the wrong way round, a GEMM whose
    activations outgrow the cache drops from 28 to 16 TFLOPs (bound by memory bandwidth, not by the tensor units), which is
    what had made batches larger than 1024 tokens slower instead of faster.
  - *Whole-prompt batches.* The MoE is the bulk of the work and its efficiency is tokens-per-expert: 2394 tokens give each
    expert ~150 instead of ~64 at the old 1024-token chunks.
  - *Short-latency weight staging.* Every simdgroup of a tile waits for the staging of the next weight tile, so what matters
    is its dependency chain, not its op count: the Q4_K block header is kept in registers and a k-step is one 16-byte load.
  - *Tensor results belong in registers.* A `matmul2d` that writes to threadgroup memory is ~2x slower than the same op into
    register accumulators followed by a store; attention went from 88 to 53 ms per 1024 tokens with that alone. The
    relative-position bias is one more small tensor op (R.B^T for all distances a tile can see) and a skewed lookup.
  - *Epilogues in registers.* The accumulator layout is fixed (verified at load), so SiTU x routing weight, the half
    conversions, the attention gate and the loop controllers' gain/forget apply all happen on the accumulators before
    anything is written; the mix and controller kernels emit their fp16 outputs directly (no conversion passes).
  - *The last layer only matters for the last tokens.* Nothing downstream reads it except the final logits and 3 rows of
    conv state, so its queries, attention, output projection and FFN run for 8 tokens instead of all (K/V still for all).
  Probabilities are carried in fp16 pre-scaled by e^8 — without that, small probabilities underflow fp16 and the first-token
  KL vs llama.cpp was 0.04; with it, 0.0001.

### Prefill on Apple M4 vs. M5

Apple M5 and Apple M4 run Metal 4's batch matrix-multiply operations on different hardware blocks inside the GPU, which changes how intermediate results are distributed across threads:

- **Apple M5** executes them on dedicated **Neural Accelerators**, giving each GPU thread **groups of four** consecutive values at a time.
- **Apple M4** executes them on its **8×8 matrix units**, giving each GPU thread **pairs of two** consecutive values instead.

Originally, the startup check only recognized M5's four-value arrangement, forcing M4 onto a slower, higher-memory fallback prefill path. Even though M4 distributes values in pairs rather than fours, each thread still holds all of the companion values needed by the fused register epilogues in its own registers—just spaced differently. At load time, the engine now detects whether the GPU arranges results in fours or pairs and selects matching prefill variants that keep all epilogue work in registers with no extra shared-memory traffic.

On an M4 Max, using the native pair-arranged fast path speeds up prompt prefill by roughly **20%** (up to ~1,260 tok/s at 2.5K tokens), reduces peak prefill working memory by about **38%**, and produces identical greedy output. Single-token generation speed is unchanged because decoding is limited by memory bandwidth and uses custom matrix-vector routines rather than batch matrix-multiply units.

### Things tried that did not help (kept out of the hot path)
- **Megakernel-style fusion** (inspired by [mvm](https://github.com/Bobliuuu/mvm)): "last threadgroup finishes the job"
  (atomic completion counter + device-scope fences; verified coherent on M5 over 20K dispatches) to fold the attention
  combine, router top-k and controller norms into their producers. Correct, but no measurable gain: on Metal,
  dependent dispatches in one concurrent encoder cost ~1.5 us; the remaining small-kernel time is genuine serial
  latency, which fusion cannot remove. (mvm's own README reports the same for decode: 0.97x vs CUDA graphs.)
- **Cross-phase weight prefetch into the system-level cache** (it is ~24-32 MB and serves 330-370 GB/s when warm):
  concurrent "touch" kernels steal bandwidth and the barriers wait for them; every variant was slower.
- **Aligned repack of the Q8_0/Q5_0 expert down-projections** (lossless, on the GPU at load): ~1.5% slower — the
  odd-sized blocks were never the bottleneck.

## Where the limits are (measured)

- **Decode floor.** ~3.2 GB of weights must be read per token; the M5 Pro reads ~290 GB/s, so exact single-stream
  decode cannot go below ~11 ms/token (~90 tok/s). We are at ~15.9 ms: ~12.5 ms of weight streaming plus ~3 ms of
  inherently serial small-kernel latency (depth mixes, controllers, attention at short context).
- **Speculative decoding** is the only way past that floor, and this MoE makes it expensive: routing is strongly
  correlated between neighbouring tokens (a 4-token verification pass touches 19.5 experts/layer vs 29 if random), but a
  4-token pass still costs ~1.85x a single step in bytes. Free prompt-lookup drafts accept too rarely on this model's
  reasoning text (simulated best case 1.02-1.12x); even perfect drafts cap at ~2.2x. A trained draft model (EAGLE-style)
  would be needed — not done here.
- **Prefill.** The GEMMs are 8.9 TFLOP per 1024 tokens (20.8 TFLOP for the 2394-token prompt). The tensor units do
  ~30 TFLOPs of fp16 with no weight staging at all and one op call per tile (`inkling kpeak`), 33 with the weights in
  threadgroup memory: so the GEMMs alone cannot take less than ~270 ms per 1024 tokens (~3800 tok/s with everything else
  free). The engine takes ~455 ms: dense GEMM tiles at 29-30 TFLOPs (at the ceiling), the ragged MoE at 24-26 (one weight
  staging per expert tile, partially filled last tile), attention ~37 ms, and ~70 ms of memory-bound glue (depth mixes,
  gathers, expert sum, convolutions). There is no faster exact mode: `relaxed_precision` changes neither speed nor
  results, fp16 accumulators are slower and 0.1% off (`inkling kmode`).
  Things that measured well in isolation and lost in the kernel:
  - *the tensor units reading the quantized weights directly* (fp16 x packed 4-bit / int8 straight from the GGUF: 32 TFLOPs
    at any tile size, exact). Q4_K scales change every 32 weights per row, so every k-step needs its partial products pulled
    out of the accumulators and rescaled; that extraction costs more than dequantizing (23 vs 30 TFLOPs);
  - int8 x int8 (57-61 TOPS raw): needs quantized activations (inexact) and the same per-block rescale;
  - wider tiles (256+ tokens per staging), larger k-steps, static extents, spreading the dequantization over more
    threads, pre-touching weight rows, reading MoE inputs through the routing permutation instead of a gathered copy.
- **K/V cache.** The payload cannot shrink without changing its numbers: the cached fp16 keys and values carry 13.7 bits of
  entropy per 16-bit value (mantissas are uniform noise, there are no zeros; xz saves 10%; `inkling kvdump` writes the payload
  out). The floor for this model is min(context, 1024) x 160 KiB + context x 32 KiB, and the cache is now within 6% of it.
  A buffer cannot be left partly unbacked either: Metal makes a no-copy buffer over untouched anonymous memory resident
  in full the first time the GPU uses it (`inkling lazytest`: 1 GiB resident after a one-word write), and the process
  footprint does not count those pages, so footprint numbers for such a buffer would look smaller than the memory used.
  Nothing exact gets near half of that: split into bytes, the low byte of every cached value (8 mantissa bits, exactly half
  the cache) does not compress at all (xz: 100.0%), nor do the 2 mantissa bits above it, so 10 of every 16 bits are
  irreducible and any exact store is above 62.5%; sign and exponent bring the practical floor to 85%. Every key dimension is
  read by a query head (`inkling qclip`: no zero query scales), so no dimension can be left out. Also measured and ruled out: the three loop passes' K/V differ by 50-106% (no sharing);
  K and V are full rank (a rank-64 subspace loses 5-20% of the energy); the full-attention layers really use their history
  (bounding it to the last 2048 / 1024 / 512 keys leaves 22 / 18 / 17 of 25 next-token choices intact); attention there is
  diffuse (`inkling kvscreen`: 69% of all keys score within 12 logits of the best one, 96% within 17), so no exact scheme can
  skip keys either.
- **Dropping keys instead of bits (measured, not in the engine yet).** `inkling kvprune -f prompt.txt` prefills a prompt,
  decodes with the whole cache, then decodes the same tokens again with only a share of the full-attention layers' far keys
  (older than one window; they carry no position information, so they are an unordered set): the ones that received the
  most attention from a sample of the prompt's own queries. At 14.5K tokens, keeping half of them takes the cache from 614 to
  404 MiB with KL 0.003 mean / 0.016 worst and 64/64 next-token choices unchanged; a quarter takes it to 298 MiB (KL 0.009 /
  0.071, 63/64). On a 16.2K-token source file half costs 3-4 of 64 next-token choices (KL 0.002 / 0.03). Decode gets 3-8%
  faster (fewer keys). Random eviction is 10x worse, scores from distant queries matter most, and nothing is saved below
  ~1K tokens, where every key is inside the window.
- **`--kv-disk` halves the cache's memory without touching a number, and pays for it in generation speed.** Since the
  values cannot be made smaller, half of them are moved out of memory: while generating, each of the 48 cache slots keeps
  only its newest keys in a ring, and every older key a layer still attends to lives in an unlinked temporary file (4 KiB
  per position and slot, uncached I/O) and is read back for every token into a small staging buffer, two halves in turn so
  the GPU works on one while the next reads land in the other. Memory = hot rings + staging <= half of the in-memory cache
  at every context from 512 tokens up (process footprint while generating at 15K tokens: 952 -> 622 MB). The attention
  keeps the split boundaries of the in-memory path (the file runs one split ahead of the ring so a boundary always lies
  where both have the keys), so every partial sum covers the same keys in the same order: logits are bit-identical
  (`inkling check` prints a hash of all logit bits), and greedy, sampled and multi-turn outputs are byte-identical to the
  in-memory engine. The cost is the SSD: half the cache is read per token at 7-9 GiB/s.

  | context | in memory | `--kv-disk` memory | decode in memory | decode `--kv-disk` |
  |---|---|---|---|---|
  | 2.5K tokens | 246 MiB | 118 MiB | 58.7 tok/s | 36.3 tok/s |
  | 5.5K tokens | 336 MiB | 164 MiB | 57.4 tok/s | 30.8 tok/s |
  | 15K tokens | 642 MiB | 314 MiB | 53.2 tok/s | 17.8 tok/s |

  Not covered: a prefill brings the whole cache back into memory for its duration (it sits next to the 1-1.6 GB prefill
  scratch then) and prefill speed is unchanged; every generated token writes its 192 KiB of K/V to the SSD once.
- **`--kv8`, the half-size cache, is a quantized cache.** It stores one signed byte per value and a per-vector scale that takes
  no space (its 8 bits ride in the lowest bits of the vector's first 8 bytes), so it is exactly half at every context length
  (120 instead of 240 MiB at 2.4K tokens, 336 instead of 672 MiB at 16K) at unchanged speed (the tensor units take int8
  operands natively). It is off by default because it is not the model's numbers any more. Against the llama.cpp reference
  logits: KL 0.0013 mean / 0.012 worst after a short prompt (fp16: 0.00001 / 0.00014), 0.0030 / 0.023 after a 2.4K-token
  prompt (fp16: 0.0008 / 0.0069); all 66 checked next-token choices agree; a greedy generation departs from the fp16 one
  after about 20 tokens and stays coherent. This model reacts to cache precision more than the usual 8-bit-cache experience
  suggests: the same check run token by token flips one prediction inside the prompt (KL 2.9), and in simulation
  (`INK_KVQ=kbits,vbits inkling check`, which rounds the fp16 cache in place) even 12 bits per value flip two of 61.
- Multi-sequence batching would roughly double aggregate decode throughput at ~16 concurrent streams (not implemented:
  this engine targets single-stream latency).

## Layout

```
src/main.rs           CLI (chat / run / bench / check + dev tools: prof, prof_pf, kbench, kmoe, bw, debug)
src/session.rs        context management, chunked prefill, pipelined greedy decode, sampling, output rendering
src/engine.rs         weight binding, GPU memory layout, per-token decode schedule and batched prefill schedule
src/kernels.metal     decode kernels
src/prefill.metal     batched / tensor-core prefill kernels
src/bench_kernels.metal  matvec variants for micro-benchmarks
src/tokenizer.rs      byte-level BPE + hand-written GPT-4o pre-tokenizer
src/gguf.rs           zero-copy GGUF reader
tools/ref_logits.cpp  llama.cpp reference-logit dumper (validation oracle)
```

`inkling prof -n POS` / `inkling prof_pf -f prompt.txt` attribute GPU time per kernel class (skip-mask method);
`inkling kmoepf -f prompt.txt` times the MoE GEMMs alone on the prompt's real routing, `inkling ksg` the tile kernel vs
threadgroup size and fill, `inkling kpeak` / `kmode` the tensor units' ceiling and precision modes, `inkling kvdump` dumps the K/V payload, `inkling kvscreen` measures how
concentrated full attention is over its keys, `INK_KVQ=kbits,vbits inkling check` simulates a coarser cache; `INK_TIER_MIN=n` / `INK_TIER_HOT=n` make `--kv-disk` start at a smaller context / with tiny rings for tests. `INK_CHUNK=N` caps the prefill batch (default 4096 tokens). If the tensor ops' accumulator layout is not the one the register epilogues expect, the engine
falls back to a slower tensor path (`INK_NO_HGEMM=1` forces it); `INK_NO_TENSOR=1` uses simdgroup matrices instead of the
neural accelerators.
