# F16 compute roadmap

This document describes how qmog.cpp will evolve from **f16 storage + f32 compute** (current) toward **native f16 compute on Apple ARM**, while keeping perplexity as the stability gate.

For the on-disk format, see [model-binary.md](model-binary.md).

---

## Current state (phase 1)

**Q8F16** export stores:

| Weights | Format |
|---------|--------|
| `mlp.gate_proj`, `mlp.up_proj` | int8 + per-group f32 scales |
| Everything else | IEEE float16 (2 bytes/elem) |

At runtime:

- Weights are mmap'd as f16 views (zero-copy).
- Activations, KV cache, softmax, RoPE, and SiLU all run in **float32**.
- F16 weights are promoted to f32 at use time (`fp16_to_f32` in matmul and `Tensor::get` for embedding/norm lookups).

This gives an immediate file-size win (~18 GB → ~10 GB for Mistral 7B) without changing numerical behavior beyond the bf16→f16 cast at export.

---

## Why Apple ARM is the target

Apple Silicon (M-series) runs fp16 SIMD at full rate:

- 8-wide fp16 loads (`float16x8_t`)
- Native f16 FMA (`vfmaq_f16`)
- Fast f16↔f32 conversion (`vcvt_f32_f16`, `vcvt_f16_f32`)

x86 Linux builds keep f32 compute with a scalar promote fallback. F16 compute optimizations are gated behind `#if defined(__ARM_NEON)` and tested on Apple hardware.

---

## Iteration phases

Each phase is a separate PR. Do not skip the validation loop.

| Phase | What stays f16 | Expected gain | Stability risk |
|-------|----------------|---------------|----------------|
| **1 (now)** | Weights only | ~45% smaller `.mog` | Low |
| **2** | Matmul inner loop (f16×f16 MAC, f32 output) | Moderate tok/s | Low–medium |
| **3** | Attention QK dot + softmax input | Moderate | Medium |
| **4** | KV cache in f16 | Memory bandwidth | Medium–high |
| **5** | Full f16 activations through MLP | Highest tok/s | High |

Phases 4–5 need extra scrutiny: error can compound over context length and through residual connections.

---

## Speed vs stability gates

Every phase must pass this loop before merge:

```
change kernel → unit test (kernel golden) → perplexity.sh vs baseline → tok/s benchmark
```

**Stability gate:** `./perplexity.sh` must stay within tolerance recorded in `perplexity_baseline.json`. A perplexity regression blocks merge unless traced to a pre-existing issue.

**Speed gate:** Measure tok/s on target hardware (M4 MacBook, same prompt/temp as README). A phase only ships if it measurably improves tok/s or memory without failing stability.

**Rollback:** Each phase should be independently toggleable (compile flag or runtime config) so regressions can be bisected without reverting f16 storage.

---

## Kernel candidates (ordered by payoff / risk)

1. **`dot_f16_f16`** — f16 weight × f16 activation, f32 output. Covers the largest matmuls: q/k/v/o proj, down_proj, lm_head.
2. **`row_matmul` f16 variant** — attention V weighted sum when activations are f16.
3. **Mixed-precision RMSNorm** — f16 hidden state with f32 reduction for sum-of-squares.
4. **F16 KV cache** — store K/V as f16; promote to f32 at attention dot products.

Phase 1 already implements `dot_f16_f32` (f16 weights, f32 activations) with an ARM NEON widen path in `kernels.cpp`.

---

## Open questions (resolve per phase)

- Per-layer vs global f16 activation policy.
- Whether residual connections must stay f32.
- Acceptable PPL delta per phase (suggested starting point: ≤0.05 for phases 2–3; evaluate case-by-case for 4–5).

---

## Explicitly deferred

- bfloat16 on-disk or bf16 compute (HF Mistral source is bf16; we store IEEE f16).
- GPU / Accelerate / Metal (CPU-only for now).
- Automatic mixed-precision (manual phase-by-phase only).
