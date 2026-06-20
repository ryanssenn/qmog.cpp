# Architecture

mistral.cpp is a from-scratch C++ inference engine for Mistral 7B. At a high level it does one thing: **predict the next token, append it, repeat**.

```
  prompt text
       │
       ▼
  ┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
  │ Tokenizer   │ ──▶ │ Transformer (×32)│ ──▶ │ Sample next │
  │ text → IDs  │     │ ID → logits      │     │ token       │
  └─────────────┘     └──────────────────┘     └─────────────┘
       ▲                                              │
       └──────── append token, run again ─────────────┘
```

For the on-disk format, see [model-binary.md](model-binary.md).

---

## 1. Load the model

Hugging Face checkpoints are a folder of config, tokenizer, and sharded weights. `export_mistral.py` packs them into one **`.mog`** file.

At startup, `Parameters::load_parameters` memory-maps that file and exposes each weight as a strided `Tensor` view. The ~18 GB payload is not copied into heap RAM.

| Step | Where |
|------|-------|
| HF → `.mog` | `export_mistral.py` |
| mmap + tensor index | `src/loader/parameters.cpp` |
| Tensor views | `src/common/tensor.cpp` |

---

## 2. Tokenize the prompt

The model operates on integer token IDs, not strings. The tokenizer applies Mistral-compatible BPE: Metaspace pre-tokenization, merge-by-rank, byte fallback for uncovered bytes.

Example: `"Paris is"` → `[1, 782, 312]`

| Step | Where |
|------|-------|
| encode / decode | `src/tokenizer/tokenizer.cpp` |

---

## 3. Run the transformer

Each token ID flows through 32 decoder layers. A shared `InferenceState` holds the running `hidden_state`, KV cache, and output logits.

**Per token:**

1. **Embedding:** look up the token’s vector in the embedding table.
2. **For each layer:**
   - **RMSNorm:** stabilize activations.
   - **Attention (GQA):** project to Q/K/V, apply RoPE, attend over cached keys/values from prior tokens, write new K/V into the **KV cache** (so past tokens are not recomputed).
   - **MLP (SwiGLU):** two linear projections with SiLU gating, then a down projection.
   - Residual connections around attention and MLP.
3. **LM head:** final linear map to vocabulary-sized **logits**.

| Component | Where |
|-----------|-------|
| Embedding, layers, LM head | `src/model/mistral/modules.cpp` |
| matmul, softmax, RoPE, SIMD | `src/backend/cpu/kernels.cpp` |
| per-forward scratch + cache | `include/inference_state.h` |

---

## 4. Pick the next token

Logits are raw scores over ~32k vocabulary entries. The sampler turns them into a choice:

- **`--temp 0`:** greedy (argmax).
- **`--temp > 0`:** temperature-scaled softmax, then multinomial sample.

The chosen ID is decoded to text, printed, and fed back into step 3 until a stop condition or token limit.

| Step | Where |
|------|-------|
| CLI, generation loop, sampling | `src/main.cpp` |

---

## Source map

| Path | Role |
|------|------|
| `src/main.cpp` | CLI: load model, tokenize, generate or `--ppl` perplexity |
| `src/loader/parameters.cpp` | Parse `.mog` header, mmap weights |
| `src/tokenizer/tokenizer.cpp` | BPE encode/decode |
| `src/common/tensor.cpp` | f32 and int8 tensor views (on-the-fly dequant) |
| `src/backend/cpu/kernels.cpp` | Core ops (OpenMP + NEON/AVX2) |
| `src/model/mistral/modules.cpp` | Full Mistral forward pass |
| `test/mistral/` | Parity tests vs Hugging Face reference |

---

## Performance notes

- **Quantization:** export supports per-group int8 (`Q8_K`); weights dequantize during matmul.
- **CPU:** OpenMP parallelizes matmul rows and attention heads; SIMD dot products on Apple Silicon (NEON) and x86 (AVX2).
