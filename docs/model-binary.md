# MOG format (`mistral.mog`)

MOG (**M**odel **O**bject **G**raph) is the single-file format this engine loads at startup. One `.mog` file holds everything the runtime needs: model config, tokenizer, a tensor index, and all weight bytes.

`export_mistral.py` builds it from a Hugging Face checkpoint; `Parameters::load_parameters` in C++ memory-maps it and exposes tensor views into the payload.

```bash
python3 export_mistral.py --model_dir ../Mistral-7B-v0.1 --out ./mistral.mog
```

**Inputs from HF:** `config.json`, `tokenizer.json`, `model.safetensors.index.json` + shard files.

**Output:** `mistral.mog` (~18 GB for Mistral 7B int8).

---

## What’s in the file?

Think of a `.mog` file as three parts stacked back-to-back:

```
┌─────────────────────────────────────────┐
│  File prefix (16 bytes)                 │  magic, version, header length
├─────────────────────────────────────────┤
│  Header (metadata)                      │  architecture, config, tokenizer, tensor table
├─────────────────────────────────────────┤
│  Padding (0–63 bytes)                   │  align next section to 64-byte boundary
├─────────────────────────────────────────┤
│  Weight payload (~18 GB)                │  raw tensor bytes; mmap'd, not copied
└─────────────────────────────────────────┘
```

The header is small (a few MB, mostly tokenizer vocab). The payload is almost the entire file. Tensor entries in the header store **offsets into the payload**, so the loader can jump straight to each weight without scanning.

---

## File prefix (first 16 bytes)

All multi-byte integers are **little-endian**.

| Offset | Size | Field | Value |
|--------|------|-------|-------|
| 0 | 4 | Magic | `MOG\0` (bytes `4D 4F 47 00`) |
| 4 | 4 | Version | `1` |
| 8 | 8 | `header_size` | Length of the header blob only (not including prefix or padding) |

If magic or version does not match, the loader rejects the file. Old JSON-header `mistral.bin` files are not supported.

**Payload start offset:**

```
payload_base = align_up(16 + header_size, 64)
```

---

## Header blob (read in this order)

Strings everywhere use the same encoding: **`u32` byte length** + UTF-8 bytes (no null terminator).

### 1. Architecture

One string identifying the model family, e.g. `"mistral"`. Reserved for future multi-architecture support; the current loader reads and discards it.

### 2. Config (key–value)

```
u32 count
repeat count times:
    string key
    u8  type
    value (depends on type)
```

| Type byte | Meaning | Value encoding |
|-----------|---------|----------------|
| 0 | STRING | length-prefixed string |
| 1 | UINT32 | `u32` |
| 2 | FLOAT32 | `f32` |

**Keys written by the exporter:**

| Key | Type | Example |
|-----|------|---------|
| `hidden_size` | UINT32 | 4096 |
| `intermediate_size` | UINT32 | 14336 |
| `n_layers` | UINT32 | 32 |
| `n_heads` | UINT32 | 32 |
| `n_kv_heads` | UINT32 | 8 |
| `vocab_size` | UINT32 | 32000 |
| `sliding_window` | UINT32 | 4096 |
| `max_position_embeddings` | UINT32 | 32768 |
| `rope_theta` | FLOAT32 | 10000.0 |
| `norm_eps` | FLOAT32 | 1e-5 |
| `quant` | STRING | `"int8"` or `"f32"` |

`head_dim` is not stored; C++ computes it as `hidden_size / n_heads`.

### 3. Tokenizer

```
u32 vocab_count
repeat vocab_count times:
    string token
    u32    id

u32 merge_count
repeat merge_count times:
    string merge_rule    # e.g. "▁the ▁th"
```

Loaded into `Tokenizer`: `token_to_id`, `id_to_token`, and BPE merge tables.

### 4. Tensor table

One entry per weight tensor (291 for Mistral 7B: 3 global + 32 layers × 9).

```
u32 count
repeat count times:
    string name          # full HF name, e.g. "model.layers.0.self_attn.q_proj.weight"
    u8   dtype           # 0 = f32, 1 = int8
    u8   ndim            # number of shape dimensions (1–4)
    u32  dims[4]         # shape; unused slots are 0
    u64  offset          # byte offset from payload start
    u64  scale_offset    # int8 only: offset of f32 scales in payload
    u32  scale_size      # int8 only: number of f32 scale values
```

For **f32** tensors, `scale_offset` and `scale_size` are zero.

**After load, names are routed:**

| MOG name pattern | Stored as |
|------------------|-----------|
| `model.layers.{i}.{suffix}` | `layer_weights[i]["{suffix}"]` |
| `model.embed_tokens.weight`, `model.norm.weight`, `lm_head.weight` | `global_weights["{name}"]` |

`get_tensor(layer, name)` uses the short per-layer name (e.g. `get_tensor(0, "self_attn.q_proj.weight")`). Use `layer = -1` for globals (e.g. `"lm_head.weight"`).

---

## Weight payload

Tensors are written in `weight_map` order during export. Each tensor’s data is followed by enough padding to align the **next** tensor’s offset to a 64-byte boundary.

### f32 tensors

Contiguous `float32` values. C++ wraps them as:

```cpp
Tensor<float>(float* at offset, shape)
```

### int8 tensors (default export)

Only **`mlp.gate_proj`** and **`mlp.up_proj`** are quantized per layer. Everything else (attention, `down_proj`, embeddings, norms) stays f32.

Quantization is **symmetric per group of 64** weights, mapped to `[-127, 127]`:

1. Weight bytes at `offset`
2. `scale_size = numel / 64` float32 scales at `scale_offset`

C++ copies scales into a `vector<float>` and wraps:

```cpp
Tensor<int8_t>(int8* at offset, scales, shape)
```

Matmul dequantizes on the fly using `GROUP_SIZE = 64` in `kernels.cpp`.

Use `--quant f32` at export time for an all-float file.

---

## Export pipeline

```
HF config.json        →  config KV section
HF tokenizer.json     →  vocab + merges
HF safetensors shards →  tensor table (offsets) + payload bytes
```

Steps in `export_mistral.py`:

1. Walk `weight_map` and assign each tensor an offset, dtype, and (for int8) scale location.
2. Build the header blob (architecture → config → tokenizer → tensor table).
3. Write prefix + header + padding.
4. Write tensor bytes at the assigned offsets.

---

## Load pipeline (C++)

```
open mistral.mog
mmap entire file
verify MOG magic + version
parse header blob  →  Config, Tokenizer, tensor table
payload = mmap_base + align_up(16 + header_size, 64)
for each tensor entry:
    create Tensor view pointing into payload (no copy of weights)
```

Tensors live in `global_weights` or `layer_weights[layer]` as `variant<Tensor<float>, Tensor<int8_t>>`.

When `config.quant == "int8"`, the model uses int8 matmul for gate/up projections only; attention and `down_proj` still run in f32.

---

## Quick reference

| Topic | Detail |
|-------|--------|
| Magic | `MOG\0` |
| Version | `1` |
| Default export | int8 gate/up, f32 elsewhere |
| Mistral 7B size | ~18 GB |
| Header | Small; tokenizer dominates |
| Weights | mmap'd; offsets in tensor table |
