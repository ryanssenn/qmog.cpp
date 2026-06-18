# mistral.cpp

From-scratch C++ implementation of Mistral 7B for CPU inference.

The goal of this project is to understand how modern LLM inference works by building the major pieces directly: model loading, tokenization, transformer execution, KV caching, quantization, and text generation.

Current features:

* Mistral 7B weight loading
* SentencePiece tokenization
* Rotary positional embeddings (RoPE)
* RMSNorm
* Grouped-query attention
* KV cache
* Greedy and temperature-based sampling
* Export pipeline from Hugging Face weights
* Float32 and int8 inference paths
* Validation against Hugging Face reference outputs

This is an educational project focused on correctness and understanding rather than production deployment.

Independent project, not affiliated with Mistral AI.

<br>

![demo2](https://github.com/user-attachments/assets/1711dc3e-9ab2-4f73-8c35-b7ac3aabec55)

# Running

| | Minimum |
|---|---|
| RAM | 16 GiB |
| Disk | 40 GiB |
| Python | 3.10+ |
| CMake | 3.20+ |
| Compiler | C++17 (gcc 11+ or clang) |

These commands assume the Mistral Hugging Face checkout and this repo are sibling directories:

```text
parent-directory/
  Mistral-7B-v0.1/
  mistral.cpp/
```

#### 1. Download Mistral 7B v0.1 and mistral.cpp

```bash
git lfs install
git clone https://huggingface.co/mistralai/Mistral-7B-v0.1
git clone https://github.com/ryanssenn/mistral.cpp.git
cd mistral.cpp
```

If the model download fails, make sure your Hugging Face account has access to `mistralai/Mistral-7B-v0.1` and that Git LFS is installed.

#### 2. Create the Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 3. Export the model binary expected by the C++ app and tests

The app and tests read quantization mode from the binary header, so export the format you want. Both paths write to `./mistral.bin` in the repo root (re-export to switch formats).

**Default - int8 (smaller, faster inference)**

Per-group symmetric int8 quantization on MLP gate/up weights (~18 GB). `down_proj`, attention, embeddings, norms, and `lm_head` stay f32 for generation quality.

```bash
python3 export_mistral.py \
  --model_dir ../Mistral-7B-v0.1 \
  --out ./mistral.bin
```

**Option - f32 (full parity tests)**

Full-precision weights (~27 GB). Best for validating correctness against Hugging Face. Runs 21 parity tests.

```bash
python3 export_mistral.py \
  --model_dir ../Mistral-7B-v0.1 \
  --out ./mistral.bin \
  --quant f32
```

Expected result (either option):

```text
Completed
```

#### 4. Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Expected result:

```text
Built target mistral.cpp
Built target test_exec
```

#### 5. Run text completion

Same command for f32 and int8 - the runtime picks the path from the binary header.

```bash
./build/mistral.cpp ./mistral.bin "Paris is the capital of" --temp 0.7
```

`--temp` controls sampling: `0` is greedy (default), values like `0.7` add randomness. A repetition penalty is applied on every step regardless of temperature, so even greedy decoding discourages repeats.

The program prints up to 50 generated tokens and then a throughput line like:

```text
throughput: <number> tok/s
```

The default int8 export is much faster on CPU than f32. Use f32 mainly for correctness checks.

# Testing

The implementation is validated against Hugging Face reference outputs at multiple levels, including tokenizer behavior, CPU kernels, decoder modules, hidden states, and logits.

Reference tensors are generated from the Hugging Face Mistral implementation and compared against the corresponding C++ outputs. The test suite supports both f32 and int8 model exports.
Run the tests from the repo root after creating `./mistral.bin`:

```bash
cmake --build build --target test_exec
./build/test_exec
```

**Expected result (default int8 export)** — the logits diagnostic fails on int8:

```text
====================================================
  mistral.cpp · test suite            model: int8
====================================================

  ✗  test logits multi top10           17287.3 ms
          [paris] step 0 top1 f32=4843 int8=4843 | top1=OK  top10_overlap=7/10
          [paris] step 5 top1 f32=9504 int8=9504 | top1=OK  top10_overlap=3/10
          [sky] step 1 top1 f32=4672 int8=3534 | top1=FLIP top10_overlap=5/10
          [sky] step 3 top1 f32=28723 int8=28725 | top1=FLIP top10_overlap=5/10
          [sky] step 4 top1 f32=415 int8=13 | top1=FLIP top10_overlap=7/10
          [sky] step 5 top1 f32=4376 int8=3181 | top1=FLIP top10_overlap=3/10
  ✓  test layer stack prefill              0.0 ms
  ✓  load config                           0.0 ms
  ✓  load weights                          0.3 ms
  ✓  test attention feedforward mlp        4.8 ms
  ✓  tokenizer encode                      0.5 ms
  ✓  tokenizer encode fallback             0.0 ms

----------------------------------------------------
  FAILED   6 / 7        1 failed        17293.0 ms
====================================================
```

**Expected result (f32 export):**

```text
====================================================
  mistral.cpp · test suite             model: f32
====================================================

  ✗  test logits multi top10           17000.0 ms
  ✓  test layer stack prefill              0.0 ms
  ✓  test rope                             0.0 ms
  ✓  test matmul                           0.0 ms
  ✓  test row matmul                       0.0 ms
  ✓  test softmax                          0.0 ms
  ✓  test silu                             0.0 ms
  ✓  load config                           0.0 ms
  ✓  load weights                         22.9 ms
  ✓  test layer                           45.1 ms
  ✓  test attention                        2.1 ms
  ✓  test attention feedforward mlp       48.7 ms
  ✓  test kv cache                         1.4 ms
  ✓  test embedding                        0.1 ms
  ✓  test rotary embedding inv freq        0.0 ms
  ✓  test rotary embedding                 0.0 ms
  ✓  test rmsnorm                          0.1 ms
  ✓  test lm head                          6.3 ms
  ✓  tokenizer encode                      0.4 ms
  ✓  tokenizer encode fallback             0.0 ms
  ✓  tokenizer decode                      0.0 ms

----------------------------------------------------
  FAILED   20 / 21        1 failed        198.4 ms
====================================================
```

If you see this:

```text
Model binary open failed
```

then `./mistral.bin` does not exist at the repo root. Run the export command in step 3, or copy the exported model binary to `./mistral.bin`.

# Numerical drift (int8)

The int8 engine is much less accurate than the HuggingFace fp32 reference, and the error grows with context length (`scripts/perplexity.py` scores the engine against HF fp32, bucketed by position):

| Token positions | int8 engine perplexity | HF fp32 |
| --------------- | ---------------------- | ------- |
| 0–16    | ~32 | ~3.7 |
| 32–64   | ~54 | ~3.7 |
| 160–192 | ~90 | ~3.7 |

Only the MLP gate/up projections are quantized (per-group symmetric int8, group size 64), so the weight format is a small perturbation. The likely dominant cause is the int8 compute path (matmul accumulation, dequant, KV cache feedback) compounding per-token error.

### Solutions

1. Accumulate int8 matmuls in int32, scale once.
2. Add a kernel test comparing int8 matmul vs f32 matmul of the dequantized weights.
3. Bisect by layer with `DUMP_LAYER_STACK=1` goldens.
4. Confirm KV cache is f32.
5. Try per-token activation scaling.

# Roadmap

Full progress tracker: [ROADMAP.md](ROADMAP.md). Still todo: terminal chat interface, fp8, SIMD, CUDA.


# Resources

Reading and reference material used while building mistral.cpp.

### Machine learning theory

- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) - Original transformer paper
- [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) - Andrej Karpathy
- [Rotary Embeddings](https://www.youtube.com/watch?v=V8r__fXx7tU) - RoPE walkthrough

### Systems and performance

- [PyTorch Internals](https://blog.ezyang.com/2019/05/pytorch-internals/) - Edward Z. Yang
- [C++ Vtables](https://shaharmike.com/cpp/vtable-part1/) - Shahar Mike
- [yalm](https://andrewkchan.dev/posts/yalm.html) - Andrew Chan
- [LLM inference speed of light](https://zeux.io/2024/03/15/llm-inference-sol/) - Arseny Kapoulkine
- [Quantize llama models with ggml and llama.cpp](https://medium.com/data-science/quantize-llama-models-with-ggml-and-llama-cpp-3612dfbcc172) - Maxime Labonne

### Reference implementations

- [Hugging Face Mistral model](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py)
- [calm](https://github.com/zeux/calm/tree/main) - Arseny Kapoulkine
- [llama.cpp](https://github.com/ggml-org/llama.cpp/) - Georgi Gerganov
- [llama2.c export and quantization](https://github.com/karpathy/llama2.c/blob/master/export.py) - Andrej Karpathy

