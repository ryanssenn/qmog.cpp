# mistral.cpp

From-scratch C++ implementation of [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) base model for CPU inference.

The project was built by hand up to the first usable int8 generation path: model loading, tokenization, decoder execution, text generation, and Hugging Face validation. After that, AI tools, mostly Cursor Agent, began helping with debugging, refactors, docs, tests, and performance.

Educational project for understanding LLM inference, not a production engine.

Current status: int8 runs at ~4.9 tok/s on Apple M4; quality is still WIP due to numerical drift (int8 engine perplexity ~90 vs ~3.7 for f32, see [Numerical drift](#numerical-drift-int8)).

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

Full-precision weights (~27 GB). Best for validating correctness against Hugging Face. Runs 19 parity tests.

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

`--temp` controls sampling: `0` is greedy (default), values like `0.7` reduce repetition.

The program prints up to 50 generated tokens and then a throughput line like:

```text
throughput: <number> tok/s
```

The default int8 export is much faster on CPU than f32. Use f32 mainly for correctness checks.

# Testing

To check that the C++ code matches the real Mistral implementation, I validated each component separately rather than only checking end-to-end output.

First, the Python scripts in `scripts/test/mistral/` run individual pieces, attention, RMSNorm, RoPE, MLP, etc. using Hugging Face's Mistral with the actual weights. Each script dumps its output tensors into `test/mistral/expected.txt` as named float arrays.

Then the C++ tests in `test/mistral/` load those values and compare them against the output of the corresponding mistral.cpp code. For example, an attention test copies a known `hidden_state` from the golden file, runs `Attention::forward`, and checks that Q/K/V and the output match. The same pattern is used for the tokenizer, CPU kernels (matmul, softmax, RoPE, SiLU), and each decoder module. Comparisons use a tolerance of ±0.05.

**End-to-end logits tests** (`test/mistral/logits_expected.txt`) compare multi-token greedy decoding against Hugging Face f32 reference:

- Top-10 token IDs and logit values after the last prompt token and each of the next 5 greedy steps
- Per-layer hidden states after the last prompt token (finds where int8 drift starts)

Regenerate golden logits after changing prompts or the model (needs ~16 GB free RAM; quit any running `mistral.cpp` first):

```bash
python scripts/test/mistral/logits.py
```

Optional per-layer hidden-state goldens (much heavier on memory):

```bash
DUMP_LAYER_STACK=1 python scripts/test/mistral/logits.py
```

Run the tests from the repo root after creating `./mistral.bin`:

```bash
cmake --build build --target test_exec
./build/test_exec
```

Tests are filtered by the quantization mode in `mistral.bin`. Golden values come from Hugging Face f32 weights. int8 runs component tests plus logits/layer-stack diagnostics (these fail when int8 drift breaks generation). The project is still dealing with numerical drift in the quantized path, so int8 output quality and long generations should be treated as work in progress. Re-export with `--quant f32` to run the full 19-test component suite.

The runner prints a report (green checks on pass, red on fail) with per-test timing.

**Expected result (default int8 export):**

```text
====================================================
  mistral.cpp · test suite            model: int8
====================================================

  ✓  test logits multi top10               0.0 ms
  ✓  test layer stack prefill              0.0 ms
  ✓  load config                           0.0 ms
  ✓  load weights                          9.9 ms
  ✓  test attention feedforward mlp      290.2 ms
  ✓  tokenizer encode                      0.3 ms
  ✓  tokenizer encode fallback             0.0 ms

----------------------------------------------------
  PASSED   7 / 7        0 failed        300.4 ms
====================================================
```

**Expected result (f32 export):**

```text
====================================================
  mistral.cpp · test suite             model: f32
====================================================

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
  PASSED   19 / 19        0 failed        198.4 ms
====================================================
```

If you see this:

```text
Model binary open failed
```

then `./mistral.bin` does not exist at the repo root. Run the export command in step 3, or copy the exported model binary to `./mistral.bin`.

# Numerical drift (int8)

The int8 path loses accuracy, but **not** because of the quantization itself. Swapping the dequantized int8 gate/up weights back into the reference Hugging Face model changes perplexity by only **+0.08%** (3.726 → 3.729 on a 193-token passage), so per-group symmetric int8 is essentially lossless as a weight format.

The loss comes from the **int8 inference engine**, and it **compounds with sequence length**: each token's small error feeds the KV cache and accumulates. Early tokens stay close to the f32 reference, then perplexity climbs steeply. Measured on the same passage (f32 reference perplexity ≈ 3.7 throughout):

| Context length | int8 engine perplexity |
| -------------- | ---------------------- |
| first 16 tokens  | ~32  |
| first 32 tokens  | ~27  |
| first 64 tokens  | ~54  |
| first 128 tokens | ~87  |
| first 192 tokens | ~90  |

So the engine is roughly **24× worse** than f32 at next-token prediction over a full passage (~90 vs ~3.7), driven almost entirely by drift accumulation in the int8 matmul path rather than the quantized weights. Reducing this drift is the main open work item for the quantized path.

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

# License

MIT
