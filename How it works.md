# How it works

If you are new to LLM internals, an inference engine is essentially a loop that predicts the next word in a sequence, adds it to the history, and repeats. Here is the full lifecycle:

## Loading
Before we can run any math, we need the weights. The `export_mistral.py` script takes the complex Hugging Face folder structure and packs the weights into a single standardized binary file. The C++ engine loads this entire file into RAM at startup so the data is mapped and ready for computation.

## Tokenization
The model performs math on numbers, not strings. When you type a prompt like "Paris is", the `tokenizer` breaks it down using byte-pair Encoding (BPE). It looks up these chunks in the Mistral vocabulary and converts them into a list of integer IDs (e.g. [1, 782, 312]).

## Transformer Loop
We feed these IDs into the model one by one. The goal is to update a single vector, the `hidden_state`, as it passes through the network.

* **Embedding:** We take the input `token ID` and look up its specific floating-point vector in the embedding table. This turns a simple integer into a dense vector representing the token's initial semantic meaning.
* **Layers:** This state travels through 32 identical layers. In every layer, we first apply `RMSNorm` to stabilize the numbers. Then the state enters the `attention` module. It projects the state into `query`, `key`, and `value` vectors. The query "looks back" at the Keys of previous tokens to find relevant information (values). We apply `RoPE` (Rotary Positional Embeddings) so the model understands relative distance between words, then store the key and value in the `KV Cache`. This cache acts as the model's short-term memory, saving us from recalculating the history for every new word.
* **MLP:** Finally, the state goes through the `feedforward` module (a SwiGLU block). If Attention gathers context from the past, the MLP processes that information. It projects the vector to a higher dimension (14,336) to untangle complex relationships, applies a non-linear activation (SiLU), and projects it back down.

## Prediction
After 32 layers of processing, the final `hidden_state` holds the "meaning" of the next predicted token. We project this vector against the entire vocabulary to get `logits` raw confidence scores for all 32,000 possible next tokens. We run a `softmax` operation to turn these scores into probabilities and `sample` the result (either choosing the most likely token or picking randomly based on the probability distribution). We decode that ID back into text, print it, and feed it back into the transformer.

## Source files

The C++ engine lives under `src/`. Each file maps to one layer of the stack above.

### `src/main.cpp`
The CLI entry point. Parses arguments (`<model_path>`, `<prompt>`, `--temp`, `--ppl`), loads the model, tokenizes the prompt, and either runs text generation or teacher-forced perplexity. Generation prefills the prompt through the model, then loops up to 70 tokens: forward pass, repetition penalty, sampling (greedy or temperature-scaled multinomial), decode, and stream to stdout.

### `src/loader/parameters.cpp`
Loads `mistral.mog`. Parses the MOG (Model Object Graph) header for config, vocabulary, merges, and tensor offsets; memory-maps the weight blob; and exposes `Tensor` views for every layer weight. Also wires up the tokenizer tables from the header.

### `src/tokenizer/tokenizer.cpp`
Mistral-compatible BPE tokenizer. Applies Metaspace pre-tokenization, merges token pairs by rank, and falls back to per-byte vocab entries (`<0x0A>`, etc.) for bytes not covered by merges. `decode` reverses metaspace spacing and byte-fallback tokens back into UTF-8 text.

### `src/common/tensor.cpp`
Lightweight strided tensor views over raw memory. Supports f32 weights directly and int8 weights with per-group scale arrays for on-the-fly dequantization during matmul.

### `src/backend/cpu/kernels.cpp`
Core math ops: matrix multiply (f32 and int8), row matmul, softmax, RoPE, and elementwise helpers. Uses OpenMP for parallelism and SIMD dot products (NEON on Apple Silicon, AVX2 on x86).

### `src/model/mistral/modules.cpp`
The transformer itself. Implements `Embedding`, `RMSNorm`, `RotaryEmbedding`, grouped-query `Attention` (with KV cache), SwiGLU `MLP`, stacked `Layer`s, the full `Model` forward pass, and the `LM Head` that writes vocabulary logits into `InferenceState`.

## Roadmap

### Loading

#### Model Loader
- [x] **Model binary converter** *(export_mistral.py)*  
  Converts a Hugging Face Mistral model (config, vocab/merges, and weights) into a single standardized MOG file (`mistral.mog`), optionally applying quantization. It uses a compact MOG binary header to store model metadata, vocabulary, merges, and tensor index information, followed by all model weights packed sequentially as contiguous floating point data.
- [x] **In-memory loader** *(src/loader/parameters.cpp)*  
  Memory-maps the binary, loads the config and provides direct tensor views.

#### Tensor & Ops
- [x] **Tensor** *(src/common/tensor.cpp)*  
  Implements a strided view over memory supporting f32 and int8, with on-the-fly dequantization during compute
- [x] **Math operations** *(src/backend/cpu/kernels.cpp)*  
  Implementation of matmul, softmax and RoPE to be optimized later

### Text In, Tokens Out

#### Tokenizer
- [x] **Tokenizer** *(src/tokenizer/tokenizer.cpp)*  
  Implements full byte-pair encoding (BPE) compatible with Mistral's vocabulary. It loads tokenizer.json, builds vocab and merge maps, applies Metaspace pre-tokenization, encodes UTF-8 text by merging token pairs by rank, and supports byte fallback

#### Token Generation & Sampler
- [x] Basic text completion with greedy decoding
- [x] Multinomial sampling
- [x] Temperature scaling

#### CLI I/O
- [ ] Build a terminal chat inferface

### Core Transformer
The architecture *(src/model/mistral/modules.cpp)* is broken into independent C++ structs using a shared inference state to manage memory and cache. The implementation encodes relative positions using rotary embeddings (RoPE), applies gated SwiGLU in the feed-forward layers, and utilizes grouped-query attention (GQA) which assigns multiple query heads to share a single key-value head pair.

#### Inference State
- [x] Temporary memory and cache *(include/inference_state.h)* used to hold all intermediate tensors for a single token's computation during the forward pass

#### Modules
- [x] **Embedding** - Looks up initial embedding from token and copies it to `infer.hidden_state`
- [x] **RMSNorm** - Initializes inverse frequencies based on rope theta and generates cosine/sine tables dynamically based on the current `infer.pos`
- [x] **Rotary Embedding** - precomputes inverse frequencies from rope_theta and fills cos/sin tensors for RoPE for each position
- [x] **Attention** - Projects to Q/K/V, applies rotary embeddings to Q/K, pushes to the KV cache, runs the grouped-query attention mechanism (reusing KV heads 4x), and projects the result.
- [x] **Feedforward MLP** - Implements the SwiGLU feedforward: linear projections + SiLU
- [x] **Layer** - Runs norm, attention, and MLP with residuals around each subblock
- [x] **Model** - Embeds input token and runs it through all decoder layers
- [x] **LM Head** - Projects the final `infer.hidden_state` onto the vocabulary dimension to populate `infer.logits`

#### Parity Tests
- [x] Comprehensive validation in *(test/mistral)* of all inference components (tokenizer, modules, ops) by checking that their outputs match those produced by the Hugging Face Mistral implementation

### Gotta go fast

#### Quantization
- [ ] Support fp8 with a cast during model export
- [x] Q8_K Per-group symmetric quantization - split tensor into groups, for each group, finds max abs value, computes scale and produces quantized weights

#### CPU Multithreading
- [x] OpenMP parallel matmul rows, attention heads, and row_matmul

#### SIMD
- [x] NEON (Apple Silicon) and AVX2 (x86) dot products for f32 and int8 matmul

#### Custom CUDA Kernels
- [ ] Todo
