# qmog.cpp

A compact C++ inference engine optimized for Apple platforms.

Load a single `.mog` (Model Object Graph) file and run inference locally. No runtime dependencies. A small C++ codebase focused on readability and simplicity.

## Supported models

Benchmarks on M4 MacBook, Q8F16 (~10 GB).

| Model | tok/s | perplexity |
|-------|------:|-----------:|
| [Mistral-7B-v0.1 Q8F16](https://huggingface.co/QmogAI/Mistral-7B-Q8F16) | 5.73 | 5.24 |

## Run it

Built and tested on macOS. Linux builds are supported but not the primary target.

1. Clone this repo and the pre-exported model from Hugging Face:

```bash
git lfs install
git clone https://github.com/ryanssenn/qmog.cpp.git
git clone https://huggingface.co/QmogAI/Mistral-7B-Q8F16
cd qmog.cpp
```

The model repo contains `mistral-7B-Q8F16.mog`, a single file with config, tokenizer, and Q8F16 weights (~10 GB). Requires `git-lfs`.

2. Build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

3. Run:

```bash
./build/qmog-cli ../Mistral-7B-Q8F16/mistral-7B-Q8F16.mog "Paris is the capital of" --temp 0.7
```

Use `--temp 0` for greedy decoding.

To export your own `.mog` from a Hugging Face checkpoint, see `export_mistral.py`.

## Testing

Perplexity is the main correctness check. `./perplexity.sh` runs the engine against a Hugging Face reference on a fixed prompt. Regenerate unit-test goldens with `python scripts/test/mistral/goldens.py`.

```bash
./perplexity.sh
./perplexity.sh --check
./build/test_exec
```
