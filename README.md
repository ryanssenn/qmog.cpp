# mistral.cpp

C++ Mistral 7B inference on CPU. On an Apple M4 Mac: 4.36 tok/s, perplexity 5.24.

<img width="1200" height="331" alt="mistral_demo" src="https://github.com/user-attachments/assets/2660a8e4-c444-44da-8e19-bd70ea76449a" />

## Run it

**Requirements:** macOS or Linux, ~16 GiB RAM, [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) (Git LFS).

**1. Clone the model and this repo** into the same parent directory:

```bash
git lfs install
git clone https://huggingface.co/mistralai/Mistral-7B-v0.1
git clone https://github.com/ryanssenn/mistral.cpp.git
cd mistral.cpp
```

**2. Build a `.mog` file.** The engine reads a single `.mog` (Model Object Graph) file with config, tokenizer, and weights. Use `export_mistral.py` to convert the Hugging Face checkpoint:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 export_mistral.py --model_dir ../Mistral-7B-v0.1 --out ./mistral.mog
```

**3. Build:**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

**4. Run:**

```bash
./build/mistral.cpp ./mistral.mog "Paris is the capital of" --temp 0.7
```

Use `--temp 0` for greedy decoding.

## Testing

Perplexity is the main correctness check. `./perplexity.sh` runs the engine against a Hugging Face reference on a fixed prompt.

```bash
./perplexity.sh
./build/test_exec
```

## Docs

[Architecture](docs/architecture.md) · [MOG format](docs/model-binary.md)
