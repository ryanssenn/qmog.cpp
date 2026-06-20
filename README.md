# mistral.cpp

From-scratch C++ Mistral 7B inference on CPU. On an Apple M4 Mac it runs at 4.36 tok/s with perplexity 5.24.

<img width="1200" height="331" alt="mistral_demo" src="https://github.com/user-attachments/assets/2660a8e4-c444-44da-8e19-bd70ea76449a" />

Educational project, not a production engine. Not affiliated with Mistral AI.

# Running

Works on macOS and Linux. You need about 16 GiB RAM. Put `Mistral-7B-v0.1` and this repo in the same parent directory. The model requires [Hugging Face access](https://huggingface.co/mistralai/Mistral-7B-v0.1) and Git LFS.

```bash
git lfs install
git clone https://huggingface.co/mistralai/Mistral-7B-v0.1
git clone https://github.com/ryanssenn/mistral.cpp.git
cd mistral.cpp
```

Export `mistral.mog` (~18 GB) from the Hugging Face weights. The file uses the MOG (Model Object Graph) binary format; see [docs/model-binary.md](docs/model-binary.md).

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 export_mistral.py --model_dir ../Mistral-7B-v0.1 --out ./mistral.mog
```

Build and run.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/mistral.cpp ./mistral.mog "Paris is the capital of" --temp 0.7
```

Use `--temp 0` for greedy decoding.

# Testing

Perplexity is the main quality check. `./perplexity.sh` compares the engine against a Hugging Face reference on a fixed prompt.

```bash
./perplexity.sh
./perplexity.sh --hf      # recompute the HF reference
./perplexity.sh --save    # update the saved baseline
```

There is also a unit test suite: `./build/test_exec`.

# Resources

- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
- [Rotary Embeddings](https://www.youtube.com/watch?v=V8r__fXx7tU)
- [PyTorch Internals](https://blog.ezyang.com/2019/05/pytorch-internals/)
- [LLM inference speed of light](https://zeux.io/2024/03/15/llm-inference-sol/)
- [Hugging Face Mistral model](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py)
- [llama.cpp](https://github.com/ggml-org/llama.cpp/)
- [llama2.c export](https://github.com/karpathy/llama2.c/blob/master/export.py)
