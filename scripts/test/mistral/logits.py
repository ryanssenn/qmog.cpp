"""
Generate test/mistral/logits_expected.txt for multi-token logits and layer-stack tests.

Matches mog-cli greedy decoding (temp=0).

Weights load in bfloat16 (Mistral's native dtype) so the ~7B model fits in RAM
without swapping; a full f32 copy is ~28 GB and thrashes on a 32 GB machine.
Override with LOGITS_DTYPE=float32 / LOGITS_DEVICE=cpu if desired.

Layer stack dumps are off by default (set DUMP_LAYER_STACK=1 to enable).

Usage (from repo root, with ../Mistral-7B-v0.1 present):
  python scripts/test/mistral/logits.py
"""
import gc
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
MODEL = os.environ.get(
    "MISTRAL_MODEL",
    os.path.abspath(os.path.join(REPO_ROOT, "../Mistral-7B-v0.1")),
)
OUT = os.path.join(REPO_ROOT, "test/mistral/logits_expected.txt")
PROMPTS = {
    "sky": "The color of the sky is",
    "paris": "Paris is the capital of",
}
NUM_GEN_STEPS = 5
TOPK = 10
# Off by default: output_hidden_states for 32 layers needs a lot of extra memory.
DUMP_LAYER_STACK = os.environ.get("DUMP_LAYER_STACK", "0") == "1"


def dump_vector(f, name, tensor):
    f.write(name + "\n")
    f.write(" ".join(str(float(v)) for v in tensor.flatten()) + "\n")


def dump_topk(f, prefix, step, ids, vals):
    dump_vector(f, f"{prefix}_step{step}_top{TOPK}_ids", torch.tensor(ids, dtype=torch.float32))
    dump_vector(f, f"{prefix}_step{step}_top{TOPK}_vals", torch.tensor(vals, dtype=torch.float32))


def topk(logits):
    vals, ids = torch.topk(logits, TOPK)
    return ids.tolist(), [float(v) for v in vals.tolist()]


@torch.inference_mode()
def dump_logits_trace(f, model, input_ids, prefix):
    # Explicit greedy loop with plain forward calls. We avoid model.generate()
    # (it can hang on MPS) and KV cache; recomputing the short sequence each step
    # is cheap and mirrors the teacher-forced loop in test_logits.cpp exactly.
    seq = input_ids
    for step in range(NUM_GEN_STEPS + 1):
        attention_mask = torch.ones_like(seq)
        out = model(seq, attention_mask=attention_mask, use_cache=False)
        logits = out.logits[0, -1]

        kid, kval = topk(logits)
        dump_topk(f, f"logits_{prefix}", step, kid, kval)

        next_id = torch.tensor([[kid[0]]], dtype=seq.dtype, device=seq.device)
        seq = torch.cat([seq, next_id], dim=1)


@torch.inference_mode()
def dump_layer_stack(f, model, input_ids, prefix):
    attention_mask = torch.ones_like(input_ids)
    pre_norm = {}

    def capture_pre_norm(_module, inp, _output):
        # HF hidden_states[num_layers] equals last_hidden_state (post-norm), not the
        # final decoder block output. Hook the norm input for layer L{n-1} goldens.
        pre_norm["last"] = inp[0][0, -1].detach().float().cpu()

    handle = model.model.norm.register_forward_hook(capture_pre_norm)
    out = model.model(
        input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    handle.remove()

    n_layers = model.config.num_hidden_layers
    for layer in range(n_layers):
        if layer == n_layers - 1:
            dump_vector(f, f"layer_stack_{prefix}_L{layer}", pre_norm["last"])
        else:
            dump_vector(f, f"layer_stack_{prefix}_L{layer}", out.hidden_states[layer + 1][0, -1])

    dump_vector(f, f"layer_stack_{prefix}_norm", out.last_hidden_state[0, -1])


def main():
    if not os.path.isdir(MODEL):
        raise SystemExit(f"Model directory not found: {MODEL}")

    device = os.environ.get("LOGITS_DEVICE")
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = DTYPES[os.environ.get("LOGITS_DTYPE", "bfloat16")]
    print(f"Loading model from {MODEL} on {device} as {dtype}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    with open(OUT, "w") as f:
        f.write(f"# Golden values from Hugging Face Mistral-7B-v0.1 ({dtype})\n")
        f.write("# Regenerate: python scripts/test/mistral/logits.py\n\n")
        f.flush()

        for prefix, prompt in PROMPTS.items():
            print(f"Processing {prefix!r}", flush=True)
            ids = tokenizer.encode(prompt, add_special_tokens=True)
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)

            dump_logits_trace(f, model, input_ids, prefix)
            f.flush()

            if DUMP_LAYER_STACK:
                dump_layer_stack(f, model, input_ids, prefix)
                f.flush()

            if device == "mps":
                torch.mps.empty_cache()
            gc.collect()

    del model
    gc.collect()
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
