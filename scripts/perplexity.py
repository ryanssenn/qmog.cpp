"""
Compare int8-engine perplexity against the f32 Hugging Face reference, bucketed
by sequence position to show how int8 drift accumulates with context length.

Usage:
  1. Produce the engine's per-token output (token ids + per-token NLL):

       ./build/mistral.cpp ./mistral.bin "<some passage>" --ppl > /tmp/int8_ppl.txt

  2. Run this script (loads HF Mistral in fp32 and scores the SAME token ids):

       python scripts/perplexity.py

Paths assume this repo and the Mistral-7B-v0.1 checkout are sibling directories.
"""

import re
import math
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "../Mistral-7B-v0.1"
INT8_OUT = "/tmp/int8_ppl.txt"

# Parse the engine's --ppl output: token ids and its per-token NLL.
text = open(INT8_OUT).read()
token_ids = [int(x) for x in re.search(r"token_ids:(.*)", text).group(1).split()]
int8_nll = [float(x) for x in re.search(r"per_token_nll:(.*)", text).group(1).split()]
print("tokens:", len(token_ids), "int8 nll points:", len(int8_nll), flush=True)

# f32 reference per-token NLL on the SAME token ids.
print("loading HF model (fp32)...", flush=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
model.eval()

ids = torch.tensor([token_ids])
with torch.no_grad():
    logits = model(ids).logits[0]  # [N, vocab]

logprobs = torch.log_softmax(logits.float(), dim=-1)
f32_nll = [float(-logprobs[i, token_ids[i + 1]]) for i in range(len(token_ids) - 1)]


# Bucket by position and report perplexity = exp(mean NLL) per bucket.
def ppl(nlls):
    return math.exp(sum(nlls) / len(nlls)) if nlls else float("nan")


buckets = [(0, 16), (16, 32), (32, 64), (64, 96), (96, 128), (128, 160), (160, 192)]
print(f"\n{'position':>12} | {'f32 PPL':>10} | {'int8 PPL':>10} | {'int8 mean NLL':>13}")
print("-" * 56)
for a, b in buckets:
    f = f32_nll[a:b]
    q = int8_nll[a:b]
    if not q:
        continue
    print(f"{f'{a:3d}-{b:3d}':>12} | {ppl(f):>10.3f} | {ppl(q):>10.3f} | {np.mean(q):>13.4f}")

print("-" * 56)
print(f"{'overall':>12} | {ppl(f32_nll):>10.3f} | {ppl(int8_nll):>10.3f} |")
