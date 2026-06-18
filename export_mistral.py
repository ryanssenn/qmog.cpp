import os
import json
import struct

import safetensors
import torch

import argparse

"""
Usage:
  python export_mistral.py --model_dir /path/to/Mistral-7B-v0.1 [--out model.bin] [--quant f32]

Arguments:
  --model_dir   Required. Path to the Hugging Face model directory.
  --out         Optional. Output file path. Defaults to ./model.bin
  --quant       Optional. Quantization mode. Defaults to int8. Accepts f32 or int8.
                int8 quantizes MLP gate/up projections only; down_proj and attention stay f32.
  

python export_mistral.py --model_dir ../Mistral-7B-v0.1 --out ../mistral.bin

python export_mistral.py --model_dir ../Mistral-7B-v0.1 --out ../mistral.bin --quant f32

---------

This script converts a Hugging Face Mistral model into one standardized binary file that can be fed into the inference engine.

Inputs (from the downloaded model directory):
  - config.json              : model hyperparameters
  - tokenizer.json           : vocabulary
  - model.safetensors.index.json + shard files : weights

Output:
  model.bin with layout:
    [8-byte uint64: size of JSON header]
    [JSON header]: 
        - config
        - vocab/merges
        - tensor info:
            - data type
            - shape
            - tensor start index
            - scales size
            - scales start index
            
    [payload: All tensors as continuous data with quantization scales]
"""

# Per-group symmetric quantization
# Splits tensor in groups, finds the max abs value and computes the scale that maps values -> int range
# With int8 (n_bits=8) the usable range is [-127, 127].
# Returns the quantized tensor and the scales
def quantize(x: torch.Tensor, n_bits: int, group_size: int):
    assert (x.numel() % group_size == 0)

    x = x.to(torch.float32)

    # Split tensor in groups
    x = x.reshape(-1, group_size)

    # Max int range
    int_max = 2 ** (n_bits - 1) - 1

    # Compute scale for each group
    scales = int_max / x.abs().max(dim=-1).values.unsqueeze(-1)

    # Quantize
    quant = (x * scales).round()

    return quant, scales

def load_config(header):
    config_path = os.path.join(IN_PATH, "config.json")

    with open(config_path, 'r') as f:
        cfg = json.load(f)
        header["metadata"] = {
            "hidden_size": str(cfg["hidden_size"]),
            "intermediate_size": str(cfg["intermediate_size"]),
            "n_layers": str(cfg["num_hidden_layers"]),
            "n_heads": str(cfg["num_attention_heads"]),
            "n_kv_heads": str(cfg["num_key_value_heads"]),
            "vocab_size": str(cfg["vocab_size"]),
            "max_position_embeddings": str(cfg["max_position_embeddings"]),
            "sliding_window": str(cfg["sliding_window"]),
            "rope_theta": str(cfg["rope_theta"]),
            "norm_eps": str(cfg["rms_norm_eps"]),
            "act_type": cfg["hidden_act"],
            "quant": str(args.quant)
        }

def load_tokenizer(header):
    # Insert the vocab in header["vocab"]
    tokenizer_path = os.path.join(IN_PATH, "tokenizer.json")
    header["tokenizer"] = {}

    with open(tokenizer_path, 'r') as f:
        t = json.load(f)
        header["tokenizer"]["vocab"] = t["model"]["vocab"]
        header["tokenizer"]["merges"] = t["model"]["merges"]

def pad_to_64(offset):
    r = offset % 64
    if r == 0:
        return 0

    return 64 - r

def should_quantize(tensor_name):
    # Quantize MLP up/gate only. down_proj feeds the residual directly and int8
    # error there causes bad logits and greedy collapse (e.g. repeating "0").
    # Attention stays f32 for the same reason.
    return args.quant != "f32" and any(
        key in tensor_name for key in ("mlp.gate_proj", "mlp.up_proj")
    )

def load_tensor_map(header):
    # Loop through each tensor and add info to header["tensors"]
    header["tensors"] = {}
    start = 0

    for tensor_name in weight_map:
        tensor_file_path = os.path.join(IN_PATH, weight_map[tensor_name])

        with safetensors.safe_open(tensor_file_path, framework="pt") as f:
            tensor = f.get_tensor(tensor_name)

            # Quantize
            if should_quantize(tensor_name):
                header["tensors"][tensor_name] = {"dtype": args.quant, "shape": list(tensor.shape)[:4], "offset": start}
                start += tensor.numel() * DATA_SIZE
                start += pad_to_64(start)

                # We store scales
                scale_size = tensor.numel() // GROUP_SIZE
                header["tensors"][tensor_name]["scale_offset"] = start
                header["tensors"][tensor_name]["scale_size"] = scale_size
                start += tensor.numel() // GROUP_SIZE * 4
                start += pad_to_64(start)

            # Full float
            else:
                header["tensors"][tensor_name] = {"dtype": "f32", "shape": list(tensor.shape)[:4], "offset": start}
                start += tensor.numel() * 4
                start += pad_to_64(start)

def write_tensor(out, tensor, base_offset, tensor_offset):
        tensor_bytes = tensor.numpy().tobytes()
        out.seek(base_offset + tensor_offset, 0)
        out.write(tensor_bytes)


def write_binary(header):
    with open(OUT_PATH, "wb") as out:
        header_bytes = json.dumps(header).encode("utf-8")
        padding_size = pad_to_64(8 + len(header_bytes))

        header_size = struct.pack("<Q", len(header_bytes) + padding_size)

        out.write(header_size)
        out.write(header_bytes)

        out.seek(padding_size, 1)
        base_offset = out.tell()

        total = len(header["tensors"])
        i = 0
        bar_width = 40

        # Dump all the tensors in the same order as header
        for tensor_name in header["tensors"]:
            i += 1
            print("[" + "#" * int(bar_width * i/total) + "-" * (bar_width - int(bar_width * i/total)) + f"] {int(i/total*100)}%", end="\r")

            tensor_file_path = os.path.join(IN_PATH, weight_map[tensor_name])
            with safetensors.safe_open(tensor_file_path, framework="pt") as f:
                tensor = f.get_tensor(tensor_name)
                scales = None

                if "_proj" in tensor_name and header["tensors"][tensor_name]["dtype"] != "f32":
                    tensor, scales = quantize(tensor, DATA_SIZE * 8, GROUP_SIZE)
                    tensor = tensor.to(DATA_TYPE)
                    scales = scales.to(torch.float32)
                else:
                    tensor = tensor.to(torch.float32)

                write_tensor(out, tensor, base_offset, header["tensors"][tensor_name]["offset"])

                if scales is not None:
                    write_tensor(out, scales, base_offset, header["tensors"][tensor_name]["scale_offset"])



parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", required=True)
parser.add_argument("--out", default="./model.bin")
parser.add_argument("--quant", default="int8", choices=["f32", "int8"])
args = parser.parse_args()

IN_PATH = args.model_dir
OUT_PATH = args.out
DATA_SIZE = 4
DATA_TYPE = torch.float32
GROUP_SIZE = 64

if args.quant == "int8":
    DATA_SIZE = 1
    DATA_TYPE = torch.int8

# Load weight map
tensor_index_path = os.path.join(IN_PATH, "model.safetensors.index.json")
with open(tensor_index_path, 'r') as f:
    index = json.load(f)
    weight_map = index["weight_map"]

print("\033[1m\033[4mModel Export\033[0m\n"
      f"\033[1mModel Directory:\033[0m {IN_PATH}\n"
      f"\033[1mOutput File:\033[0m     {OUT_PATH}\n"
      f"\033[1mQuantization:\033[0m    {args.quant}\n")

header = {}
load_config(header)
load_tensor_map(header)
load_tokenizer(header)
write_binary(header)

#print(header["tensors"])

print("\nCompleted")
