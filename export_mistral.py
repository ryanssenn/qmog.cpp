import os
import json
import struct

import safetensors
import torch

import argparse

"""
Usage:
  python export_mistral.py --model_dir /path/to/Mistral-7B-v0.1 [--out mistral.mog] [--quant f32]

Arguments:
  --model_dir   Required. Path to the Hugging Face model directory.
  --out         Optional. Output file path. Defaults to ./mistral.mog
  --quant       Optional. Quantization mode. Defaults to int8. Accepts f32 or int8.
                int8 quantizes MLP gate/up projections only; down_proj and attention stay f32.


python export_mistral.py --model_dir ../Mistral-7B-v0.1 --out ../mistral.mog

python export_mistral.py --model_dir ../Mistral-7B-v0.1 --out ../mistral.mog --quant f32

---------

This script converts a Hugging Face Mistral model into one standardized binary file that can be fed into the inference engine.

Inputs (from the downloaded model directory):
  - config.json              : model hyperparameters
  - tokenizer.json           : vocabulary
  - model.safetensors.index.json + shard files : weights

Output:
  mistral.mog with layout:
    MOG magic + version + header_size
    binary header (architecture, config KV, tokenizer, tensor table)
    padding to 64-byte alignment
    payload: all tensors as continuous data with quantization scales
"""

MAGIC = b"MOG\x00"
FORMAT_VERSION = 1
FILE_PREFIX_SIZE = 16

KV_STRING = 0
KV_UINT32 = 1
KV_FLOAT32 = 2

DTYPE_F32 = 0
DTYPE_INT8 = 1


def write_u8(buf, v):
    buf.extend(struct.pack("<B", v))


def write_u32(buf, v):
    buf.extend(struct.pack("<I", v))


def write_u64(buf, v):
    buf.extend(struct.pack("<Q", v))


def write_f32(buf, v):
    buf.extend(struct.pack("<f", v))


def write_string(buf, s):
    b = s.encode("utf-8")
    write_u32(buf, len(b))
    buf.extend(b)


def write_kv_uint32(buf, key, value):
    write_string(buf, key)
    write_u8(buf, KV_UINT32)
    write_u32(buf, value)


def write_kv_float32(buf, key, value):
    write_string(buf, key)
    write_u8(buf, KV_FLOAT32)
    write_f32(buf, float(value))


def write_kv_string(buf, key, value):
    write_string(buf, key)
    write_u8(buf, KV_STRING)
    write_string(buf, value)


def quantize(x: torch.Tensor, n_bits: int, group_size: int):
    assert (x.numel() % group_size == 0)

    x = x.reshape(-1, group_size).float()

    int_max = 2 ** (n_bits - 1) - 1

    scales = int_max / x.abs().max(dim=-1).values.unsqueeze(-1)

    quant = (x * scales).round().clamp(-int_max, int_max)

    return quant, scales


def load_config():
    config_path = os.path.join(IN_PATH, "config.json")

    with open(config_path, "r") as f:
        cfg = json.load(f)
        return {
            "hidden_size": cfg["hidden_size"],
            "intermediate_size": cfg["intermediate_size"],
            "n_layers": cfg["num_hidden_layers"],
            "n_heads": cfg["num_attention_heads"],
            "n_kv_heads": cfg["num_key_value_heads"],
            "vocab_size": cfg["vocab_size"],
            "max_position_embeddings": cfg["max_position_embeddings"],
            "sliding_window": cfg["sliding_window"] if cfg["sliding_window"] is not None else 0,
            "rope_theta": cfg["rope_theta"],
            "norm_eps": cfg["rms_norm_eps"],
            "quant": args.quant,
        }


def load_tokenizer():
    tokenizer_path = os.path.join(IN_PATH, "tokenizer.json")

    with open(tokenizer_path, "r") as f:
        t = json.load(f)
        return t["model"]["vocab"], t["model"]["merges"]


def pad_to_64(offset):
    r = offset % 64
    if r == 0:
        return 0

    return 64 - r


def should_quantize(tensor_name):
    return args.quant != "f32" and any(
        key in tensor_name for key in ("mlp.gate_proj", "mlp.up_proj")
    )


def load_tensor_map():
    tensors = {}
    start = 0

    for tensor_name in weight_map:
        tensor_file_path = os.path.join(IN_PATH, weight_map[tensor_name])

        with safetensors.safe_open(tensor_file_path, framework="pt") as f:
            tensor = f.get_tensor(tensor_name)

            if should_quantize(tensor_name):
                tensors[tensor_name] = {
                    "dtype": DTYPE_INT8,
                    "shape": list(tensor.shape)[:4],
                    "offset": start,
                }
                start += tensor.numel() * DATA_SIZE
                start += pad_to_64(start)

                scale_size = tensor.numel() // GROUP_SIZE
                tensors[tensor_name]["scale_offset"] = start
                tensors[tensor_name]["scale_size"] = scale_size
                start += tensor.numel() // GROUP_SIZE * 4
                start += pad_to_64(start)
            else:
                tensors[tensor_name] = {
                    "dtype": DTYPE_F32,
                    "shape": list(tensor.shape)[:4],
                    "offset": start,
                    "scale_offset": 0,
                    "scale_size": 0,
                }
                start += tensor.numel() * 4
                start += pad_to_64(start)

    return tensors


def build_header_blob(config, vocab, merges, tensors):
    buf = bytearray()

    write_string(buf, "mistral")

    write_u32(buf, 11)
    write_kv_uint32(buf, "hidden_size", config["hidden_size"])
    write_kv_uint32(buf, "intermediate_size", config["intermediate_size"])
    write_kv_uint32(buf, "n_layers", config["n_layers"])
    write_kv_uint32(buf, "n_heads", config["n_heads"])
    write_kv_uint32(buf, "n_kv_heads", config["n_kv_heads"])
    write_kv_uint32(buf, "vocab_size", config["vocab_size"])
    write_kv_uint32(buf, "sliding_window", config["sliding_window"])
    write_kv_uint32(buf, "max_position_embeddings", config["max_position_embeddings"])
    write_kv_float32(buf, "rope_theta", config["rope_theta"])
    write_kv_float32(buf, "norm_eps", config["norm_eps"])
    write_kv_string(buf, "quant", config["quant"])

    write_u32(buf, len(vocab))
    for token, token_id in vocab.items():
        write_string(buf, token)
        write_u32(buf, int(token_id))

    write_u32(buf, len(merges))
    for merge in merges:
        write_string(buf, merge)

    write_u32(buf, len(tensors))
    for tensor_name in weight_map:
        info = tensors[tensor_name]
        shape = info["shape"]
        ndim = len(shape)
        write_string(buf, tensor_name)
        write_u8(buf, info["dtype"])
        write_u8(buf, ndim)
        for d in range(4):
            write_u32(buf, shape[d] if d < ndim else 0)
        write_u64(buf, info["offset"])
        write_u64(buf, info.get("scale_offset", 0))
        write_u32(buf, info.get("scale_size", 0))

    return bytes(buf)


def write_tensor(out, tensor, base_offset, tensor_offset):
    tensor_bytes = tensor.numpy().tobytes()
    out.seek(base_offset + tensor_offset, 0)
    out.write(tensor_bytes)


def write_binary(config, vocab, merges, tensors, header_blob):
    with open(OUT_PATH, "wb") as out:
        out.write(MAGIC)
        out.write(struct.pack("<I", FORMAT_VERSION))
        out.write(struct.pack("<Q", len(header_blob)))
        out.write(header_blob)

        padding_size = pad_to_64(out.tell())
        out.seek(padding_size, 1)
        base_offset = out.tell()

        total = len(tensors)
        i = 0
        bar_width = 40

        for tensor_name in weight_map:
            i += 1
            print("[" + "#" * int(bar_width * i / total) + "-" * (bar_width - int(bar_width * i / total)) + f"] {int(i / total * 100)}%", end="\r")

            tensor_file_path = os.path.join(IN_PATH, weight_map[tensor_name])
            with safetensors.safe_open(tensor_file_path, framework="pt") as f:
                tensor = f.get_tensor(tensor_name)
                scales = None

                if "_proj" in tensor_name and tensors[tensor_name]["dtype"] != DTYPE_F32:
                    tensor, scales = quantize(tensor, DATA_SIZE * 8, GROUP_SIZE)
                    tensor = tensor.to(DATA_TYPE)
                    scales = scales.to(torch.float32)
                else:
                    tensor = tensor.to(torch.float32)

                write_tensor(out, tensor, base_offset, tensors[tensor_name]["offset"])

                if scales is not None:
                    write_tensor(out, scales, base_offset, tensors[tensor_name]["scale_offset"])


def verify_header(path, expected_tensors):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError("missing MOG magic")

        version = struct.unpack("<I", f.read(4))[0]
        if version != FORMAT_VERSION:
            raise ValueError(f"unexpected version {version}")

        header_size = struct.unpack("<Q", f.read(8))[0]
        header_blob = f.read(header_size)
        if len(header_blob) != header_size:
            raise ValueError("truncated header")

        payload_base = f.tell() + pad_to_64(f.tell())
        f.seek(payload_base)
        payload_size = os.path.getsize(path) - payload_base

        pos = 0
        arch_len = struct.unpack_from("<I", header_blob, pos)[0]
        pos += 4 + arch_len

        kv_count = struct.unpack_from("<I", header_blob, pos)[0]
        pos += 4
        for _ in range(kv_count):
            key_len = struct.unpack_from("<I", header_blob, pos)[0]
            pos += 4 + key_len
            kv_type = header_blob[pos]
            pos += 1
            if kv_type == KV_STRING:
                val_len = struct.unpack_from("<I", header_blob, pos)[0]
                pos += 4 + val_len
            elif kv_type == KV_UINT32:
                pos += 4
            elif kv_type == KV_FLOAT32:
                pos += 4

        vocab_count = struct.unpack_from("<I", header_blob, pos)[0]
        pos += 4
        for _ in range(vocab_count):
            token_len = struct.unpack_from("<I", header_blob, pos)[0]
            pos += 4 + token_len + 4

        merge_count = struct.unpack_from("<I", header_blob, pos)[0]
        pos += 4
        for _ in range(merge_count):
            merge_len = struct.unpack_from("<I", header_blob, pos)[0]
            pos += 4 + merge_len

        tensor_count = struct.unpack_from("<I", header_blob, pos)[0]
        pos += 4
        if tensor_count != len(expected_tensors):
            raise ValueError(f"tensor count mismatch: {tensor_count} vs {len(expected_tensors)}")

        max_end = 0
        for _ in range(tensor_count):
            name_len = struct.unpack_from("<I", header_blob, pos)[0]
            pos += 4 + name_len
            dtype = header_blob[pos]
            ndim = header_blob[pos + 1]
            pos += 2
            dims = struct.unpack_from("<IIII", header_blob, pos)
            pos += 16
            offset, scale_offset, scale_size = struct.unpack_from("<QQI", header_blob, pos)
            pos += 20
            shape = [dims[i] for i in range(ndim)]
            if dtype == DTYPE_F32:
                nbytes = 4
                for d in shape:
                    nbytes *= d
                max_end = max(max_end, offset + nbytes)
            else:
                nbytes = 1
                for d in shape:
                    nbytes *= d
                max_end = max(max_end, offset + nbytes, scale_offset + scale_size * 4)

        if max_end > payload_size:
            raise ValueError(f"payload too small: need {max_end}, have {payload_size}")


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", required=True)
parser.add_argument("--out", default="./mistral.mog")
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

tensor_index_path = os.path.join(IN_PATH, "model.safetensors.index.json")
with open(tensor_index_path, "r") as f:
    index = json.load(f)
    weight_map = index["weight_map"]

print("\033[1m\033[4mModel Export\033[0m\n"
      f"\033[1mModel Directory:\033[0m {IN_PATH}\n"
      f"\033[1mOutput File:\033[0m     {OUT_PATH}\n"
      f"\033[1mQuantization:\033[0m    {args.quant}\n")

config = load_config()
tensors = load_tensor_map()
vocab, merges = load_tokenizer()
header_blob = build_header_blob(config, vocab, merges, tensors)
write_binary(config, vocab, merges, tensors, header_blob)
verify_header(OUT_PATH, tensors)

print("\nCompleted")
