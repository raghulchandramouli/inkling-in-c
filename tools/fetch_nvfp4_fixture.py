#!/usr/bin/env python3
"""Regenerate the small real-byte NVFP4 oracle fixture (requires torch==2.8.0).

Ordinary C tests consume the committed JSON offline and do not import PyTorch.
"""
from __future__ import annotations

import argparse
import json
import platform
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fetch_checkpoint_metadata import (
    BASE_URL, MODEL, REVISION, request, sha256, url_for, validate_index_headers,
    write_atomic,
)
from inkling_ref import REFERENCES, SOURCE_PINS, SOURCE_DIR, bits, reference_oracles

METADATA = Path("tests/fixtures/checkpoint")
OUTPUT = Path("tests/fixtures/nvfp4.json")


def catalogue():
    manifest = json.loads((METADATA / "metadata-manifest.json").read_text())
    if manifest["revision"] != REVISION or manifest["model"] != MODEL:
        raise ValueError("metadata revision/model mismatch")
    for filename, expected in manifest["files"].items():
        if sha256((METADATA / filename).read_bytes()) != expected["sha256"]:
            raise ValueError(f"modified metadata: {filename}")
    index = json.loads((METADATA / "model.safetensors.index.json").read_text())
    headers = {}
    for shard in manifest["shards"]:
        raw = (METADATA / "headers" / shard["name"]).read_bytes()
        if sha256(raw) != shard["sha256"]:
            raise ValueError(f"modified header: {shard['name']}")
        headers[shard["name"]] = raw
    validate_index_headers(index, headers)
    parsed = {name: json.loads(raw[8:]) for name, raw in headers.items()}
    return {
        name: dict(parsed[shard][name], shard=shard, header_size=len(headers[shard]) - 8)
        for name, shard in index["weight_map"].items()
    }


def fetch_range(tensors, name, offset=0, length=None):
    tensor = tensors[name]
    start, end = tensor["data_offsets"]
    if length is None:
        length = end - start
    if offset < 0 or length <= 0 or offset + length > end - start or length > 4096:
        raise ValueError(f"unsafe sample range: {name} {offset} {length}")
    file_offset = 8 + tensor["header_size"] + start + offset
    raw = request(url_for(BASE_URL, tensor["shard"]),
                  (file_offset, file_offset + length - 1))
    return dict(tensor=name, tensor_offset=offset, data_offset=start + offset,
                file_offset=file_offset, length=length, sha256=sha256(raw), hex=raw.hex())


def recorded_ranges(path, tensors):
    """Validate B2 extraction records and return a reader with no network path."""
    fixture = json.loads(path.read_bytes())
    if (fixture["schema_version"] != 1 or fixture["model"] != MODEL
            or fixture["revision"] != REVISION
            or fixture["references"] != {key: SOURCE_PINS[key] for key in REFERENCES}):
        raise ValueError("recorded fixture identity mismatch")
    names = [f"model.llm.layers.3.mlp.experts.{projection}"
             for projection in ("w13_weight", "w2_weight")]
    if [p["weight"] for p in fixture["projections"]] != names:
        raise ValueError("recorded projection coverage mismatch")
    expected_tensors = {name: metadata for name, metadata in tensors.items()
                        if any(name == weight or name.startswith(weight + ".") for weight in names)}
    if fixture["tensors"] != expected_tensors:
        raise ValueError("recorded tensor metadata mismatch")
    records = {}

    def add(record, name, offset, length):
        tensor = tensors[name]
        start, end = tensor["data_offsets"]
        raw = bytes.fromhex(record["hex"])
        if (record["tensor"] != name or record["tensor_offset"] != offset
                or record["length"] != length or len(raw) != length
                or record["sha256"] != sha256(raw) or offset < 0
                or offset + length > end - start
                or record["data_offset"] != start + offset
                or record["file_offset"] != 8 + tensor["header_size"] + start + offset
                or (name, offset) in records):
            raise ValueError(f"invalid recorded sample: {name} at {offset}")
        records[name, offset] = record
        return raw

    for p in fixture["projections"]:
        name = p["weight"]
        shape = list(struct.unpack("<3q", add(p["original_shape"], name + ".original_shape", 0, 24)))
        columns = 4096 if name == names[0] else 2048
        if shape != [256, 4096, columns] or p["logical_shape"] != shape:
            raise ValueError("recorded logical shape mismatch")
        add(p["scale2"], name + ".scale2", 0, 1024)
        add(p["input_amax"], name + ".input_amax", 0, 2)
        positions = [(e, r, b) for e in (0, 7, 255)
                     for r, b in ((0, 0), (1, 3), (4095, columns // 16 - 2))]
        if [(s["expert"], s["row"], s["block"]) for s in p["samples"]] != positions:
            raise ValueError("recorded sample coverage mismatch")
        for sample in p["samples"]:
            linear = (sample["expert"] * 4096 + sample["row"]) * columns + sample["block"] * 16
            add(sample["packed"], name, linear // 2, 16)
            add(sample["scales"], name + ".scale", linear // 16, 2)

    def read_range(tensors, name, offset=0, length=None):
        record = records[name, offset]
        if length is not None and length != record["length"]:
            raise ValueError(f"recorded sample length mismatch: {name}")
        return dict(record)

    return read_range


def generate(output, *, recorded=None, source_dir=SOURCE_DIR, allow_download=True):
    import torch
    if torch.__version__.split("+")[0] != "2.8.0":
        raise ValueError("regeneration requires torch==2.8.0")
    tensors = catalogue()
    read_range = fetch_range if recorded is None else recorded_ranges(recorded, tensors)
    te_decode, modelopt_class, references = reference_oracles(
        torch, source_dir=source_dir, allow_download=allow_download)
    projections = []
    for projection in ("w13_weight", "w2_weight"):
        name = f"model.llm.layers.3.mlp.experts.{projection}"
        auxiliaries = {suffix: read_range(tensors, name + "." + suffix)
                       for suffix in ("original_shape", "scale2", "input_amax")}
        shape = list(struct.unpack("<3q", bytes.fromhex(auxiliaries["original_shape"]["hex"])))
        experts, rows, columns = shape
        if (experts, rows, columns) != (256, 4096, 4096 if projection == "w13_weight" else 2048):
            raise ValueError(f"unexpected original_shape: {shape}")
        if (tensors[name]["dtype"] != "U8" or tensors[name]["shape"] != [experts, rows, columns // 2]
            or tensors[name + ".scale"]["dtype"] != "F8_E4M3"
            or tensors[name + ".scale"]["shape"] != [experts, rows, columns // 16]
            or tensors[name + ".scale2"]["dtype"] != "F32"
            or tensors[name + ".scale2"]["shape"] != [experts]):
            raise ValueError("unsupported stored layout")
        positions = [(e, r, b) for e in (0, 7, 255)
                     for r, b in ((0, 0), (1, 3), (rows - 1, columns // 16 - 2))]

        def sample(position):
            expert, row, block = position
            linear = (expert * rows + row) * columns + block * 16
            return dict(expert=expert, row=row, block=block,
                        packed=read_range(tensors, name, linear // 2, 16),
                        scales=read_range(tensors, name + ".scale", linear // 16, 2))

        with ThreadPoolExecutor(max_workers=4) as pool:
            samples = list(pool.map(sample, positions))
        global_scales = struct.unpack("<256f", bytes.fromhex(auxiliaries["scale2"]["hex"]))
        for sample_data in samples:
            raw = torch.tensor(list(bytes.fromhex(sample_data["packed"]["hex"])), dtype=torch.uint8).reshape(1, -1)
            scale = torch.tensor(list(bytes.fromhex(sample_data["scales"]["hex"])), dtype=torch.uint8).view(torch.float8_e4m3fn).reshape(1, -1)
            global_scale = torch.tensor(global_scales[sample_data["expert"]], dtype=torch.float32)
            # TE supplies nibble semantics, PyTorch FP8 conversion supplies E4M3FN.
            # Preserve ModelOpt's two float32 multiplications and their order.
            decoded = (te_decode(raw, torch.float32).reshape(1, 2, 16)
                       * (scale.float() * global_scale).unsqueeze(-1)).reshape(1, 32)
            modelopt = modelopt_class()
            modelopt._quantized_data = raw
            modelopt.metadata = {"shape": (1, 32), "dtype": torch.float32}
            independent = modelopt.dequantize(dtype=torch.float32, scale=scale,
                                              double_scale=global_scale, block_sizes={-1: 16})
            if not torch.equal(decoded, independent):
                raise ValueError("pinned independent decoders disagree on numerical values")
            sample_data["expected_f32_bits"] = bits(decoded, torch)
        projections.append(dict(weight=name, logical_shape=shape, **auxiliaries, samples=samples))
        print(f"validated {len(samples)} real samples for {name}", flush=True)
    codes = torch.tensor([lo | ((lo + 1) << 4) for lo in range(0, 16, 2)], dtype=torch.uint8).reshape(1, 8)
    result = dict(
        schema_version=1, model=MODEL, revision=REVISION,
        tools={"python": platform.python_version(), "torch": torch.__version__,
               "torch_git_revision": torch.version.git_version},
        references=references,
        tables={"e2m1_f32_bits": bits(te_decode(codes, torch.float32), torch),
                "e4m3_f32_bits": bits(torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float(), torch)},
        tensors={name: metadata for name, metadata in tensors.items()
                 if any(name == p["weight"] or name.startswith(p["weight"] + ".") for p in projections)},
        projections=projections,
    )
    write_atomic(output, (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    print(f"wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    generate(args.output)
