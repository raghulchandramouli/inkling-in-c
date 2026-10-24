#!/usr/bin/env python3
"""Copy the text trunk into one verified, atomically published SafeTensors file."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile

REVISION = "b6a99534467840620d411e4cd4ad5819b2610d9c"
METADATA_FILES = ("config.json", "model.safetensors.index.json", "hf_quant_config.json")
CHUNK_BYTES = 1024 * 1024
MAX_HEADER_BYTES = 100 * 1024 * 1024
DTYPE_BYTES = {"BF16": 2, "F32": 4, "I64": 8, "F8_E4M3": 1, "U8": 1}
ORDER = (
    "attn_norm.weight", "attn.wq_du.weight", "attn.q_norm.weight",
    "attn.wk_dv.weight", "attn.k_sconv.weight", "attn.k_norm.weight",
    "attn.wv_dv.weight", "attn.v_sconv.weight", "attn.wr_du.weight",
    "attn.rel_logits_proj.proj", "attn.wo_ud.weight", "attn_sconv.weight",
    "mlp_norm.weight", "mlp.gate.weight", "mlp.gate.bias", "mlp.gate.global_scale",
    "mlp.w13_dn.weight", "mlp.w2_md.weight", "mlp.global_scale",
    "mlp.shared_experts.shared_w13_weight", "mlp.shared_experts.shared_w2_weight",
    "mlp_sconv.weight",
)


def unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key {key}")
        result[key] = value
    return result


def load_json(raw):
    return json.loads(raw, object_pairs_hook=unique_object)


def read_metadata(path):
    with path.open("rb") as source:
        raw = source.read(MAX_HEADER_BYTES + 1)
    if len(raw) > MAX_HEADER_BYTES:
        raise ValueError(f"oversized metadata {path.name}")
    return raw


def is_trunk(name):
    return name.startswith("model.llm.") and ".mlp.experts." not in name


def execution_order(name):
    if name == "model.llm.embed.weight":
        return (0, 0, 0, name)
    if name == "model.llm.embed_norm.weight":
        return (0, 0, 1, name)
    match = re.fullmatch(r"model\.llm\.layers\.(\d+)\.(.+)", name)
    if match:
        suffix = match[2]
        return (1, int(match[1]), ORDER.index(suffix) if suffix in ORDER else len(ORDER), name)
    return (2, 0, 0 if name == "model.llm.norm.weight" else 1, name)


def metadata_identity(directory):
    digest = hashlib.sha256()
    for name in METADATA_FILES:
        path = directory / name
        size = path.stat().st_size
        if size < 0 or size > MAX_HEADER_BYTES:
            raise ValueError(f"oversized metadata {name}")
        digest.update(name.encode() + b"\0" + struct.pack("<Q", size))
        with path.open("rb") as source:
            remaining = size
            while remaining:
                data = source.read(min(remaining, CHUNK_BYTES))
                if not data:
                    raise ValueError(f"truncated metadata {name}")
                digest.update(data)
                remaining -= len(data)
            if source.read(1):
                raise ValueError(f"metadata grew during packing {name}")
    return digest.hexdigest()


def read_catalogue(directory):
    index = load_json(read_metadata(directory / METADATA_FILES[1]))
    weight_map = index["weight_map"]
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("empty weight map")
    shards = {}
    catalogue = {}
    total = 0
    for shard in sorted(set(weight_map.values())):
        if not isinstance(shard, str) or not shard or Path(shard).name != shard or "\\" in shard or shard in (".", ".."):
            raise ValueError("invalid shard name")
        # MTP is outside the text execution graph and may be absent.
        if shard == "mtp.safetensors":
            continue
        path = directory / shard
        with path.open("rb") as source:
            prefix = source.read(8)
            if len(prefix) != 8:
                raise ValueError(f"truncated shard {shard}")
            header_size, = struct.unpack("<Q", prefix)
            if not 0 < header_size <= MAX_HEADER_BYTES:
                raise ValueError("invalid source header size")
            raw_header = source.read(header_size)
        if len(raw_header) != header_size:
            raise ValueError("truncated source header")
        header = load_json(raw_header)
        size = path.stat().st_size
        ranges = []
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            if weight_map.get(name) != shard or set(tensor) != {"dtype", "shape", "data_offsets"}:
                raise ValueError("source index/header mismatch")
            shape, dtype = tensor["shape"], tensor["dtype"]
            offsets = tensor["data_offsets"]
            if dtype not in DTYPE_BYTES or not isinstance(shape, list) or len(shape) > 8 or any(type(x) is not int or x < 0 for x in shape):
                raise ValueError("invalid source dtype/shape")
            if not isinstance(offsets, list) or len(offsets) != 2 or any(type(x) is not int or x < 0 for x in offsets):
                raise ValueError("invalid source offsets")
            start, end = offsets
            if end < start or end - start != math.prod(shape) * DTYPE_BYTES[dtype]:
                raise ValueError("source tensor size mismatch")
            ranges.append((start, end))
            total += end - start
            catalogue[name] = (shard, header_size, tensor)
        end = 0
        for start, stop in sorted(ranges):
            if start != end:
                raise ValueError("source gaps or overlaps")
            end = stop
        if size != 8 + header_size + end:
            raise ValueError("source shard length mismatch")
        shards[shard] = path
    expected = {name for name, shard in weight_map.items() if shard != "mtp.safetensors"}
    if set(catalogue) != expected:
        raise ValueError("source index is incomplete")
    if "mtp.safetensors" not in weight_map.values() and index["metadata"]["total_size"] != total:
        raise ValueError("source index total_size mismatch")
    return catalogue, shards


def copy_bytes(source, target, count, digest):
    while count:
        data = source.read(min(count, CHUNK_BYTES))
        if not data:
            raise ValueError("source tensor became truncated")
        target.write(data)
        digest.update(data)
        count -= len(data)


def pack(directory, destination):
    directory, destination = Path(directory), Path(destination)
    catalogue, shards = read_catalogue(directory)
    indexed_shards = set(load_json(read_metadata(directory / METADATA_FILES[1]))["weight_map"].values())
    indexed_shards.add("mtp.safetensors")
    inputs = [directory / name for name in METADATA_FILES] + [directory / name for name in indexed_shards]
    resolved = destination.resolve()
    if any(resolved == path.resolve() or (destination.exists() and path.exists() and os.path.samefile(destination, path)) for path in inputs):
        raise ValueError("destination aliases a source checkpoint file")
    identity = metadata_identity(directory)
    names = sorted((name for name in catalogue if is_trunk(name)), key=execution_order)
    if not names:
        raise ValueError("checkpoint contains no text trunk")
    header = {}
    source_ranges = {}
    offset = 0
    for name in names:
        shard, _, tensor = catalogue[name]
        start, end = tensor["data_offsets"]
        size = end - start
        header[name] = dict(dtype=tensor["dtype"], shape=tensor["shape"], data_offsets=[offset, offset + size])
        source_ranges[name] = [shard, start, size]
        offset += size
    metadata = dict(schema="inkling-trunk-v1", checkpoint_revision=REVISION, source_sha256=identity,
                    payload_sha256="0" * 64, header_sha256="0" * 64, source_ranges=json.dumps(source_ranges, separators=(",", ":")))
    header["__metadata__"] = metadata
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    if len(raw) > MAX_HEADER_BYTES:
        raise ValueError("packed header is too large")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False) as target:
            temporary = Path(target.name)
            target.write(struct.pack("<Q", len(raw)))
            target.write(raw)
            digest = hashlib.sha256()
            for name in names:
                shard, header_size, tensor = catalogue[name]
                start, end = tensor["data_offsets"]
                with shards[shard].open("rb") as source:
                    source.seek(8 + header_size + start)
                    copy_bytes(source, target, end - start, digest)
            metadata["payload_sha256"] = digest.hexdigest()
            final = json.dumps(header, separators=(",", ":")).encode()
            final += b" " * (-len(final) % 8)
            metadata["header_sha256"] = hashlib.sha256(final).hexdigest()
            final = json.dumps(header, separators=(",", ":")).encode()
            final += b" " * (-len(final) % 8)
            if len(final) != len(raw) or metadata_identity(directory) != identity:
                raise ValueError("source metadata changed during packing")
            target.seek(8)
            target.write(final)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return len(names), offset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        count, size = pack(args.source, args.destination)
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        parser.exit(1, f"pack-trunk: {error}\n")
    print(f"packed {count} tensors, {size} payload bytes into {args.destination}")


if __name__ == "__main__":
    main()
