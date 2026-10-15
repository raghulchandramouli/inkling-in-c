#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import struct
import time
from pathlib import Path

import fetch_nvfp4_fixture
from fetch_checkpoint_metadata import BASE_URL, MODEL, REVISION, request, sha256, url_for, write_atomic
from inkling_ref import SOURCE_DIR, reference_oracles
from inkling_model_ref import (MODEL_SOURCE_PINS, LAYER_MAPPING, _bind, load_reference,
                               checkpoint_config, causal_masks, storage_tensor,
                               decode_nvfp4, build_layer, capture)

GENERATOR_FILES = ("tools/inkling_model_ref.py", "tools/fetch_model_layers.py", "tools/inkling_ref.py",
                   "tools/fetch_checkpoint_metadata.py", "tools/fetch_nvfp4_fixture.py")


def generator_hashes():
    project = Path(__file__).resolve().parents[1]
    return {name: sha256((project / name).read_bytes()) for name in GENERATOR_FILES}


def tolerances(*, real=False):
    projection = dict(atol=1e-3 if real else 1e-4, rtol=2e-4 if real else 1e-4)
    limits = {name: dict(projection) for name in ("output", "q_proj", "k_proj", "v_proj", "r_proj",
        "attention_output", "post_attention", "routed_output", "shared_output", "mlp_output",
        "attn_conv", "mlp_conv", "k_conv", "v_conv", "relative_bias", "relative_scaled")}
    limits.update({name: dict(atol=2e-5, rtol=2e-5) for name in
        ("attn_norm", "mlp_norm", "q_norm", "k_norm", "query_scaled")})
    limits.update({name: dict(atol=2e-6, rtol=2e-5) for name in ("attention_probs", "router_weights", "shared_gammas")})
    limits["router_logits"] = dict(projection)
    return limits


CHUNK = 8 * 1024 * 1024
BUDGET = 2 * 1024 * 1024 * 1024
SIZES = {"BF16": 2, "F32": 4, "U8": 1, "F8_E4M3": 1, "I64": 8}


def json_write(path, value):
    write_atomic(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


class RangeStore:
    def __init__(self, directory, layer, tensors, *, offline=False):
        self.directory, self.tensors, self.offline = directory, tensors, offline
        self.path = directory / "ranges.json"
        self.manifest = json.loads(self.path.read_text()) if self.path.exists() else dict(
            schema=1, checkpoint_revision=REVISION, model=MODEL, layer=layer, ranges=[])
        if (self.manifest.get("schema") != 1 or self.manifest.get("checkpoint_revision") != REVISION
                or self.manifest.get("layer") != layer or self.manifest.get("model") != MODEL):
            raise ValueError(f"wrong range manifest identity: {self.path}")
        for record in self.manifest["ranges"]:
            self.check(record)

    def check(self, record):
        name = record["tensor"]
        tensor = self.tensors[name]
        relative = Path(record["file"])
        if (relative.is_absolute() or ".." in relative.parts or "\\" in str(relative)
                or record["file"] != f"blobs/{record['sha256']}.bin"
                or record.get("dtype") != tensor["dtype"] or record.get("shape") != tensor["shape"]
                or not name.startswith(f"model.llm.layers.{self.manifest['layer']}.")
                or record["offset"] < 0 or record["bytes"] <= 0
                or record["offset"] + record["bytes"] > tensor["data_offsets"][1] - tensor["data_offsets"][0]):
            raise ValueError(f"unsafe recorded range: {record}")
        data = (self.directory / relative).read_bytes()
        if len(data) != record["bytes"] or sha256(data) != record["sha256"]:
            raise ValueError(f"modified recorded range: {relative}")
        start = 8 + tensor["header_size"] + tensor["data_offsets"][0] + record["offset"]
        if record.get("file_offset") != start or record.get("url") != url_for(BASE_URL, tensor["shard"]):
            raise ValueError(f"wrong range provenance: {relative}")
        return data

    def fetch(self, name, expert=None):
        tensor = self.tensors[name]
        length = tensor["data_offsets"][1] - tensor["data_offsets"][0]
        offset = 0
        if expert is not None:
            if len(tensor["shape"]) != 3 or not 0 <= expert < tensor["shape"][0]:
                raise ValueError(f"invalid expert slice: {name}[{expert}]")
            length //= tensor["shape"][0]
            offset = expert * length
        for chunk_offset in range(offset, offset + length, CHUNK):
            size = min(CHUNK, offset + length - chunk_offset)
            old = next((r for r in self.manifest["ranges"] if r["tensor"] == name and r["offset"] == chunk_offset), None)
            if old is not None:
                if old["bytes"] != size:
                    raise ValueError("incompatible cached chunk boundaries")
                self.check(old)
                continue
            if self.offline:
                raise ValueError(f"missing offline payload: {name} at {chunk_offset}")
            total = sum(r["bytes"] for p in self.directory.parent.glob("layer-*/ranges.json")
                        for r in json.loads(p.read_text())["ranges"])
            if total + size > BUDGET:
                raise ValueError("real layer payload would exceed the 2 GiB aggregate budget")
            start = 8 + tensor["header_size"] + tensor["data_offsets"][0] + chunk_offset
            url = url_for(BASE_URL, tensor["shard"])
            for attempt in range(3):
                try:
                    data = request(url, (start, start + size - 1), max_bytes=CHUNK)
                    break
                except RuntimeError:
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
            digest = sha256(data)
            filename = f"blobs/{digest}.bin"
            write_atomic(self.directory / filename, data)
            self.manifest["ranges"].append(dict(tensor=name, offset=chunk_offset, bytes=size,
                file=filename, sha256=digest, url=url, file_offset=start, dtype=tensor["dtype"],
                shape=tensor["shape"]))
            self.manifest["ranges"].sort(key=lambda r: (r["tensor"], r["offset"]))
            json_write(self.path, self.manifest)
            print(f"layer {self.manifest['layer']} fetched {name.rsplit('.', 2)[-2:]} offset={chunk_offset} bytes={size}", flush=True)

    def raw(self, name, expert=None):
        tensor = self.tensors[name]
        size = tensor["data_offsets"][1] - tensor["data_offsets"][0]
        offset = 0
        if expert is not None:
            size //= tensor["shape"][0]
            offset = expert * size
        end, parts = offset + size, []
        for record in sorted((r for r in self.manifest["ranges"] if r["tensor"] == name), key=lambda r: r["offset"]):
            a, b = record["offset"], record["offset"] + record["bytes"]
            if b <= offset or a >= end:
                continue
            if a > offset:
                raise ValueError(f"uncovered payload: {name} at {offset}")
            raw = self.check(record)
            stop = min(end, b)
            parts.append(raw[offset - a:stop - a])
            offset = stop
            if offset == end:
                return b"".join(parts)
        raise ValueError(f"uncovered payload: {name} at {offset}")

    def reader(self, torch, modelopt):
        def read(name, expert=None):
            metadata = self.tensors[name]
            shape = metadata["shape"][1:] if expert is not None else metadata["shape"]
            value = storage_tensor(self.raw(name, expert), metadata["dtype"], shape, torch)
            if metadata["dtype"] == "U8":
                if expert is None:
                    raise ValueError("NVFP4 decoding requires one original expert ID")
                logical = struct.unpack("<3q", self.raw(name + ".original_shape"))
                if tuple(metadata["shape"]) != (logical[0], logical[1], logical[2] // 2):
                    raise ValueError("invalid original_shape")
                scales = read(name + ".scale", expert=expert)
                scale2 = read(name + ".scale2")[expert]
                value = decode_nvfp4(value, scales, scale2, logical[1:], torch, modelopt)
            return value
        return read


def trace_arrays(records, torch):
    result, indices = {}, {}
    aliases = {"input_layernorm": "attn_norm", "self_attn.q_proj": "q_proj",
        "self_attn.k_proj": "k_proj", "self_attn.v_proj": "v_proj", "self_attn.r_proj": "r_proj",
        "self_attn.k_sconv": "k_conv", "self_attn.v_sconv": "v_conv", "self_attn.q_norm": "q_norm",
        "self_attn.k_norm": "k_norm", "self_attn.rel_logits_proj": "relative_bias",
        "attn_sconv": "attn_conv", "post_attention_layernorm": "mlp_norm",
        "mlp.experts": "routed_output", "mlp.shared_experts": "shared_output", "mlp": "mlp_output",
        "mlp_sconv": "mlp_conv", "": "output"}
    for key, label in aliases.items():
        if key in records and isinstance(records[key]["output"], torch.Tensor):
            value = records[key]["output"]
            if label == "relative_bias":
                value = value.transpose(1, 2)
            result[label] = value.flatten().tolist()
    if "self_attn" in records:
        output, probs = records["self_attn"]["output"]
        result["attention_output"] = output.flatten().tolist()
        result["attention_probs"] = probs.transpose(1, 2).flatten().tolist()
    if "post_attention_layernorm" in records:
        result["post_attention"] = records["post_attention_layernorm"]["inputs"][0].flatten().tolist()
    if "mlp.gate" in records:
        logits, weights, ids, shared = records["mlp.gate"]["output"]
        order = (logits.sigmoid() + records["mlp.gate"]["correction"]).gather(-1, ids).argsort(dim=-1, descending=True, stable=True)
        ids, weights = ids.gather(-1, order), weights.gather(-1, order)
        for label, tensor in (("router_logits", records["mlp.gate"]["router_logits"]), ("router_weights", weights), ("shared_gammas", shared)):
            result[label] = tensor.flatten().tolist()
        indices["router_selected"] = ids.flatten().tolist()
    return result, indices


def generate(directory, metadata, source_dir, *, offline=False, layers=(0, 2, 5)):
    import torch
    if torch.__version__.split("+")[0] != "2.8.0":
        raise ValueError("requires torch==2.8.0")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fetch_nvfp4_fixture.METADATA = metadata
    tensors = fetch_nvfp4_fixture.catalogue()
    config = checkpoint_config(json.loads((metadata / "config.json").read_text()))
    reference = load_reference(torch, source_dir=source_dir, allow_download=not offline)
    _, modelopt, _ = reference_oracles(torch, source_dir=source_dir, allow_download=not offline)
    x = torch.tensor([((i * 73) % 257 - 128) / 128 for i in range(config.hidden_size)]).reshape(1, 1, -1)
    for layer in layers:
        store = RangeStore(directory / f"layer-{layer}", layer, tensors, offline=offline)
        prefix = f"model.llm.layers.{layer}."
        for suffix in LAYER_MAPPING:
            store.fetch(prefix + suffix)
        read = store.reader(torch, modelopt)
        selected = None
        if config.mlp_layer_types[layer] == "sparse":
            for suffix in ("weight", "bias", "global_scale"):
                store.fetch(prefix + "mlp.gate." + suffix)
            with torch.device("meta"):
                stage = reference["InklingDecoderLayer"](config, layer)
            for source, target in LAYER_MAPPING.items():
                _bind(stage, target, read(prefix + source), torch)
            for source, target in (("weight", "weight"), ("bias", "e_score_correction_bias"), ("global_scale", "global_scale")):
                _bind(stage, "mlp.gate." + target, read(prefix + "mlp.gate." + source), torch)
            stage.eval()
            with torch.no_grad():
                masks = causal_masks(reference, config, 1, torch)
                mask = masks["sliding_attention" if layer in config.local_layer_ids else "full_attention"]
                attn = stage.self_attn(stage.input_layernorm(x), attention_mask=mask)[0]
                normalized = stage.post_attention_layernorm(x + stage.attn_sconv(attn))
                logits, mixture, ids, shared = stage.mlp.gate(normalized)
                selected = sorted(set(ids.flatten().tolist()))
                choice = logits.sigmoid() + stage.mlp.gate.e_score_correction_bias
                ranked = choice.sort(descending=True).values
                margin = float((ranked[:, config.num_experts_per_tok - 1] - ranked[:, config.num_experts_per_tok]).min())
                if margin <= 0:
                    raise ValueError("real selection has a top-k boundary tie")
            store.manifest["selection"] = dict(expert_ids=selected, margin=margin,
                router_indices=ids.flatten().tolist(), router_weights=mixture.flatten().tolist(),
                shared_gammas=shared.flatten().tolist())
            json_write(store.path, store.manifest)
            del stage
            for name in ("shared_w13_weight", "shared_w2_weight"):
                store.fetch(prefix + "mlp.shared_experts." + name)
            for projection in ("w13_weight", "w2_weight"):
                name = prefix + "mlp.experts." + projection
                if tensors[name]["dtype"] == "U8":
                    for suffix in ("scale2", "original_shape", "input_amax"):
                        store.fetch(name + "." + suffix)
                for expert in selected:
                    store.fetch(name, expert)
                    if tensors[name]["dtype"] == "U8":
                        store.fetch(name + ".scale", expert)
        else:
            for suffix in ("w13_dn.weight", "w2_md.weight", "global_scale"):
                store.fetch(prefix + "mlp." + suffix)
        module, mapping = build_layer(config, layer, read, reference, torch, expert_ids=selected)
        records, handles = capture(module, torch)
        with torch.no_grad():
            mask = causal_masks(reference, config, 1, torch)["sliding_attention" if layer in config.local_layer_ids else "full_attention"]
            result = module(x, attention_mask=mask)
        for handle in handles:
            handle.remove()
        traces, indices = trace_arrays(records, torch)
        store.manifest.update(references=MODEL_SOURCE_PINS, mappings=mapping, tolerances=tolerances(real=True), generators=generator_hashes(),
            input_sha256=sha256(struct.pack("<4096f", *x.flatten().tolist())),
            expected_sha256=sha256(struct.pack("<4096f", *result.flatten().tolist())),
            input_recipe="x[i] = ((i * 73) % 257 - 128) / 128 for i in range(4096)",
            input_contract="Recorded nonzero hidden vector supplied independently to this layer; zero convolution history.",
            case=dict(name=f"real_layer_{layer}", kind="layer", layer=layer, tokens=1,
                position_start=0, input=x.flatten().tolist(), expected=result.flatten().tolist(),
                shape=list(result.shape), traces=traces, indices=indices),
            tools=dict(torch=torch.__version__, torch_git_revision=torch.version.git_version))
        json_write(store.path, store.manifest)
        print(f"COMPLETE layer={layer} experts={selected} payload_bytes={sum(r['bytes'] for r in store.manifest['ranges'])}", flush=True)
        del module, records


def main():
    parser = argparse.ArgumentParser(description='Fetch bounded real layer ranges and emit independent pinned CPU expectations.')
    parser.add_argument("--output-dir", type=Path, default=Path("weights/c16-real"))
    parser.add_argument("--metadata", type=Path, default=Path("tests/fixtures/checkpoint"))
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 2, 5])
    args = parser.parse_args()
    if any(layer not in (0, 2, 5) for layer in args.layers):
        parser.error("real acceptance is bounded to layers 0, 2, 5")
    generate(args.output_dir, args.metadata, args.source_dir, offline=args.offline, layers=args.layers)


if __name__ == "__main__":
    main()
