#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import struct
from pathlib import Path

from fetch_checkpoint_metadata import sha256, write_atomic
from inkling_ref import SOURCE_DIR, reference_oracles
from inkling_model_ref import (MODEL_SOURCE_PINS, LAYER_MAPPING, load_reference,
    checkpoint_config, causal_masks, storage_tensor, decode_nvfp4, build_model, capture)
from fetch_model_layers import json_write, trace_arrays, tolerances

SEED = 932184


def generate(output, metadata, source_dir, *, offline=False):
    import torch
    if torch.__version__.split("+")[0] != "2.8.0":
        raise ValueError("requires torch==2.8.0")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rng = random.Random(SEED)
    reference = load_reference(torch, source_dir=source_dir, allow_download=not offline)
    te_decode, modelopt, _ = reference_oracles(torch, source_dir=source_dir, allow_download=not offline)
    raw = json.loads((metadata / "config.json").read_text())
    raw["eos_token_id"] = 6
    raw["text_config"].update(hidden_size=32, num_hidden_layers=3, vocab_size=40,
        unpadded_vocab_size=37, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        swa_num_attention_heads=4, swa_num_key_value_heads=2, swa_head_dim=8,
        d_rel=4, rel_extent=1024, sliding_window_size=512, local_layer_ids=[0, 1],
        dense_mlp_idx=1, dense_intermediate_size=48, intermediate_size=16,
        n_routed_experts=4, n_shared_experts=2, num_experts_per_tok=2)
    config = checkpoint_config(raw)
    tensors = {}

    def add(name, shape, dtype="BF16", values=None):
        if values is None:
            values = [rng.uniform(-0.13, 0.13) for _ in range(math.prod(shape))]
            if "norm.weight" in name:
                values = [v + 1 for v in values]
            if name.endswith("global_scale"):
                values = [1.375] * math.prod(shape)
            if name.endswith("mlp.gate.bias"):
                values = [-0.11, 0.09, -0.03, 0.04]
        types = {"BF16": torch.bfloat16, "F32": torch.float32, "U8": torch.uint8,
                 "F8_E4M3": torch.uint8, "I64": torch.int64}
        value = torch.tensor(values, dtype=types[dtype]).reshape(shape)
        if name.endswith("attn.rel_logits_proj.proj"):
            value[:, -1] = 8
        blob = bytes(value.contiguous().view(torch.uint8).flatten().tolist())
        tensors[name] = dict(dtype=dtype, shape=list(shape), data=blob)

    for name, shape in (("embed.weight", (40, 32)), ("embed_norm.weight", (32,)),
                        ("norm.weight", (32,)), ("unembed.weight", (40, 32))):
        add("model.llm." + name, shape)
    for i in range(3):
        with torch.device("meta"):
            module = reference["InklingDecoderLayer"](config, i)
        prefix = f"model.llm.layers.{i}."
        for source, target in LAYER_MAPPING.items():
            add(prefix + source, module.get_parameter(target).shape)
        if i == 0:
            for name, shape in (("mlp.w13_dn.weight", (96, 32)), ("mlp.w2_md.weight", (32, 48)), ("mlp.global_scale", (1,))):
                add(prefix + name, shape)
        else:
            for name, shape, dtype in (("mlp.gate.weight", (6, 32), "BF16"),
                    ("mlp.gate.bias", (4,), "F32"), ("mlp.gate.global_scale", (1,), "F32"),
                    ("mlp.shared_experts.shared_w13_weight", (2, 32, 32), "BF16"),
                    ("mlp.shared_experts.shared_w2_weight", (2, 32, 16), "BF16")):
                add(prefix + name, shape, dtype)
            for projection, columns in (("w13_weight", 32), ("w2_weight", 16)):
                name = prefix + "mlp.experts." + projection
                if i == 1:
                    add(name, (4, 32, columns))
                else:
                    add(name, (4, 32, columns // 2), "U8", [rng.randrange(256) for _ in range(4 * 32 * columns // 2)])
                    add(name + ".scale", (4, 32, columns // 16), "F8_E4M3", [rng.randrange(32, 49) for _ in range(4 * 32 * columns // 16)])
                    add(name + ".scale2", (4,), "F32", [0.0625, 0.09375, 0.125, 0.15625])
                    add(name + ".original_shape", (3,), "I64", [4, 32, columns])
                    add(name + ".input_amax", (1,), "BF16", [1])

    def read(name, expert=None):
        item = tensors[name]
        value = storage_tensor(item["data"], item["dtype"], item["shape"], torch)
        if expert is not None:
            value = value[expert]
        if item["dtype"] == "U8":
            shape = tuple(read(name + ".original_shape").tolist())[1:]
            scales, scale2 = read(name + ".scale", expert), read(name + ".scale2")[expert]
            decoded = decode_nvfp4(value, scales, scale2, shape, torch, modelopt)
            te_values = te_decode(value, torch.float32).float().reshape(shape)
            independently_decoded = te_values * scales.float().repeat_interleave(16, dim=-1) * scale2
            torch.testing.assert_close(decoded, independently_decoded, atol=0, rtol=0)
            return decoded
        return value

    index = {"metadata": {"total_size": sum(len(t["data"]) for t in tensors.values())}, "weight_map": {}}
    names = sorted(tensors)
    for shard_index in range(2):
        filename = f"model-{shard_index + 1:05d}-of-00002.safetensors"
        header, payload = {}, bytearray()
        for name in names[shard_index::2]:
            tensor = tensors[name]
            start = len(payload)
            payload.extend(tensor["data"])
            header[name] = {k: tensor[k] for k in ("dtype", "shape")}
            header[name]["data_offsets"] = [start, len(payload)]
            index["weight_map"][name] = filename
        encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        write_atomic(output / filename, struct.pack("<Q", len(encoded)) + encoded + payload)
    json_write(output / "model.safetensors.index.json", index)
    json_write(output / "config.json", raw)
    quant = json.loads((metadata / "hf_quant_config.json").read_text())
    quant["quantization"]["exclude_modules"] = sorted(
        name.removesuffix(".weight") for name, tensor in tensors.items()
        if tensor["dtype"] == "BF16" and len(tensor["shape"]) >= 2)
    json_write(output / "hf_quant_config.json", quant)
    model, mappings = build_model(config, read, reference, torch)
    cases = []

    def random_input(tokens):
        return torch.tensor([rng.uniform(-1.25, 1.25) for _ in range(tokens * config.hidden_size)]).reshape(1, tokens, -1)

    for tokens in (1, 3, 6):
        ids = [2, 5, 7, 39, 8, 9][:tokens]
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids]), attention_mask=causal_masks(reference, config, tokens, torch), use_cache=False).logits
        cases.append(dict(name=f"prefix_{tokens}", kind="model", tokens=tokens, position_start=0,
            input_ids=ids, expected=logits.flatten().tolist(), shape=list(logits.shape), traces={}, indices={}))
    for i, layer in enumerate(model.model.layers):
        x = random_input(6)
        records, handles = capture(layer, torch)
        with torch.no_grad():
            mask = causal_masks(reference, config, 6, torch)["sliding_attention" if i in config.local_layer_ids else "full_attention"]
            result = layer(x, attention_mask=mask)
        for handle in handles:
            handle.remove()
        traces, indices = trace_arrays(records, torch)
        cases.append(dict(name=f"layer_{i}", kind="layer", layer=i, tokens=6, position_start=0,
            input=x.flatten().tolist(), expected=result.flatten().tolist(), shape=list(result.shape), traces=traces, indices=indices))
        records, handles = capture(layer, torch)
        with torch.no_grad():
            result = layer.mlp(x)
        for handle in handles:
            handle.remove()
        traces, indices = trace_arrays(records, torch)
        cases.append(dict(name=f"mlp_{i}", kind="mlp", layer=i, tokens=6, position_start=0,
            input=x.flatten().tolist(), expected=result.flatten().tolist(), shape=list(result.shape), traces=traces, indices=indices))

    class PositionWindow:
        def __init__(self, length, offset):
            self.length, self.offset = length, offset
        def has_previous_state(self, *_):
            return False
        def update_conv_state(self, hidden, *_args, **_kwargs):
            return hidden
        def get_mask_sizes(self, *_):
            return self.length, self.offset
        def get_query_offset(self, *_):
            return self.offset
        def update(self, key, value, *_):
            return key, value

    for name, layer_id, tokens, position in (("local_window_512", 1, 513, 0),
            ("global_extent_1024", 2, 1025, 0), ("tau_floor", 2, 2, 127999), ("tau_after_floor", 2, 2, 128000), ("tau_far", 2, 2, 512000)):
        x = random_input(tokens)
        layer = model.model.layers[layer_id]
        if tokens > 512:
            x[:, :4] *= 8
            x[:, -2:] = layer.self_attn.r_proj.weight[:config.d_rel].sum(dim=0) * 8
        records, handles = capture(layer, torch) if tokens < 3 else ({}, [])
        attention_inputs = {}
        heads = layer.self_attn.num_heads if hasattr(layer.self_attn, "num_heads") else (config.swa_num_attention_heads if layer_id in config.local_layer_ids else config.num_attention_heads)
        coordinates = [(0, 0, 1), (tokens - 2, 0, 0), (tokens - 1, 0, 0),
            (tokens - 1, 0, 1), (tokens - 1, 0, tokens - 1),
            (tokens - 2, heads - 1, 0), (tokens - 1, heads - 1, 0), (tokens - 1, heads - 1, 1)]
        sample_ids = sorted({(q * heads + h) * tokens + k for q, h, k in coordinates}) if tokens > 512 else []
        original = reference["eager_attention_forward"]
        def recording_attention(module, query, key, value, mask, **kwargs):
            result = original(module, query, key, value, mask, **kwargs)
            if sample_ids:
                for stage, tensor in (("relative_bias", kwargs["position_bias"]), ("attention_probs", result[1])):
                    flat = tensor.detach().transpose(1, 2).reshape(-1)
                    attention_inputs[stage] = flat[torch.tensor(sample_ids)].tolist()
            else:
                attention_inputs.update(query_scaled=query.detach().transpose(1, 2).flatten().tolist(),
                                        relative_scaled=kwargs["position_bias"].detach().transpose(1, 2).flatten().tolist())
            return result
        reference["ALL_ATTENTION_FUNCTIONS"].get_interface = lambda _, fallback: recording_attention
        with torch.no_grad():
            mask = causal_masks(reference, config, tokens, torch)["sliding_attention" if layer_id in config.local_layer_ids else "full_attention"]
            result = layer.self_attn(x, attention_mask=mask, past_key_values=PositionWindow(tokens, position))[0]
        reference["ALL_ATTENTION_FUNCTIONS"].get_interface = lambda _, fallback: fallback
        for handle in handles:
            handle.remove()
        traces, indices = trace_arrays(records, torch)
        traces.update(attention_inputs)
        cases.append(dict(name=name, kind="attention", layer=layer_id, tokens=tokens, position_start=position,
            input=x.flatten().tolist(), expected=result.flatten().tolist(), shape=list(result.shape), traces=traces, indices=indices,
            history="Zero convolution prehistory; query and key positions both start at position_start."))
        if sample_ids:
            cases[-1]["trace_samples"] = {stage: sample_ids for stage in attention_inputs}

    for name, logits, bias in (("biased_unbiased_mixture", [0.5, -0.25, 0.9, -1, 1.5, 2], [0.3, 0.4, -0.5, 0]),
            ("negative_logits", [-1000, -1001, -1002, -1003, -1004, -999], [0.4, 0.3, 0.2, 0.1]),
            ("stable_ties", [0, 0, 0, 0, 0.2, 0.4], [0, 0, 0, 0])):
        router = reference["InklingTopkRouter"](config)
        with torch.no_grad():
            router.weight.zero_()
            router.weight[:, 0] = torch.tensor(logits)
            router.global_scale.fill_(1.375)
            router.e_score_correction_bias.copy_(torch.tensor(bias))
            x = torch.zeros(1, config.hidden_size)
            x[0, 0] = 1
            _, mixture, ids, shared = router(x)
            order = (torch.tensor(logits[:4]).sigmoid() + torch.tensor(bias))[ids].argsort(dim=-1, descending=True, stable=True)
            ids, mixture = ids.gather(-1, order), mixture.gather(-1, order)
            if name == "stable_ties":
                ids = torch.argsort(torch.tensor(logits[:4]).sigmoid() + torch.tensor(bias), descending=True, stable=True)[:2].reshape(1, 2)
        cases.append(dict(name=name, kind="router", layer=1, tokens=1, position_start=0, input=logits, bias=bias, route_scale=config.route_scale,
            global_scale=1.375, expected=torch.cat([mixture, shared], dim=-1).flatten().tolist(), shape=[4], traces={},
            indices={"router_selected": ids.flatten().tolist()}, tie_contract="ascending original index; not HF topk parity" if name == "stable_ties" else None))
    oracle = dict(schema=1, seed=SEED, references=MODEL_SOURCE_PINS, mappings=mappings, tolerances=tolerances(),
        tools=dict(torch=torch.__version__, torch_git_revision=torch.version.git_version),
        cases=cases)
    json_write(output / "oracle.json", oracle)
    files = ["config.json", "hf_quant_config.json", "model.safetensors.index.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "oracle.json"]
    generators = ["tools/inkling_model_ref.py", "tools/emit_model_fixtures.py", "tools/fetch_model_layers.py", "tools/inkling_ref.py", "tools/fetch_checkpoint_metadata.py", "tools/fetch_nvfp4_fixture.py"]
    json_write(output / "manifest.json", dict(schema=1, files={name: sha256((output / name).read_bytes()) for name in files}, generators={name: sha256(Path(name).read_bytes()) for name in generators}))
    print(f"wrote {output}: {len(tensors)} native tensors, {len(cases)} numerical cases")


def main():
    parser = argparse.ArgumentParser(description='Emit the miniature native checkpoint and pinned full-prefix/block oracles.')
    parser.add_argument("--output-dir", type=Path, default=Path("tests/fixtures/model"))
    parser.add_argument("--metadata", type=Path, default=Path("tests/fixtures/checkpoint"))
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    generate(args.output_dir, args.metadata, args.source_dir, offline=args.offline)


if __name__ == "__main__":
    main()
