#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import tempfile
from pathlib import Path

import fetch_nvfp4_fixture
from fetch_checkpoint_metadata import sha256
from fetch_model_layers import GENERATOR_FILES, RangeStore, BUDGET, tolerances, generator_hashes
from inkling_model_ref import MODEL_SOURCE_PINS, LAYER_MAPPING


def load(path):
    return json.loads(path.read_text())


def audit_tiny(directory):
    manifest = load(directory / "manifest.json")
    if manifest.get("schema") != 1:
        raise ValueError("unsupported miniature manifest")
    required = {"config.json", "hf_quant_config.json", "model.safetensors.index.json",
                "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "oracle.json"}
    if set(manifest["files"]) != required:
        raise ValueError("incomplete miniature file manifest")
    for name, digest in manifest["files"].items():
        if sha256((directory / name).read_bytes()) != digest:
            raise ValueError(f"modified miniature fixture: {name}")
    if set(manifest["generators"]) != {"tools/emit_model_fixtures.py", *GENERATOR_FILES}:
        raise ValueError("incomplete miniature generator manifest")
    project = Path(__file__).resolve().parents[1]
    for name, digest in manifest["generators"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "tools":
            raise ValueError("unsafe generator manifest path")
        if sha256((project / relative).read_bytes()) != digest:
            raise ValueError(f"miniature generator changed without regeneration: {name}")
    oracle = load(directory / "oracle.json")
    if oracle.get("schema") != 1 or oracle["references"] != MODEL_SOURCE_PINS:
        raise ValueError("wrong miniature reference provenance")
    audit_cases(oracle, load(directory / "config.json")["text_config"])
    return oracle


def finite_array(value, size, label, *, integer=False):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"wrong array length: {label}")
    for item in value:
        if integer:
            valid = type(item) is int
        else:
            valid = type(item) in (int, float) and math.isfinite(item)
        if not valid:
            raise ValueError(f"invalid scalar in {label}")


def audit_cases(oracle, config):
    if oracle.get("seed") != 932184 or oracle.get("tools") != {
            "torch": "2.8.0", "torch_git_revision": "a1cb3cc05d46d198467bebbb6e8fba50a325d4e7"}:
        raise ValueError("wrong deterministic tool identity")
    geometry = dict(hidden_size=32, num_hidden_layers=3, vocab_size=40, unpadded_vocab_size=37,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        swa_num_attention_heads=4, swa_num_key_value_heads=2, swa_head_dim=8,
        d_rel=4, rel_extent=1024, sliding_window_size=512, local_layer_ids=[0, 1],
        dense_mlp_idx=1, dense_intermediate_size=48, intermediate_size=16,
        n_routed_experts=4, n_shared_experts=2, num_experts_per_tok=2)
    if any(config.get(key) != value for key, value in geometry.items()):
        raise ValueError("wrong miniature dimensions")
    expected = [(f"prefix_{n}", "model", n, None, 0) for n in (1, 3, 6)]
    for layer in range(3):
        expected.extend([(f"layer_{layer}", "layer", 6, layer, 0), (f"mlp_{layer}", "mlp", 6, layer, 0)])
    expected.extend([("local_window_512", "attention", 513, 1, 0),
        ("global_extent_1024", "attention", 1025, 2, 0),
        ("tau_floor", "attention", 2, 2, 127999), ("tau_after_floor", "attention", 2, 2, 128000), ("tau_far", "attention", 2, 2, 512000)])
    expected.extend((name, "router", 1, 1, 0) for name in
        ("biased_unbiased_mixture", "negative_logits", "stable_ties"))
    actual = [(c.get("name"), c.get("kind"), c.get("tokens"), c.get("layer"), c.get("position_start")) for c in oracle["cases"]]
    if actual != expected:
        raise ValueError("missing or reordered miniature coverage cases")
    if not oracle.get("mappings"):
        raise ValueError("missing storage adapter provenance")
    for name, limit in oracle["tolerances"].items():
        if set(limit) != {"atol", "rtol"} or any(type(v) not in (int, float) or not 0 <= v <= 0.001 for v in limit.values()):
            raise ValueError(f"invalid tolerance: {name}")
    for case in oracle["cases"]:
        name, kind, tokens = case["name"], case["kind"], case["tokens"]
        width = 37 if kind == "model" else 4 if kind == "router" else 32
        finite_array(case["expected"], tokens * width, name + ".expected")
        expected_shape = [4] if kind == "router" else [1, tokens, width]
        if case["shape"] != expected_shape:
            raise ValueError(f"wrong expected shape: {name}")
        if kind == "model":
            finite_array(case["input_ids"], tokens, name + ".input_ids", integer=True)
            if any(i < 0 or i >= 40 for i in case["input_ids"]):
                raise ValueError("invalid miniature token")
        else:
            finite_array(case["input"], 6 if kind == "router" else tokens * 32, name + ".input")
        layer = case.get("layer", 0)
        heads, kv_heads, dim = (4, 2, 8) if layer < 2 else (2, 1, 16)
        lengths = {key: tokens * 32 for key in ("output", "attn_norm", "mlp_norm", "attention_output", "post_attention", "attn_conv", "mlp_conv", "routed_output", "shared_output", "mlp_output")}
        lengths.update({key: tokens * heads * dim for key in ("q_proj", "q_norm", "query_scaled")})
        lengths.update({key: tokens * kv_heads * dim for key in ("k_proj", "k_norm", "v_proj", "k_conv", "v_conv")})
        lengths.update({key: tokens * heads * tokens for key in ("relative_bias", "relative_scaled", "attention_probs")})
        lengths.update(r_proj=tokens * heads * 4, router_logits=tokens * 6, router_weights=tokens * 2, shared_gammas=tokens * 2)
        for key, values in case["traces"].items():
            if key not in lengths or key not in oracle["tolerances"]:
                raise ValueError(f"unknown trace {name}.{key}")
            size = len(case.get("trace_samples", {}).get(key, [])) if "trace_samples" in case else lengths[key]
            finite_array(values, size, name + "." + key)
        required = set()
        if kind == "layer" or (kind == "attention" and tokens == 2):
            required.update(("q_proj", "k_proj", "v_proj", "r_proj", "q_norm", "k_norm", "k_conv", "v_conv", "relative_bias", "attention_probs", "attention_output"))
        if kind == "layer":
            required.update(("attn_norm", "attn_conv", "post_attention", "mlp_norm", "mlp_output", "mlp_conv", "output"))
        if kind in ("mlp", "layer") and layer > 0:
            required.update(("router_logits", "router_weights", "shared_gammas", "routed_output", "shared_output"))
        if kind == "mlp":
            required.add("mlp_output")
        if kind == "attention" and tokens == 2:
            required.update(("query_scaled", "relative_scaled"))
        if kind == "attention" and tokens > 512:
            required = {"attention_probs", "relative_bias"}
            if set(case["traces"]) != required:
                raise ValueError(f"missing discriminating trace coverage: {name}")
            coordinates = [(0, 0, 1), (tokens - 2, 0, 0), (tokens - 1, 0, 0),
                (tokens - 1, 0, 1), (tokens - 1, 0, tokens - 1),
                (tokens - 2, heads - 1, 0), (tokens - 1, heads - 1, 0), (tokens - 1, heads - 1, 1)]
            samples = sorted({(q * heads + h) * tokens + k for q, h, k in coordinates})
            if case.get("trace_samples") != {stage: samples for stage in required}:
                raise ValueError("missing or invalid attention boundary samples")
            for stage in required:
                finite_array(case["trace_samples"][stage], len(samples), stage + ".samples", integer=True)
                for index, value in zip(samples, case["traces"][stage]):
                    q, k = index // (heads * tokens), index % tokens
                    masked = k > q or (stage == "attention_probs" and layer < 2 and q - k >= 512)
                    outside = stage == "relative_bias" and q - k >= (512 if layer < 2 else 1024)
                    if (masked or outside) and value != 0:
                        raise ValueError("oracle has nonzero masked boundary sample")
                    if stage == "relative_bias" and q - k == tokens - 2 and value == 0:
                        raise ValueError("oracle is missing visible in-range relative bias")
                    if stage == "attention_probs" and not masked and value <= 0:
                        raise ValueError("oracle is missing visible boundary attention")
        elif "trace_samples" in case:
            raise ValueError("unexpected sampled trace")
        if set(case["traces"]) != required:
            raise ValueError(f"missing discriminating trace coverage: {name}")
        selected = kind == "router" or (kind in ("mlp", "layer") and layer > 0)
        if set(case["indices"]) != ({"router_selected"} if selected else set()):
            raise ValueError(f"missing router index coverage: {name}")
        if selected:
            finite_array(case["indices"]["router_selected"], tokens * 2, name + ".router_selected", integer=True)
            if any(i not in range(4) for i in case["indices"]["router_selected"]):
                raise ValueError("invalid router expert ID")
        if kind == "router":
            finite_array(case["bias"], 4, name + ".bias")
            if case["route_scale"] != 8 or case["global_scale"] != 1.375:
                raise ValueError("wrong router fixture scales")
        if "attention_probs" in case["traces"] and "trace_samples" not in case:
            probs = case["traces"]["attention_probs"]
            for q in range(tokens):
                for h in range(heads):
                    row = probs[(q * heads + h) * tokens:(q * heads + h + 1) * tokens]
                    if any(v != 0 for k, v in enumerate(row) if k > q or (layer < 2 and q - k >= 512)):
                        raise ValueError("oracle has nonzero masked attention probability")


def compare(actual, expected, limit, label):
    if len(actual) != len(expected):
        raise ValueError(f"{label}: length {len(actual)} != {len(expected)}")
    absolute, relative, worst_ratio, failures = 0.0, 0.0, 0.0, []
    for index, (a, e) in enumerate(zip(actual, expected)):
        if not math.isfinite(a) or not math.isfinite(e):
            raise ValueError(f"{label}[{index}]: nonfinite value")
        error = abs(a - e)
        absolute = max(absolute, error)
        if e != 0:
            relative = max(relative, error / abs(e))
        bound = limit["atol"] + limit["rtol"] * abs(e)
        worst_ratio = max(worst_ratio, error / bound if bound else float(error != 0))
        exact_zero = label.endswith(("attention_probs", "relative_bias")) and e == 0 and a != 0
        lost_probability = label.endswith("attention_probs") and e > 0 and a <= 0
        if error > bound or exact_zero or lost_probability:
            if len(failures) < 3:
                failures.append(f"[{index}] actual={a:.9g}, expected={e:.9g}, abs={error:.3g}, limit={bound:.3g}")
    if failures:
        raise ValueError(f"{label}: " + "; ".join(failures))
    return dict(count=len(actual), max_absolute_error=absolute, max_relative_error=relative,
                max_tolerance_fraction=worst_ratio, limits=limit)


def run_cases(probe, directory, oracle_path, oracle, *, ranges=None):
    results = []
    for index, case in enumerate(oracle["cases"]):
        with tempfile.TemporaryDirectory(prefix="inkling-model-probe-") as temporary:
            output = Path(temporary) / "actual.json"
            command = [str(probe.resolve()), str(directory.resolve()), str(oracle_path.resolve()), str(index), str(output)]
            if ranges is not None:
                command.append(str(ranges.resolve()))
            process = subprocess.run(command, capture_output=True, text=True)
            if process.returncode:
                raise ValueError(f"{case['name']}: probe failed ({process.returncode})\n{process.stderr}")
            actual = load(output)
        limits = oracle["tolerances"]
        output_limit = limits["router_weights"] if case["kind"] == "router" else limits["output"]
        record = dict(name=case["name"], output=compare(actual["output"], case["expected"], output_limit, case["name"] + ".output"), traces={})
        values, indices = {}, {}
        for event in actual.get("events", []):
            table, field = (indices, "indices") if "indices" in event else (values, "values")
            table.setdefault(event["name"], []).extend(event[field])
        for name, expected in case.get("traces", {}).items():
            if name not in values:
                raise ValueError(f"{case['name']}: missing trace {name}")
            record["traces"][name] = compare(values[name], expected, limits[name], case["name"] + "." + name)
        for name, expected in case.get("indices", {}).items():
            if indices.get(name) != expected:
                raise ValueError(f"{case['name']}.{name}: {indices.get(name)} != {expected}")
        results.append(record)
        print(f"PASS {case['name']}: {record['output']['count']} outputs, max_abs={record['output']['max_absolute_error']:.6g}, {len(record['traces'])} traces", flush=True)
    return results


def audit_real_case(manifest):
    layer = manifest.get("layer")
    if type(layer) is not int or layer not in (0, 2, 5):
        raise ValueError("unsupported real layer")
    if manifest.get("tools") != {"torch": "2.8.0", "torch_git_revision": "a1cb3cc05d46d198467bebbb6e8fba50a325d4e7"}:
        raise ValueError("wrong real deterministic tool identity")
    if manifest.get("references") != MODEL_SOURCE_PINS or manifest.get("tolerances") != tolerances(real=True):
        raise ValueError("wrong real provenance or fixed tolerances")
    case = manifest["case"]
    if (case.get("name") != f"real_layer_{layer}" or case.get("layer") != layer
            or case.get("tokens") != 1 or case.get("kind") != "layer"
            or case.get("position_start") != 0 or case.get("shape") != [1, 1, 4096]
            or "trace_samples" in case):
        raise ValueError("wrong real case contract")
    for key in ("input", "expected"):
        finite_array(case[key], 4096, "real." + key)
        if sha256(struct.pack("<4096f", *case[key])) != manifest[key + "_sha256"]:
            raise ValueError("modified real " + key)
    if case["input"] != [((i * 73) % 257 - 128) / 128 for i in range(4096)]:
        raise ValueError("wrong deterministic real input")
    lengths = {key: 4096 for key in ("output", "attn_norm", "mlp_norm", "attention_output", "post_attention",
        "attn_conv", "mlp_conv", "mlp_output", "q_proj", "q_norm")}
    lengths.update({key: 1024 for key in ("k_proj", "k_norm", "v_proj", "k_conv", "v_conv")})
    lengths.update(r_proj=512, relative_bias=32, attention_probs=32)
    if layer:
        lengths.update(router_logits=258, router_weights=6, shared_gammas=2, routed_output=4096, shared_output=4096)
    if set(case["traces"]) != set(lengths):
        raise ValueError("missing real trace coverage")
    for key, size in lengths.items():
        finite_array(case["traces"][key], size, "real." + key)
    if case["traces"]["output"] != case["expected"] or case["traces"]["attention_probs"] != [1.0] * 32:
        raise ValueError("inconsistent real output or attention probabilities")
    if set(case["indices"]) != ({"router_selected"} if layer else set()):
        raise ValueError("missing real router index coverage")
    selected = None
    if layer:
        ids = case["indices"]["router_selected"]
        finite_array(ids, 6, "real.router_selected", integer=True)
        if len(set(ids)) != 6 or any(i not in range(256) for i in ids):
            raise ValueError("invalid real expert IDs")
        selected = sorted(ids)
        selection = manifest.get("selection", {})
        if set(selection) != {"expert_ids", "margin", "router_indices", "router_weights", "shared_gammas"}:
            raise ValueError("missing real selection provenance")
        for key, size in (("expert_ids", 6), ("router_indices", 6), ("router_weights", 6), ("shared_gammas", 2)):
            finite_array(selection[key], size, "real.selection." + key, integer=key in ("expert_ids", "router_indices"))
        margin = selection["margin"]
        if type(margin) not in (int, float) or not math.isfinite(margin) or not 0 < margin <= 1:
            raise ValueError("invalid real selection margin")
        if (selection["expert_ids"] != selected or sorted(selection["router_indices"]) != selected
                or selection["shared_gammas"] != case["traces"]["shared_gammas"]
                or dict(zip(selection["router_indices"], selection["router_weights"])) != dict(zip(ids, case["traces"]["router_weights"]))):
            raise ValueError("real selection provenance mismatch")
    elif "selection" in manifest:
        raise ValueError("unexpected dense selection provenance")
    prefix = f"model.llm.layers.{layer}."
    mappings = [dict(source=prefix + source, target=target) for source, target in LAYER_MAPPING.items()]
    pairs = (("mlp.w13_dn.weight", "mlp.gate_proj.weight"), ("mlp.w13_dn.weight", "mlp.up_proj.weight"),
        ("mlp.w2_md.weight", "mlp.down_proj.weight"), ("mlp.global_scale", "mlp.global_scale")) if not layer else (
        ("mlp.gate.weight", "mlp.gate.weight"), ("mlp.gate.bias", "mlp.gate.e_score_correction_bias"),
        ("mlp.gate.global_scale", "mlp.gate.global_scale"),
        ("mlp.shared_experts.shared_w13_weight", "mlp.shared_experts.gate_proj"),
        ("mlp.shared_experts.shared_w13_weight", "mlp.shared_experts.up_proj"),
        ("mlp.shared_experts.shared_w2_weight", "mlp.shared_experts.down_proj"))
    mappings.extend(dict(source=prefix + source, target=target) for source, target in pairs)
    if layer:
        mappings.extend(dict(source=prefix + "mlp.experts." + source, target="mlp.experts." + target, expert_ids=selected)
            for source, target in (("w13_weight", "gate_up_proj"), ("w2_weight", "down_proj")))
    if manifest.get("mappings") != mappings:
        raise ValueError("wrong real storage mapping coverage")
    return dict(schema=1, cases=[case], tolerances=manifest["tolerances"])


def audit_real(directory, metadata):
    manifest = load(directory / "ranges.json")
    oracle = audit_real_case(manifest)
    if manifest.get("generators") != generator_hashes():
        raise ValueError("real generator changed without regeneration")
    fetch_nvfp4_fixture.METADATA = metadata
    tensors = fetch_nvfp4_fixture.catalogue()
    RangeStore(directory, manifest["layer"], tensors, offline=True)
    total = sum(r["bytes"] for r in manifest["ranges"])
    if total > BUDGET:
        raise ValueError("invalid real payload budget")
    seen = {}
    for record in manifest["ranges"]:
        previous = seen.get(record["tensor"], -1)
        if record["offset"] < previous:
            raise ValueError("overlapping or unordered real ranges")
        seen[record["tensor"]] = record["offset"] + record["bytes"]
    return oracle, total


def main():
    parser = argparse.ArgumentParser(description='Audit model fixtures, run the C probe, and compare every recorded scalar.')
    parser.add_argument("--probe", type=Path, default=Path("bin/model_probe"))
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--real", type=Path, action="append", default=[])
    parser.add_argument("--metadata", type=Path, default=Path("tests/fixtures/checkpoint"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    reports = []
    if args.fixtures is not None or not args.real:
        directory = args.fixtures or Path("tests/fixtures/model")
        oracle = audit_tiny(directory)
        reports.extend(run_cases(args.probe, directory, directory / "oracle.json", oracle))
    total = 0
    for directory in args.real:
        oracle, payload = audit_real(directory, args.metadata)
        total += payload
        if total > BUDGET:
            raise ValueError("aggregate real payload exceeds 2 GiB")
        with tempfile.TemporaryDirectory(prefix="inkling-real-oracle-") as temporary:
            path = Path(temporary) / "oracle.json"
            path.write_text(json.dumps(oracle, allow_nan=False))
            reports.extend(run_cases(args.probe, args.metadata, path, oracle, ranges=directory / "ranges.json"))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(dict(schema=1, cases=reports, real_payload_bytes=total), sort_keys=True, indent=2) + "\n")
    print(f"PASS {len(reports)} model/block cases; verified real payload bytes={total}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"model acceptance failed: {error}") from None
