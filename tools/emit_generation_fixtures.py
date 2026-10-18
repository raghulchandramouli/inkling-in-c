#!/usr/bin/env python3
"""Regenerate generation expectations from authenticated native model bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

from check_model import audit_tiny
from inkling_model_ref import (MODEL_SOURCE_PINS, build_model, causal_masks,
    checkpoint_config, decode_nvfp4, load_reference, storage_tensor)
from inkling_ref import reference_oracles

PROJECT = Path(__file__).resolve().parents[1]
MODEL_MANIFEST_SHA256 = "03ba854b30b14bf46f3f1436fcf66729b713e62b0b0f8e50b701488d693e79d0"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def authenticated_model(directory):
    if digest(directory / "manifest.json") != MODEL_MANIFEST_SHA256:
        raise ValueError("wrong original miniature model manifest")
    return audit_tiny(directory)


class NativeCheckpoint:
    def __init__(self, directory):
        authenticated_model(directory)
        index = json.loads((directory / "model.safetensors.index.json").read_text())
        self.tensors = {}
        for filename in sorted(set(index["weight_map"].values())):
            raw = (directory / filename).read_bytes()
            size = struct.unpack_from("<Q", raw)[0]
            header = json.loads(raw[8:8 + size])
            for name, item in header.items():
                if name == "__metadata__":
                    continue
                if index["weight_map"].get(name) != filename or name in self.tensors:
                    raise ValueError("inconsistent native tensor index")
                start, end = item["data_offsets"]
                if not 0 <= start <= end <= len(raw) - 8 - size:
                    raise ValueError("invalid native tensor range")
                self.tensors[name] = (item["dtype"], item["shape"], raw[8 + size + start:8 + size + end])
        if set(self.tensors) != set(index["weight_map"]):
            raise ValueError("incomplete native tensor payload")

    def reader(self, torch, modelopt, te_decode):
        def read(name, expert=None):
            dtype, shape, raw = self.tensors[name]
            value = storage_tensor(raw, dtype, shape, torch)
            if expert is not None:
                value = value[expert]
            if dtype == "U8":
                shape = tuple(read(name + ".original_shape").tolist())[1:]
                scales = read(name + ".scale", expert)
                scale2 = read(name + ".scale2")[expert]
                decoded = decode_nvfp4(value, scales, scale2, shape, torch, modelopt)
                independent = te_decode(value, torch.float32).float().reshape(shape)
                independent = independent * scales.float().repeat_interleave(16, dim=-1) * scale2
                torch.testing.assert_close(decoded, independent, atol=0, rtol=0)
                return decoded
            return value
        return read


def native_reference(directory, source_dir):
    import torch
    if torch.__version__ != "2.8.0" or torch.version.git_version != "a1cb3cc05d46d198467bebbb6e8fba50a325d4e7":
        raise ValueError("requires original torch 2.8.0 CPU oracle tool identity")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoint = NativeCheckpoint(directory)
    reference = load_reference(torch, source_dir=source_dir, allow_download=False)
    te_decode, modelopt, _ = reference_oracles(torch, source_dir=source_dir, allow_download=False)
    config = checkpoint_config(json.loads((directory / "config.json").read_text()))
    model, mappings = build_model(config, checkpoint.reader(torch, modelopt, te_decode), reference, torch)
    def logits(ids):
        with torch.no_grad():
            value = model(input_ids=torch.tensor([ids]),
                          attention_mask=causal_masks(reference, config, len(ids), torch),
                          use_cache=False).logits
        return value[0, -1].tolist()
    return logits, mappings


MASK64 = (1 << 64) - 1
RNG_VECTORS = {
    "seed": 0,
    "outputs_hex": ["e220a8397b1dcdaf", "6e789e6aa1b965f4", "06c45d188009454f",
                    "f88bb8a8724c81ec", "1b39896a51a8749b"],
}


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    value ^= value >> 31
    return state, value, (value >> 11) * (2.0 ** -53)


def rng_vectors():
    state = RNG_VECTORS["seed"]
    result = []
    for literal in RNG_VECTORS["outputs_hex"]:
        state, word, uniform = splitmix64(state)
        if word != int(literal, 16):
            raise ValueError("SplitMix64 literal test vector disagreement")
        result.append(dict(state_hex=f"{state:016x}", output_hex=literal, uniform=uniform))
    return dict(seed=RNG_VECTORS["seed"], draws=result)

RNG_BOUNDARIES = [
    {"seed": 7046029254386353131, "output_hex": "0000000000000000", "uniform": 0.0},
    {"seed": 3453682501520545093, "output_hex": "8000000000000000", "uniform": 0.5},
    {"seed": 3558559446808474027, "output_hex": "ffffffffffffffff", "uniform": 1.0 - 2.0 ** -53},
]


def rng_boundaries():
    for vector in RNG_BOUNDARIES:
        _, word, uniform = splitmix64(vector["seed"])
        if word != int(vector["output_hex"], 16) or uniform != vector["uniform"]:
            raise ValueError("SplitMix64 boundary vector disagreement")
    return RNG_BOUNDARIES


def effective_options(options, width):
    temperature = options["temperature"]
    return dict(max_new_tokens=options["max_new_tokens"], temperature=temperature if temperature else 0.0,
                top_k=min(options["top_k"] or width, width) if temperature else 1,
                top_p=options["top_p"] if temperature else 1.0, seed=options["seed"])


def sample(logits, options, state):
    ranked = sorted(range(len(logits)), key=lambda index: (-logits[index], index))
    if not options["temperature"]:
        return dict(token_id=ranked[0], state_hex=f"{state:016x}", support_ids=ranked[:1],
                    probabilities=[1.0], masses=[1.0], uniform=None, cdf_margin=None)
    ranked = ranked[:options["top_k"]]
    maximum = logits[ranked[0]]
    temperature = options["temperature"]
    masses = []
    for index in ranked:
        gap = maximum - logits[index]
        if gap == 0:
            mass = 1.0
        elif temperature < 1 and gap > float.fromhex("0x1.fffffffffffffp+1023") * temperature:
            mass = 0.0
        else:
            mass = math.exp(-gap / temperature)
        masses.append(mass)
    total = 0.0
    for mass in masses:
        total += mass
    if options["top_p"] < 1:
        cumulative = 0.0
        for count, mass in enumerate(masses, 1):
            cumulative += mass
            if cumulative >= options["top_p"] * total:
                ranked, masses = ranked[:count], masses[:count]
                break
    total = 0.0
    for mass in masses:
        total += mass
    state, _, uniform = splitmix64(state)
    threshold = uniform * total
    cumulative = 0.0
    selected = next(index for index, mass in reversed(list(zip(ranked, masses))) if mass > 0)
    boundaries = [0.0]
    for index, mass in zip(ranked, masses):
        cumulative += mass
        boundaries.append(cumulative / total)
        if cumulative > threshold:
            selected = index
            break
    return dict(token_id=selected, state_hex=f"{state:016x}", support_ids=ranked,
                probabilities=[mass / total for mass in masses], masses=masses,
                uniform=uniform, cdf_margin=min(abs(uniform - boundary) for boundary in boundaries))


def options(max_new_tokens=1, temperature=0.0, top_k=0, top_p=1.0, seed=0):
    return dict(max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p, seed=seed)


GENERATION_SPECS = [
    dict(name="eos_immediate", prompt_ids=[2], context_capacity=16, options=options(6)),
    dict(name="eos_multistep", prompt_ids=[1], context_capacity=16, options=options(6)),
    dict(name="stored_only_prompt", prompt_ids=[39], context_capacity=16, options=options(6)),
    dict(name="requested_length", prompt_ids=[0], context_capacity=16, options=options(5)),
    dict(name="eos_in_prompt", prompt_ids=[6], context_capacity=16, options=options(12)),
    dict(name="zero_output", prompt_ids=[2], context_capacity=1, options=options(0)),
    dict(name="full_context", prompt_ids=[1], context_capacity=1, options=options(4)),
    dict(name="context_cutoff", prompt_ids=[0], context_capacity=4, options=options(8)),
    dict(name="length_context_tie", prompt_ids=[0], context_capacity=4, options=options(3)),
    dict(name="eos_length_context_tie", prompt_ids=[2], context_capacity=2, options=options(1)),
    dict(name="long_prompt", prompt_ids=[2, 5, 7, 39, 8, 9], context_capacity=16, options=options(4)),
    dict(name="greedy_ignores_valid_filters", prompt_ids=[1], context_capacity=16, options=options(6, -0.0, 2, 0.01, 42)),
    dict(name="sampled_top_k_top_p", prompt_ids=[1], context_capacity=16, options=options(6, 0.7, 9, 0.8, 42)),
    dict(name="sampled_full_support", prompt_ids=[39], context_capacity=16, options=options(5, 1.2, 0, 1.0, 0)),
]


def sampler_specs():
    cases = [
        ("greedy_tie", [1, 1, -5], options(temperature=0, top_k=2, top_p=0.1, seed=42)),
        ("greedy_negative_zero", [-2, -1, -1], options(temperature=-0.0, seed=MASK64)),
        ("negative_logits", [-1000, -1001, -1002, -1003], options(temperature=1, seed=42)),
        ("top_k_one_consumes_draw", [-1, 3, 2], options(temperature=1, top_k=1, seed=42)),
        ("top_k_clamps", [1, 0, -1], options(temperature=1, top_k=MASK64, seed=42)),
        ("top_k_stable_ties", [1, 1, 1, 1], options(temperature=1, top_k=2, seed=0)),
        ("top_p_exact_prefix", [0, 0, 0, 0], options(temperature=1, top_p=0.5, seed=0)),
        ("top_p_crossing_token", [2, 1, 0], options(temperature=1, top_k=2, top_p=0.75, seed=0)),
        ("top_p_singleton", [2, 1, 0], options(temperature=1, top_k=2, top_p=0.7, seed=0)),
        ("top_p_subnormal", [0, 0], options(temperature=1, top_p=float.fromhex("0x0.0000000000001p-1022"), seed=0)),
        ("temperature_subnormal", [1, 0, -1], options(temperature=float.fromhex("0x0.0000000000001p-1022"), seed=0)),
        ("temperature_huge", [-3.4028234663852886e38, 3.4028234663852886e38, 0], options(temperature=float.fromhex("0x1.fffffffffffffp+1023"), seed=0)),
        ("finite_float_extremes", [-3.4028234663852886e38, 3.4028234663852886e38, 0], options(temperature=1, seed=0)),
        ("subnormal_mass_retained", [0, -745.1], options(temperature=1, seed=0)),
    ]
    for label, vector in zip(("zero", "half", "maximum"), RNG_BOUNDARIES):
        cases.append(("cdf_" + label, [0, 0], options(temperature=1, seed=vector["seed"])))
    return [dict(name=name, logits=[struct.unpack("<f", struct.pack("<f", value))[0] for value in logits], options=opts)
            for name, logits, opts in cases]


def generate_case(spec, logits, model_max_length):
    effective = effective_options(spec["options"], 37)
    ids = list(spec["prompt_ids"])
    generated, steps = [], []
    state = effective["seed"]
    usable = min(spec["context_capacity"], model_max_length)
    stop = "length" if effective["max_new_tokens"] == 0 else "context"
    while len(generated) < effective["max_new_tokens"] and len(ids) < usable:
        row = logits(ids)
        ranked = sorted(range(37), key=lambda index: (-row[index], index))
        selection = sample(row, effective, state)
        state = int(selection["state_hex"], 16)
        token = selection["token_id"]
        steps.append(dict(prefix_ids=list(ids), logits=row, token_id=token,
                          greedy_id=ranked[0], greedy_margin=row[ranked[0]] - row[ranked[1]],
                          sampler=selection))
        generated.append(token)
        ids.append(token)
        if token == 6:
            stop = "eos"
            break
        if len(generated) == effective["max_new_tokens"]:
            stop = "length"
            break
    return dict(**spec, effective=effective, generated_ids=generated, generated_count=len(generated),
                stop_reason=stop, steps=steps, final_state_hex=f"{state:016x}")


def generate(output, directory, source_dir):
    import check_generation_fixture as checker
    logits, mappings = native_reference(directory, source_dir)
    original = authenticated_model(directory)
    for case in original["cases"]:
        if case["kind"] == "model" and logits(case["input_ids"]) != case["expected"][-37:]:
            raise ValueError("native-byte reader disagrees with original model fixture")
    config = json.loads((directory / "config.json").read_text())["text_config"]
    generation = [generate_case(spec, logits, config["model_max_length"]) for spec in GENERATION_SPECS]
    sampler = []
    for spec in sampler_specs():
        effective = effective_options(spec["options"], len(spec["logits"]))
        sampler.append(dict(**spec, effective=effective, expected=sample(spec["logits"], effective, effective["seed"])))
    document = dict(schema=1, model_manifest_sha256=MODEL_MANIFEST_SHA256,
                    dependencies=checker.dependency_hashes(), references=MODEL_SOURCE_PINS,
                    tools=original["tools"], mappings=mappings, logit_tolerance=checker.LOGIT_LIMIT,
                    rng_vectors=rng_vectors(), rng_boundaries=rng_boundaries(),
                    generation=generation, sampler=sampler)
    document["payload_sha256"] = checker.payload_digest(document)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n")
    checker.audit(output, directory)
    print(f"wrote {output}: {len(generation)} native generation cases, {len(sampler)} sampler cases")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=PROJECT / "tests/fixtures/model")
    parser.add_argument("--output", type=Path, default=PROJECT / "tests/fixtures/generation.json")
    args = parser.parse_args()
    generate(args.output, args.model, args.source_dir)


if __name__ == "__main__":
    main()
