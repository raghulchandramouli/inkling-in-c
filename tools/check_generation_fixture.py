#!/usr/bin/env python3
"""Audit generation provenance offline and compare independently produced results."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import emit_generation_fixtures as reference

PROJECT = Path(__file__).resolve().parents[1]
DEPENDENCIES = (
    "tools/check_model.py", "tools/emit_generation_fixtures.py",
    "tools/check_generation_fixture.py", "tools/emit_model_fixtures.py",
    "tools/inkling_model_ref.py", "tools/fetch_model_layers.py",
    "tools/inkling_ref.py", "tools/fetch_checkpoint_metadata.py",
    "tools/fetch_nvfp4_fixture.py",
)
LOGIT_LIMIT = {"atol": 0.0001, "rtol": 0.0001}


def dependency_hashes():
    return {name: reference.digest(PROJECT / name) for name in DEPENDENCIES}


def payload_digest(document):
    payload = {key: value for key, value in document.items() if key != "payload_sha256"}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def finite_list(values, count, label):
    if not isinstance(values, list) or len(values) != count:
        raise ValueError(f"wrong array length for {label}")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError(f"invalid numeric value for {label}")


def compare_logits(actual, expected):
    finite_list(actual, len(expected), "actual logits")
    maximum = 0.0
    for index, (value, target) in enumerate(zip(actual, expected)):
        error = abs(value - target)
        if error > LOGIT_LIMIT["atol"] + LOGIT_LIMIT["rtol"] * abs(target):
            raise ValueError(f"logit {index} differs: {value} != {target}")
        maximum = max(maximum, error)
    return maximum


def compare_generation(actual, expected):
    for key in ("generated_ids", "stop_reason"):
        if actual.get(key) != expected[key]:
            raise ValueError(f"generation {key} mismatch")
    if type(actual.get("generated_count")) is not int or actual["generated_count"] != expected["generated_count"]:
        raise ValueError("generation generated_count mismatch")
    effective = actual.get("effective")
    if not isinstance(effective, dict):
        raise ValueError("generation effective options missing")
    for key, target in expected["effective"].items():
        value = effective.get(key)
        if key == "seed":
            if isinstance(value, str) and value and all("0" <= c <= "9" for c in value) and (value == "0" or value[0] != "0"):
                value = int(value)
            if type(value) is not int or not 0 <= value <= reference.MASK64:
                raise ValueError("generation effective seed invalid")
        elif key in ("max_new_tokens", "top_k"):
            if type(value) is not int:
                raise ValueError(f"generation effective {key} invalid")
        elif type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"generation effective {key} invalid")
        if value != target:
            raise ValueError(f"generation effective {key} mismatch")
    if len(actual.get("steps", [])) != len(expected["steps"]):
        raise ValueError("generation step count mismatch")
    errors = []
    for actual_step, expected_step in zip(actual["steps"], expected["steps"]):
        if actual_step.get("token_id") != expected_step["token_id"]:
            raise ValueError("generation selected token mismatch")
        errors.append(compare_logits(actual_step["logits"], expected_step["logits"]))
    return max(errors, default=0.0)


def compare_sampler(actual, expected):
    for key in ("token_id", "state_hex"):
        if actual.get(key) != expected[key]:
            raise ValueError(f"sampler {key} mismatch")
    if "support_ids" in actual and actual["support_ids"] != expected["support_ids"]:
        raise ValueError("sampler support_ids mismatch")
    for key in ("masses", "probabilities"):
        if key == "probabilities" and key not in actual:
            continue
        values = actual.get(key)
        finite_list(values, len(expected[key]), "sampler " + key)
        if any(a < 0 or (a == 0) != (b == 0) or abs(a - b) > 2e-14
               for a, b in zip(values, expected[key])):
            raise ValueError(f"sampler {key} mismatch")


def audit(path, model_directory=None):
    model_directory = model_directory or PROJECT / "tests/fixtures/model"
    original = reference.authenticated_model(model_directory)
    document = json.loads(path.read_text())
    if document.get("schema") != 1:
        raise ValueError("unsupported generation fixture schema")
    if document.get("model_manifest_sha256") != reference.MODEL_MANIFEST_SHA256:
        raise ValueError("wrong original model provenance")
    if document.get("dependencies") != dependency_hashes():
        raise ValueError("generation dependency changed without regeneration")
    if document.get("references") != reference.MODEL_SOURCE_PINS:
        raise ValueError("wrong original CPU source provenance")
    if document.get("tools") != {"torch": "2.8.0", "torch_git_revision": "a1cb3cc05d46d198467bebbb6e8fba50a325d4e7"}:
        raise ValueError("wrong generation CPU tool identity")
    if document.get("payload_sha256") != payload_digest(document):
        raise ValueError("generation payload digest mismatch")
    if document.get("logit_tolerance") != LOGIT_LIMIT:
        raise ValueError("wrong generation comparison tolerance")
    if document.get("rng_vectors") != reference.rng_vectors() or document.get("rng_boundaries") != reference.rng_boundaries():
        raise ValueError("wrong literal SplitMix64 vectors")
    if document.get("mappings") != original["mappings"]:
        raise ValueError("wrong native storage mappings")
    audit_cases(document)
    return document



def audit_cases(document):
    generation = document.get("generation", [])
    if [case.get("name") for case in generation] != [spec["name"] for spec in reference.GENERATION_SPECS]:
        raise ValueError("missing or reordered native generation coverage")
    for case, spec in zip(generation, reference.GENERATION_SPECS):
        for key, value in spec.items():
            if case.get(key) != value:
                raise ValueError("wrong native generation input " + spec["name"])
        steps = iter(case["steps"])
        def logits(prefix):
            try:
                step = next(steps)
            except StopIteration:
                raise ValueError("missing generation logit step") from None
            if step.get("prefix_ids") != prefix:
                raise ValueError("wrong generation evaluated prefix")
            finite_list(step.get("logits"), 37, "generation logits")
            return step["logits"]
        expected = reference.generate_case(spec, logits, 1048576)
        if next(steps, None) is not None or case != expected:
            raise ValueError("inconsistent generation steps or outcome " + spec["name"])
        for step in case["steps"]:
            if step["greedy_margin"] <= 2 * (LOGIT_LIMIT["atol"] + LOGIT_LIMIT["rtol"] * max(abs(v) for v in step["logits"])):
                raise ValueError("insufficient native greedy separation")
    sampler = document.get("sampler", [])
    specs = reference.sampler_specs()
    if [case.get("name") for case in sampler] != [spec["name"] for spec in specs]:
        raise ValueError("missing or reordered sampler coverage")
    for case, spec in zip(sampler, specs):
        effective = reference.effective_options(spec["options"], len(spec["logits"]))
        expected = dict(**spec, effective=effective,
                        expected=reference.sample(spec["logits"], effective, effective["seed"]))
        if case != expected:
            raise ValueError("inconsistent independent sampler case " + spec["name"])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=PROJECT / "tests/fixtures/generation.json")
    parser.add_argument("--model", type=Path, default=PROJECT / "tests/fixtures/model")
    args = parser.parse_args()
    document = audit(args.fixture, args.model)
    print(f"PASS generation fixture audit: {len(document['generation'])} native cases, {len(document['sampler'])} sampler cases")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"generation fixture audit failed: {error}") from None
