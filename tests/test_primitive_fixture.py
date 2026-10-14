"""Offline fixture provenance and schema audit; Python standard library only."""
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import struct
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import emit_fixtures
from emit_fixtures import FIXTURE_FILES, GENERATOR_FILES, SEED
from fetch_checkpoint_metadata import sha256
import fetch_nvfp4_fixture as b2_generator
import inkling_ref

TOLERANCES = {"matvec": {"atol": 5e-5, "rtol": 5e-5},
              "norm": {"atol": 2e-6, "rtol": 2e-6},
              "activation": {"atol": 2e-6, "rtol": 2e-6},
              "softmax": {"atol": 2e-7, "rtol": 2e-6},
              "conv": {"atol": 3e-6, "rtol": 3e-6}}
CASES = {
    "bf16": ["random_1x1", "random_3x5", "random_4x33", "random_2x257", "zeros",
             "cancellation", "bf16_boundary"],
    "nvfp4": ["random_1x1x16", "random_2x3x32", "random_3x4x64", "zero_scale",
              "scale_extremes", "model.llm.layers.3.mlp.experts.w13_weight",
              "model.llm.layers.3.mlp.experts.w2_weight"],
    "norm": ["singleton", "zero", "tiny", "random", "head_0", "head_1", "head_2"],
    "activation": ["random", "zero", "extremes"],
    "softmax": ["random", "singleton", "ties", "large", "masked", "extremes"],
    "topk": ["random", "singleton", "ties", "all", "infinities"],
    "conv": ["start", "impulse", "zero_kernel", "short_carry", "key", "value",
             "attention_output", "mlp_output"],
}


def integers(values, count, maximum=0xffffffff):
    assert isinstance(values, list) and len(values) == count
    assert all(type(n) is int and 0 <= n <= maximum for n in values)


def floats(values, count, *, infinities=False):
    integers(values, count)
    decoded = [struct.unpack("<f", struct.pack("<I", n))[0] for n in values]
    assert all(not math.isnan(n) and (infinities or math.isfinite(n)) for n in decoded)
    return decoded


def dimension(item, key):
    value = item[key]
    assert type(value) is int and 0 < value <= 4096
    return value


def audit_primitives(fixture, b2_raw):
    b2 = json.loads(b2_raw)
    assert set(fixture) == set(CASES) | {"schema_version", "seed", "tools", "references",
                                       "tolerances", "b2_fixture"}
    assert type(fixture["schema_version"]) is int and fixture["schema_version"] == 1
    assert type(fixture["seed"]) is int and fixture["seed"] == SEED == 932184
    assert fixture["references"] == inkling_ref.SOURCE_PINS
    assert fixture["tolerances"] == TOLERANCES
    assert fixture["tools"] == b2["tools"]
    tools = fixture["tools"]
    assert set(tools) == {"python", "torch", "torch_git_revision"}
    assert re.fullmatch(r"3\.11\.\d+", tools["python"])
    assert tools["torch"].split("+")[0] == "2.8.0"
    assert re.fullmatch(r"[0-9a-f]{40}", tools["torch_git_revision"])
    assert fixture["b2_fixture"] == {"path": "tests/fixtures/nvfp4.json", "sha256": sha256(b2_raw)}
    for kind, names in CASES.items():
        cases = fixture[kind]
        assert [item["name"] for item in cases] == names
        for item in cases:
            if kind == "bf16":
                assert set(item) == {"name", "rows", "columns", "weights", "x", "expected"}
                rows, columns = dimension(item, "rows"), dimension(item, "columns")
                integers(item["weights"], rows * columns, 65535)
                floats(item["x"], columns)
                floats(item["expected"], rows)
            elif kind == "nvfp4":
                assert set(item) == {"name", "experts", "rows", "columns", "packed", "scales",
                                     "scale2", "x", "expected"}
                experts, rows, columns = (dimension(item, key) for key in ("experts", "rows", "columns"))
                assert columns % 16 == 0
                integers(item["packed"], experts * rows * columns // 2, 255)
                integers(item["scales"], experts * rows * columns // 16, 126)
                assert all(value > 0 for value in floats(item["scale2"], experts))
                floats(item["x"], columns)
                floats(item["expected"], experts * rows)
            elif kind == "conv":
                assert set(item) == {"name", "tokens", "channels", "x", "weights", "initial",
                                     "expected", "final", "outer", "with_outer"}
                tokens, channels = dimension(item, "tokens"), dimension(item, "channels")
                for key in ("x", "expected", "outer", "with_outer"):
                    floats(item[key], tokens * channels)
                floats(item["weights"], channels * 4)
                for key in ("initial", "final"):
                    floats(item[key], channels * 3)
            else:
                n = len(item["x"])
                assert 0 < n <= 4096
                x = floats(item["x"], n, infinities=kind in ("softmax", "topk"))
                if kind == "norm":
                    assert set(item) == {"name", "x", "weights", "epsilon", "expected"}
                    floats(item["weights"], n)
                    assert type(item["epsilon"]) is float and math.isfinite(item["epsilon"]) and item["epsilon"] > 0
                    floats(item["expected"], n)
                elif kind == "activation":
                    assert set(item) == {"name", "x", "sigmoid", "silu"}
                    for key in ("sigmoid", "silu"):
                        floats(item[key], n)
                elif kind == "softmax":
                    assert set(item) == {"name", "x", "expected"}
                    assert all(value != math.inf for value in x) and any(math.isfinite(value) for value in x)
                    expected = floats(item["expected"], n)
                    assert all(0 <= value <= 1 for value in expected)
                    assert abs(sum(expected) - 1) < 2e-7
                    assert all(item["expected"][i] == 0 for i, value in enumerate(x) if value == -math.inf)
                else:
                    assert set(item) == {"name", "x", "k", "expected"}
                    k = dimension(item, "k")
                    assert k <= n
                    integers(item["expected"], k, n - 1)
                    assert item["expected"] == sorted(range(n), key=lambda i: (-x[i], i))[:k]
    for projection, item in zip(b2["projections"], fixture["nvfp4"][-2:]):
        assert item["name"] == projection["weight"]
        assert (item["experts"], item["rows"], item["columns"]) == (3, 3, 32)
        for source_key, target_key in (("packed", "packed"), ("scales", "scales")):
            assert item[target_key] == [byte for sample in projection["samples"]
                                        for byte in bytes.fromhex(sample[source_key]["hex"])]
        globals_ = struct.unpack("<256I", bytes.fromhex(projection["scale2"]["hex"]))
        assert item["scale2"] == [globals_[e] for e in (0, 7, 255)]


def audit_manifest(manifest, root, fixture_dir):
    assert set(manifest) == {"schema_version", "generators", "fixtures"}
    assert type(manifest["schema_version"]) is int and manifest["schema_version"] == 1
    assert set(manifest["generators"]) == set(GENERATOR_FILES)
    assert set(manifest["fixtures"]) == set(FIXTURE_FILES)
    for name, expected in manifest["generators"].items():
        assert sha256((root / name).read_bytes()) == expected, f"generator changed: {name}; regenerate oracles"
    for name, expected in manifest["fixtures"].items():
        assert sha256((fixture_dir / name).read_bytes()) == expected, f"fixture changed: {name}"


def rejects(call):
    try:
        call()
    except (AssertionError, ValueError, KeyError, TypeError, RuntimeError):
        return
    raise AssertionError("corrupt provenance or fixture was accepted")


def main():
    root = Path(__file__).resolve().parents[1]
    directory = root / "tests/fixtures"
    primitive = json.loads((directory / "primitives.json").read_bytes())
    b2_raw = (directory / "nvfp4.json").read_bytes()
    manifest = json.loads((directory / "oracle-manifest.json").read_bytes())
    audit_manifest(manifest, root, directory)
    audit_primitives(primitive, b2_raw)
    mutations = [
        lambda d: d.update(seed=0),
        lambda d: d.update(schema_version=True),
        lambda d: d["tools"].update(torch="2.7.0"),
        lambda d: d["references"]["modelopt"].update(sha256="0" * 64),
        lambda d: d["b2_fixture"].update(sha256="0" * 64),
        lambda d: d["tolerances"]["softmax"].update(atol=1e-3),
        lambda d: d["bf16"].pop(),
        lambda d: d["bf16"][0].update(rows=0),
        lambda d: d["bf16"][0].update(weights=[65536]),
        lambda d: d["bf16"][0].update(x=[True]),
        lambda d: d["activation"][0]["sigmoid"].pop(),
        lambda d: d["nvfp4"][-1]["packed"].__setitem__(0, 0),
        lambda d: d["nvfp4"][-1]["scale2"].reverse(),
        lambda d: d["softmax"][4]["expected"].__setitem__(0, 1),
        lambda d: d["topk"][2]["expected"].reverse(),
        lambda d: d["conv"][0]["final"].pop(),
    ]
    for mutation in mutations:
        corrupt = deepcopy(primitive)
        mutation(corrupt)
        rejects(lambda: audit_primitives(corrupt, b2_raw))
    for key in ("generators", "fixtures"):
        corrupt = deepcopy(manifest)
        corrupt[key][next(iter(corrupt[key]))] = "0" * 64
        rejects(lambda: audit_manifest(corrupt, root, directory))
    metadata = b2_generator.catalogue()
    with tempfile.TemporaryDirectory() as temporary:
        cache = Path(temporary)
        pin = inkling_ref.SOURCE_PINS["modelopt"]
        with patch.object(inkling_ref, "request") as network:
            rejects(lambda: inkling_ref.source_bytes("modelopt", source_dir=cache, allow_download=False))
            path = cache / (pin["sha256"] + ".py")
            path.write_bytes(b"corrupt source")
            rejects(lambda: inkling_ref.source_bytes("modelopt", source_dir=cache))
            network.assert_not_called()
        path.unlink()
        with patch.object(inkling_ref, "request", return_value=b"wrong download"):
            rejects(lambda: inkling_ref.source_bytes("modelopt", source_dir=cache))
        assert not path.exists()
        output = cache / "output"
        output.mkdir()
        for name in (*FIXTURE_FILES, "oracle-manifest.json"):
            (output / name).write_bytes(b"existing output")
        for failed_stage in ("b2", "b3"):
            with patch.object(b2_generator, "generate", side_effect=(RuntimeError("failed B2")
                              if failed_stage == "b2" else None)), \
                 patch.object(emit_fixtures, "generate", side_effect=RuntimeError("failed B3")):
                rejects(lambda: emit_fixtures.main(["--offline", "--output-dir", str(output)]))
            assert all((output / name).read_bytes() == b"existing output"
                       for name in (*FIXTURE_FILES, "oracle-manifest.json"))
        recorded = cache / "nvfp4.json"
        recorded.write_bytes(b2_raw)
        with patch.object(b2_generator, "request") as network:
            reader = b2_generator.recorded_ranges(recorded, metadata)
            projection = json.loads(b2_raw)["projections"][0]
            record = projection["samples"][0]["packed"]
            assert reader(metadata, record["tensor"], record["tensor_offset"], 16) == record
            rejects(lambda: reader(metadata, record["tensor"], record["tensor_offset"], 8))
            for mutate in (
                lambda d: d["projections"][0]["samples"].pop(),
                lambda d: d["projections"][0]["samples"][0]["packed"].update(file_offset=0),
                lambda d: d["projections"][0]["samples"][0]["packed"].update(sha256="0" * 64),
                lambda d: d["references"]["loader"].update(revision="wrong"),
            ):
                corrupt = json.loads(b2_raw)
                mutate(corrupt)
                recorded.write_text(json.dumps(corrupt))
                rejects(lambda: b2_generator.recorded_ranges(recorded, metadata))
            network.assert_not_called()
    print("Primitive fixtures: code identity, provenance, dimensions, real-byte linkage and corruption rejection OK")


if __name__ == "__main__":
    main()
