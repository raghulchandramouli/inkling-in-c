import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from check_model import audit_tiny, compare


def run(command, expected):
    result = subprocess.run([str(x) for x in command], capture_output=True, text=True)
    assert result.returncode == expected, (command, result.returncode, result.stderr)
    return result


def synthetic_ranges(directory):
    records = []
    suffixes = (".q_norm.weight", ".k_norm.weight", ".k_sconv.weight", ".v_sconv.weight",
                ".attn_norm.weight", ".mlp_norm.weight", ".attn_sconv.weight", ".mlp_sconv.weight",
                ".rel_logits_proj.proj", ".mlp.global_scale")
    for path in sorted((ROOT / "tests/fixtures/checkpoint/headers").glob("model-*.safetensors")):
        with path.open("rb") as file:
            header = json.loads(file.read(struct.unpack("<Q", file.read(8))[0]))
        for name, tensor in header.items():
            if not name.startswith("model.llm.layers.0.") or not name.endswith(suffixes):
                continue
            assert tensor["dtype"] == "BF16"
            value = b"\x80\x3f" if "norm.weight" in name or "global_scale" in name else b"\0\0"
            raw = value * math.prod(tensor["shape"])
            digest = hashlib.sha256(raw).hexdigest()
            relative = f"blobs/{digest}.bin"
            (directory / relative).parent.mkdir(exist_ok=True)
            (directory / relative).write_bytes(raw)
            records.append(dict(tensor=name, offset=0, bytes=len(raw), file=relative, sha256=digest))
    assert len(records) == 10
    return dict(schema=1, checkpoint_revision="b6a99534467840620d411e4cd4ad5819b2610d9c",
                layer=0, ranges=records, test_only="synthetic vectors; no real numerical evidence")


def range_guards(probe, directory):
    path = directory / "ranges.json"
    original = synthetic_ranges(directory)
    command = [probe, "--bind-layer", ROOT / "tests/fixtures/checkpoint", path]
    path.write_text(json.dumps(original))
    run(command, 0)
    mutations = [
        lambda m: m.update(schema=2),
        lambda m: m.update(checkpoint_revision="0" * 40),
        lambda m: m.update(layer=42),
        lambda m: m["ranges"][0].update(offset=-1),
        lambda m: m["ranges"][0].update(offset=1),
        lambda m: m["ranges"][0].update(bytes=2 ** 64),
        lambda m: m["ranges"][0].update(file="../escape.bin"),
        lambda m: m["ranges"][0].update(file="/tmp/escape.bin"),
        lambda m: m["ranges"][0].update(sha256="g" * 64),
        lambda m: m["ranges"].append(copy.deepcopy(m["ranges"][0])),
        lambda m: m["ranges"].pop(),
    ]
    for mutate in mutations:
        manifest = copy.deepcopy(original)
        mutate(manifest)
        path.write_text(json.dumps(manifest))
        run(command, 1)
    path.write_text(json.dumps(original))
    blob = directory / original["ranges"][0]["file"]
    raw = blob.read_bytes()
    blob.write_bytes(raw[:-1])
    run(command, 1)
    blob.write_bytes(raw)
    with tempfile.TemporaryDirectory(prefix="inkling-outside-range-") as outside:
        external = Path(outside) / "outside.bin"
        external.write_bytes(raw)
        link = directory / "blobs/escape.bin"
        link.symlink_to(external)
        manifest = copy.deepcopy(original)
        manifest["ranges"][0]["file"] = "blobs/escape.bin"
        path.write_text(json.dumps(manifest))
        run(command, 1)
    path.write_text(json.dumps(original))
    run(command, 0)


def cli_guards(cli, fixture_cli, directory):
    target = directory / "logits.json"
    target.write_text("keep")
    for ids in ("", ",", "1,", "-1", "+1", "1 2", "0x1", "4294967296", "1,,2", "a"):
        run([cli, directory / "missing", "--ids", ids, "--dump-logits", target], 2)
        assert target.read_text() == "keep"
    fixture = ROOT / "tests/fixtures/model"
    run([cli, fixture, "--ids", "2", "--dump-logits", target], 1)
    assert target.read_text() == "keep"
    oracle = audit_tiny(fixture)
    for case in oracle["cases"][:3]:
        run([fixture_cli, fixture, "--ids", ",".join(map(str, case["input_ids"])), "--dump-logits", target], 0)
        result = json.loads(target.read_text())
        assert result["tokens"] == case["tokens"] and result["vocab"] == 37
        compare(result["logits"], case["expected"], oracle["tolerances"]["output"], case["name"])
    target.write_text("keep")
    run([fixture_cli, fixture, "--ids", "40", "--dump-logits", target], 1)
    assert target.read_text() == "keep"
    run([fixture_cli, fixture, "--ids", "2", "--dump-logits", directory / "absent/out.json"], 1)
    assert not list(directory.glob("*.tmp.*"))


def main():
    assert len(sys.argv) == 4, "expected model_probe, inkling, and model_cli_probe"
    probe, cli, fixture_cli = (Path(x).resolve() for x in sys.argv[1:])
    with tempfile.TemporaryDirectory(prefix="inkling-model-guards-") as temporary:
        directory = Path(temporary)
        range_guards(probe, directory)
        cli_guards(cli, fixture_cli, directory)
    print("PASS synthetic range rejection/retry and raw-ID CLI output/failure checks")


if __name__ == "__main__":
    main()
