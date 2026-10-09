"""Offline provenance, shape, range and checksum checks; no ML dependency."""
import json
import math
from pathlib import Path
import shutil
import struct
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from fetch_nvfp4_fixture import MODEL, REVISION, REFERENCES, catalogue, sha256
import fetch_nvfp4_fixture as generator

fixture = json.loads(Path("tests/fixtures/nvfp4.json").read_text())
assert fixture["model"] == MODEL and fixture["revision"] == REVISION
assert fixture["schema_version"] == 1 and fixture["tools"]["torch"].split("+")[0] == "2.8.0"
metadata = catalogue()
for key, (repo, revision, path) in REFERENCES.items():
    ref = fixture["references"][key]
    assert ref["revision"] == revision
    assert ref["url"] == f"https://raw.githubusercontent.com/{repo}/{revision}/{path}"
    assert len(ref["sha256"]) == 64
for name, tensor in fixture["tensors"].items():
    assert tensor == metadata[name]


def validate(record, name):
    assert record["tensor"] == name
    tensor = metadata[name]
    raw = bytes.fromhex(record["hex"])
    assert len(raw) == record["length"] and sha256(raw) == record["sha256"]
    start, end = tensor["data_offsets"]
    offset = record["tensor_offset"]
    assert offset >= 0 and offset + len(raw) <= end - start
    assert record["data_offset"] == start + offset
    assert record["file_offset"] == 8 + tensor["header_size"] + start + offset
    return raw


for projection in fixture["projections"]:
    name = projection["weight"]
    logical = struct.unpack("<3q", validate(projection["original_shape"], name + ".original_shape"))
    assert list(logical) == projection["logical_shape"]
    experts, rows, columns = logical
    assert metadata[name]["shape"] == [experts, rows, columns // 2]
    assert metadata[name + ".scale"]["shape"] == [experts, rows, columns // 16]
    global_scales = struct.unpack("<256f", validate(projection["scale2"], name + ".scale2"))
    assert all(math.isfinite(s) and s > 0 for s in global_scales)
    assert len(set(global_scales[e] for e in (0, 7, 255))) == 3
    amax = validate(projection["input_amax"], name + ".input_amax")
    assert len(amax) == 2
    assert struct.unpack("<f", b"\0\0" + amax)[0] > 0
    assert {(s["expert"], s["row"], s["block"]) for s in projection["samples"]} == {
        (e, r, b) for e in (0, 7, 255) for r, b in ((0, 0), (1, 3), (rows - 1, columns // 16 - 2))}
    for sample in projection["samples"]:
        raw = validate(sample["packed"], name)
        scales = validate(sample["scales"], name + ".scale")
        assert len(raw) == 16 and len(scales) == 2
        linear = (sample["expert"] * rows + sample["row"]) * columns + sample["block"] * 16
        assert sample["packed"]["tensor_offset"] == linear // 2
        assert sample["scales"]["tensor_offset"] == linear // 16
        assert len(sample["expected_f32_bits"]) == 32
        assert all(0 <= bits <= 0xffffffff for bits in sample["expected_f32_bits"])

def rejects(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("invalid regeneration input was accepted")


name = fixture["projections"][0]["weight"]
with patch.object(generator, "request") as network:
    for offset, length in ((-1, 8), (0, 0), (0, 4097), (2**64, 8)):
        rejects(lambda: generator.fetch_range(metadata, name, offset, length))
    network.assert_not_called()

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary) / "checkpoint"
    shutil.copytree(generator.METADATA, root)
    with patch.object(generator, "METADATA", root):
        manifest_path = root / "metadata-manifest.json"
        original = manifest_path.read_bytes()
        for key in ("model", "revision"):
            manifest = json.loads(original)
            manifest[key] = "incorrect"
            manifest_path.write_text(json.dumps(manifest))
            rejects(generator.catalogue)
        manifest_path.write_bytes(original)
        for path in (root / "config.json", next((root / "headers").glob("*.safetensors"))):
            original = path.read_bytes()
            path.write_bytes(original + b" ")
            rejects(generator.catalogue)
            path.write_bytes(original)

    output = Path(temporary) / "must-not-exist.json"
    with patch.dict(sys.modules, torch=SimpleNamespace(__version__="2.7.0")):
        rejects(lambda: generator.generate(output))
    # These guards must fail before touching either network or torch operations.
    with patch.dict(sys.modules, torch=SimpleNamespace(__version__="2.8.0+cpu")), \
         patch.object(generator, "reference_oracles", return_value=(None, None, {})), \
         patch.object(generator, "fetch_range") as ranges, \
         patch.object(generator, "request") as network:
        ranges.return_value = {"hex": struct.pack("<3q", 256, 4096, 33).hex()}
        rejects(lambda: generator.generate(output))
        ranges.return_value = {"hex": struct.pack("<3q", 256, 4096, 4096).hex()}
        changed = dict(metadata)
        changed[name] = dict(metadata[name], dtype="BF16")
        with patch.object(generator, "catalogue", return_value=changed):
            rejects(lambda: generator.generate(output))
        network.assert_not_called()
    assert not output.exists()

print("NVFP4 fixture: provenance, sample coverage and guarded regeneration checks OK")

# Optional regeneration check: real pinned CPU oracles, recorded checkpoint bytes.
# Ordinary make test remains offline and does not import torch.
if "--with-oracle" in sys.argv:
    import torch
    te_decode, modelopt_class, references = generator.reference_oracles(torch)
    records = {}
    for projection in fixture["projections"]:
        ranges = [projection[key] for key in ("original_shape", "scale2", "input_amax")]
        ranges += [sample[key] for sample in projection["samples"] for key in ("packed", "scales")]
        for record in ranges:
            records[record["tensor"], record["tensor_offset"]] = record

    def recorded_range(tensors, name, offset=0, length=None):
        record = records[name, offset]
        if length is not None:
            assert length == record["length"]
        return record

    class DisagreeingOracle(modelopt_class):
        def dequantize(self, *args, **kwargs):
            return super().dequantize(*args, **kwargs) + 1.0

    with tempfile.TemporaryDirectory() as temporary, \
         patch.object(generator, "fetch_range", side_effect=recorded_range):
        output = Path(temporary) / "fixture.json"
        with patch.object(generator, "reference_oracles", return_value=(te_decode, modelopt_class, references)):
            generator.generate(output)
        regenerated = json.loads(output.read_text())
        assert {k: v for k, v in regenerated.items() if k != "tools"} == {
            k: v for k, v in fixture.items() if k != "tools"}
        previous = output.read_bytes()
        with patch.object(generator, "reference_oracles", return_value=(te_decode, DisagreeingOracle, references)):
            rejects(lambda: generator.generate(output))
        assert output.read_bytes() == previous
    print("NVFP4 regeneration: exact oracle reproduction and disagreement rejection OK")
