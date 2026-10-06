"""Network-free CLI contract checks; sparse shards contain headers, never weights."""
import json
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


BINARY = Path(sys.argv[1]).resolve()
FIXTURES = Path(sys.argv[2]).resolve()
INDEX = "model.safetensors.index.json"
GOLDEN = {
    "trunk": (798, 21505668260),
    "routed-packed": (78, 125627793408),
    "scales-and-auxiliaries": (312, 15703556076),
    "embeddings-unembed": (2, 3293577216),
    "vision": (8, 128160768),
    "audio": (2, 10493952),
    "mtp": (160, 4463824912),
}


def read_json(path):
    return json.loads(path.read_bytes())


def write_json(path, value):
    path.write_text(json.dumps(value))


def read_header(path):
    with path.open("rb") as file:
        size = struct.unpack("<Q", file.read(8))[0]
        return json.loads(file.read(size))


def write_header(path, header):
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded)


def invoke(root, *options, success=True, diagnostic=""):
    result = subprocess.run(
        [str(BINARY), str(root), "--verify-model", *options],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == (0 if success else 3), (result.returncode, result.stderr)
    if not success:
        assert diagnostic in result.stderr, result.stderr
        assert "verification OK" not in result.stdout
    return result.stdout


def metadata_case(change, diagnostic):
    with tempfile.TemporaryDirectory(prefix="inkling-verify-") as directory:
        root = Path(directory) / "checkpoint"
        shutil.copytree(FIXTURES, root)
        change(root)
        invoke(root, "--metadata-only", "--with-mtp", success=False, diagnostic=diagnostic)


def change_json(root, filename, change):
    path = root / filename
    value = read_json(path)
    change(value)
    write_json(path, value)


def change_tensor(root, name, change):
    shard = read_json(root / INDEX)["weight_map"][name]
    path = root / "headers" / shard
    header = read_header(path)
    change(header[name])
    write_header(path, header)


output = invoke(FIXTURES, "--metadata-only", "--with-mtp")
assert "full files NOT verified" in output
assert "total: tensors=1360 bytes=170733074592 indexed=1360" in output
for name, (count, size) in GOLDEN.items():
    assert f"census {name}: tensors={count} bytes={size}" in output

# Independently derive the census and every listing from the pinned headers.
counts = {name: [0, 0] for name in GOLDEN}
headers = {}
weight_map = read_json(FIXTURES / INDEX)["weight_map"]
for path in sorted((FIXTURES / "headers").glob("*.safetensors")):
    for name, tensor in read_header(path).items():
        if name == "__metadata__":
            continue
        headers[name] = tensor
        assert weight_map[name] == path.name
        start, end = tensor["data_offsets"]
        shape = ",".join(str(axis) for axis in tensor["shape"])
        assert (f"tensor {name} dtype={tensor['dtype']} shape=[{shape}] "
                f"shard={path.name} data_range=[{start},{end})") in output
        if name.startswith("model.mtp."):
            kind = "mtp"
        elif name.startswith("model.visual."):
            kind = "vision"
        elif name.startswith("model.audio."):
            kind = "audio"
        elif name.endswith((".input_amax", ".original_shape", ".scale", ".scale2")):
            kind = "scales-and-auxiliaries"
        elif name in ("model.llm.embed.weight", "model.llm.unembed.weight"):
            kind = "embeddings-unembed"
        elif tensor["dtype"] == "U8":
            kind = "routed-packed"
        else:
            kind = "trunk"
        counts[kind][0] += 1
        counts[kind][1] += end - start
assert {name: tuple(value) for name, value in counts.items()} == GOLDEN
listed = [line.split()[1] for line in output.splitlines() if line.startswith("tensor ")]
assert listed == sorted(weight_map)
assert sum(weight_map[name] != weight_map[name + ".scale"]
           for name, tensor in headers.items() if tensor["dtype"] == "U8") > 50

with tempfile.TemporaryDirectory(prefix="inkling-verify-") as directory:
    root = Path(directory) / "checkpoint"
    shutil.copytree(FIXTURES, root)
    # JSON member order must not change any reported metadata or census.
    change_json(root, INDEX, lambda value: value.update(weight_map=dict(reversed(list(value["weight_map"].items())))))
    assert invoke(root, "--metadata-only", "--with-mtp") == output
    (root / "headers" / "mtp.safetensors").unlink()
    skipped = invoke(root, "--metadata-only")
    assert "MTP: skipped" in skipped and "census mtp: tensors=0 bytes=0" in skipped
    assert "total: tensors=1200 bytes=166269249680 indexed=1360" in skipped
    invoke(root, "--metadata-only", "--with-mtp", success=False, diagnostic="required shard")

# Actual-file mode: sparse truncation allocates no tensor payloads. This also
# demonstrates that .original_shape payload values are intentionally NOT read.
with tempfile.TemporaryDirectory(prefix="inkling-sparse-") as directory:
    root = Path(directory)
    for filename in ("config.json", INDEX, "hf_quant_config.json"):
        shutil.copyfile(FIXTURES / filename, root / filename)
    for source in (FIXTURES / "headers").glob("*.safetensors"):
        destination = root / source.name
        shutil.copyfile(source, destination)
        payload_size = max(tensor["data_offsets"][1] for name, tensor in read_header(source).items()
                           if name != "__metadata__")
        with destination.open("r+b") as file:
            file.truncate(source.stat().st_size + payload_size)
    actual = invoke(root, "--with-mtp")
    assert "actual shard lengths; headers only" in actual
    assert actual.split("census ", 1)[1] == output.split("census ", 1)[1]
    (root / "mtp.safetensors").unlink()
    assert "MTP: skipped" in invoke(root)
    invoke(root, "--with-mtp", success=False, diagnostic="required shard")
    path = root / "model-00001-of-00009.safetensors"
    with path.open("r+b") as file:
        file.truncate(path.stat().st_size - 1)
    invoke(root, success=False, diagnostic="invalid shard")

invoke(FIXTURES / "does-not-exist", success=False, diagnostic="existing directory")
invoke(FIXTURES / "config.json", success=False, diagnostic="existing directory")
invoke(FIXTURES, success=False, diagnostic="required shard")
for filename in ("config.json", INDEX, "hf_quant_config.json", "headers/model-00003-of-00009.safetensors"):
    metadata_case(lambda root, filename=filename: (root / filename).unlink(), filename)

metadata_case(lambda root: change_json(root, INDEX, lambda value: value["metadata"].update(total_size=1)), "index identity")
metadata_case(lambda root: change_json(root, INDEX, lambda value: value["weight_map"].update(
    {"model.llm.embed.weight": "model-00001-of-00009.safetensors"})), "invalid shard")
for name in ("model.llm.norm.weight", "model.llm.layers.3.mlp.experts.w13_weight.scale"):
    metadata_case(lambda root, name=name: change_json(root, INDEX, lambda value: value["weight_map"].pop(name)),
                  "missing required tensor " + name)

for name in ("model.llm.norm.weight", "model.llm.layers.3.mlp.experts.w13_weight.scale"):
    metadata_case(lambda root, name=name: change_tensor(root, name, lambda tensor: tensor.update(shape=[64,64]
        if name == "model.llm.norm.weight" else [256,256,4096])), "dtype/shape mismatch")
metadata_case(lambda root: change_tensor(root, "model.llm.layers.3.mlp.gate.bias",
    lambda tensor: tensor.update(dtype="I64", shape=[128])), "dtype/shape mismatch")
metadata_case(lambda root: change_tensor(root, "model.mtp.layers.0.embed_norm.weight",
    lambda tensor: tensor.update(shape=[64,64])), "dtype/shape mismatch")

for section, key, value in (("text_config", "hidden_size", 2048), ("text_config", "q_bias", True),
                            ("vision_config", "patch_size", 20), ("audio_config", "n_mel_bins", 40),
                            ("mtp_config", "num_nextn_predict_layers", 4)):
    metadata_case(lambda root, section=section, key=key, value=value: change_json(root, "config.json",
        lambda config: config[section].update({key: value})), "unsupported config")
for section in ("text_config", "mtp_config"):
    metadata_case(lambda root, section=section: change_json(root, "config.json",
        lambda config: config[section].update(local_layer_ids=[0] * len(config[section]["local_layer_ids"]))),
        "local_layer_ids")
for key, value in (("group_size", 32), ("quant_algo", "FP8")):
    metadata_case(lambda root, key=key, value=value: change_json(root, "hf_quant_config.json",
        lambda config: config["quantization"].update({key: value})), "hf_quant_config.json")
metadata_case(lambda root: change_json(root, "hf_quant_config.json", lambda config:
    config["quantization"]["exclude_modules"].__setitem__(0, config["quantization"]["exclude_modules"][1])),
    "hf_quant_config.json")
metadata_case(lambda root: change_json(root, "hf_quant_config.json", lambda config:
    config["quantization"]["modelopt_quant_config"]["quant_cfg"]["*input_quantizer"].update(enable=False)),
    "hf_quant_config.json")

# Rename in both index and header: reconciliation alone is not enough to pass.
def rename_tensor(root, old, new):
    index = read_json(root / INDEX)
    shard = index["weight_map"].pop(old)
    index["weight_map"][new] = shard
    write_json(root / INDEX, index)
    path = root / "headers" / shard
    header = read_header(path)
    header[new] = header.pop(old)
    write_header(path, header)


metadata_case(lambda root: rename_tensor(root, "model.llm.layers.3.mlp.experts.w13_weight.scale2",
                                       "model.llm.layers.3.mlp.experts.w13_weight.extra"), "missing required tensor")
metadata_case(lambda root: rename_tensor(root, "model.mtp.layers.0.embed_norm.weight",
                                       "model.unknown.optional.weight"), "missing required tensor")


def extra_tensor(root):
    shard = "model-00001-of-00009.safetensors"
    name = "model.llm.layers.0.attn.extra.weight"
    change_json(root, INDEX, lambda index: index["weight_map"].update({name: shard}))
    path = root / "headers" / shard
    header = read_header(path)
    header[name] = {"dtype": "BF16", "shape": [0], "data_offsets": [0, 0]}
    write_header(path, header)


metadata_case(extra_tensor, "unexpected tensor in required checkpoint families")


def missing_header_tensor(root):
    path = root / "headers" / weight_map["model.llm.norm.weight"]
    header = read_header(path)
    del header["model.llm.norm.weight"]
    write_header(path, header)


metadata_case(missing_header_tensor, "invalid shard")

normal = subprocess.run([str(BINARY), str(FIXTURES / "config.json")], capture_output=True, text=True, timeout=30)
assert normal.returncode == 0 and normal.stdout.startswith("Inkling-Small configuration OK\n")
assert "local layers:        35 [0,1,2,3,4,6" in normal.stdout
for options in (("--bad-option",), ("--with-mtp", "--with-mtp"), ("--metadata-only", "--metadata-only")):
    result = subprocess.run([str(BINARY), str(FIXTURES), "--verify-model", *options], capture_output=True, timeout=30)
    assert result.returncode == 2
print("checkpoint verifier: census, sparse actual files, MTP gate, negative cases and legacy CLI OK")
