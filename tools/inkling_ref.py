"""Independent pinned CPU reference functions, loaded only for regeneration."""
from __future__ import annotations

import ast
from pathlib import Path

from fetch_checkpoint_metadata import request, sha256, write_atomic

SOURCE_DIR = Path(".context/oracle-sources")
REFERENCES = {
    "modelopt": ("NVIDIA/Model-Optimizer", "b311c054de4052df9c7f3de9409b7598f44a0dba",
                 "modelopt/torch/quantization/qtensor/nvfp4_tensor.py"),
    "transformer_engine": ("NVIDIA/TransformerEngine", "699ed6ec9e9c600c3d0acd3a488c4ca6f339e5d9",
                           "transformer_engine/pytorch/custom_recipes/reference_nvfp4.py"),
    "loader": ("vllm-project/vllm", "e378275a8f20eac9e92212e4a1fee84ec5a31dc7",
               "vllm/models/inkling/nvidia/moe.py"),
    "mapper": ("vllm-project/vllm", "e378275a8f20eac9e92212e4a1fee84ec5a31dc7",
               "vllm/models/inkling/nvidia/model.py"),
}

SOURCE_SHA256 = {
    "loader": "4aebfef3d16d7ea2d9289647a606a1893bf5f8d5f518220aa8e165bed169c217",
    "mapper": "a29cd2b83922a5eae50590dc52607888ee17f58fcc6abf28aead499d15166a8c",
    "modelopt": "e128fd62f0d0c4aa179e057b53a615bccf42f30f8ed9ef7c8f7e3dc84c5eb61e",
    "transformer_engine": "9213bd1effe9ecffc3e922e308fb4f26f6213f57150ba24ff8f42a47cc132fbb",
    "transformers": "7bb8710c9eed16786231a6923b895ec46b385aa97509ef2f1a6e93fd8a0d1338",
}

TRANSFORMERS = ("huggingface/transformers", "c587bc884db2c2e31fc2b8102314656b17aa07b1",
                "src/transformers/models/inkling/modeling_inkling.py")
SOURCE_PINS = {
    key: dict(url=f"https://raw.githubusercontent.com/{repo}/{revision}/{path}",
              revision=revision, sha256=SOURCE_SHA256[key])
    for key, (repo, revision, path) in dict(REFERENCES, transformers=TRANSFORMERS).items()
}


def source_bytes(key, *, source_dir=SOURCE_DIR, allow_download=True):
    pin = SOURCE_PINS[key]
    path = source_dir / (pin["sha256"] + ".py")
    if path.exists():
        raw = path.read_bytes()
    elif allow_download:
        raw = request(pin["url"], max_bytes=2 * 1024 * 1024)
    else:
        raise ValueError(f"missing offline reference source: {path}")
    if sha256(raw) != pin["sha256"]:
        raise ValueError(f"reference source checksum mismatch: {key} ({path})")
    if not path.exists():
        write_atomic(path, raw)
    return raw


def reference_oracles(torch, *, source_dir=SOURCE_DIR, allow_download=True):
    """Execute only the original pinned CPU functions, without GPU package imports."""
    sources, provenance = {}, {}
    for key in REFERENCES:
        raw = source_bytes(key, source_dir=source_dir, allow_download=allow_download)
        sources[key] = ast.parse(raw, filename=SOURCE_PINS[key]["url"])
        provenance[key] = dict(SOURCE_PINS[key])
    te_function = next(n for n in sources["transformer_engine"].body
                       if isinstance(n, ast.FunctionDef) and n.name == "cast_from_fp4x2")
    te_namespace = {"torch": torch}
    exec(compile(ast.Module(body=[te_function], type_ignores=[]),
                 provenance["transformer_engine"]["url"], "exec"), te_namespace)
    source_class = next(n for n in sources["modelopt"].body
                        if isinstance(n, ast.ClassDef) and n.name == "NVFP4QTensor")
    source_class.body = [n for n in source_class.body
                         if (isinstance(n, ast.FunctionDef) and n.name in
                             ("get_e2m1_values", "dequantize")) or
                         (isinstance(n, ast.Assign) and any(
                             isinstance(t, ast.Name) and t.id == "e2m1_values_on_device"
                             for t in n.targets))]
    table = next(n for n in sources["modelopt"].body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "e2m1_values" for t in n.targets))
    mo_namespace = {"torch": torch, "BaseQuantizedTensor": object}
    exec(compile(ast.Module(body=[table, source_class], type_ignores=[]),
                 provenance["modelopt"]["url"], "exec"), mo_namespace)
    return te_namespace["cast_from_fp4x2"], mo_namespace["NVFP4QTensor"], provenance


def bits(values, torch):
    return [n & 0xffffffff for n in values.contiguous().view(torch.int32).flatten().tolist()]


def architecture(torch, *, source_dir=SOURCE_DIR, allow_download=True):
    raw = source_bytes("transformers", source_dir=source_dir, allow_download=allow_download)
    selected = {"InklingRMSNorm", "InklingShortConvolution", "causal_conv1d_update",
                "causal_conv1d_fn", "apply_mask_to_padding_states"}
    nodes = [n for n in ast.parse(raw).body
             if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in selected]
    if {n.name for n in nodes} != selected:
        raise ValueError("pinned architecture functions missing")
    # Bypass only GPU/hub decorators; execute original classes/function bodies.
    for node in nodes:
        for child in ast.walk(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef)):
                child.decorator_list = []
    namespace = {"torch": torch, "nn": torch.nn, "F": torch.nn.functional}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), SOURCE_PINS["transformers"]["url"], "exec"), namespace)
    return namespace, dict(SOURCE_PINS["transformers"])
