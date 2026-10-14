#!/usr/bin/env python3
"""Regenerate B2/B3 oracles from recorded bytes and verified pinned CPU sources."""
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import struct
import tempfile
from pathlib import Path
from types import SimpleNamespace

from fetch_checkpoint_metadata import sha256, write_atomic
import fetch_nvfp4_fixture
from inkling_ref import SOURCE_DIR, bits, reference_oracles, architecture

SEED = 932184


def generate(output, *, b2_fixture=Path("tests/fixtures/nvfp4.json"),
             source_dir=SOURCE_DIR, allow_download=True):
    import torch
    if torch.__version__.split("+")[0] != "2.8.0":
        raise ValueError("regeneration requires torch==2.8.0")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rng = random.Random(SEED)
    upstream, provenance = architecture(torch, source_dir=source_dir, allow_download=allow_download)
    _, modelopt_class, nv_references = reference_oracles(torch, source_dir=source_dir, allow_download=allow_download)

    def tensor(values):
        return torch.tensor(values, dtype=torch.float32)

    def random_tensor(*shape):
        return tensor([rng.uniform(-2, 2) for _ in range(math.prod(shape))]).reshape(shape)

    def record(name, **values):
        return dict(name=name, **{key: bits(value.detach(), torch) if isinstance(value, torch.Tensor)
                                 else value for key, value in values.items()})

    result = dict(schema_version=1, seed=SEED,
                  tools=dict(python=platform.python_version(), torch=torch.__version__,
                             torch_git_revision=torch.version.git_version),
                  references=dict(transformers=provenance, **nv_references),
                  tolerances={"matvec": {"atol": 5e-5, "rtol": 5e-5},
                              "norm": {"atol": 2e-6, "rtol": 2e-6},
                              "activation": {"atol": 2e-6, "rtol": 2e-6},
                              "softmax": {"atol": 2e-7, "rtol": 2e-6},
                              "conv": {"atol": 3e-6, "rtol": 3e-6}},
                  bf16=[], nvfp4=[], norm=[], activation=[], softmax=[], topk=[], conv=[])

    for rows, columns in ((1, 1), (3, 5), (4, 33), (2, 257)):
        weights = random_tensor(rows, columns).to(torch.bfloat16)
        x = random_tensor(columns)
        result["bf16"].append(record(f"random_{rows}x{columns}", rows=rows, columns=columns,
                                    weights=[n & 65535 for n in weights.view(torch.int16).flatten().tolist()],
                                    x=x, expected=torch.mv(weights.float(), x)))
    for name, weights, x in (
        ("zeros", [[0, -0.0, 0], [1, -1, 2]], [0, 0, 0]),
        ("cancellation", [[1, -1, 1, -1], [0.5, 2, -4, 8]], [100, 100, 0.25, 0.25]),
        ("bf16_boundary", [[2**-133, 2**-126, -2**-126, -0.0]], [1, 2, 3, 4]),
    ):
        w, v = tensor(weights).to(torch.bfloat16), tensor(x)
        result["bf16"].append(record(name, rows=w.shape[0], columns=w.shape[1],
                                    weights=[n & 65535 for n in w.view(torch.int16).flatten().tolist()],
                                    x=v, expected=torch.mv(w.float(), v)))

    def nv_case(name, packed, scales, globals_, rows, columns, x):
        packed_tensor = torch.tensor(packed, dtype=torch.uint8).reshape(-1, rows, columns // 2)
        scale_tensor = torch.tensor(scales, dtype=torch.uint8).view(torch.float8_e4m3fn).reshape(-1, rows, columns // 16)
        expected = []
        for e, global_ in enumerate(globals_):
            oracle = modelopt_class()
            oracle._quantized_data = packed_tensor[e]
            oracle.metadata = {"shape": (rows, columns), "dtype": torch.float32}
            decoded = oracle.dequantize(dtype=torch.float32, scale=scale_tensor[e],
                                       double_scale=tensor(global_), block_sizes={-1: 16})
            expected.append(torch.mv(decoded, x))
        result["nvfp4"].append(record(name, experts=len(globals_), rows=rows, columns=columns,
                                     packed=packed, scales=scales, scale2=tensor(globals_),
                                     x=x, expected=torch.stack(expected)))

    for experts, rows, columns in ((1, 1, 16), (2, 3, 32), (3, 4, 64)):
        packed = [rng.randrange(256) for _ in range(experts * rows * columns // 2)]
        scales = [rng.randrange(1, 80) for _ in range(experts * rows * columns // 16)]
        nv_case(f"random_{experts}x{rows}x{columns}", packed, scales,
                [0.125 * (e + 1) for e in range(experts)], rows, columns, random_tensor(columns))
    nv_case("zero_scale", list(range(8)), [0], [1], 1, 16, random_tensor(16))
    nv_case("scale_extremes", [0x21, 0xcb] * 8, [1, 126], [0.25], 1, 32, random_tensor(32))
    real_path = Path("tests/fixtures/nvfp4.json")
    real_raw = b2_fixture.read_bytes()
    real = json.loads(real_raw)
    result["b2_fixture"] = dict(path=str(real_path), sha256=sha256(real_raw))
    for p in real["projections"]:
        global_values = struct.unpack("<256f", bytes.fromhex(p["scale2"]["hex"]))
        packed = [b for sample in p["samples"] for b in bytes.fromhex(sample["packed"]["hex"])]
        scales = [b for sample in p["samples"] for b in bytes.fromhex(sample["scales"]["hex"])]
        nv_case(p["weight"], packed, scales, [global_values[e] for e in (0, 7, 255)],
                3, 32, random_tensor(32))

    for name, x, weights, eps in (
        ("singleton", tensor([-3]), tensor([2]), 1e-6),
        ("zero", torch.zeros(8), random_tensor(8), 1e-6),
        ("tiny", tensor([1e-20, -1e-20, 0]), tensor([1, 2, -1]), 1e-6),
        ("random", random_tensor(128), random_tensor(128), 1e-5),
        *[(f"head_{head}", random_tensor(8), random_tensor(8), 1e-6) for head in range(3)],
    ):
        model = upstream["InklingRMSNorm"](x.numel(), eps=eps)
        model.weight.data.copy_(weights)
        result["norm"].append(record(name, x=x, weights=weights, epsilon=eps, expected=model(x)))

    for name, x in (("random", random_tensor(37)), ("zero", tensor([0, -0.0])),
                    ("extremes", tensor([-1000, -100, -20, -1, 0, 1, 20, 100, 1000,
                                         -3.4028234663852886e38, 3.4028234663852886e38]))):
        result["activation"].append(record(name, x=x, sigmoid=torch.sigmoid(x), silu=torch.nn.functional.silu(x)))
    for name, x in (("random", random_tensor(37)), ("singleton", tensor([7])),
                    ("ties", tensor([0, -0.0, 0])), ("large", tensor([1000, 1001, 999])),
                    ("masked", tensor([-float("inf"), 2, -float("inf"), 1])),
                    ("extremes", tensor([-3.4028234663852886e38, 3.4028234663852886e38]))):
        result["softmax"].append(record(name, x=x, expected=torch.softmax(x, dim=0)))
    for name, x, k in (("random", random_tensor(23), 6), ("singleton", tensor([2]), 1),
                       ("ties", tensor([3, 1, 3, 3, -1, 3]), 3),
                       ("all", tensor([0, -0.0, 0, -1]), 4),
                       ("infinities", tensor([-float("inf"), float("inf"), 1, float("inf")]), 4)):
        # torch.topk does not promise stable ties; stable argsort specifies our
        # portable tie contract independently of the C insertion algorithm.
        indices = torch.argsort(x, descending=True, stable=True)[:k]
        result["topk"].append(record(name, x=x, k=k, expected=indices.tolist()))

    conv_cases = [("start", torch.ones(1, 1), tensor([[1, 2, 3, 4]]), torch.zeros(1, 3)),
                  ("impulse", tensor([[1, 0], [0, 1], [0, 0], [0, 0], [0, 0]]),
                   tensor([[1, 2, 3, 4], [-2, 0.5, 3, -1]]), torch.zeros(2, 3)),
                  ("zero_kernel", random_tensor(2, 3), torch.zeros(3, 4), random_tensor(3, 3)),
                  ("short_carry", random_tensor(2, 3), random_tensor(3, 4), random_tensor(3, 3))]
    for stream in ("key", "value", "attention_output", "mlp_output"):
        conv_cases.append((stream, random_tensor(7, 5), random_tensor(5, 4), torch.zeros(5, 3)))
    for name, x, weights, initial in conv_cases:
        tokens, channels = x.shape
        state = initial.unsqueeze(0).clone()
        expected = upstream["causal_conv1d_update"](x.T.unsqueeze(0), state, weights).squeeze(0).T + x
        model = upstream["InklingShortConvolution"](channels, 4, 0, 0)
        model.conv1d.weight.data.copy_(weights.unsqueeze(1))
        if torch.count_nonzero(initial) == 0:
            torch.testing.assert_close(model(x.unsqueeze(0)).squeeze(0), expected, atol=3e-6, rtol=3e-6)
        carried = initial.unsqueeze(0).clone()
        cache = SimpleNamespace(layers=[SimpleNamespace(conv_states=[carried], record_past=False)],
                                has_previous_state=lambda *_: True)
        step_outputs = torch.cat([model(x[t:t + 1].unsqueeze(0), past_key_values=cache).squeeze(0)
                                  for t in range(tokens)])
        torch.testing.assert_close(step_outputs, expected, atol=3e-6, rtol=3e-6)
        assert torch.equal(carried, state)
        outer = random_tensor(tokens, channels) if name in ("attention_output", "mlp_output") else torch.zeros_like(x)
        result["conv"].append(record(name, tokens=tokens, channels=channels, x=x, weights=weights,
                                    initial=initial, expected=expected, final=state,
                                    outer=outer, with_outer=expected + outer))
    write_atomic(output, (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    print(f"wrote {output}: " + ", ".join(f"{key}={len(result[key])}" for key in
          ("bf16", "nvfp4", "norm", "activation", "softmax", "topk", "conv")))


GENERATOR_FILES = (
    "tools/inkling_ref.py",
    "tools/emit_fixtures.py",
    "tools/fetch_nvfp4_fixture.py",
    "tools/fetch_checkpoint_metadata.py",
)
FIXTURE_FILES = ("nvfp4.json", "primitives.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("tests/fixtures"))
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--offline", action="store_true",
                        help="require verified cached reference sources; never access network")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="inkling-oracles-") as temporary:
        stage = Path(temporary)
        fetch_nvfp4_fixture.generate(stage / "nvfp4.json", recorded=fetch_nvfp4_fixture.OUTPUT,
                                    source_dir=args.source_dir, allow_download=not args.offline)
        generate(stage / "primitives.json", b2_fixture=stage / "nvfp4.json",
                 source_dir=args.source_dir, allow_download=not args.offline)
        manifest = dict(schema_version=1,
                        generators={name: sha256(Path(name).read_bytes()) for name in GENERATOR_FILES},
                        fixtures={name: sha256((stage / name).read_bytes()) for name in FIXTURE_FILES})
        for name in FIXTURE_FILES:
            write_atomic(args.output_dir / name, (stage / name).read_bytes())
        write_atomic(args.output_dir / "oracle-manifest.json",
                     (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        raise SystemExit(f"oracle regeneration failed: {error}") from None
