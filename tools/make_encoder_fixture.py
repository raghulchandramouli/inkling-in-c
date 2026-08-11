"""Build a tiny fixture mirroring Inkling's audio/vision encoder tensors.

Encoder weights ship as BF16, norms as F32 -- in both the BF16 and the
NVFP4 checkpoints. A decoy model.llm.* tensor is included so the
enumerate-by-prefix test can prove it filters correctly.
"""
from pathlib import Path
import struct


def bf16(value):
    # BF16 is the top 16 bits of the float32 bit pattern (round toward
    # zero). Values chosen below are all exactly representable.
    return struct.pack("<f", value)[2:4]


def bf16_row_major(values):
    return b"".join(bf16(v) for row in values for v in row)


def f32(values):
    return b"".join(struct.pack("<f", v) for v in values)


tensors = [
    # name, dtype, byte payload
    ("model.audio.encoder.weight", "BF16",
     bf16_row_major([[1.0, -2.0, 0.5], [1.5, -0.5, 3.0]]), [2, 3]),
    ("model.audio.final_norm.weight", "F32",
     f32([1.0, 2.0, 3.0]), [3]),
    ("model.visual.layers.linear_0.weight", "BF16",
     bf16_row_major([[0.5, -0.5], [2.0, -4.0]]), [2, 2]),
    ("model.visual.final_norm.weight", "F32",
     f32([1.0, -1.0]), [2]),
    # decoy: must NOT be returned for the audio/visual prefixes
    ("model.llm.embed.weight", "F32", f32([7.0, 8.0]), [2]),
]

entries = {}
offset = 0
data = b""
for name, dtype, payload, shape in tensors:
    entries[name] = {
        "dtype": dtype,
        "shape": shape,
        "data_offsets": [offset, offset + len(payload)],
    }
    data += payload
    offset += len(payload)

import json
header = json.dumps(entries, separators=(",", ":")).encode("utf-8")

output = Path("tests/fixtures/encoders.safetensors")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_bytes(struct.pack("<Q", len(header)) + header + data)

print(f"wrote {output}")
print(f"header: {len(header)} bytes")
print(f"data:   {len(data)} bytes")
print(f"total:  {output.stat().st_size} bytes")
