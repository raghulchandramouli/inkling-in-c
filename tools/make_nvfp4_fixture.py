"""Build a tiny NVFP4 fixture matching Inkling's on-disk convention.

An NVFP4 weight is three tensors:
  - <w>          : U8, two E2M1 4-bit codes per byte (low nibble first)
  - <w>.scale    : F8_E4M3, one value per 16 logical elements
  - <w>.scale2   : F32, a single per-tensor global scale

Reconstruction: value = e2m1(code) * e4m3(block_scale) * global_scale

This fixture uses 32 logical elements = 2 blocks of 16, so both the
nibble unpacking and block indexing are exercised. Expected values are
printed for cross-checking against the C test.
"""
from pathlib import Path
import struct
import json

BLOCK_SIZE = 16
N = 32  # logical elements -> 16 packed bytes, 2 block scales

# E2M1 magnitudes; index 0..7, sign bit adds negation (codes 8..15).
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def e2m1(code):
    mag = E2M1[code & 0x7]
    return -mag if (code & 0x8) else mag


def e4m3_decode(byte):
    sign = (byte >> 7) & 1
    exp = (byte >> 3) & 0xF
    mant = byte & 0x7
    if exp == 0:
        mag = (mant / 8.0) * (2.0 ** -6)
    else:
        mag = (1.0 + mant / 8.0) * (2.0 ** (exp - 7))
    return -mag if sign else mag


# codes: cycle 0..15 so we hit every magnitude and both signs, twice.
codes = [i % 16 for i in range(N)]

# pack two nibbles per byte, low nibble = even index
packed = bytearray()
for b in range(N // 2):
    low = codes[2 * b] & 0xF
    high = codes[2 * b + 1] & 0xF
    packed.append(low | (high << 4))

# block scales: block 0 -> 1.0 (0x38), block 1 -> 2.0 (0x40)
block_scale_bytes = bytes([0x38, 0x40])
assert e4m3_decode(0x38) == 1.0
assert e4m3_decode(0x40) == 2.0

global_scale = 0.5
scale2 = struct.pack("<f", global_scale)

# expected f32 output (printed for reference / test cross-check)
expected = []
for i in range(N):
    bs = e4m3_decode(block_scale_bytes[i // BLOCK_SIZE])
    expected.append(e2m1(codes[i]) * bs * global_scale)

tensors = [
    ("w.weight", "U8", bytes(packed), [N // 2]),
    ("w.weight.scale", "F8_E4M3", block_scale_bytes, [2]),
    ("w.weight.scale2", "F32", scale2, [1]),
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

header = json.dumps(entries, separators=(",", ":")).encode("utf-8")

output = Path("tests/fixtures/nvfp4.safetensors")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_bytes(struct.pack("<Q", len(header)) + header + data)

print(f"wrote {output}  ({output.stat().st_size} bytes)")
print("codes:   ", codes)
print("expected:", [round(v, 4) for v in expected])
