from pathlib import Path
import struct

header = (
    b'{"test.weight":{"dtype":"F32","shape":[1],'
    b'"data_offsets":[0,4]}}'
)

tensor_data = struct.pack("<f", 1.5)

assert len(header) == 64

output = Path("tests/fixtures/tiny.safetensors")
output.parent.mkdir(parents=True, exist_ok=True)

output.write_bytes(
    struct.pack("<Q", len(header))
    + header
    + tensor_data
)

print(f"wrote {output}")