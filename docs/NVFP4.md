# Proven Small NVFP4 storage and scalar decoding

Phase B2 uses real payload ranges from `thinkingmachines/Inkling-Small-NVFP4`
revision `b6a99534467840620d411e4cd4ad5819b2610d9c`. The committed fixture
[`tests/fixtures/nvfp4.json`](../tests/fixtures/nvfp4.json) contains 2,424 original
checkpoint bytes as lossless hex, 576 independently decoded float32 bit patterns,
and the complete 16-code E2M1 / 256-code E4M3FN reference tables.

## Evidence and scope

The fixture samples layer 3's routed `w13_weight` and `w2_weight`, experts 0, 7,
and 255. Each expert contributes two adjacent blocks at row 0/block 0, row 1/block
3, and the last row's last two blocks: 18 samples / 36 blocks overall. Layer 2
is excluded from quantization and stores BF16, so it is unsuitable for this proof.

Every sampled range records its tensor name, offset within the tensor, offset
within the shard payload, absolute file offset, length, and SHA-256. The fixture's
`tensors` map records each tensor's shard, dtype, stored shape, full payload
interval, and header size. It includes the actual I64 `.original_shape`, complete
F32 `.scale2` vectors, and BF16 `.input_amax` bytes. All names, including
auxiliaries, resolve independently through the pinned global index. For example,
layer 3 `w13_weight` is in shard 4, `.scale` and `.scale2` in shard 5,
`.original_shape` in shard 9, and `.input_amax` in shard 3.

The generator verifies the checked-in metadata hashes and reuses the existing
bounded HTTP Range client. It requires exact 206 responses and never accepts an
entire shard in place of a range. Each checkpoint request is at most 4 KiB.
This proves the sampled exported layouts, not every payload value in the model.

## Layout and scaling

The logical order is contiguous `[expert, row, column]`. There is no padding or
scale swizzle in these exported tensors. SafeTensors shapes and byte counts agree
with the raw `.original_shape` values and the pinned loaders.

| Projection | Logical shape | Packed row / expert bytes | Scale row / expert bytes |
|---|---|---|---|
| `w13_weight` | `[256, 4096, 4096]` | 2,048 / 8,388,608 | 256 / 1,048,576 |
| `w2_weight` | `[256, 4096, 2048]` | 1,024 / 4,194,304 | 128 / 524,288 |

For logical shape `[E, R, K]`, expert `e`, row `r`, and 16-value block `b`:

```text
packed tensor offset = ((e * R + r) * (K / 16) + b) * 8
scale tensor offset  =  (e * R + r) * (K / 16) + b
scale2 tensor offset = e * 4
file offset          = 8 + shard header size + tensor data_start + tensor offset
```

Each U8 packs the even column in its low nibble and the odd column in its high
nibble. A single E4M3FN byte scales 16 adjacent values along the final dimension.
`.scale2` stores one float32 multiplier per expert, shared across that expert's
rows/blocks and, for `w13`, both gate/up halves. Selected experts have distinct
scales: the `w13` values for experts 0, 7, and 255 are approximately 0.0004040,
0.0003197, and 0.0002659, respectively. The loader confirms that `w13` rows are
interleaved `[gate0, up0, gate1, up1, ...]`; deinterleaving for a serving kernel
happens after loading and must not be applied to checkpoint byte offsets.

The reference operation order, including float32 rounding, is:

```text
combined_scale = float32(E4M3FN(scale_byte) * scale2[expert])
value          = float32(E2M1(nibble) * combined_scale)
```

Global scaling multiplies, never divides. `.input_amax` is an activation statistic:
vLLM uses `max(input_amax) / (448 * 6)` for its activation input scale. It is not
part of scalar weight reconstruction. vLLM ignores `.original_shape` after binding
its serving layout; this implementation validates the original dimensions.

## Independent references and special values

The fixture records exact source URLs, revisions, SHA-256 hashes, Python version,
PyTorch version, and PyTorch git revision. Pins are unchanged from
[`SOURCES.md`](SOURCES.md).

- [Transformer Engine `cast_from_fp4x2`](https://github.com/NVIDIA/TransformerEngine/blob/699ed6ec9e9c600c3d0acd3a488c4ca6f339e5d9/transformer_engine/pytorch/custom_recipes/reference_nvfp4.py)
  establishes low-nibble-first E2M1 decoding, including negative zero.
- [Model Optimizer `NVFP4QTensor.dequantize`](https://github.com/NVIDIA/Model-Optimizer/blob/b311c054de4052df9c7f3de9409b7598f44a0dba/modelopt/torch/quantization/qtensor/nvfp4_tensor.py)
  independently validates every real sample, including block-scale/global-scale
  multiplication order. The generator executes the original CPU method, table,
  and lookup through AST extraction, avoiding unrelated GPU imports.
- PyTorch 2.8.0's CPU `float8_e4m3fn` conversion supplies all 256 FP8 reference
  bit patterns. E4M3FN uses exponent bias 7, subnormals in multiples of `2^-9`,
  signed zeros, maximum finite magnitude 448, and NaNs only at `0x7f`/`0xff`.
  It has no infinities. The NaN float32 patterns here are `0x7ff00000`/`0xfff00000`.
- [vLLM's Inkling loader](https://github.com/vllm-project/vllm/blob/e378275a8f20eac9e92212e4a1fee84ec5a31dc7/vllm/models/inkling/nvidia/moe.py)
  establishes per-expert global indexing, `w13` row order, and activation-amax use;
  [its model mapper](https://github.com/vllm-project/vllm/blob/e378275a8f20eac9e92212e4a1fee84ec5a31dc7/vllm/models/inkling/nvidia/model.py)
  maps `.scale` and `.scale2` to those loader parameters.

There is one explicit reference difference: Model Optimizer's table turns E2M1
code 8 into positive zero; Transformer Engine preserves negative zero. The C
primitives and committed expected bits follow Transformer Engine. Regeneration
also requires exact numerical equality with Model Optimizer (which equates signed
zeros). C tests compare every committed result bit-for-bit, including zero signs;
no tolerance is needed for these scalar float32 samples. This is a scalar reference
contract, not a claim about GPU fused-kernel accumulation.

## API and validation

`src/core/inkling_nvfp4.h` provides layout validation, checked byte offsets,
E2M1/E4M3FN conversion, and reconstruction of exactly one 16-value block. It reuses
the size-arithmetic and BF16 helpers; it allocates no memory. Unsupported shapes,
padded scales, incomplete groups, invalid coordinates, arithmetic overflow,
truncated input/output buffers, invalid scales, and nonfinite results fail without
partial output. Local scales must be finite/nonnegative; global scales must be
finite/positive. Primitive E4M3 conversion still exposes all special encodings.

Tests verify all reference codes and samples, expert/row/block offsets, and error
paths. They also require the real fixture to distinguish swapped nibbles, switched
block scales, reciprocal global scaling, and the wrong expert's global scale.
Padded layouts are explicitly rejected.

## Regeneration and offline checks

Only regeneration needs PyTorch and network access. Use a Python 3.11 environment:

```sh
python3.11 -m venv .context/nvfp4-venv
.context/nvfp4-venv/bin/pip install 'torch==2.8.0'
.context/nvfp4-venv/bin/python -B tools/fetch_nvfp4_fixture.py
.context/nvfp4-venv/bin/python -B tests/test_nvfp4_fixture.py --with-oracle
make test
make test-sanitize
```

The generator accepts `--output PATH` for comparison without replacing the fixture.
The optional `--with-oracle` check reproduces the fixture from the recorded raw
bytes and pinned source functions, then verifies that an oracle disagreement
cannot overwrite an existing fixture. It fetches reference source files only.
Ordinary tests require neither weights nor an ML framework: C consumes the
committed oracle, and a standard-library Python check validates metadata, ranges,
checksums, and sample coverage offline.
