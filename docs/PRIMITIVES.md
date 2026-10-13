# Scalar numerical primitives (Phase B3)

`src/core/inkling_math.h` provides the scalar operations needed before building the
text transformer. Runtime code is C99 plus the C math library. It performs no
allocation and does not depend on Python, model weights, or an ML framework.

## Operations and storage

| Operation | Contract |
|---|---|
| BF16 matvec | Row-major `[rows, columns]` native `uint16_t` weights; widen each element inside the dot product, multiply and accumulate in float32, in column order. |
| NVFP4 matvec | B2's contiguous `[expert,row,column]` layout and separate packed/scales/global buffers; select one expert and reconstruct one 16-value block at a time. |
| RMSNorm | Normalize one float32 vector/head per call; compute `weight * normalized_x`, without adding one to the weight. Positive finite epsilon; double sum of squares and reciprocal square root, normalized result and learned-weight multiplication in float32. |
| SiLU / sigmoid | Elementwise float32 inputs/outputs; evaluate sigmoid with a sign branch so large negative values do not overflow `exp`. |
| Softmax | Subtract the maximum, compute exponentials and their sum with double precision, store float32 outputs. `-INFINITY` is a mask with exactly zero probability. |
| Top-k | Return indices in descending value order, ties in ascending original index order; signed zeros compare equal. Infinities are permitted, NaNs are rejected. |
| Short convolution | Depthwise causal cross-correlation with kernel `[channels,4]`, input/output `[tokens,channels]`, state `[channels,3]`. Kernel/state order is oldest to newest. No bias or activation. |

NVFP4 callers provide a layout returned by `inkling_nvfp4_layout`; the matvec
also checks its dimensions and stored byte totals. Input capacities cover the
complete expert tensors, while the output holds only the selected expert's rows.
The three payloads may originate in different shards. Global scales are native
float32 values supplied by the caller; this API does not perform file I/O.
Packing, rounding order, and scale interpretation remain those proven in
[NVFP4.md](NVFP4.md).

The convolution computes, for channel `c`:

```text
y[t,c] = w[c,0]*x[t-3,c] + w[c,1]*x[t-2,c]
       + w[c,2]*x[t-1,c] + w[c,3]*x[t,c] + x[t,c]
```

All four pinned `InklingShortConvolution` streams (key, value, attention output,
MLP output) include this input residual. The decoder layer adds its separate
pre-attention/pre-MLP residual **after** the output convolution. The primitive
does not add that outer residual; fixtures check the two additions separately.
Each stream needs its own state initialized to zeros. The state stores input
samples, never convolution outputs; zeroing it resets the stream. A single call
handles prefill or a chunk; repeated `tokens=1` calls use the same arithmetic and
produce bit-identical outputs and final state. Padding masks and model/cache
execution belong to later model integration.

## Boundaries and failure behavior

All functions return 1 on success or 0 on failure. Dimensions, token counts and
`k` must be positive; zero-length calls are rejected, including `k=0`. Top-k
requires `k <= n`. Sizes use checked arithmetic and reject required byte spans
beyond `PTRDIFF_MAX`. The pointer-plus-capacity arguments describe storage owned
by the caller; capacities cannot authenticate an invalid pointer.

Inputs and weights must be finite except top-k's infinities and softmax's negative
infinity masks. Softmax rejects NaN, positive infinity, and all-masked vectors.
Matvec, norm and convolution reject nonfinite results, including overflow from
finite inputs. RMSNorm's wider accumulation avoids overflow when squaring large
finite float32 values. NVFP4 retains B2's scale validity rules.

Invalid dimensions, capacities and NULL pointers are rejected before writes.
Numerical errors may occur after earlier rows/tokens have been written. Discard
all outputs and reset convolution state after any failure; failure is not a
retryable partial-progress result. No invalid-input path reads beyond the supplied
required extent.

Buffers must be disjoint except that RMSNorm, activations, softmax, and convolution
support **exact** `out == x`. Partial overlaps are unsupported, as are output
aliases with weights, convolution state, or matvec inputs. Read-only inputs may
share storage with each other. Oversized output buffers retain their unused tail.

## Independent fixtures and tolerances

[`tests/fixtures/primitives.json`](../tests/fixtures/primitives.json) contains 43
cases with seed `932184`, shapes, inputs and expected float32 values encoded as
uint32 bit patterns. It records Python/PyTorch versions, source revisions and
source hashes, plus the SHA-256 of B2's real-byte fixture. The generator never
calls the C implementation.

- The pinned Transformers `InklingRMSNorm`, `InklingShortConvolution`,
  `causal_conv1d_fn`, and `causal_conv1d_update` bodies execute on the CPU after
  removing only hub/GPU decorators. Convolution prefill and cached incremental
  reference paths agree, including carried state and both output residual cases.
- PyTorch 2.8.0 supplies BF16 widening/matvec, activations, softmax and stable
  descending `argsort`. The pinned model uses `topk(sorted=False)`, which does not
  define stable ties; lower-index-first is our explicit portable tie policy,
  rather than a claim to reproduce unspecified PyTorch tie ordering.
- The pinned NVIDIA Model Optimizer decoder reconstructs NVFP4 values for PyTorch
  matvec. Two cases assemble B2's sampled 32-value slices into compact
  `[3 experts,3 rows,32 columns]` matrices, retaining the globals for original
  experts 0, 7 and 255. These are sampled slices, not full checkpoint rows. Their
  original positions and extraction provenance remain in B2's unchanged fixture.

Comparisons use `abs(actual-expected) <= atol + rtol*abs(expected)`:

| Operation | Absolute tolerance | Relative tolerance |
|---|---:|---:|
| BF16 / NVFP4 matvec | `5e-5` | `5e-5` |
| RMSNorm, SiLU, sigmoid | `2e-6` | `2e-6` |
| Softmax | `2e-7` | `2e-6` |
| Convolution, including outer residual | `3e-6` | `3e-6` |

Matvec permits float32 sequential reduction versus the independent backend's
reduction order (up to 257 columns here); its absolute term handles cancellation.
The other tolerances allow a few float32 rounding differences from library
transcendentals, wider norm/softmax accumulation, and four-tap reductions. These
are small-fixture contracts, not full-model parity thresholds. Top-k indices and
convolution state are exact; C prefill/chunk/incremental parity is bit-exact.
B1/B2's exhaustive encoding checks remain exact and are still run.

Tests include deterministic random inputs, singleton/zero/tiny/extreme values,
masked probabilities, signed-zero and infinity ties, asymmetric impulse kernels,
nonzero carried state, reset/reuse, and invalid/overflowing/truncated buffers.
They verify the NVFP4 matvec fixtures distinguish swapped nibbles, wrong block
scales and wrong expert globals. Non-square BF16 and multi-row/multi-block NVFP4
cases check strides. Every possible chunk split in the convolution cases is checked.

## Reproduce and run

Ordinary tests need a C99 compiler, C math library, make and Python's standard
library. They are offline and consume committed fixtures:

```sh
make test
make test-sanitize
```

Regeneration needs Python 3.11, PyTorch 2.8.0, and network access only for pinned
source files (no checkpoint download). The B2 environment can be reused:

```sh
python3.11 -m venv .context/nvfp4-venv
.context/nvfp4-venv/bin/pip install 'torch==2.8.0'
# Regenerate and compare without overwriting the committed fixture:
.context/nvfp4-venv/bin/python -B tools/emit_primitive_fixtures.py --output .context/primitives.json
cmp tests/fixtures/primitives.json .context/primitives.json
```

The generator sets a fixed seed, single-threaded deterministic CPU execution,
and writes atomically only after its independent convolution checks pass. Output
is deterministic in the recorded environment; other package builds/platforms may
have different provenance or floating-point bits and require review.
Phase B4 still owns consolidated regeneration tooling and final Gate 1 sign-off.
