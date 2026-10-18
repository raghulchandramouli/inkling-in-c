# Text runtime and validation

The runtime evaluates one complete text prefix with scalar C99 arithmetic. The
public opener requires the pinned native checkpoint described in [SOURCES.md](SOURCES.md).
The committed miniature checkpoint exercises the same binder and arithmetic through
private fixture entry points. Phase C passed the checks in
[the validation record](PHASE_C_VALIDATION.md) on 2026-10-15.
Raw-ID generation uses this same complete-prefix graph. Its API, sampling policy,
CLI options, and checks are documented in [GENERATION.md](GENERATION.md).

## Public API

Include `inkling/inkling_model.h` and link the model, I/O, core sources, and libm.
The API declares these opaque objects and functions:

```c
typedef struct InklingModel InklingModel;
typedef struct InklingContext InklingContext;
int inkling_model_open(const char *directory, InklingModel **out);
void inkling_model_close(InklingModel *model);
size_t inkling_model_logits_width(const InklingModel *model);
int inkling_context_create(const InklingModel *model, size_t max_tokens,
                           InklingContext **out);
void inkling_context_reset(InklingContext *context);
void inkling_context_free(InklingContext *context);
int inkling_forward(InklingContext *context, const uint32_t *ids, size_t tokens,
                    float *logits, size_t logits_count);
```

Constructors clear `*out` and publish an owned object only on success. Functions
that return `int` return 1 on success and 0 on failure, with diagnostics on stderr.
Free every context before closing its model. The model owns immutable bindings and
small widened vectors. Each context owns its mutable activation and read buffers.
Separate contexts can share a model. A context must not serve concurrent calls.

Allocate at least `tokens * inkling_model_logits_width(model)` floats with checked
size arithmetic. `logits_count` counts floats, not bytes. Output is token-major.
The pinned width is 200058, although input IDs can address all 201024 stored rows.
Each ID must be less than the stored vocabulary size. Prefix length must be
positive, fit the context capacity, and not exceed `model_max_length`.
Input IDs and output storage must not overlap. The caller owns both arrays.

Every `inkling_forward` starts at position zero and resets sequence state.
`inkling_context_reset` explicitly clears state without freeing its buffers.
Discard all logits after a failed forward, since an earlier output row may already
have been written. A subsequent complete forward resets state and can retry after
the underlying failure is resolved. This API has no incremental decoding contract.

The opener checks the pinned metadata contract and actual shard bounds. It does
not authenticate payload hashes. Keep checkpoint files unchanged throughout the
model lifetime. Bounded reads check exact byte counts and return failure on short
reads. Extracted fixtures reject missing ranges instead of substituting zeros.
The scalar projection scratch is 1 MiB per context. A BF16 row or a packed NVFP4
row plus its block-scale row must fit that limit. All pinned checkpoint rows fit.
Activation storage grows with context capacity. There is no configurable total
memory budget in this API.

## Run the CLI

```sh
make
bin/inkling MODEL_DIR --ids 1,2,3 --dump-logits logits.json
```

The comma-separated IDs contain decimal digits without spaces. The command writes
JSON with `tokens`, `vocab`, and a flat `logits` array. It publishes the file by
renaming a temporary file after a successful forward and serialization. A usage
error returns 2. A runtime or output-file failure returns 1.

## Run the committed miniature checks

Ordinary checks need a C99 compiler, make, libm, and Python's standard library.
They fetch no data and import no PyTorch.

```sh
make test
make test-sanitize
```

For a focused comparison and an elementwise error report, run:

```sh
make bin/model_probe
python3 -B tools/check_model.py --probe bin/model_probe \
	--fixtures tests/fixtures/model --report .context/model-report.json
```

The miniature fixture has hidden size 32, three layers, four routed experts, two
shared experts, and stored/output vocabularies of 40/37. It covers full-prefix
logits at lengths 1, 3, and 6, individual layers and MLPs, local distance 511/512,
global relative distance 1023/1024, the long-context tau boundary and a farther
position that detects omitted query or bias scaling, router bias,
extreme negative logits, and the local tie policy. It also uses distinct local
and global head dimensions and both BF16 and NVFP4 expert storage.

`bin/model_probe` is a private diagnostic executable. It accepts the reduced
fixture through internal openers that the product CLI does not expose. For
example, case zero emits one miniature full-prefix result:

```sh
bin/model_probe tests/fixtures/model tests/fixtures/model/oracle.json 0 .context/probe.json
```

The general diagnostic form is `model_probe FIXTURE_DIR CASES_JSON CASE_INDEX
OUTPUT_JSON [RANGE_MANIFEST]`. The optional range manifest binds one original
checkpoint layer. `model_probe --bind-layer METADATA_DIR RANGE_MANIFEST` checks
that binding without a forward. These forms do not authorize a partial checkpoint
for `inkling_model_open`.

## Regenerate identical miniature artifacts

Explicit regeneration needs Python 3.11 and PyTorch 2.8.0. The recorded environment
uses Python 3.11.16 and torch git revision
`a1cb3cc05d46d198467bebbb6e8fba50a325d4e7`. Source files are fetched only by explicit
regeneration and are checked against immutable SHA-256 pins before execution.

```sh
python3.11 -m venv .context/model-venv
.context/model-venv/bin/pip install 'torch==2.8.0'
make model-fixtures ORACLE_PYTHON=.context/model-venv/bin/python
```

With all pinned sources already cached, regenerate without network access:

```sh
make model-fixtures ORACLE_PYTHON=.context/model-venv/bin/python ORACLE_FLAGS=--offline
```

The seed is 932184. The generator fixes CPU thread count to one and enables torch
deterministic algorithms. `tests/fixtures/model/manifest.json` records file hashes
and all six transitive generator dependencies. For two independent output directories, run:

```sh
.context/model-venv/bin/python -B tools/emit_model_fixtures.py --offline --output-dir .context/model-a
.context/model-venv/bin/python -B tools/emit_model_fixtures.py --offline --output-dir .context/model-b
diff -qr tests/fixtures/model .context/model-a
diff -qr .context/model-a .context/model-b
```

Run these commands from the repository root because generator identities use
repository-relative paths. A changed generator requires explicit regeneration.
The ordinary checker audits identities, shapes, coverage, and expected outputs.

## Fetch and check bounded real layers

Use the same Python/PyTorch environment for explicit real-payload extraction and
oracle execution. The default plan fetches only required ranges for dense layer 0,
local MoE layer 2, and global MoE layer 5:

```sh
.context/model-venv/bin/python -B tools/fetch_model_layers.py
make test-model-real
```

The first recorded plan contains 1,259,738,686 payload bytes, about 1.260 GB or
1.173 GiB. The aggregate limit is 2 GiB. This is not a full-checkpoint download.
The fetcher requires proper HTTP Range responses and reuses checksum-verified
blobs. `--offline` requires all needed blobs and pinned sources to exist locally.

```sh
.context/model-venv/bin/python -B tools/fetch_model_layers.py --offline
python3 -B tools/check_model.py --probe bin/model_probe \
	--real weights/c16-real/layer-0 --real weights/c16-real/layer-2 \
	--real weights/c16-real/layer-5 --metadata tests/fixtures/checkpoint \
	--report .context/real-model-report.json
```

Each `ranges.json` records source revision, tensor-relative ranges, payload hashes,
nonzero input, selected experts, source identities, tool versions, and expected
output. The acceptance checker verifies recorded payload hashes before invoking C.
C validates lengths, tensor bounds, and complete range coverage for every read.

Gate 4 checks one token at position zero with zero convolution prehistory in each
of the three isolated layers. Every 4096-element layer output and recorded
intermediate is compared. The layers receive independent recorded hidden vectors.
They do not form a sequence of three layers or establish full-model logits parity.
Long-context and multi-token behavior are covered by the miniature cases.

## Numerical contract

The oracle executes pinned Transformers CPU bodies after native storage adapters
widen BF16 values exactly to float32 and decode NVFP4 with pinned NVIDIA references.
It explicitly maps checkpoint dense and expert widths, converts interleaved native
rows to the logical HF layout, and supplies CPU import infrastructure. Expected
arrays do not come from C. This contract differs from fused BF16 backend arithmetic
because float32 activations do not undergo intermediate BF16 rounding.

C matvec accumulates in float32 in column order. Norm and softmax reductions use
wider arithmetic. PyTorch reductions can differ in accumulation order. Comparisons
use `abs(actual - expected) <= atol + rtol * abs(expected)` with these limits:

| Operation | Miniature atol / rtol | Real atol / rtol |
|---|---|---|
| Projections, router logits, relative bias, convolution, residuals, MLP and final outputs | `1e-4 / 1e-4` | `1e-3 / 2e-4` |
| RMSNorm and scaled queries | `2e-5 / 2e-5` | `2e-5 / 2e-5` |
| Attention probabilities, routed weights, shared gammas | `2e-6 / 2e-5` | `2e-6 / 2e-5` |

Limits are operation-specific allowances for the scalar and PyTorch reduction
orders. Reports keep observed maximum absolute and relative errors separate from
those limits and record the maximum fraction of the allowed error. Large relative
errors near zero do not alone fail a comparison. Masked probabilities must equal
zero exactly. Selected expert IDs must match exactly for unique scores. For equal
selection scores, C chooses the lower original expert index. The tied fixture
checks this local policy and does not claim equivalence to unspecified torch top-k
tie ordering.

The two near-floor cases cannot resolve the first scaling increment at these
tolerances. Source inspection establishes the exact floor formula; `tau_far`
independently rejects omitted query and relative-bias scaling. Long attention
cases compare eight indexed probabilities and relative biases apiece, including
the exact included and excluded distances. Global attention remains positive
beyond the relative-bias extent. Zero expectations must match exactly, and a
positive expected attention probability cannot become zero.

`make test-mutations` also rebuilds seven deliberately damaged model graphs and
requires the intended numerical comparison to reject each one. The fixture audit
has 75 rejection cases. CLI tests compile the actual CLI source with only its
model opener redirected to the private miniature opener. This exercises parsing,
forward execution, and atomic output while the production executable retains
strict checkpoint verification.
