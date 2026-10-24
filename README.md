# inkling-in-c

The scalar C99 text runtime accepts raw token IDs and supports greedy or seeded
generation with temperature, top-k, and top-p. Generation prefills once and then decodes with per-layer KV and convolution
state. An explicit full-prefix mode remains available for comparison. It also returns float32 logits for every position through its forward API
and binds the native pinned checkpoint without a conversion step. See
[the forward contract](docs/TEXT.md) and [generation guide](docs/GENERATION.md).

Implementation references and immutable upstream revisions are recorded in
[`docs/SOURCES.md`](docs/SOURCES.md).

Run the network-free tests with `make test`. They validate all 1,360 indexed
tensors across the pinned checkpoint headers, including the five stored dtypes,
hash collisions, shape/byte arithmetic, payload gaps/overlaps, trailing bytes,
and truncated shards. Tensor ranges must cover the full payload, as required by
the [SafeTensors format](https://github.com/safetensors/safetensors#format).

Phase B's first primitives are in `src/core/`: checked size arithmetic, buffer
allocation with explicit ownership/error behavior, and bit-exact BF16-to-F32
widening. Tests cover allocation boundaries and forced failure, plus all 65,536
BF16 bit patterns. The default build uses the strict reference warnings and
`-ffp-contract=off`. Run `make test-sanitize` for the complete suite with ASan/UBSan;
its binaries are kept separately under `bin/sanitize`.

NVFP4 scalar decoding now has a real-byte oracle: 18 samples from routed weights
and their separately indexed auxiliaries, checked against pinned NVIDIA CPU
references. Offline tests cover all E2M1/E4M3FN encodings, byte offsets, malformed
layouts and exact block outputs. See [the proven layout and regeneration steps](docs/NVFP4.md).

Phase B3 adds scalar BF16/NVFP4 matvec, RMSNorm, SiLU, sigmoid, stable softmax,
deterministic top-k, and stateful kernel-4 convolution. Tests compare 43 independent
CPU reference cases, check buffer/overflow failures, and verify convolution state
across prefill and incremental calls. See [primitive contracts, tolerances, and
fixture regeneration](docs/PRIMITIVES.md).

Explicit `make oracle-fixtures` regeneration uses recorded checkpoint bytes and
checksum-verified upstream CPU sources. It requires the documented Python/PyTorch
environment; ordinary tests remain offline with standard-library Python only.
`make test-mutations` checks that relevant numerical regressions reject altered
implementations in temporary builds. See the same primitive guide for cache-only
regeneration, generator identity checks and compiler selection.

`InklingIndex` is the tensor catalogue. Load the index with `inkling_index_load`,
bind each real shard with `inkling_index_load_shard`, then resolve tensors with
`inkling_index_find_tensor`. Entries have UNKNOWN dtype until bound. Free the
catalogue with `inkling_index_free` before reloading or discarding it.

The checked-in headers contain no weights. Tests explicitly supply their declared
payload extent to `inkling_index_bind_header`; the real-file loader rejects these
header-only files as truncated.

## Verify a local checkpoint

```sh
make
bin/inkling MODEL_DIR --verify-model
bin/inkling MODEL_DIR --verify-model --with-mtp
# Offline header fixtures only (does NOT verify full shard files):
bin/inkling tests/fixtures/checkpoint --verify-model --metadata-only --with-mtp
```

Verification reads metadata and shard headers, never tensor payloads. It checks
the pinned Small NVFP4 architecture, quantization settings/exclusions, every
required tensor's stored dtype and shape, index/header agreement, actual file
bounds, and the byte census. Unknown tensor families are rejected. MTP may be
absent unless `--with-mtp` is requested; a present MTP file is always validated.
The index must still contain its pinned MTP entries when that file is omitted.

The deterministic report lists every verified tensor's dtype, shape, shard and
half-open byte range, relative to its shard's payload start (`8 + header_size`).
Classes are mutually exclusive: BF16/F32 trunk, packed routed weights, scales
and auxiliaries, embeddings/unembed, vision, audio, and MTP. With MTP, totals are
1,360 tensors / 170,733,074,592 bytes; without it, 1,200 / 166,269,249,680.
Usage errors return 2 and verification failures return 3.

This verifies the metadata contract of the revision in `docs/SOURCES.md`, not
weight hashes or the payload values in `.original_shape`. Metadata-only mode uses
the pinned original shard lengths and clearly labels its limited result. Tests
also exercise actual-file mode with sparse local shards, without downloading or
allocating weights. The legacy `bin/inkling config.json` command is unchanged.

## Run a raw token prefix

```sh
make
bin/inkling MODEL_DIR --ids 1,2,3 --dump-logits logits.json
```

`MODEL_DIR` must contain the complete pinned text checkpoint shards and metadata.
IDs address the 201024 stored embedding rows. The JSON contains `tokens`, `vocab`,
and a flat token-major `logits` array with 200058 columns per token. This command
recomputes the complete prefix from position zero.

Ordinary `make test` includes the committed miniature native checkpoint and
independent CPU oracle outputs. It needs a C99 compiler, make, libm, and Python's
standard library. Explicit oracle regeneration needs Python 3.11 and PyTorch 2.8.0.
Real-layer validation fetches 1,259,738,686 payload bytes for isolated one-token
layers 0, 2, and 5. These checks establish layer parity, not full-checkpoint logits.
See [TEXT.md](docs/TEXT.md) for the API, private diagnostic probe, tolerances,
and bounded download commands. [The Phase C validation record](docs/PHASE_C_VALIDATION.md)
records the completed checks and their limits.

## Generate raw token IDs

```sh
bin/inkling MODEL_DIR --ids 1,2,3 --gen 16
bin/inkling MODEL_DIR --ids 1,2,3 --gen 16 \
	--temperature 0.8 --top-k 20 --top-p 0.9 --seed 42 \
	--dump-tokens generated.txt --out run.json
```

The JSON report separates prompt and generated IDs and records effective options
and the stop reason. Temperature zero selects greedy decoding. The committed
miniature tests prove continuation parity against the pinned CPU oracle;
incremental/full-prefix parity, expert-cache budgets, and packed/native storage
are verified with miniature weights. Full-checkpoint generation remains D5.
See [GENERATION.md](docs/GENERATION.md) for the API, RNG, output failure behavior,
and reproducible checks. Text tokenization belongs to Phase E.


## Use bounded weight buffers and an expert cache

```sh
bin/inkling MODEL_DIR --ids 1,2,3 --gen 16 \
    --expert-cache-bytes 67108864 --weight-buffer-bytes 1048576 \
    --decode-mode incremental --trace run.jsonl
python3 tools/pack_trunk.py MODEL_DIR trunk.safetensors
bin/inkling MODEL_DIR --ids 1,2,3 --gen 16 --packed-trunk trunk.safetensors
```

The cache stores complete routed NVFP4 projections and includes entry metadata
in its byte limit. Larger entries stream through the weight buffer. The optional
trunk preserves native bytes and leaves routed experts in their original shards.
See [memory and storage contracts](docs/MEMORY.md) for APIs, traces, validation,
and the checkpoint-dependent gates.
