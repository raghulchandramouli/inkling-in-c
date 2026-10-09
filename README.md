# inkling-in-c

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
