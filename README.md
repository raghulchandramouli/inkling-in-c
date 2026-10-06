# inkling-in-c

Implementation references and immutable upstream revisions are recorded in
[`docs/SOURCES.md`](docs/SOURCES.md).

Run the network-free tests with `make test`. They validate all 1,360 indexed
tensors across the pinned checkpoint headers, including the five stored dtypes,
hash collisions, shape/byte arithmetic, overlapping ranges, and truncated shards.

`InklingIndex` is the tensor catalogue. Load the index with `inkling_index_load`,
bind each real shard with `inkling_index_load_shard`, then resolve tensors with
`inkling_index_find_tensor`. Entries have UNKNOWN dtype until bound. Free the
catalogue with `inkling_index_free` before reloading or discarding it.

The checked-in headers contain no weights. Tests explicitly supply their declared
payload extent to `inkling_index_bind_header`; the real-file loader rejects these
header-only files as truncated. Full model verification and `--verify-model` are
the next milestone.
