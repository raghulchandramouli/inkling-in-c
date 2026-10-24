# Memory and storage contracts

D2 through D4 use the committed miniature native checkpoint. These steps do not
require the released checkpoint or a Python ML framework. D5 still requires
full-model first-token logits, at least 32 greedy tokens, and parity across
execution and memory choices on the released weights.

## Incremental contexts

`inkling_context_create` enables incremental generation with a 1 MiB weight
buffer and a zero-byte expert cache. The options constructor accepts explicit
limits.

```c
InklingContextOptions options = INKLING_CONTEXT_OPTIONS_INIT;
options.max_tokens = 2048;
options.expert_cache_bytes = 64 * 1024 * 1024;
options.weight_buffer_bytes = 1024 * 1024;
InklingContext *context = NULL;
int ok = inkling_context_create_with_options(model, &options, &context);
```

Each layer owns separate key, value, attention-output, and MLP-output convolution
histories. Keys enter the cache after convolution and head normalization. Values
enter after convolution. Local layers retain at most `min(window, max_tokens)`
positions. Global layers retain `max_tokens` positions. Attention uses absolute
positions for relative bias and query scaling, with chronological accumulation.

`inkling_prefill` resets sequence state, consumes a nonempty prompt, and returns
the last token's logit row. `inkling_decode` appends one token and returns its row.
`inkling_context_position` reports the consumed count. A failed incremental call
invalidates continuation. Reset or prefill before retrying. Validation failures
leave caller logits unchanged. Discard logits after any runtime failure.

Generation prefills once and then calls decode between sampling steps. Setting
`incremental` to zero selects the complete-prefix reference loop. Public
`inkling_forward` always recomputes a fresh complete prefix. It clears incremental
state and lazily allocates prefix scratch when needed. Contexts sharing a model
own independent sequence state, expert entries, scratch, and counters.

## Expert cache accounting

The context caches complete routed NVFP4 projections, keyed by bound matrix and
expert ID. Each allocation includes packed nibbles, E4M3 scales, and its LRU entry
metadata. Resident `scale2` belongs to the model. The byte limit includes every
requested byte in cache allocations. It excludes allocator bookkeeping, model
metadata, KV state, generation scratch, and weight scratch.

A hit promotes the entry. A miss evicts the oldest entries before allocating.
An entry larger than the limit streams in row tiles. Allocation failure also
falls back to streaming. A failed data or scale read never publishes an entry.
Sequence reset preserves cached immutable weights. Statistics reset preserves
current entries and sets peak occupancy to current occupancy.

`InklingRuntimeStats` reports successful numerical weight bytes read, read calls,
hits, misses, evictions, tokens processed, and current, peak, and allowed cache
bytes. These counters exclude model-open vector reads and packed-file validation.
Read calls include failed calls. Counters saturate at `UINT64_MAX`.

`inkling_context_cache_trace` accepts a callback for hit, miss, bypass, and eviction
events. Each event identifies original layer ID, expert, and projection, where
zero is w13 and one is down. `bytes` is the cache allocation charge, including
entry metadata. A miss reports zero before determining that charge.
`resident_bytes` is occupancy when the event fires. Callbacks must not reenter or
mutate their context.

## Bounded native and packed reads

Embedding reads fetch only the requested native BF16 row. Unembedding reads only
the unpadded output rows through the common matvec tile. `weight_buffer_bytes`
must fit every complete matrix row, including the scale bytes for an NVFP4 row.
The constructor rejects a smaller buffer. Tile choices preserve dot-product order.

`tools/pack_trunk.py MODEL_DIR DESTINATION` copies non-routed text tensors into a
single SafeTensors file in execution order. It uses bounded copy buffers and
atomically replaces the destination after completion. It refuses checkpoint
input aliases and removes temporary files after failure. Original model shards
and metadata remain required because routed experts continue to use them.

`InklingModelOptions.packed_trunk_path` selects the optional trunk at model open.
The loader validates schema, pinned revision, source metadata identity, exact
tensor coverage, dtype, shape, original ranges, payload length, and checksums.
The checksums detect pack corruption. They do not authenticate arbitrary source
weights against an independently published weight digest. Validation reads the
whole packed payload once with bounded scratch.

The model holds an open descriptor and uses positional reads. Replacing the path
after open cannot redirect reads. In-place changes to open weight files violate
the immutable-source contract. Truncation still fails loudly. A corrupt selected
trunk fails model open instead of falling back to native trunk bytes.

## CLI options

All options take a value. Byte limits are decimal integers.

| Option | Behavior |
|---|---|
| `--decode-mode incremental` | Prefill once, then carry KV and convolution state. Default. |
| `--decode-mode full` | Recompute each prefix. |
| `--expert-cache-bytes N` | Set the total cache allocation limit. Default zero. |
| `--weight-buffer-bytes N` | Set row tile scratch. Default 1048576. |
| `--packed-trunk PATH` | Validate and read the optional local trunk. |
| `--trace PATH` | Write cache events and a final statistics object as JSONL during generation. |

Trace output is staged before publication, as are tokens and the generation
report. Trace, token, and report destinations must be distinct, including aliases.
Publication across separate output files is sequential. A later rename failure
reports any files already published.

These limits govern specific allocations. They are not a total process RSS cap.
Named presets, resident-weight modes, asynchronous prefetch, real-model RSS,
and performance measurements remain later work. No speedup is claimed.

## Offline validation

Run `make test` and `make test-sanitize`. The added checks cover every logit row
across 521 positions, the production 512-position local boundary, reduced
relative-extent and tau boundaries, four carried histories, independent contexts,
failure invalidation, and reset or retry. Existing independent CPU oracle fixtures
still check the numerical reference and generated IDs.

Expert-cache tests cover exact capacity boundaries, eviction order, allocation
failure, failed scale reads, trace accounting, read reduction, and generation
parity under several cache and tile limits. Packed-file tests cover deterministic
copies, native parity, source aliases, interrupted writes, corruption rejection,
descriptor stability, and runtime truncation. CLI checks combine both execution
modes, cache limits, small row buffers, and native or packed storage.

These miniature gates establish the implementation paths. They do not establish
full-checkpoint parity, full-model RSS, portability to another operating system,
tokenizer behavior, or calibrated memory presets.


The recorded macOS arm64 run passed the complete normal suite, ASan and UBSan,
and all 17 existing numerical mutations. Every added gate also passed an
independent review run. The existing 14 generation and 17 sampler cases retained
a maximum absolute step-logit error of `2.24e-08`. Machine and source identities
are in [the D2-D4 validation record](validation/phase-d-results.json).
