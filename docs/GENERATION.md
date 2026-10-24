# Raw-ID generation

`inkling_generate` prefills once and then decodes one token at a time by default.
Per-layer KV and convolution state preserve the scalar full-prefix result.
`inkling_forward` still starts at position zero and returns every input logit row.
Set `InklingContextOptions.incremental` to zero, or use `--decode-mode full`,
to select complete-prefix generation. [Memory and storage](MEMORY.md) describes
context options, cache budgets, packed trunks, and incremental APIs.

## C API

Include `inkling/inkling_model.h`. The API uses these types:

```c
typedef struct {
    size_t max_new_tokens;
    double temperature;
    size_t top_k;
    double top_p;
    uint64_t seed;
} InklingGenerateOptions;

#define INKLING_GENERATE_OPTIONS_INIT {0, 0.0, 0, 1.0, UINT64_C(0)}

typedef enum {
    INKLING_STOP_LENGTH = 1,
    INKLING_STOP_CONTEXT,
    INKLING_STOP_EOS
} InklingStopReason;

typedef struct {
    size_t generated_count;
    InklingStopReason stop_reason;
    InklingGenerateOptions effective;
} InklingGenerateResult;

size_t inkling_model_max_length(const InklingModel *model);
int inkling_generate(InklingContext *context,
    const uint32_t *prompt, size_t prompt_count,
    const InklingGenerateOptions *options,
    uint32_t *output, size_t output_capacity,
    InklingGenerateResult *result);
```

Set the requested count explicitly:

```c
uint32_t prompt[] = {1, 2, 3};
uint32_t generated[16];
InklingGenerateOptions options = INKLING_GENERATE_OPTIONS_INIT;
InklingGenerateResult result;
options.max_new_tokens = 16;
if (inkling_generate(context, prompt, 3, &options,
                     generated, 16, &result)) {
    consume_ids(generated, result.generated_count);
}
```

The caller owns the prompt, options, generated-ID array, and result. The context
owns lazy generation scratch and its RNG. Free the context before its model.
A model can serve several independent contexts. A context supports one active
call at a time. Each generation call restarts from its explicit seed, including
after another generation, a reset, a public forward, or a failed call.

A successful call returns 1. Every failure returns 0 with a diagnostic and leaves
both caller outputs unchanged. Generated IDs publish only after the run succeeds.
The result is valid only after success. Its `effective` options describe the
normalized sampler settings, so callers do not need to reproduce normalization.

## Limits and stops

A prompt must be nonempty and fit both the context capacity and the model maximum
length. IDs may address any stored embedding row, including padded rows. Generated
IDs come only from the unpadded output vocabulary. For the pinned checkpoint,
these widths are 201024 and 200058 respectively.

Let `usable` be the smaller of context capacity and model maximum length. Reserve
at least `min(max_new_tokens, usable - prompt_count)` output IDs. A NULL output
requires zero declared and required capacity. All byte counts and the requested
prompt-plus-output total must fit checked size arithmetic. Output and result
storage cannot overlap each other or the input objects.

A zero request validates the inputs, performs no model evaluation, and reports
`length`. A positive request with an already full prompt reports `context` with
zero generated IDs. Otherwise, stops have this precedence after each appended ID:

1. A generated EOS reports `eos` and remains in the returned generated IDs.
2. The requested count reports `length`.
3. The usable total sequence capacity reports `context`.

An EOS inside the prompt does not stop generation. No extra forward runs after a
stop. Every successful prompt-plus-output sequence fits the usable capacity.

## Sampling policy

Temperature must be finite and nonnegative. Top-p must be finite in `(0, 1]`.
Top-k is an unsigned size, and zero means the complete output vocabulary. Every
64-bit unsigned seed is valid. All options are validated even for greedy or
zero-length runs.

Temperature zero selects greedy argmax over finite raw logits, with the lower ID
on ties. It consumes no random words. Effective top-k and top-p become 1.

For positive temperature:

1. Rank raw float logits by descending value, then ascending ID.
2. Retain top-k candidates, with zero or oversized top-k normalized to vocabulary width.
3. Compute max-shifted exponential masses in double precision, with the logit gap
   widened before subtraction. A zero gap has mass 1. Guard division overflow
   before division for extremely small positive temperatures.
4. Retain the shortest nonempty prefix whose cumulative mass reaches top-p times
   total mass. Include the crossing candidate. Top-p 1 retains all top-k candidates.
5. Draw one random word, including for singleton support, and select the first
   cumulative retained mass strictly greater than the uniform draw times its total.

Zero masses cannot win. If floating-point rounding leaves no strict crossing,
selection uses the last positive mass. Nonfinite model logits fail explicitly.

The RNG uses [SplitMix64's published integer transition](https://prng.di.unimi.it/splitmix64.c)
with context-owned state initialized directly from the supplied seed. It adds
`0x9e3779b97f4a7c15`, then applies the xor shifts 30, 27, and 31, with multipliers
`0xbf58476d1ce4e5b9` and `0x94d049bb133111eb`. Unsigned arithmetic wraps modulo
2^64. The uniform draw is `(word >> 11) * 0x1p-53`, in `[0, 1)`.

Integer RNG words and their uniform conversion have a portable contract. Seeded
sampling replay requires the same build, platform, math library, and rounding
mode. Different `exp` implementations can disagree near sampling thresholds.
Greedy fixture choices have recorded margins from the independent reference.

## CLI

```sh
make
bin/inkling MODEL_DIR --ids 1,2,3 --gen 16
bin/inkling MODEL_DIR --ids 1,2,3 --gen 16 \
	--temperature 0.8 --top-k 20 --top-p 0.9 --seed 42 \
	--dump-tokens generated.txt --out run.json
```

Generation requires `--gen`. Defaults are temperature 0, top-k 0, top-p 1, and seed
0. Counts and seeds use decimal digits. Float options require finite decimal
values. Representable positive subnormals are accepted; a positive decimal that
underflows to zero is rejected. Duplicate, missing, and unknown options fail
before checkpoint loading.

The report separates `prompt_ids` from `generated_ids` and includes count, stop
reason, and effective options. Its seed is a decimal JSON string to preserve all
64 bits in readers that use floating-point JSON numbers. With no `--out`, the
report goes to stdout. `--dump-tokens` writes only generated decimal IDs separated
by commas and a final newline. An empty generated sequence writes one newline.

The existing `--dump-logits` mode retains its prompt-logit schema. Combining it
with `--gen` is a usage error. Config inspection and `--verify-model` retain their
existing behavior. The product opener continues to require the pinned checkpoint.

All requested output files finish staging before publication. Destination
collisions are rejected before checkpoint loading, including case and Unicode
aliases according to the containing filesystem. Each file replacement is atomic.
Multiple destinations
are not a single filesystem transaction. The token file publishes first and the
JSON report last. A later rename failure can leave the new token file and the old
report; the command reports that partial publication and fails. Computation and
staging failures preserve existing destinations and remove temporary files.

## Verification

The independent generation fixture uses the committed native miniature weights
and the original pinned CPU model bodies. It records each continuation prefix,
last-row logits, selected ID, greedy margin, and stop reason. Separate sampler
cases cover integer RNG vectors, exact CDF boundaries, filtering, ties, and extreme
finite values. Ordinary checks need no model download, cached source, or PyTorch.

Run the ordinary checks from the repository root:

```sh
make test CC=clang
make test-sanitize CC=clang
make test-mutations CC=clang
python3 -B -S tests/test_generation_fixture.py
```

The suite includes 14 native generation cases with 48 evaluated prefix steps and
17 independent sampler cases. It checks immediate and prompted EOS, length and
context limits, stop ties, padded prompt IDs, zero output, seeded continuations,
exact CDF boundaries, singleton draws, tied logits, and subnormal temperatures,
top-p values, and masses. C tests also compare private last-row execution bit for
bit with full forward, force all four generation allocation failures, and force a
read failure after one staged token. Caller outputs remain unchanged on failure
and a subsequent call succeeds. The actual CLI source is tested with only its
opener redirected at compile time to the private miniature opener.

Explicit generation requires the Python 3.11/PyTorch 2.8.0 environment and pinned
source cache described in [TEXT.md](TEXT.md#regenerate-identical-miniature-artifacts).
The generation tool never downloads sources. It authenticates the original model
manifest, native bytes, source hashes, and exact torch version before it computes
expectations. Its native reader must first reproduce all three original model
prefix cases exactly. Expected logits and IDs come from the original CPU model
bodies, independently of C. Missing or modified sources fail regeneration.

For two separate outputs, using the existing recorded environment:

```sh
.context/nvfp4-venv/bin/python -B tools/emit_generation_fixtures.py \
	--source-dir .context/oracle-sources --output .context/d1-root-regen-a.json
.context/nvfp4-venv/bin/python -B tools/emit_generation_fixtures.py \
	--source-dir .context/oracle-sources --output .context/d1-root-regen-b.json
cmp tests/fixtures/generation.json .context/d1-root-regen-a.json
cmp tests/fixtures/generation.json .context/d1-root-regen-b.json
```

`make generation-fixtures ORACLE_PYTHON=YOUR_PYTHON
ORACLE_FLAGS='--source-dir .context/oracle-sources'` explicitly replaces the
committed fixture. Ordinary tests audit generator identities and expected results
with standard-library Python only. A generator or checker change requires honest
regeneration. The comparator requires actual returned count/effective options and
actual sampler masses, preserving exact zero versus positive classification.

Full-checkpoint generation remains D5. The D1 results below predate the D2-D4 miniature validation in
[MEMORY.md](MEMORY.md). Full-model RSS, tokenizer behavior, and cross-platform
sampling identity remain unverified. No performance claim follows from this complete-prefix reference loop.

## Recorded D1 results

On 2026-10-18, macOS arm64 with Apple Clang 21.0.0 passed
`make test CC=clang`, `make test-sanitize CC=clang`, and the existing 10 primitive/seven
model mutation checks. All 14 generation and 17 sampler cases matched. The
maximum absolute per-step logit error was `2.24e-08`, against the unchanged
`1e-4 + 1e-4 * abs(expected)` allowance. All 21 generation fixture audit tests
passed with standard-library-only Python.

A fresh source export passed normal and ASan/UBSan suites with networking denied
and no torch installed in its system Python interpreter. Two independent
network-denied regenerations matched the committed JSON byte for byte. Its
SHA-256 is `f47d3a281e7d986d3679f71fd6e276e70579b53a8ee8c0d9eda37ea0d829f23b`.

Direct CLI checks produced `14,2,6`, including EOS, for miniature prompt ID 1,
kept all bits of a UINT64_MAX seed in the report, and replayed the same sampled
sequence. Production checkpoint validation still rejected the miniature model.
File tests cover staging failures, partial publication, existing output
preservation, filesystem case/Unicode aliases, and temporary-file cleanup.

These runs used miniature weights, not full-checkpoint generation. No Linux
compiler run or separate case-sensitive volume was part of this validation.
