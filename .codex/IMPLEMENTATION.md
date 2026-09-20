# Inkling-Small-NVFP4 in portable C: implementation blueprint

Status: working specification  
Target checkpoint: `thinkingmachines/Inkling-Small-NVFP4`  
Reference implementation style: `FareedKhan-dev/kimi-k3-in-c`  
Language target: portable C99, CPU inference, no Python or ML framework at runtime

This file is the implementation contract for this repository. It describes the
intended code layout, numerical behavior, memory model, command-line interface,
and validation gates. Update it when evidence from the pinned checkpoint or an
independent oracle changes an assumption.

## 1. Goal and first deliverable

Build a text-first, exact-enough-to-validate CPU inference engine for the released
NVFP4 checkpoint. Follow the Kimi K3 reference project's useful ideas:

- parse the checkpoint rather than hard-coding a private conversion;
- keep packed expert weights packed and stream them from disk;
- separate I/O, kernels, model binding, model execution, caching, tokenizer, and CLI;
- validate every kernel against an independent Python/PyTorch oracle;
- provide weightless tests and a tiny end-to-end model before requiring the full
  171 GB checkpoint;
- make memory a configurable budget without changing generated token IDs.

The first usable milestone is text-only greedy generation. Image, audio, and MTP
(multi-token prediction) follow only after text logits and incremental decoding
match the oracle. The architecture must leave clean module boundaries for those
features; it must not pretend they work before their gates pass.

## 2. Sources of truth

Use sources in this order:

1. The exact files in the pinned Hugging Face revision of
   `thinkingmachines/Inkling-Small-NVFP4`.
2. Hugging Face Transformers' `configuration_inkling.py` and
   `modeling_inkling.py` at a recorded commit.
3. SGLang and vLLM Inkling implementations at recorded commits, especially for
   checkpoint tensor layout and serving behavior.
4. NVIDIA's NVFP4 documentation for E2M1 encoding and two-level scaling.
5. `kimi-k3-in-c` for repository structure, streaming, cache, CLI, and validation
   patterns. Do not copy its MXFP4 decoder into this project.

Record all upstream commit hashes in `docs/SOURCES.md` before numerical kernel
work begins. A moving `main` branch is not a reproducible oracle.

Exact checkpoint and upstream revisions, immutable file links, source precedence,
and the pin-update procedure are recorded in `docs/SOURCES.md`.

## 3. Repository audit: what exists now

Current files:

```text
include/inkling/inkling.h          public types and current I/O/config API
src/cli/inkling_run.c              config-inspection CLI stub
src/io/inkling_config.c            strict scalar config reader
src/io/inkling_index.c             SafeTensors shard-index reader
src/io/inkling_json.c/.h           strict shared JSON DOM parser
src/io/inkling_safetensors.c       header scan and payload read
tests/test_config.c                config validation
tests/test_index.c                 index lookup and real-header lookup
tests/test_json.c                  JSON syntax, bounds, and real-fixture tests
tests/test_safetensors.c           synthetic payload and real-header metadata
tests/fixtures/checkpoint/         pinned config/index/all shard headers
tools/fetch_checkpoint_metadata.py metadata-only checkpoint fetcher
tools/make_tiny_fixture.py         tiny SafeTensors generator
makefile                           C99 build and current tests
```

Keep these modules and evolve their APIs. Do not replace them with a framework or
an opaque external runtime.

### Pinned metadata fixtures

`tests/fixtures/checkpoint/` is generated from revision
`b6a99534467840620d411e4cd4ad5819b2610d9c` of the Small NVFP4 checkpoint. It
contains the config, quantization config, index, and metadata-only headers for all
9 model shards plus the separate `mtp.safetensors` file. The index contains 1,360
tensors and reports `total_size == 170733074592` for the model shards. Tests assert
that identity and resolve a real BF16 tensor through the index into its pinned
header. Regenerate the fixtures with `tools/fetch_checkpoint_metadata.py`; ordinary
tests remain network-free.

## 4. Target model facts

Values below come from the current released Small config and must still be parsed
from `config.json` at runtime:

| Property | Value |
|---|---:|
| Decoder layers | 42 |
| Hidden size | 4096 |
| Vocabulary size | 201024 |
| Unpadded vocabulary | 200058 |
| Maximum configured sequence | 1048576 |
| Attention heads | 32 |
| KV heads | 8 |
| Head dimension | 128 |
| Sliding window | 512 |
| Local/global pattern | five local layers, then one global layer |
| Local layers | 35 |
| Global layers | 7 |
| Relative feature width | 16 |
| Global relative extent | 1024 |
| Short-convolution kernel | 4 |
| Dense MLP layers | layers 0 and 1 |
| Dense intermediate size | 16384 |
| Routed experts | 256 |
| Experts selected per token | 6 |
| Shared experts | 2 |
| Expert intermediate size | 2048 |
| Router activation | sigmoid |
| Route scale | 8.0 |
| RMSNorm epsilon | 1e-6 |
| EOS token | 200006 |

The model card describes 276B total parameters and 12B active parameters. The
checkpoint is natively multimodal, but the decoder is the first implementation
target.

## 5. Target repository structure

Adopt this layout incrementally:

```text
include/inkling/
  inkling.h                     stable public API and shared public types
  inkling_config.h              parsed model configuration
  inkling_model.h               model/context/generation API

src/
  cache/
    inkling_expert_cache.c/.h   byte-budgeted LRU for packed routed experts
    inkling_kv_cache.c/.h       global and sliding KV plus convolution state
  cli/
    inkling_run.c               argument parsing, generation, report output
  core/
    inkling_alloc.c/.h          checked aligned allocation and overflow helpers
    inkling_bf16.c/.h           lossless BF16-to-F32 widening
    inkling_nvfp4.c/.h          E2M1/scales and packed matrix-vector kernels
    inkling_ops.c/.h            norms, SiLU, softmax, top-k, convolution
    inkling_threads.c/.h        optional OpenMP scheduling helpers
  io/
    inkling_config.c            complete nested config parser
    inkling_index.c             tensor-to-shard catalogue
    inkling_safetensors.c       indexed metadata and positional reads
    inkling_model_files.c/.h    directory validation and shard handles
    inkling_portable_io.h       pread/alignment/platform compatibility
    inkling_trunk.c/.h          optional packed sequential trunk
  model/
    inkling_bind.c/.h           tensor-name census and typed weight views
    inkling_attention.c/.h      hybrid GQA attention and relative logits
    inkling_moe.c/.h            dense MLP, router, shared/routed experts
    inkling_decoder.c/.h        layer and full decoder forward paths
    inkling_generate.c/.h       prefill, decode, sampling, stop conditions
  tokenizer/
    inkling_tokenizer.c/.h      tokenizer.json BPE and special tokens
    inkling_chat.c/.h           message/content/tool token rendering
  multimodal/                   later gated phase
    inkling_vision.c/.h
    inkling_audio.c/.h

tests/
  fixtures/
    checkpoint/                 pinned config/index/header metadata only
    ops/                        small oracle inputs and expected outputs
    tiny/                       generated miniature Inkling graph
    golden/                     token IDs and logits from independent oracle
  unit/
    test_config.c
    test_safetensors.c
    test_index.c
    test_bf16.c
    test_nvfp4.c
    test_ops.c
    test_attention.c
    test_router.c
    test_moe.c
    test_cache.c
    test_tokenizer.c
    test_model.c

tools/
  inkling_ref.py                readable independent PyTorch oracle
  emit_fixtures.py              deterministic operation fixtures
  make_tiny_checkpoint.py       miniature graph with production tensor names
  verify_checkpoint.py          shard/header/tensor census
  compare_logits.py             elementwise C/oracle comparison
  pack_trunk.py                 optional sequential layout packer
  budget.py                     RAM/disk/cache estimator

scripts/
  download-model.sh             resumable pinned-revision download and verify
  inkling-doctor.sh             compiler, ISA, RAM, storage, and model checks
  pack-trunk.sh                 wrapper with manifest and atomic output

docs/
  ARCHITECTURE.md
  CHECKPOINT.md
  VALIDATION.md
  PERFORMANCE.md
  SOURCES.md
```

One implementation file may temporarily hold several small primitives, but public
ownership must follow these boundaries. This keeps tests and later optimization
work from depending on CLI internals.

## 6. Public and internal data model

### Configuration

Extend `InklingConfig` instead of using constants hidden in kernels. Parse:

- every scalar currently present;
- the exact `local_layer_ids` array rather than assuming a stride of six;
- `dense_mlp_idx`, interpreted from upstream behavior and checked against the
  tensor census;
- `log_scaling_n_floor`, `log_scaling_alpha`, `unpadded_vocab_size`,
  `logits_mup_width_multiplier`, `use_embed_norm`, `use_sconv`,
  `shared_expert_sink`, `use_gate_bias`, `gate_activation`, `norm_after_topk`,
  and `use_global_scale`;
- vision, audio, and MTP sub-configs into separate structs, even while execution
  for those branches is disabled.

The config reader must parse nested JSON and arrays correctly. The current `strstr`
reader can confuse repeated keys in nested objects and cannot validate the local
layer list. Replace it with a small checked JSON tokenizer/parser or vendor one tiny
C JSON parser with its license. Missing, duplicate, wrong-type, overflowing, or
inconsistent required fields are fatal configuration errors.

### Tensor catalogue

Replace the linear index lookup with an open-addressed FNV-1a hash table once the
correct Small NVFP4 fixture is installed. Each `InklingTensor` stores:

```c
typedef struct {
    const char *name;
    InklingDataType dtype;
    uint32_t rank;
    uint64_t shape[INKLING_MAX_TENSOR_RANK];
    uint64_t data_offset;
    uint64_t byte_length;
    uint32_t shard_id;
} InklingTensor;
```

All additions and multiplications used to compute offsets, element counts, and byte
sizes require checked overflow. SafeTensors offsets are relative to the data section,
which begins at `8 + header_size`. Validate intervals against the actual shard size,
reject overlap, and reject a dtype/shape/byte-count mismatch.

Add dtypes required by the pinned headers. At minimum expect F32, BF16, packed byte
storage for FP4 payloads, scale tensors, and integer `original_shape` metadata. Derive
the exact dtype enum set from a full header census; do not guess from tensor names.

### Model and execution context

Separate immutable checkpoint data from sequence state:

```c
typedef struct InklingModel InklingModel;       /* config, catalogue, weights, files */
typedef struct InklingContext InklingContext;   /* KV cache, conv state, scratch */
typedef struct InklingSampler InklingSampler;   /* RNG and sampling parameters */
```

Required lifecycle:

```c
int  inkling_model_open(InklingModel **out, const char *model_dir,
                        const InklingModelOptions *options);
void inkling_model_close(InklingModel *model);
int  inkling_context_create(InklingContext **out, const InklingModel *model,
                            const InklingContextOptions *options);
void inkling_context_reset(InklingContext *context);
void inkling_context_free(InklingContext *context);
int  inkling_forward(...);
int  inkling_generate(...);
```

No process-global mutable model state. Two contexts must be able to share one model.

## 7. Checkpoint binding contract

Bind by complete tensor name and validate every expected shape. The principal decoder
families are:

```text
model.llm.embed.weight
model.llm.embed_norm.weight
model.llm.layers.L.attn.{wq_du,wk_dv,wv_dv,wr_du,wo_ud}.weight
model.llm.layers.L.attn.{q_norm,k_norm}.weight
model.llm.layers.L.attn.{k_sconv,v_sconv}.weight
model.llm.layers.L.attn.rel_logits_proj.proj
model.llm.layers.L.{attn_norm,mlp_norm}.weight
model.llm.layers.L.{attn_sconv,mlp_sconv}.weight
model.llm.layers.L.mlp.global_scale
model.llm.layers.L.mlp.{w13_dn,w2_md}.weight             # dense layers
model.llm.layers.L.mlp.gate.{weight,bias,global_scale}   # MoE layers
model.llm.layers.L.mlp.experts.{w13_weight,w2_weight}
model.llm.layers.L.mlp.shared_experts.{shared_w13_weight,shared_w2_weight}
model.llm.norm.weight
model.llm.unembed.weight
```

For each quantized base tensor `X`, the current Small index also maps auxiliary
tensors such as:

```text
X.input_amax
X.original_shape
X.scale
X.scale2
```

These auxiliaries can live in different shards from `X`; resolution must always go
through the global index. `hf_quant_config.json` is also required input because its
exclude list determines which modules remain BF16 and which use NVFP4. Never infer
quantization solely from a layer number or payload dtype.

At model open, emit a census with counts and bytes by class: BF16/F32 trunk,
quantized routed weights, scales, embeddings/unembed, vision, audio, and MTP. Fail
before inference on missing, extra-within-a-required-family, or shape-inconsistent
tensors. Unknown optional families may be reported and ignored only by an explicit
feature gate.

## 8. Numerical contract

### BF16

Store BF16 weights as `uint16_t` and widen inside matrix-vector loops. Widening is
bit exact: move the 16 bits into the high half of an IEEE-754 `uint32_t`, then
`memcpy` to `float`. Accumulate dot products in float32 initially. Compile with
`-ffp-contract=off` for reference builds so compiler FMA choices do not move gates or
top-k boundaries.

### NVFP4

Inkling's NVFP4 is not Kimi K3's MXFP4. The required model is described by the
checkpoint quant config as:

- E2M1 signed 4-bit values, representable magnitudes up to 6;
- group size 16 along the final logical dimension;
- local FP8 E4M3 block scales in `X.scale`;
- a global higher-precision scale in `X.scale2`;
- a logical shape recorded in `X.original_shape`;
- packed nibbles in `X`.

Conceptually, for logical element `i`:

```text
w[i] = E2M1[nibble(i)] * E4M3(block_scale[i / 16]) * global_scale
```

The exact nibble order, E4M3 variant, scale layout/padding, `scale2` convention, and
matrix strides must be proven from real Small checkpoint bytes and the pinned NVIDIA
loader. Create a fixture containing raw bytes plus independently decoded FP32 values.
The fixture must detect swapped nibbles, wrong block axis, ignored padding, wrong
E4M3 special-value behavior, and applying `scale2` in the wrong direction.

Implement in two stages:

1. Scalar decoder and scalar packed matrix-vector multiply, used as the correctness
   oracle on every platform.
2. Architecture-specific AVX2/NEON kernels behind runtime dispatch. They must match
   the scalar path within the gate tolerance and select the same experts/tokens.

Never unpack an entire expert to FP32. Decode one block into registers, multiply,
and discard it. The expert cache stores checkpoint-native packed bytes and scales.

### RMSNorm

For vector `x` and learned weight `w`:

```text
y = w * x / sqrt(mean(x*x) + eps)
```

Compute the variance in float32 or wider reference accumulation. Attention q/k
normalization is per head. Because q and k are normalized, attention scaling is
`1 / head_dim`, not `1 / sqrt(head_dim)`.

### Short convolution

Each layer has four causal depthwise kernel-4 convolutions:

1. attention key projection;
2. attention value projection;
3. attention sublayer output before residual addition;
4. MLP output before residual addition.

The attention-output and MLP-output convolutions include the input residual inside
the upstream short-convolution module's result. Match the pinned oracle exactly.
Incremental state stores the previous three values for every channel and every one
of the four streams. Prefill and token-by-token execution must produce matching
outputs.

### Hybrid grouped-query attention

For layer `L`:

1. RMS-normalize the layer input.
2. Project q, k, v, and relative features r.
3. Apply causal short convolution to k and v.
4. RMS-normalize q and k per 128-wide head.
5. Repeat each of 8 KV heads across 4 query heads logically; avoid physical copies.
6. Compute causal scores with scale `1/128`.
7. Add token-conditioned relative logits.
8. Apply the 512-token window for local layers; global layers use the complete cache.
9. Softmax in float32, multiply values, apply output projection.
10. Apply output short convolution and residual.

Relative logits use `r @ rel_logits_proj.proj` to produce a bias profile by backward
distance. Gather distance `q_position - k_position`. Bias is zero for future positions
and for distances outside the layer's extent. The attention mask still enforces
causality. For global layers beyond position 128000, multiply q and position bias by:

```text
tau = 1 + 0.1 * log(max(1, (position + 1) / 128000))
```

### Dense MLP

Layers 0 and 1 use the dense path:

```text
gate, up = split(w13_dn * x)
y = w2_md * (SiLU(gate) * up)
y *= global_scale
```

Validate the packed orientation of `w13_dn` against its real shape.

### MoE router and experts

Layers 2 through 41 use routed and shared experts. For one token:

```text
logits = gate.weight * x
scores = sigmoid(logits)
selection_scores = routed_scores + correction_bias
selected = top6(selection_scores)
combined_logits = selected routed logits + the two shared logits
weights = sigmoid(combined_logits) / sum(sigmoid(combined_logits))
weights *= route_scale * gate.global_scale
```

Selection bias changes which routed experts win; combination weights come from the
unbiased logits. The two shared experts participate in the same normalization and
receive their own weights. Preserve the oracle's tie behavior and ordering.

For every chosen expert:

```text
gate, up = split(w13 * x)
expert_y = w2 * (SiLU(gate) * up)
```

Sum the six routed outputs and two shared outputs using their router weights. Tests
must isolate selection, normalization, shared-expert weighting, route scale, and the
gate global scale.

### Decoder order and logits

Per layer:

```text
x = x + shortconv(attention(rmsnorm(x)))
x = x + shortconv(mlp(rmsnorm(x)))
```

Embedding lookup is followed by embed RMSNorm. After all 42 layers apply final
RMSNorm and the untied unembed matrix. Apply the configured muP/logit scaling and
any soft cap exactly as the pinned upstream implementation does. Do not add a final
softmax for greedy argmax.

## 9. Memory and storage design

The checkpoint payload is about 171 GB plus the optional 4.46 GB MTP file. A CPU
engine should support multiple memory budgets with identical math:

- Keep metadata, small norms, biases, convolution kernels, router weights, and
  required global scales resident.
- Keep embeddings/unembed resident when the budget allows. A low-memory mode reads
  exact embedding rows and chunks the unembed matvec.
- Store routed expert matrices in NVFP4 and stream only the selected experts.
- Cache complete packed expert tensor slices with an LRU keyed by
  `(layer, expert, projection)` and a strict byte capacity.
- Initially memory-map or position-read the dense/attention trunk. After correctness,
  add an optional packed trunk arranged in layer execution order so each layer uses
  sequential reads.
- Use double-buffered aligned I/O so layer `L+1` can be fetched while layer `L`
  computes. Keep a synchronous path as the reference.

Named presets belong in the CLI only after `tools/budget.py` measures actual Small
tensor bytes. Each preset resolves to explicit trunk/cache/scratch budgets. Report
peak RSS, bytes read, expert-cache hit rate, prefill time, and per-token time.

At one million tokens, a naïve global KV cache is large even with only 8 KV heads.
The first release may impose a lower default context limit, but the requested limit
must be explicit and validated. Local layers retain 512 positions; global layers
retain the requested context. Convolution history is fixed size and separate from KV.

## 10. Tokenizer and chat rendering

Implement raw token-ID input early because it decouples model validation from the
tokenizer. Then implement the released `tokenizer.json` byte-level BPE:

- treat token pieces and merge operations as byte sequences;
- preserve the checkpoint's token IDs exactly;
- load and validate all added/special tokens;
- encode prompt files as bytes, avoiding shell re-encoding;
- provide exact decode round trips for ordinary text and control-token-aware decode.

The chat format contains dedicated role and content tokens such as
`<|message_user|>`, `<|message_model|>`, `<|content_text|>`,
`<|content_thinking|>`, and `<|content_model_end_sampling|>`. Render chat messages
directly to token IDs following the pinned template/renderer. Do not assume that a
plain-text concatenation is faithful. Keep raw `--ids` and raw `--prompt` paths for
reproducible engine tests.

## 11. CLI contract

Target synopsis:

```text
inkling MODEL_DIR [prompt options] [memory options] [generation options] [diagnostics]
```

Prompt options:

```text
--prompt TEXT
--prompt-file PATH
--ids ID,ID,...
--chat-file PATH             JSON messages; later supports image/audio content
```

Memory options:

```text
--preset NAME
--memory-gb N
--expert-cache-gb N
--trunk DIR
--context N
```

Generation options:

```text
--gen N
--temperature X              0 means greedy
--top-p X
--top-k N
--seed N
--incremental                default after its parity gate passes
```

Diagnostics:

```text
--config PATH
--layers N
--dump-logits PATH
--dump-tokens PATH
--dump-cache-trace DIR
--out PATH                   JSON run report
--verify-model
--list-presets
```

Exit codes: `0` success, `1` load/compute failure, `2` usage/config failure,
`3` checkpoint verification failure, `4` incomplete/corrupt streamed expert read.
No partial expert read may silently contribute zeros.

## 12. Validation ladder

Correctness must not depend on owning the full checkpoint. Follow these gates in
order; do not optimize a layer whose gate is red.

### Gate 0: parsers and checkpoint identity

- strict nested config and array parsing;
- current 9-shard Small index and total-size assertion;
- dtype/shape/offset census from all real headers;
- required NVFP4 auxiliaries resolve across shards;
- malformed JSON, overflow, truncation, overlap, wrong dtype, and missing file tests.

### Gate 1: scalar primitives

- BF16 widening bit patterns;
- E2M1 table and nibble order;
- E4M3 scale decode and two-level NVFP4 reconstruction;
- RMSNorm, SiLU, softmax, top-k/ties, and causal depthwise convolution;
- BF16 and NVFP4 matrix-vector products.

Use generated random cases plus at least one fixture cut from real Small checkpoint
bytes.

### Gate 2: isolated architecture blocks

- local attention at beginning, middle, and beyond the 512-token window;
- global attention before and after log-scaling starts;
- relative logits for in-range, future, and out-of-range distances;
- dense MLP;
- router with bias-changing selections;
- one routed expert, shared experts, and complete MoE;
- all four convolution streams with carried state.

Compare arrays elementwise against `tools/inkling_ref.py`, with documented absolute
and relative tolerances. Also assert discrete selections exactly.

### Gate 3: miniature end-to-end graph

Generate a small model that keeps the same graph and tensor naming but reduces
hidden size, heads, layers, experts, vocabulary, and context. Commit the fixture or
generate it deterministically. Validate:

1. teacher-forced logits;
2. greedy generated token IDs;
3. full-prefix generation;
4. incremental generation with KV and convolution caches;
5. identical IDs under several expert-cache capacities.

### Gate 4: real checkpoint slices

- bind and run one dense layer from real weights;
- bind and run one local MoE layer;
- bind and run one global MoE layer;
- compare sampled intermediates and logits to the pinned oracle.

### Gate 5: full text model

- first-token logits, with elementwise error report;
- at least 32 greedy tokens matching exactly;
- incremental/full-recompute token parity;
- cache-budget token parity;
- tokenizer parity across ASCII, multilingual text, emoji, code, JSON, and invalid
  UTF-8 byte sequences where supported.

### Gate 6: performance and portability

- ASan and UBSan build;
- GCC and Clang on Linux x86-64;
- Clang on macOS arm64;
- scalar reference path on machines without AVX2/NEON;
- measured RSS never exceeds requested budget beyond a documented fixed overhead;
- benchmark warm/cold storage, prefill, decode, and cache hit rates.

## 13. Build system

Keep GNU Make as the reference developer path and add CMake for portability/IDE use.

Reference build flags:

```text
-std=c99 -Wall -Wextra -Wpedantic -Werror -Wshadow -Wconversion
-Wpointer-arith -ffp-contract=off
```

Link `libm`, threads, and OpenMP when available. OpenMP must be optional; the scalar
single-threaded build is the correctness baseline. Add targets:

```text
make
make test
make test-sanitize
make oracle-fixtures
make portable
make native
```

Generated fixtures must be reproducible but ordinary `make test` must require no
network, checkpoint, Python package installation, or model weights.

## 14. Implementation sequence

### Phase A: make metadata trustworthy

1. Pin upstream revisions in `docs/SOURCES.md`.
2. Replace the mismatched index/header fixtures with Small NVFP4 metadata.
3. Implement a real nested JSON parser and exact local-layer list.
4. Extend dtype support and build a hash-indexed tensor catalogue.
5. Add full checkpoint census and `--verify-model`.

Exit: the program can prove it has the correct checkpoint and print every required
tensor's dtype, shape, shard, and byte range without loading tensor payloads.

### Phase B: establish numerical primitives

1. Add checked allocation/overflow and BF16 helpers.
2. Prove NVFP4 layout from real bytes.
3. Add scalar BF16/NVFP4 matvec, norm, activation, softmax, top-k, and convolution.
4. Generate and commit small oracle fixtures.

Exit: all primitive tests match the independent oracle and sanitizers pass.

### Phase C: text transformer

1. Bind embedding, dense layers, attention weights, routers, experts, final norm,
   and unembed.
2. Implement local/global attention and relative logits.
3. Implement dense MLP, router, shared experts, and routed experts.
4. Implement layer/full-model forward with raw token IDs.
5. Pass tiny-model and real-layer gates.

Exit: one full-prefix forward pass produces matching logits.

### Phase D: generation and bounded memory

1. Add greedy decode, then configurable sampling.
2. Add local/global KV caches and four convolution histories per layer.
3. Add packed expert LRU and trace reporting.
4. Add trunk streaming/packing and low-memory embedding/unembed paths.
5. Prove identical greedy IDs across incremental/full and memory presets.

Exit: text prompt to text output on the full checkpoint with reported, bounded RSS.

### Phase E: tokenizer and usable CLI

1. Implement tokenizer parity and decode.
2. Implement direct token-ID chat rendering.
3. Add download, doctor, pack, and verification scripts.
4. Document setup, CLI, memory presets, and expected performance.

Exit: a new user can verify the weightless engine, download the pinned checkpoint,
and generate text using documented commands.

### Phase F: multimodal and MTP

1. Vision HMLP patch encoder and image preprocessing.
2. Audio discrete-mel preprocessing/embedding path.
3. Mixed-modality token replacement and chat rendering.
4. MTP loading and speculative decoding, guarded by exact acceptance tests.

Each feature ships only when it has independent oracle fixtures and end-to-end output
parity. `mtp.safetensors` remains optional for ordinary text generation.

## 15. Non-negotiable invariants

1. Runtime dimensions come from validated config and tensor shapes.
2. Local layers come from the explicit `local_layer_ids` list.
3. SafeTensors payload offsets are relative to `8 + header_size`.
4. Quantized base weights and each auxiliary tensor may live in different shards.
5. NVFP4 group size is 16 and uses both local and global scales.
6. q/k are normalized per head and attention scale is `1/head_dim`.
7. Relative position logits are token-conditioned and distance-bounded.
8. Global attention applies configured long-context log scaling to q and relative bias.
9. Router correction bias affects selection, not the gathered unbiased logits.
10. Six routed and two shared experts participate in the normalized mixture.
11. Four separate causal-convolution histories are carried per layer.
12. Incremental decoding and full-prefix recomputation select identical greedy IDs.
13. Changing memory budgets may change time and I/O, never model outputs.
14. Any missing or short streamed weight read fails the run loudly.
15. Scalar kernels remain available as the executable correctness reference.

## 16. Definition of done

The project is complete for the text-first release when:

- `make test` passes without network or model weights;
- sanitizer and portability builds pass;
- the correct pinned Small NVFP4 checkpoint is verified before inference;
- scalar fixtures prove BF16 and NVFP4 decoding from real checkpoint bytes;
- tiny-model teacher forcing and generation match the independent oracle;
- full-model first logits and at least 32 greedy tokens match the oracle;
- full-prefix and incremental decoding match;
- at least three memory budgets emit identical greedy IDs;
- the CLI accepts text, prompt files, and raw token IDs and writes a JSON run report;
- setup, validation, memory use, and measured performance are documented with commands
  that were actually run.

Multimodal and MTP completion are separate gates after this text release. Their code
must not be advertised as supported until their corresponding Phase F checks pass.
