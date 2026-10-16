# Phase C validation record

Issue [16](https://github.com/raghulchandramouli/inkling-in-c/issues/16) completed
C1 through C5 on 2026-10-15 against merged B4 baseline `57626a2`.
The [runtime guide](TEXT.md) defines the API, native layout, numerical limits,
and reproduction commands. The [results](validation/phase-c-results.json) contain
source and artifact hashes, per-stage errors, and mutation rejection details.
The [decision trail](validation/phase-c-decisions.tsv) records the design and fixes.
Its timestamps record log-entry times. Rows 9 through 20 summarize earlier work
retrospectively and do not preserve the original decision times.

## Execution environment

The machine used macOS arm64 and Apple Clang 21.0.0, with the repository's strict
C99 warnings and `-ffp-contract=off`. Ordinary tests used system Python 3.9.6 with
no PyTorch installed. Explicit oracle generation used Python 3.11.16, PyTorch
2.8.0, and torch revision `a1cb3cc05d46d198467bebbb6e8fba50a325d4e7`.
Genuine GCC and Linux runs were unavailable.

## Completed checks

| Check | Observed result |
|---|---|
| `make test CC=clang` in a fresh source export | Passed all existing Phase A/B regressions and new model tests |
| `make test-sanitize CC=clang` in that export | Passed with ASan and UBSan |
| Miniature graph and architecture comparisons | All 17 cases passed, including complete logits at prefix lengths 1, 3, and 6 |
| Fixture structure and provenance audit | All 75 rejection cases passed without PyTorch or real weights |
| Native range and CLI failure checks | Passed positive bind, damaged-range rejection, retry, raw-ID parsing, forward, and atomic output checks |
| `make test-mutations CC=clang` | All ten existing numerical faults and seven new model faults failed at their intended comparisons |
| Two explicit offline fixture regenerations | All seven files matched each other and the committed fixture byte for byte |
| Real layer acceptance | All three original layers, complete outputs, 64 stage traces, and exact routed IDs passed |

The fresh export contained no ignored weights, source cache, virtual environment,
or build products. Both make commands ran under
`sandbox-exec -p '(version 1)(allow default)(deny network*)'` with only system
tools on PATH. A separate connection attempt returned `EPERM` under that policy.
The first export runner stopped on a quoting error before tests began. Its
replacement passed argument arrays directly and completed both suites.

The CLI checks use the actual CLI source with its model opener replaced only in a
dedicated test object. The product executable continues to reject the miniature
checkpoint. Full-checkpoint CLI inference was not run.

## Actual checkpoint payloads

Extraction used checkpoint revision
`b6a99534467840620d411e4cd4ad5819b2610d9c`. The fetcher required exact HTTP Range
responses, authenticated cached chunks by SHA-256, and enforced a 2 GiB aggregate
cap. It downloaded 1,259,738,686 payload bytes. The files remain in ignored
`weights/c16-real/`, with immutable URLs, original tensor offsets, payload hashes,
inputs, selected experts, and reference outputs in each `ranges.json`.

| Original layer | Architecture | Payload bytes | Maximum output absolute error | Maximum output fraction of allowed error |
|---|---|---:|---:|---:|
| 0 | Dense local attention | 490,848,770 | 0.0042751172 | 0.036223 |
| 2 | BF16 local MoE | 492,963,332 | 0.0002861027 | 0.147636 |
| 5 | NVFP4 global MoE | 275,926,584 | 0.0003963545 | 0.178203 |

Output comparisons use `atol=1e-3` and `rtol=2e-4`, element by element.
Norm and probability comparisons use the stricter limits in the runtime guide.
Absolute error alone is not the acceptance limit. The report retains relative
errors and the fraction of each element's allowed error separately.

Each real run uses one independent nonzero 4096-element hidden vector, one token
at position zero, and zero convolution prehistory. Layer 2 selects original
experts `[67,30,69,120,25,146]`; layer 5 selects `[21,20,219,110,58,145]`.
The runs compare the selected experts and both shared experts. They do not execute
unselected real experts or propagate one real layer into another.

Reproduce with the real-layer commands in the [runtime guide](TEXT.md#fetch-and-check-bounded-real-layers).

## Review and corrections

An independent C review by `gpt-6-sol` found no proven P1 or P2 defect. An
independent oracle review by `gpt-6.1-sol` required three corrections. The initial
near-floor cases could miss omitted tau scaling, real-case audits allowed missing
traces and enlarged tolerances, and long attention cases lacked exact edge samples.
The implementation added a farther tau case, strict real-case coverage and limit
validation, and sampled probabilities and biases at the exact mask boundaries.
A fresh `gpt-6.1-sol` review found no remaining proven P1 or P2 defect in those fixes.

Separate comment reviews removed one redundant C comment and fourteen docstrings.
An AST comparison verified unchanged executable behavior, including literal CLI
help text. Source identity changes then required honest offline regeneration of
both miniature and real manifests. Final offline and real checks used those files.

A final trail review requested `gpt-6-luna` and checked the actual run transcript,
all 74 recorded source hashes, and all 12 then-recorded artifact hashes. Those
hashes matched. The reviewer flagged retrospective timestamps, the isolated-real
coverage limit, and PR delivery that was still pending during review. This record
labels the timestamps and coverage explicitly. The requested review used another
available Codex model, so diversity across model families remained limited.

## Remaining milestones

This evidence establishes widened-float32 scalar graph parity for the committed
miniature model and isolated original layers. It does not establish fused BF16
backend parity, full-checkpoint logits, real long-history attention, or generation.

Phase D still owns incremental caches, generation, memory budgets, and cache
parity. Phase E owns tokenizer and chat support. Phase F owns multimodal and MTP
execution.
