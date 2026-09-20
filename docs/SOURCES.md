# Pinned upstream sources

This document records the immutable upstream revisions used to implement and
validate Inkling-Small-NVFP4. Links to a moving `main` branch are useful for
discovery, but they are not implementation evidence for this repository.

Pins were reviewed on **2026-09-20**. A newer upstream revision must not silently
change behavior here; follow the update procedure at the end of this document.

## Source precedence

When sources disagree, use them in this order:

1. The pinned Inkling-Small-NVFP4 checkpoint determines file names, tensor names,
   tensor shapes, dtypes, stored offsets, quantization configuration, tokenizer
   assets, and chat assets.
2. The pinned Hugging Face Transformers implementation determines portable model
   semantics: configuration interpretation, operation order, attention, routing,
   short convolution, multimodal projection, and MTP behavior.
3. The pinned SGLang and vLLM implementations provide independent serving-time
   evidence for checkpoint loading, tensor transformations, cache layout, and
   optimized execution. They do not override the checkpoint or portable model
   semantics merely because an optimized kernel has extra constraints.
4. The pinned NVIDIA Transformer Engine material determines the generic NVFP4
   numerical format. Checkpoint-specific layout still has to be proven from the
   checkpoint and loaders.
5. Kimi K3 is a structural reference only. Its MXFP4 numerical behavior must not
   be used for Inkling's NVFP4 tensors.

## Checkpoint

### Thinking Machines Lab Inkling-Small-NVFP4

- Repository: `thinkingmachines/Inkling-Small-NVFP4`
- Revision: `b6a99534467840620d411e4cd4ad5819b2610d9c`
- Immutable tree:
  <https://huggingface.co/thinkingmachines/Inkling-Small-NVFP4/tree/b6a99534467840620d411e4cd4ad5819b2610d9c>
- Local identity record:
  [`tests/fixtures/checkpoint/metadata-manifest.json`](../tests/fixtures/checkpoint/metadata-manifest.json)
- Local quantization record:
  [`tests/fixtures/checkpoint/hf_quant_config.json`](../tests/fixtures/checkpoint/hf_quant_config.json)

Use this revision for all fixture generation and real-checkpoint validation. Its
index contains 1,360 tensors in nine model shards plus `mtp.safetensors`; model
shard payload size is `170733074592` bytes. The checked-in fixtures contain only
configuration, index, and SafeTensors headers, not tensor payloads.

The checkpoint quantization record identifies NVFP4, a group size of 16, and a
single `-1` block axis. That checkpoint evidence takes precedence over generic
NVFP4 documentation that describes 2D weight scaling as a default option.

## Architecture reference

### Hugging Face Transformers

- Repository: `huggingface/transformers`
- Commit: `c587bc884db2c2e31fc2b8102314656b17aa07b1`
- Commit date: 2026-09-18
- Immutable commit:
  <https://github.com/huggingface/transformers/commit/c587bc884db2c2e31fc2b8102314656b17aa07b1>
- Configuration:
  <https://github.com/huggingface/transformers/blob/c587bc884db2c2e31fc2b8102314656b17aa07b1/src/transformers/models/inkling/configuration_inkling.py>
- Model:
  <https://github.com/huggingface/transformers/blob/c587bc884db2c2e31fc2b8102314656b17aa07b1/src/transformers/models/inkling/modeling_inkling.py>

Use the configuration file for nested text/audio/vision/MTP configuration,
`local_layer_ids`, dense-versus-sparse MLP selection, expert counts, route scale,
and compatibility key mappings. Use the model file for RMSNorm, relative logits,
hybrid attention, logarithmic global-attention scaling, routing normalization,
shared experts, short-convolution state, residual order, embeddings, and
multimodal execution.

Transformers operates on logical tensors and is not by itself evidence for the
packed NVFP4 on-disk nibble order or auxiliary-tensor storage.

## Serving implementation references

### SGLang

- Repository: `sgl-project/sglang`
- Commit: `5c69e32abe013fa1b913022682a3104c79105f37`
- Commit date: 2026-09-20
- Immutable commit:
  <https://github.com/sgl-project/sglang/commit/5c69e32abe013fa1b913022682a3104c79105f37>
- Model and checkpoint loader:
  <https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/inkling.py>
- Configuration adapter:
  <https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/configs/inkling.py>
- Attention:
  <https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/inkling_common/attn.py>
- MoE:
  <https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/inkling_common/moe.py>
- Short convolution:
  <https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/inkling_common/sconv.py>
- Inkling NVFP4 configuration:
  <https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/inkling_common/quantization/config.py>

Use SGLang to cross-check production tensor-name remapping, NVFP4 auxiliary
loading, group-size enforcement, expert layout, attention projection preparation,
relative logits, and incremental short-convolution state.

### vLLM

- Repository: `vllm-project/vllm`
- Commit: `e378275a8f20eac9e92212e4a1fee84ec5a31dc7`
- Commit date: 2026-09-20
- Immutable commit:
  <https://github.com/vllm-project/vllm/commit/e378275a8f20eac9e92212e4a1fee84ec5a31dc7>
- Inkling configuration:
  <https://github.com/vllm-project/vllm/blob/e378275a8f20eac9e92212e4a1fee84ec5a31dc7/vllm/models/inkling/configs.py>
- NVIDIA model and checkpoint loader:
  <https://github.com/vllm-project/vllm/blob/e378275a8f20eac9e92212e4a1fee84ec5a31dc7/vllm/models/inkling/nvidia/model.py>
- NVIDIA attention:
  <https://github.com/vllm-project/vllm/blob/e378275a8f20eac9e92212e4a1fee84ec5a31dc7/vllm/models/inkling/nvidia/attention.py>
- NVIDIA MoE and router:
  <https://github.com/vllm-project/vllm/blob/e378275a8f20eac9e92212e4a1fee84ec5a31dc7/vllm/models/inkling/nvidia/moe.py>
- NVIDIA short convolution:
  <https://github.com/vllm-project/vllm/blob/e378275a8f20eac9e92212e4a1fee84ec5a31dc7/vllm/models/inkling/nvidia/short_conv.py>
- Common multimodal preprocessing:
  <https://github.com/vllm-project/vllm/blob/e378275a8f20eac9e92212e4a1fee84ec5a31dc7/vllm/models/inkling/common/mm_preprocess.py>

Use vLLM as a second independent check on checkpoint name mapping, expert tensor
loading, selection-bias versus combination-weight behavior, shared-expert weights,
relative attention, short-convolution caching, and multimodal preprocessing.

## NVFP4 numerical references

### NVIDIA Transformer Engine

- Repository: `NVIDIA/TransformerEngine`
- Commit: `699ed6ec9e9c600c3d0acd3a488c4ca6f339e5d9`
- Commit date: 2026-09-19
- Immutable commit:
  <https://github.com/NVIDIA/TransformerEngine/commit/699ed6ec9e9c600c3d0acd3a488c4ca6f339e5d9>
- NVFP4 format documentation:
  <https://github.com/NVIDIA/TransformerEngine/blob/699ed6ec9e9c600c3d0acd3a488c4ca6f339e5d9/docs/features/low_precision_training/nvfp4/nvfp4.rst>
- Reference implementation:
  <https://github.com/NVIDIA/TransformerEngine/blob/699ed6ec9e9c600c3d0acd3a488c4ca6f339e5d9/transformer_engine/pytorch/custom_recipes/reference_nvfp4.py>
- Core CUDA definitions:
  <https://github.com/NVIDIA/TransformerEngine/blob/699ed6ec9e9c600c3d0acd3a488c4ca6f339e5d9/transformer_engine/common/cast/nvfp4/core_nvfp4.cuh>

Use these sources for E2M1 values, maximum magnitude 6, E4M3 block scales,
16-element blocks, hierarchical global scaling, and row/column scale-layout
concepts. Do not assume Transformer Engine's training-time transforms, stochastic
rounding, 2D default, padding, swizzling, or transposed GEMM layout are present in
the released checkpoint without direct checkpoint and loader evidence.

### NVIDIA Model Optimizer (supplementary)

- Repository: `NVIDIA/Model-Optimizer`
- Commit: `b311c054de4052df9c7f3de9409b7598f44a0dba`
- Commit date: 2026-09-19
- Immutable commit:
  <https://github.com/NVIDIA/Model-Optimizer/commit/b311c054de4052df9c7f3de9409b7598f44a0dba>
- NVFP4 tensor implementation:
  <https://github.com/NVIDIA/Model-Optimizer/blob/b311c054de4052df9c7f3de9409b7598f44a0dba/modelopt/torch/quantization/qtensor/nvfp4_tensor.py>
- Hugging Face quantization integration:
  <https://github.com/NVIDIA/Model-Optimizer/blob/b311c054de4052df9c7f3de9409b7598f44a0dba/modelopt/torch/quantization/plugins/huggingface.py>

Model Optimizer is supporting evidence for the exported quantized-tensor format.
The pinned Inkling checkpoint and the two serving loaders remain authoritative for
the exact auxiliary names and shapes in this model.

## Structural reference

### Kimi K3 in C

- Repository: `FareedKhan-dev/kimi-k3-in-c`
- Commit: `ac1584a70205c3a00d5346f736834818f4cc11b4`
- Commit date: 2026-09-10
- Immutable commit:
  <https://github.com/FareedKhan-dev/kimi-k3-in-c/commit/ac1584a70205c3a00d5346f736834818f4cc11b4>

Use this only for C repository organization, bounded-memory streaming, cache and
CLI patterns, and validation workflow. Kimi K3's MXFP4 decoder is not an NVFP4
oracle and must not be copied into Inkling numerical code.

## Updating a pin

An upstream pin may be updated only in a dedicated change that:

1. records the old and new commits and summarizes relevant upstream differences;
2. checks the pinned checkpoint revision and metadata manifest have not changed;
3. compares architecture, tensor loading, routing, attention, short convolution,
   and NVFP4 behavior affected by the update;
4. updates oracle fixtures or expected values only with an explained source change;
5. runs the complete parser, numerical, tiny-model, and real-slice validation gates
   available at that point; and
6. updates this document and the implementation blueprint together.

Never replace an immutable permalink here with a moving branch URL.
