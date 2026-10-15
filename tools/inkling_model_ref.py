from __future__ import annotations

import ast
import math
import sys
from types import SimpleNamespace

from fetch_checkpoint_metadata import request, sha256, write_atomic
from inkling_ref import SOURCE_DIR, SOURCE_PINS, source_bytes

EXTRA_SOURCES = {
    "config": ("models/inkling/configuration_inkling.py", "ec8c7edf135c514103213829701f0151e339ce5f92f556ef57995de064ea0e91"),
    "mapping": ("conversion_mapping.py", "67deef358d7d09df8eb579ef5b7e79f6a4e42137f4d1cdba63891bfb85c3b187"),
    "conversion": ("core_model_loading.py", "c5b6bcdb6401a3cfdf825bc979d41ac3e7e075e09459021363dce5e68c54de22"),
    "mask": ("masking_utils.py", "50a737f63d8c778a5597fa34ac139049af921f44958205e3e1c29fe2bae77254"),
}
MODEL_SOURCE_PINS = dict(SOURCE_PINS)
for _name, (_path, _hash) in EXTRA_SOURCES.items():
    MODEL_SOURCE_PINS[_name] = dict(
        url=f"https://raw.githubusercontent.com/huggingface/transformers/{SOURCE_PINS['transformers']['revision']}/src/transformers/{_path}",
        revision=SOURCE_PINS["transformers"]["revision"], sha256=_hash)

BODIES = {
    "InklingRMSNorm", "InklingRelativeLogits", "repeat_kv", "eager_attention_forward",
    "InklingAttention", "InklingMLP", "InklingExperts", "InklingTopkRouter",
    "InklingSharedExperts", "InklingMoE", "apply_mask_to_padding_states",
    "causal_conv1d_update", "causal_conv1d_fn", "InklingShortConvolution",
    "InklingDecoderLayer", "InklingNormedEmbedding", "InklingTextModel", "InklingForCausalLM",
}
MASK_BODIES = {"and_masks", "causal_mask_function", "sliding_window_overlay",
               "sliding_window_causal_mask_function"}


def _execute(raw, names, namespace, filename):
    nodes = [n for n in ast.parse(raw).body
             if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    if {n.name for n in nodes} != names:
        raise ValueError(f"missing original reference bodies in {filename}")
    for node in nodes:
        for child in ast.walk(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef)):
                child.decorator_list = []
    future = ast.parse("from __future__ import annotations").body
    exec(compile(ast.Module(body=future + nodes, type_ignores=[]), filename, "exec"), namespace)


def load_reference(torch, *, source_dir=SOURCE_DIR, allow_download=False):
    sources = {"transformers": source_bytes("transformers", source_dir=source_dir,
                                             allow_download=allow_download)}
    for key in EXTRA_SOURCES:
        pin = MODEL_SOURCE_PINS[key]
        path = source_dir / (pin["sha256"] + ".py")
        raw = path.read_bytes() if path.exists() else (
            request(pin["url"], max_bytes=2 * 1024 * 1024) if allow_download else None)
        if raw is None or sha256(raw) != pin["sha256"]:
            raise ValueError(f"missing or modified pinned {key} source: {path}")
        if not path.exists():
            write_atomic(path, raw)
        sources[key] = raw

    class InferenceBase(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

        def post_init(self):
            pass

    class Output(SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(hidden_states=None, attentions=None, **kwargs)

    namespace = dict(torch=torch, nn=torch.nn, F=torch.nn.functional,
                     GradientCheckpointingLayer=torch.nn.Module,
                     InklingPreTrainedModel=InferenceBase, GenerationMixin=object,
                     BaseModelOutputWithPast=Output,
                     InklingCausalLMOutputWithPast=SimpleNamespace,
                     ACT2FN={"silu": torch.nn.functional.silu}, ConversionOps=object)
    namespace["ALL_ATTENTION_FUNCTIONS"] = SimpleNamespace(get_interface=lambda _, fallback: fallback)
    _execute(sources["transformers"], BODIES, namespace, MODEL_SOURCE_PINS["transformers"]["url"])
    _execute(sources["conversion"], {"Interleave"}, namespace, MODEL_SOURCE_PINS["conversion"]["url"])
    _execute(sources["mask"], MASK_BODIES, namespace, MODEL_SOURCE_PINS["mask"]["url"])
    return namespace


def checkpoint_config(raw):
    cfg = dict(raw.get("text_config", raw))
    required = {"q_bias": False, "o_bias": False, "use_embed_norm": True,
                "use_sconv": True, "shared_expert_sink": True, "use_gate_bias": True,
                "gate_activation": "sigmoid", "norm_after_topk": True,
                "use_global_scale": True, "final_logit_softcapping": None,
                "hidden_act": "silu", "attention_dropout": 0.0}
    for key, value in required.items():
        if key in cfg and cfg[key] != value:
            raise ValueError(f"unsupported reference config: {key}={cfg[key]!r}")
    cfg["moe_intermediate_size"] = cfg["intermediate_size"]
    cfg["intermediate_size"] = cfg["dense_intermediate_size"]
    cfg["conv_kernel_size"] = cfg["sconv_kernel_size"]
    cfg["layer_types"] = ["hybrid_sliding" if i in cfg["local_layer_ids"] else "hybrid"
                          for i in range(cfg["num_hidden_layers"])]
    cfg["mlp_layer_types"] = ["dense" if i < cfg["dense_mlp_idx"] else "sparse"
                              for i in range(cfg["num_hidden_layers"])]
    cfg.update(hidden_act="silu", attention_dropout=0.0, _attn_implementation="eager",
               pad_token_id=None)
    return SimpleNamespace(**cfg)


def causal_masks(reference, config, length, torch):
    positions = torch.arange(length)
    q, k = positions[:, None], positions[None, :]
    full = reference["causal_mask_function"](0, 0, q, k)
    local = reference["sliding_window_causal_mask_function"](config.sliding_window_size)(0, 0, q, k)
    return {"full_attention": torch.zeros(1, 1, length, length).masked_fill(~full, -torch.inf),
            "sliding_attention": torch.zeros(1, 1, length, length).masked_fill(~local, -torch.inf),
            "linear_attention": None}


def storage_tensor(raw, dtype, shape, torch):
    if sys.byteorder != "little":
        raise ValueError("oracle requires a little-endian host")
    types = {"BF16": torch.bfloat16, "F32": torch.float32, "U8": torch.uint8,
             "F8_E4M3": torch.float8_e4m3fn, "I64": torch.int64}
    item_size = torch.empty((), dtype=types[dtype]).element_size()
    if len(raw) != math.prod(shape) * item_size:
        raise ValueError(f"wrong payload length for {dtype} {shape}")
    value = torch.frombuffer(bytearray(raw), dtype=types[dtype]).reshape(shape)
    return value.float() if dtype in ("BF16", "F32") else value


def decode_nvfp4(packed, scales, scale2, shape, torch, modelopt_class):
    rows, columns = shape
    if (columns % 16 or tuple(packed.shape) != (rows, columns // 2)
            or tuple(scales.shape) != (rows, columns // 16) or scale2.numel() != 1):
        raise ValueError("invalid NVFP4 expert storage")
    oracle = modelopt_class()
    oracle._quantized_data = packed
    oracle.metadata = {"shape": shape, "dtype": torch.float32}
    return oracle.dequantize(dtype=torch.float32, scale=scales,
                             double_scale=scale2.reshape(()), block_sizes={-1: 16})


LAYER_MAPPING = {
    "attn_norm.weight": "input_layernorm.weight",
    "mlp_norm.weight": "post_attention_layernorm.weight",
    "attn.wq_du.weight": "self_attn.q_proj.weight",
    "attn.wk_dv.weight": "self_attn.k_proj.weight",
    "attn.wv_dv.weight": "self_attn.v_proj.weight",
    "attn.wr_du.weight": "self_attn.r_proj.weight",
    "attn.wo_ud.weight": "self_attn.o_proj.weight",
    "attn.q_norm.weight": "self_attn.q_norm.weight",
    "attn.k_norm.weight": "self_attn.k_norm.weight",
    "attn.rel_logits_proj.proj": "self_attn.rel_logits_proj.proj",
    "attn.k_sconv.weight": "self_attn.k_sconv.conv1d.weight",
    "attn.v_sconv.weight": "self_attn.v_sconv.conv1d.weight",
    "attn_sconv.weight": "attn_sconv.conv1d.weight",
    "mlp_sconv.weight": "mlp_sconv.conv1d.weight",
}


def _bind(module, name, value, torch):
    parent_name, _, field = name.rpartition(".")
    parent = module.get_submodule(parent_name) if parent_name else module
    old = getattr(parent, field)
    if tuple(old.shape) != tuple(value.shape):
        raise ValueError(f"shape mismatch for {name}: {tuple(value.shape)} != {tuple(old.shape)}")
    value = value.detach().to(dtype=torch.float32, device="cpu")
    if not torch.isfinite(value).all():
        raise ValueError(f"nonfinite weight: {name}")
    if isinstance(old, torch.nn.Parameter):
        value = torch.nn.Parameter(value, requires_grad=False)
    setattr(parent, field, value)


class ExpertTable:
    def __init__(self, name, read, ids, shape, convert):
        self.name, self.read, self.ids = name, read, frozenset(ids)
        self.shape, self.convert = tuple(shape), convert

    def __getitem__(self, index):
        index = int(index)
        if index not in self.ids:
            raise ValueError(f"missing selected expert {index} in {self.name}")
        matrix = self.read(self.name, expert=index)
        if tuple(matrix.shape) != self.shape:
            raise ValueError(f"invalid selected expert shape: {self.name}[{index}]")
        if matrix.device.type != "cpu" or not matrix.is_floating_point():
            raise ValueError(f"expert reader must supply decoded CPU floats: {self.name}[{index}]")
        return self.convert(matrix.float())


def bind_layer(module, config, layer_id, read, reference, torch, *, expert_ids=None):
    prefix = f"model.llm.layers.{layer_id}."
    mappings = []

    def bind(source, target, value=None):
        if value is None:
            value = read(prefix + source)
        _bind(module, target, value, torch)
        mappings.append(dict(source=prefix + source, target=target))

    def deinterleave(value, dim=0):
        return reference["Interleave"](dim=dim).convert({"weight": value}, [], ["weight"])["weight"]

    for source, target in LAYER_MAPPING.items():
        bind(source, target)
    if config.mlp_layer_types[layer_id] == "dense":
        gate, up = deinterleave(read(prefix + "mlp.w13_dn.weight")).chunk(2, dim=0)
        bind("mlp.w13_dn.weight", "mlp.gate_proj.weight", gate)
        bind("mlp.w13_dn.weight", "mlp.up_proj.weight", up)
        bind("mlp.w2_md.weight", "mlp.down_proj.weight")
        bind("mlp.global_scale", "mlp.global_scale")
    else:
        bind("mlp.gate.weight", "mlp.gate.weight")
        bind("mlp.gate.bias", "mlp.gate.e_score_correction_bias")
        bind("mlp.gate.global_scale", "mlp.gate.global_scale")
        gate, up = deinterleave(read(prefix + "mlp.shared_experts.shared_w13_weight"), 1).chunk(2, dim=1)
        bind("mlp.shared_experts.shared_w13_weight", "mlp.shared_experts.gate_proj", gate)
        bind("mlp.shared_experts.shared_w13_weight", "mlp.shared_experts.up_proj", up)
        bind("mlp.shared_experts.shared_w2_weight", "mlp.shared_experts.down_proj")
        ids = range(config.n_routed_experts) if expert_ids is None else tuple(expert_ids)
        if any(i < 0 or i >= config.n_routed_experts for i in ids):
            raise ValueError("invalid selected expert ID")
        for source, target, convert in (("w13_weight", "gate_up_proj", deinterleave),
                                        ("w2_weight", "down_proj", lambda x: x)):
            old = getattr(module.mlp.experts, target)
            delattr(module.mlp.experts, target)
            name = prefix + "mlp.experts." + source
            setattr(module.mlp.experts, target, ExpertTable(name, read, ids, old.shape[1:], convert))
            mappings.append(dict(source=name, target="mlp.experts." + target,
                                 expert_ids=list(ids)))
    module.eval()
    return mappings


def build_layer(config, layer_id, read, reference, torch, *, expert_ids=None):
    with torch.device("meta"):
        module = reference["InklingDecoderLayer"](config, layer_id)
    mappings = bind_layer(module, config, layer_id, read, reference, torch, expert_ids=expert_ids)
    return module, mappings


def build_model(config, read, reference, torch):
    with torch.device("meta"):
        module = reference["InklingForCausalLM"](config)
    mappings = []
    for source, target in (("embed.weight", "model.embed_tokens.weight"),
                           ("embed_norm.weight", "model.embed_tokens.embed_norm.weight"),
                           ("norm.weight", "model.norm.weight"),
                           ("unembed.weight", "lm_head.weight")):
        name = "model.llm." + source
        _bind(module, target, read(name), torch)
        mappings.append(dict(source=name, target=target))
    for index, layer in enumerate(module.model.layers):
        mappings.extend(bind_layer(layer, config, index, read, reference, torch))
    module.eval()
    return module, mappings


def capture(module, torch):
    records, handles = {}, []

    def snapshot(value):
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if isinstance(value, (list, tuple)):
            return tuple(snapshot(v) for v in value)
        return value

    def hook(name):
        def save(_module, inputs, output):
            records[name] = dict(inputs=snapshot(inputs), output=snapshot(output))
            if hasattr(_module, "e_score_correction_bias"):
                records[name]["correction"] = snapshot(_module.e_score_correction_bias)
                records[name]["router_logits"] = torch.nn.functional.linear(
                    inputs[0].reshape(-1, _module.hidden_dim), _module.weight).detach().clone()
        return save

    for name, child in module.named_modules():
        handles.append(child.register_forward_hook(hook(name)))
    return records, handles
