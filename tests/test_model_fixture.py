import copy
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from check_model import audit_cases, audit_tiny, audit_real_case, compare
from fetch_checkpoint_metadata import sha256
from fetch_model_layers import tolerances


def representative_real(oracle, layer):
    source_layer = 1 if layer else 0
    tiny = next(case for case in oracle["cases"] if case["name"] == f"layer_{source_layer}")
    lengths = {key: 4096 for key in tiny["traces"]}
    lengths.update({key: 1024 for key in ("k_proj", "k_norm", "v_proj", "k_conv", "v_conv")})
    lengths.update(r_proj=512, relative_bias=32, attention_probs=32)
    if layer:
        lengths.update(router_logits=258, router_weights=6, shared_gammas=2)
    traces = {key: [values[0]] * lengths[key] for key, values in tiny["traces"].items()}
    traces["attention_probs"] = [1.0] * 32
    case = dict(name=f"real_layer_{layer}", kind="layer", layer=layer, tokens=1, position_start=0,
        shape=[1, 1, 4096], input=[((i * 73) % 257 - 128) / 128 for i in range(4096)],
        expected=list(traces["output"]), traces=traces, indices={"router_selected": list(range(6))} if layer else {})
    mappings = copy.deepcopy([m for m in oracle["mappings"] if m["source"].startswith(f"model.llm.layers.{source_layer}.")])
    for mapping in mappings:
        mapping["source"] = mapping["source"].replace(f"layers.{source_layer}.", f"layers.{layer}.")
        if "expert_ids" in mapping:
            mapping["expert_ids"] = list(range(6))
    result = dict(layer=layer, case=case, tools=oracle["tools"], references=oracle["references"],
        mappings=mappings, tolerances=tolerances(real=True),
        input_sha256=sha256(struct.pack("<4096f", *case["input"])),
        expected_sha256=sha256(struct.pack("<4096f", *case["expected"])))
    if layer:
        result["selection"] = dict(expert_ids=list(range(6)), router_indices=list(range(6)),
            router_weights=list(traces["router_weights"]), shared_gammas=list(traces["shared_gammas"]), margin=0.25)
    return result


def rejects(audit, original, mutations):
    for mutate in mutations:
        changed = copy.deepcopy(original)
        mutate(changed)
        try:
            audit(changed)
        except ValueError:
            continue
        raise AssertionError("incomplete or invalid oracle was accepted")
    return len(mutations)


def main():
    directory = ROOT / "tests/fixtures/model"
    oracle = audit_tiny(directory)
    config = json.loads((directory / "config.json").read_text())["text_config"]
    mutations = [
        lambda x: x.update(cases=[]),
        lambda x: x["cases"][0].update(expected=[]),
        lambda x: x["cases"][0]["expected"].__setitem__(0, float("nan")),
        lambda x: x["cases"][3]["traces"].pop("q_proj"),
        lambda x: x["cases"][3]["traces"]["attention_probs"].__setitem__(1, 0.1),
        lambda x: x["cases"][5]["indices"]["router_selected"].__setitem__(0, 0.5),
        lambda x: x.update(tools={}),
        lambda x: x.update(mappings=[]),
        lambda x: x["cases"][9].pop("trace_samples"),
        lambda x: x["cases"][9]["trace_samples"]["relative_bias"].reverse(),
        lambda x: x["cases"][9]["trace_samples"]["attention_probs"].__setitem__(0, 1.0),
        lambda x: x["cases"][9]["traces"]["attention_probs"].__setitem__(0, 1e-12),
        lambda x: x["cases"][9]["traces"]["relative_bias"].__setitem__(0, 1e-12),
        lambda x: x["cases"][10]["traces"].update(relative_bias=[]),
    ]
    count = rejects(lambda changed: audit_cases(changed, config), oracle, mutations)
    for layer in (0, 2, 5):
        real = representative_real(oracle, layer)
        audit_real_case(real)
        mutations = [
            lambda x: x.update(tools={}),
            lambda x: x.update(mappings=[]),
            lambda x: x["mappings"].pop(),
            lambda x: x["case"].update(traces={}),
            lambda x: x["case"]["traces"].pop("q_proj"),
            lambda x: x["case"]["traces"]["q_norm"].pop(),
            lambda x: x["case"]["traces"]["q_norm"].__setitem__(0, float("nan")),
            lambda x: x["case"]["expected"].__setitem__(0, float("inf")),
            lambda x: x["case"].update(shape=[4096]),
            lambda x: x["tolerances"]["output"].update(atol=1e100),
            lambda x: x["tolerances"]["output"].update(atol=float("nan")),
            lambda x: x["tolerances"]["output"].update(rtol=float("inf")),
        ]
        if layer:
            mutations.extend([
                lambda x: x["case"].update(indices={}),
                lambda x: x.pop("selection"),
                lambda x: x["selection"].update(margin=0),
                lambda x: x["selection"].update(margin=float("nan")),
                lambda x: x["selection"].update(margin=float("inf")),
                lambda x: x["selection"]["router_weights"].__setitem__(0, 1.0),
                lambda x: x["selection"]["shared_gammas"].__setitem__(0, 1.0),
                lambda x: x["selection"]["expert_ids"].__setitem__(0, 255),
                lambda x: x["case"]["indices"]["router_selected"].__setitem__(0, 256),
                lambda x: x["case"]["indices"]["router_selected"].__setitem__(0, 0.5),
            ])
        count += rejects(audit_real_case, real, mutations)
    far = next(case for case in oracle["cases"] if case["name"] == "tau_far")
    for unscaled, scaled in (("q_norm", "query_scaled"), ("relative_bias", "relative_scaled")):
        try:
            compare(far["traces"][unscaled], far["traces"][scaled], oracle["tolerances"][scaled], "tau_far." + scaled)
        except ValueError:
            count += 1
        else:
            raise AssertionError("tau scaling omission was accepted")
    for stage in ("relative_bias", "attention_probs"):
        try:
            compare([1e-12], [0.0], oracle["tolerances"][stage], "boundary." + stage)
        except ValueError:
            count += 1
        else:
            raise AssertionError("nonzero masked sample was accepted")
    try:
        compare([0.0], [1e-24], oracle["tolerances"]["attention_probs"], "global_extent_1024.attention_probs")
    except ValueError:
        count += 1
    else:
        raise AssertionError("lost global attention beyond relative extent was accepted")
    assert "torch" not in sys.modules
    print(f"PASS model fixture audit and {count} rejection cases without torch or real weights")


if __name__ == "__main__":
    main()
