#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "src/model/inkling_forward.c"
MUTATIONS = [
    ("query scaling", [("context->q[t * q_width + j] *= tau;", "context->q[t * q_width + j] *= tau * 0.0f + 1.0f;")], "tau_far"),
    ("relative scaling", [("context->bias[s] * tau;", "context->bias[s];"),
                          ("context->bias[s] *= tau;", "context->bias[s] *= 1.0f;")], "tau_far"),
    ("local window", [("t - s >= layer->window", "t - s > layer->window")], "local_window_512"),
    ("relative extent", [("t - s < layer->extent", "t - s + 1 < layer->extent")], "local_window_512"),
    ("gate/up order", [("context->pair[2 * j + 1]", "context->pair[2 * j]")], "prefix_1"),
    ("shared gamma", [("if (shared) context->activation[j] *= gamma;", "if (shared) context->activation[j] *= gamma + 1.0f;")], "prefix_1"),
    ("logit divisor", [("x[j] /= config->logits_mup_width_multiplier;", "x[j] *= config->logits_mup_width_multiplier;")], "prefix_1"),
]


def command(args, cwd):
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    return result, result.stdout + result.stderr


def main():
    parser = argparse.ArgumentParser(description='Require numerical comparisons to reject deliberate production graph faults.')
    parser.add_argument("--cc", default="cc")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    original = (ROOT / SOURCE).read_text()
    reference, incremental = original.split("static int token_attention(", 1)
    records = []
    with tempfile.TemporaryDirectory(prefix="inkling-model-mutations-") as temporary:
        work = Path(temporary)
        for directory in ("src", "include"):
            shutil.copytree(ROOT / directory, work / directory)
        (work / "tests").mkdir()
        shutil.copyfile(ROOT / "tests/model_probe.c", work / "tests/model_probe.c")
        shutil.copyfile(ROOT / "makefile", work / "makefile")
        build = ["make", "-B", "bin/model_probe", f"CC={args.cc}"]
        check = [sys.executable, "-B", str(ROOT / "tools/check_model.py"),
                 "--probe", str(work / "bin/model_probe"), "--fixtures", str(ROOT / "tests/fixtures/model")]
        built, output = command(build, work)
        if built.returncode:
            raise RuntimeError("pristine model build failed\n" + output)
        baseline, output = command(check, ROOT)
        if baseline.returncode:
            raise RuntimeError("pristine model comparisons failed\n" + output)
        for name, replacements, marker in MUTATIONS:
            changed = reference
            for old, new in replacements:
                if changed.count(old) != 1:
                    raise RuntimeError(f"stale or ambiguous mutation: {name}: {old}")
                changed = changed.replace(old, new)
            (work / SOURCE).write_text(changed + "static int token_attention(" + incremental)
            built, output = command(build, work)
            if built.returncode:
                raise RuntimeError(f"{name}: build failure is not a numerical rejection\n{output}")
            checked, output = command(check, ROOT)
            if (checked.returncode != 1 or marker not in output or "probe failed" in output or
                    not all(word in output for word in ("actual=", "expected=", "limit="))):
                raise RuntimeError(f"{name}: expected a numerical mismatch in {marker}\n{output}")
            record = dict(name=name, case=marker, rejection=output.strip().splitlines()[-1])
            records.append(record)
            print(f"PASS {name}: rejected by {marker}", flush=True)
    report = dict(schema=1, source_sha256=hashlib.sha256(original.encode()).hexdigest(),
                  probe_sha256=hashlib.sha256((ROOT / "tests/model_probe.c").read_bytes()).hexdigest(),
                  mutations=records)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
