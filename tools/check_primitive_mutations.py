#!/usr/bin/env python3
"""Prove committed primitive tests reject isolated implementation mutations."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

# name, source, original, replacement, test, fixture, required failure marker
MUTATIONS = (
    ("packing_order", "inkling_nvfp4.c", "((i % 2) * 4)", "((1 - i % 2) * 4)",
     "test_nvfp4", "nvfp4.json", "float_bits(actual[i]) =="),
    ("scale_direction", "inkling_nvfp4.c", "local_scale * scale2", "local_scale / scale2",
     "test_nvfp4", "nvfp4.json", "float_bits(actual[i]) =="),
    ("block_scale_index", "inkling_math.c", "offsets.scale++;", "offsets.scale += 0;",
     "test_math", "primitives.json", "matvec["),
    ("expert_scale_index", "inkling_math.c", "scale2[expert]", "scale2[0]",
     "test_math", "primitives.json", "matvec["),
    ("bf16_stride", "inkling_math.c", "weights[row * columns + col]", "weights[col * rows + row]",
     "test_math", "primitives.json", "matvec["),
    ("topk_ties", "inkling_math.c", "x[indices[position]] >= x[i]", "x[indices[position]] > x[i]",
     "test_math", "primitives.json", "indices[i] == integer(inkling_json_array_at(values, i))"),
    ("convolution_state", "inkling_math.c", "history[2] = current;", "history[2] = sum;",
     "test_math", "primitives.json", "memcmp(state, final, history * sizeof(float)) == 0"),
    ("convolution_residual", "inkling_math.c", "sum += current;", "sum += 0;",
     "test_math", "primitives.json", "conv["),
    ("convolution_kernel", "inkling_math.c", "history[tap] * kernel[tap]", "history[tap] * kernel[3 - tap]",
     "test_math", "primitives.json", "conv["),
    ("masked_probability", "inkling_math.c", "(float)exp((double)x[i] - maximum)",
     "(x[i] == -INFINITY ? 1e-12f : (float)exp((double)x[i] - maximum))",
     "test_math", "primitives.json", "out[i] == 0.0f"),
)


def run(command, cwd):
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=120)


def build(directory, compiler, target, binary_dir):
    result = run(["make", f"CC={compiler}", f"BIN_DIR={binary_dir}",
                  f"{binary_dir}/{target}"], directory)
    if result.returncode != 0:
        raise RuntimeError(f"build failed, not a mutation kill:\n{result.stdout}{result.stderr}")
    return directory / binary_dir / target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    compiler = run(shlex.split(args.cc) + ["--version"], root)
    if compiler.returncode != 0:
        raise RuntimeError(compiler.stderr)
    report = {"compiler": compiler.stdout.splitlines()[0], "mutations": []}
    print(report["compiler"], flush=True)
    with tempfile.TemporaryDirectory(prefix="inkling-mutations-") as temporary:
        directory = Path(temporary)
        for name in ("src", "include", "tests"):
            shutil.copytree(root / name, directory / name)
        shutil.copy2(root / "makefile", directory / "makefile")
        for target, fixture in sorted({(m[4], m[5]) for m in MUTATIONS}):
            binary = build(directory, args.cc, target, "bin/baseline")
            result = run([str(binary), f"tests/fixtures/{fixture}"], directory)
            if result.returncode != 0:
                raise RuntimeError(f"pristine {target} failed:\n{result.stdout}{result.stderr}")
        for name, source, old, new, target, fixture, marker in MUTATIONS:
            path = directory / "src/core" / source
            original = path.read_text()
            if original.count(old) != 1:
                raise RuntimeError(f"stale mutation {name}: expected one source match")
            try:
                path.write_text(original.replace(old, new))
                binary = build(directory, args.cc, target, f"bin/{name}")
                result = run([str(binary), f"tests/fixtures/{fixture}"], directory)
                if result.returncode != 1 or marker not in result.stderr:
                    raise RuntimeError(f"{name} survived or failed for another reason "
                                       f"(exit {result.returncode}):\n{result.stdout}{result.stderr}")
                report["mutations"].append({"name": name, "result": "rejected", "evidence": result.stderr.strip()})
                print(f"{name}: rejected ({result.stderr.strip()})", flush=True)
            finally:
                path.write_text(original)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        raise SystemExit(f"mutation acceptance failed: {error}") from None
