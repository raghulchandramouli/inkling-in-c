import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from check_generation_fixture import audit, compare_generation, compare_sampler


def run(command, expected=0):
    result = subprocess.run([str(x) for x in command], capture_output=True, text=True)
    assert result.returncode == expected, (command, result.returncode, result.stdout, result.stderr)
    return result


def normalized(actual):
    actual = dict(actual)
    actual["seed"] = int(actual["seed"])
    return actual


def main():
    production, cli, probe = [Path(p).resolve() for p in sys.argv[1:]]
    fixture = ROOT / "tests/fixtures/model"
    cases_path = ROOT / "tests/fixtures/generation.json"
    cases = audit(cases_path)
    with tempfile.TemporaryDirectory(prefix="inkling-generation-") as temporary:
        directory = Path(temporary)
        actual_path = directory / "actual.json"
        maximum_error = 0
        for i, case in enumerate(cases["generation"]):
            run([probe, fixture, cases_path, i, actual_path])
            actual = json.loads(actual_path.read_text())
            maximum_error = max(maximum_error, compare_generation(actual, case))
            assert actual["generated_count"] == case["generated_count"]
            assert normalized(actual["effective"]) == case["effective"]
        for i, case in enumerate(cases["sampler"]):
            run([probe, "--sampler", cases_path, i, actual_path])
            actual = json.loads(actual_path.read_text())
            compare_sampler(actual, case["expected"])
            assert normalized(actual["effective"]) == case["effective"]
            if "masses" in actual:
                assert len(actual["masses"]) == len(case["expected"]["masses"])
                for a, b in zip(actual["masses"], case["expected"]["masses"]):
                    assert math.isclose(a, b, rel_tol=2e-14, abs_tol=0), (case["name"], a, b)
        token_path, report_path = directory / "tokens.txt", directory / "report.json"
        for case in cases["generation"]:
            if case["context_capacity"] < len(case["prompt_ids"]) + case["options"]["max_new_tokens"]:
                continue
            args = [cli, fixture, "--ids", ",".join(map(str, case["prompt_ids"])),
                    "--gen", case["options"]["max_new_tokens"]]
            for flag in ("temperature", "top_k", "top_p", "seed"):
                args += ["--" + flag.replace("_", "-"), str(case["options"][flag])]
            stdout = json.loads(run(args).stdout)
            assert stdout["prompt_ids"] == case["prompt_ids"]
            for key in ("generated_ids", "generated_count", "stop_reason"):
                assert stdout[key] == case[key], (case["name"], key)
            assert normalized(stdout["effective"]) == case["effective"]
            run(args + ["--dump-tokens", token_path, "--out", report_path])
            assert json.loads(report_path.read_text()) == stdout
            assert token_path.read_text() == ",".join(map(str, case["generated_ids"])) + "\n"
        base = [cli, fixture, "--ids", "1", "--gen", "6"]
        report_path.write_text("old-report")
        token_path.write_text("old-tokens")
        run([production, fixture, "--ids", "1", "--gen", "6", "--out", report_path], 1)
        assert report_path.read_text() == "old-report"
        run(base + ["--dump-tokens", token_path, "--out", directory / "missing/report.json"], 1)
        assert token_path.read_text() == "old-tokens"
        assert report_path.read_text() == "old-report"
        assert not list(directory.glob("*.tmp.*"))
        bad_args = [["--gen", "1", "--gen", "2"], ["--temperature", "1"],
                    ["--out", report_path], ["--dump-tokens", token_path], ["--gen"],
                    ["--gen", "1", "--dump-logits", report_path], ["--unknown", "x"],
                    ["--gen", str(2**64 - 1)], ["--gen", "1", "--seed", str(2**64)],
                    ["--gen", "1", "--top-k", str(2**64)]]
        for flag, values in {
            "--gen": ["-1", "+1", "1.0", "0x1", " 1", "1 "],
            "--seed": ["-1", "+1", "1.0", "0x1", " 1"],
            "--temperature": ["nan", "inf", "-inf", "-1", "0x1p0", "1e-9999", "1e9999", " 1"],
            "--top-p": ["0", "-0", "1.1", "nan", "inf", "1e-9999"],
        }.items():
            for value in values:
                bad_args.append((["--gen", "1"] if flag != "--gen" else []) + [flag, value])
        for args in bad_args:
            run([production, directory / "missing-model", "--ids", "1"] + args, 2)
        for args in (["--out", token_path, "--dump-tokens", token_path],
                     ["--out", directory / "./tokens.txt", "--dump-tokens", token_path]):
            run([production, directory / "missing-model", "--ids", "1", "--gen", "1"] + list(args), 2)
        alias = directory / "alias"
        alias.symlink_to(directory, target_is_directory=True)
        run([production, directory / "missing-model", "--ids", "1", "--gen", "1",
             "--out", alias / "new.txt", "--dump-tokens", directory / "new.txt"], 2)
        for left_name, right_name in (
            ("Generated.TXT", "generated.txt"),
            ("Ä-result.txt", "ä-result.txt"),
            ("café.txt", "cafe\u0301.txt"),
            ("different-tokens.txt", "different-report.txt"),
        ):
            left, right = directory / left_name, directory / right_name
            left.write_text("old-output")
            aliases = right.exists() and left.samefile(right)
            left.unlink()
            args = ["--dump-tokens", left, "--out", right]
            failure = run([production, directory / "missing-model", "--ids", "1",
                           "--gen", "6"] + args, 2 if aliases else 1)
            assert not left.exists() and not right.exists()
            assert not list(directory.glob(".inkling-destination-*"))
            result = run(base + args, 2 if aliases else 0)
            if aliases:
                assert "invalid raw token options" in failure.stderr
                assert "invalid raw token options" in result.stderr
                assert not left.exists() and not right.exists()
                left.write_text("old-output")
                run(base + args, 2)
                assert left.read_text() == right.read_text() == "old-output"
            else:
                assert left.read_text() == "14,2,6\n"
                assert json.loads(right.read_text())["generated_ids"] == [14, 2, 6]
                right.unlink()
            left.unlink(missing_ok=True)
            assert not list(directory.glob(".inkling-destination-*"))
            assert not list(directory.glob("*.tmp.*"))
        oversized_name = directory / ("x" * (os.pathconf(directory, "PC_NAME_MAX") + 1))
        for left, right in ((oversized_name, report_path), (token_path, oversized_name)):
            run([production, directory / "missing-model", "--ids", "1", "--gen", "6",
                 "--dump-tokens", left, "--out", right], 2)
            assert token_path.read_text() == "old-tokens"
            assert report_path.read_text() == "old-report"
            assert not list(directory.glob(".inkling-destination-*"))
        hardlink = directory / "hardlink.txt"
        os.link(token_path, hardlink)
        run([production, directory / "missing-model", "--ids", "1", "--gen", "1",
             "--out", hardlink, "--dump-tokens", token_path], 2)
        for flag in ("--temperature", "--top-p"):
            report = json.loads(run(base + [flag, "5e-324"]).stdout)
            assert report["effective"][flag[2:].replace("-", "_")] > 0
        report = json.loads(run(base + ["--temperature", "1", "--top-k", str(2**64-1),
                                      "--seed", str(2**64-1)]).stdout)
        assert report["effective"]["seed"] == str(2**64-1)
        assert report["effective"]["top_k"] == 37
        zero = json.loads(run([cli, fixture, "--ids", "1", "--gen", "0"]).stdout)
        assert zero["generated_ids"] == [] and zero["stop_reason"] == "length"
        child = subprocess.Popen([str(x) for x in base], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        child.stdout.close()
        error = child.stderr.read().decode()
        assert child.wait() == 1 and "output failed" in error
        child.stderr.close()
        bad_dir = directory / "destination-directory"
        bad_dir.mkdir()
        failure = run(base + ["--dump-tokens", token_path, "--out", bad_dir], 1)
        assert token_path.read_text() == "14,2,6\n"
        assert f"already published: {token_path}" in failure.stderr
        assert bad_dir.is_dir() and not list(directory.glob("*.tmp.*"))
    print(f"PASS generation/CLI: {len(cases['generation'])} native cases, {len(cases['sampler'])} sampler cases; max logit error {maximum_error:.3g}")


if __name__ == "__main__":
    main()
