import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run(command, expected=0):
    result = subprocess.run([str(x) for x in command], capture_output=True, text=True)
    assert result.returncode == expected, (command, result.returncode, result.stderr)
    return result


def main():
    cli = Path(sys.argv[1]).resolve()
    fixture = ROOT / "tests/fixtures/model"
    with tempfile.TemporaryDirectory(prefix="inkling-memory-cli-") as temporary:
        directory = Path(temporary)
        packed = directory / "trunk.safetensors"
        run([sys.executable, ROOT / "tools/pack_trunk.py", fixture, packed])
        prompt = "2,3,39"
        base = [cli, fixture, "--ids", prompt, "--gen", "8"]
        reference = json.loads(run(base + ["--decode-mode", "full"]).stdout)
        full_trace = directory / "full.jsonl"
        run(base + ["--decode-mode", "full", "--trace", full_trace])
        full_stats = json.loads(full_trace.read_text().splitlines()[-1])
        for mode in ("full", "incremental"):
            for cache in (0, 1, 1024, 16384):
                trace = directory / "trace.jsonl"
                args = base + ["--decode-mode", mode, "--expert-cache-bytes", cache,
                               "--weight-buffer-bytes", 96, "--trace", trace,
                               "--packed-trunk", packed]
                assert json.loads(run(args).stdout) == reference
                events = [json.loads(line) for line in trace.read_text().splitlines()]
                stats = events[-1]
                assert stats["event"] == "summary" and stats["cache_capacity"] == cache
                assert stats["cache_bytes"] <= stats["cache_peak_bytes"] <= cache
                for event in events[:-1]:
                    assert event["event"] in {"hit", "miss", "bypass", "evict"}
                    assert event["resident_bytes"] <= cache
                    assert event["projection"] in {"w13", "down"}
                if mode == "incremental":
                    assert stats["tokens_processed"] < full_stats["tokens_processed"]
        native, low = directory / "native.json", directory / "low.json"
        forward = [cli, fixture, "--ids", prompt, "--dump-logits"]
        run(forward + [native])
        run(forward + [low, "--weight-buffer-bytes", 96, "--packed-trunk", packed])
        assert native.read_bytes() == low.read_bytes()
        for args, code in ((["--decode-mode", "unknown"], 2),
                           (["--weight-buffer-bytes", "0"], 2),
                           (["--weight-buffer-bytes", "95"], 1),
                           (["--expert-cache-bytes", "-1"], 2),
                           (["--expert-cache-bytes", "18446744073709551615"], 2),
                           (["--decode-mode", "full", "--decode-mode", "incremental"], 2)):
            run(base + args, code)
        target = directory / "output.json"
        target.write_text("keep")
        for other in ("--out", "--dump-tokens"):
            run(base + ["--trace", target, other, target], 2)
            assert target.read_text() == "keep"
        run(base + ["--trace", target, "--out", directory / "missing/out.json"], 1)
        assert target.read_text() == "keep"
        assert not list(directory.glob("*.tmp.*"))
    print("PASS memory CLI: native/packed, full/incremental, cache/tile budgets and trace guards")


if __name__ == "__main__":
    main()
