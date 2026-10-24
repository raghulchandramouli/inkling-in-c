import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import pack_trunk


def run(command, code=0):
    result = subprocess.run([str(item) for item in command], capture_output=True, text=True)
    assert result.returncode == code, (command, result.returncode, result.stderr)
    return result


def unpack(raw):
    size, = struct.unpack("<Q", raw[:8])
    return json.loads(raw[8:8 + size]), raw[8 + size:]


def encode(header, payload):
    header = copy.deepcopy(header)
    header["__metadata__"]["header_sha256"] = "0" * 64
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    header["__metadata__"]["header_sha256"] = hashlib.sha256(raw).hexdigest()
    final = json.dumps(header, separators=(",", ":")).encode()
    final += b" " * (-len(final) % 8)
    return struct.pack("<Q", len(final)) + final + payload


def main():
    probe = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "bin/packed_probe")
    with tempfile.TemporaryDirectory(prefix="inkling-packed-") as temporary:
        directory = Path(temporary)
        source = directory / "source"
        shutil.copytree(ROOT / "tests/fixtures/model", source)
        target = directory / "trunk.safetensors"
        second = directory / "second.safetensors"
        pack_trunk.pack(source, target)
        pack_trunk.pack(source, second)
        raw = target.read_bytes()
        assert raw == second.read_bytes()
        header, payload = unpack(raw)
        assert hashlib.sha256(payload).hexdigest() == header["__metadata__"]["payload_sha256"]
        assert not any(".mlp.experts." in name for name in header)
        run([probe, "--compare", source, target])
        original = copy.deepcopy(header)
        tensor_names = [name for name in header if name != "__metadata__"]
        name = tensor_names[0]
        changes = [
            lambda h: h["__metadata__"].update(schema="inkling-trunk-v2"),
            lambda h: h["__metadata__"].update(checkpoint_revision="0" * 40),
            lambda h: h["__metadata__"].update(source_sha256="0" * 64),
            lambda h: h["__metadata__"].update(payload_sha256="0" * 64),
            lambda h: h[name].update(dtype="U8"),
            lambda h: h[name].update(shape=list(reversed(h[name]["shape"]))),
            lambda h: h[name].update(data_offsets=[1, h[name]["data_offsets"][1] + 1]),
            lambda h: h[name].update(data_offsets=[0, 2 ** 64]),
            lambda h: h.update({"model.llm.unknown.weight": h.pop(name)}),
            lambda h: h.pop(name),
            lambda h: h["__metadata__"].update(source_ranges="{}"),
        ]
        for change in changes:
            header = copy.deepcopy(original)
            change(header)
            target.write_bytes(encode(header, payload))
            run([probe, "--open", source, target], 1)
        missing = copy.deepcopy(original)
        last = tensor_names[-1]
        start = missing.pop(last)["data_offsets"][0]
        reduced_payload = payload[:start]
        missing_mapping = json.loads(missing["__metadata__"]["source_ranges"])
        missing_mapping.pop(last)
        missing["__metadata__"].update(source_ranges=json.dumps(missing_mapping),
                                        payload_sha256=hashlib.sha256(reduced_payload).hexdigest())
        target.write_bytes(encode(missing, reduced_payload))
        run([probe, "--open", source, target], 1)
        mapping = json.loads(original["__metadata__"]["source_ranges"])
        mapping[name][1] += 1
        header = copy.deepcopy(original)
        header["__metadata__"]["source_ranges"] = json.dumps(mapping)
        target.write_bytes(encode(header, payload))
        run([probe, "--open", source, target], 1)
        for invalid in (raw[:-1], raw + b"trailing", raw[:8], b"\xff" * 8 + raw[8:], raw.replace(original["__metadata__"]["header_sha256"].encode(), b"0" * 64)):
            target.write_bytes(invalid)
            run([probe, "--open", source, target], 1)
        broken = bytearray(payload)
        broken[len(broken) // 2] ^= 1
        target.write_bytes(encode(original, broken))
        run([probe, "--open", source, target], 1)
        text = json.dumps(original, separators=(",", ":"))
        duplicate = text[:-1] + "," + json.dumps(name) + ":" + json.dumps(original[name]) + "}"
        target.write_bytes(struct.pack("<Q", len(duplicate)) + duplicate.encode() + payload)
        run([probe, "--open", source, target], 1)
        target.write_bytes(raw)
        config = source / "config.json"
        saved = config.read_bytes()
        config.write_bytes(saved + b"\n")
        run([probe, "--open", source, target], 1)
        config.write_bytes(saved)
        for input_path in [source / item for item in pack_trunk.METADATA_FILES] + list(source.glob("*.safetensors")):
            saved = input_path.read_bytes()
            try:
                pack_trunk.pack(source, input_path)
                raise AssertionError("source alias accepted")
            except ValueError:
                pass
            assert input_path.read_bytes() == saved
        mtp = source / "mtp.safetensors"
        mtp.write_bytes(b"optional checkpoint data")
        for alias in (mtp,):
            try:
                pack_trunk.pack(source, alias)
                raise AssertionError("optional source alias accepted")
            except ValueError:
                pass
        assert mtp.read_bytes() == b"optional checkpoint data"
        hardlink = directory / "hardlink"
        os.link(config, hardlink)
        symlink = directory / "symlink"
        symlink.symlink_to(config)
        mtp_hardlink = directory / "mtp-hardlink"
        os.link(mtp, mtp_hardlink)
        mtp_symlink = directory / "mtp-symlink"
        mtp_symlink.symlink_to(mtp)
        for alias in (hardlink, symlink, mtp_hardlink, mtp_symlink):
            try:
                pack_trunk.pack(source, alias)
                raise AssertionError("linked source alias accepted")
            except ValueError:
                pass
        target.write_bytes(b"existing destination")
        with mock.patch.object(pack_trunk, "copy_bytes", side_effect=OSError("interrupted copy")):
            try:
                pack_trunk.pack(source, target)
                raise AssertionError("interrupted pack succeeded")
            except OSError:
                pass
        assert target.read_bytes() == b"existing destination"
        assert not list(directory.glob(".trunk.safetensors.*.tmp"))
        pack_trunk.pack(source, target)
        assert target.read_bytes() == raw
        run([probe, "--compare", source, target])
        run([probe, "--replace", source, second])
    print("packed trunk byte, logits, generation, corruption, alias and retry checks passed")


if __name__ == "__main__":
    main()
