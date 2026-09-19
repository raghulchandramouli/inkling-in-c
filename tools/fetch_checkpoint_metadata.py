#!/usr/bin/env python3
"""Fetch checkpoint metadata without downloading checkpoint tensor payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


MODEL = "thinkingmachines/Inkling-Small-NVFP4"
REVISION = "b6a99534467840620d411e4cd4ad5819b2610d9c"

BASE_URL = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/"

OUTPUT_DIR = Path("tests/fixtures/checkpoint")
HEADER_DIR = OUTPUT_DIR / "headers"

SMALL_FILES = (
    "config.json",
    "hf_quant_config.json",
    "model.safetensors.index.json",
)

INDEX_FILE = "model.safetensors.index.json"
MANIFEST_FILE = "metadata-manifest.json"
README_FILE = "README.md"
MAX_SMALL_FILE_SIZE = 32 * 1024 * 1024
MAX_HEADER_SIZE = 128 * 1024 * 1024
CONTENT_RANGE_RE = re.compile(r"bytes (\d+)-(\d+)/(\d+|\*)\Z", re.IGNORECASE)


def request(
    url: str,
    byte_range: tuple[int, int] | None = None,
    *,
    max_bytes: int | None = None,
) -> bytes:
    """Return a URL's bytes, enforcing an exact HTTP range when requested."""
    headers = {
        "Accept-Encoding": "identity",
        "User-Agent": "inkling-in-c-metadata-fetcher/1.0",
    }

    if byte_range is not None:
        start, end = byte_range
        if start < 0 or end < start:
            raise ValueError(f"invalid byte range: {start}-{end}")
        headers["Range"] = f"bytes={start}-{end}"

    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            status = response.getcode()

            if byte_range is not None:
                if status != 206:
                    raise RuntimeError(
                        f"server ignored Range request for {url}; "
                        "refusing to download the full model shard"
                    )

                content_range = response.headers.get("Content-Range", "")
                match = CONTENT_RANGE_RE.fullmatch(content_range.strip())
                if match is None:
                    raise RuntimeError(
                        f"missing or invalid Content-Range for {url}: "
                        f"{content_range!r}"
                    )

                actual_start, actual_end = map(int, match.group(1, 2))
                if (actual_start, actual_end) != byte_range:
                    raise RuntimeError(
                        f"server returned bytes {actual_start}-{actual_end} for "
                        f"requested range {start}-{end}: {url}"
                    )

            limit = max_bytes
            if byte_range is not None:
                range_size = end - start + 1
                limit = range_size if limit is None else min(limit, range_size)

            data = response.read() if limit is None else response.read(limit + 1)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code} while fetching {url}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"could not fetch {url}: {error.reason}") from error

    if limit is not None and len(data) > limit:
        raise RuntimeError(f"response from {url} exceeds {limit} bytes")

    if byte_range is not None:
        expected_size = end - start + 1
        if len(data) != expected_size:
            raise RuntimeError(
                f"short Range response from {url}: expected {expected_size} "
                f"bytes, received {len(data)}"
            )

    return data


def parse_json_object(data: bytes, source: str) -> dict[str, Any]:
    """Decode a UTF-8 JSON object and provide a source-specific error."""
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid JSON in {source}: {error}") from error

    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object in {source}")
    return value


def url_for(base_url: str, filename: str) -> str:
    return base_url.rstrip("/") + "/" + urllib.parse.quote(filename, safe="/")


def shard_names(index: dict[str, Any]) -> list[str]:
    """Return the unique, safe SafeTensors shard names in an HF index."""
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("checkpoint index has no non-empty weight_map object")

    names: set[str] = set()
    for tensor_name, shard_name in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise RuntimeError("checkpoint index contains an invalid tensor name")
        if not isinstance(shard_name, str) or not shard_name:
            raise RuntimeError(f"invalid shard name for tensor {tensor_name!r}")
        if (
            Path(shard_name).name != shard_name
            or "\\" in shard_name
            or not shard_name.endswith(".safetensors")
        ):
            raise RuntimeError(f"unsafe or unsupported shard name: {shard_name!r}")
        names.add(shard_name)

    return sorted(names)


def validate_safetensors_header(header: bytes, shard_name: str) -> set[str]:
    document = parse_json_object(header, shard_name)
    tensor_count = 0

    for name, metadata in document.items():
        if name == "__metadata__":
            continue
        if not isinstance(metadata, dict):
            raise RuntimeError(f"invalid tensor metadata for {name!r} in {shard_name}")
        offsets = metadata.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(offset) is not int or offset < 0 for offset in offsets)
            or offsets[0] > offsets[1]
        ):
            raise RuntimeError(f"invalid data_offsets for {name!r} in {shard_name}")
        tensor_count += 1

    if tensor_count == 0:
        raise RuntimeError(f"SafeTensors header contains no tensors: {shard_name}")
    return {name for name in document if name != "__metadata__"}


def fetch_safetensors_header(base_url: str, shard_name: str) -> tuple[bytes, int]:
    """Fetch and validate one shard's 8-byte prefix and JSON header."""
    url = url_for(base_url, shard_name)
    prefix = request(url, (0, 7))
    (header_size,) = struct.unpack("<Q", prefix)

    if header_size < 2 or header_size > MAX_HEADER_SIZE:
        raise RuntimeError(
            f"implausible SafeTensors header size for {shard_name}: {header_size}"
        )

    header = request(url, (8, 7 + header_size), max_bytes=MAX_HEADER_SIZE)
    validate_safetensors_header(header, shard_name)
    return prefix + header, header_size


def validate_index_headers(
    index: dict[str, Any], headers: dict[str, bytes]
) -> None:
    """Prove that each index entry exists in exactly its declared shard."""
    weight_map = index["weight_map"]
    actual_locations: dict[str, str] = {}

    for shard_name, fixture in headers.items():
        for tensor_name in validate_safetensors_header(fixture[8:], shard_name):
            previous = actual_locations.setdefault(tensor_name, shard_name)
            if previous != shard_name:
                raise RuntimeError(
                    f"tensor {tensor_name!r} appears in both {previous} "
                    f"and {shard_name}"
                )

    indexed_names = set(weight_map)
    actual_names = set(actual_locations)
    missing = indexed_names - actual_names
    extra = actual_names - indexed_names

    if missing:
        example = min(missing)
        raise RuntimeError(
            f"{len(missing)} indexed tensor(s) are absent from the headers; "
            f"first: {example!r}"
        )
    if extra:
        example = min(extra)
        raise RuntimeError(
            f"{len(extra)} header tensor(s) are absent from the index; "
            f"first: {example!r}"
        )

    for tensor_name, indexed_shard in weight_map.items():
        actual_shard = actual_locations[tensor_name]
        if actual_shard != indexed_shard:
            raise RuntimeError(
                f"index maps {tensor_name!r} to {indexed_shard}, "
                f"but its header is in {actual_shard}"
            )


def write_atomic(path: Path, data: bytes) -> None:
    """Replace a file atomically, leaving no partial fixture on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", dir=path.parent, delete=False
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(data)
            temporary.flush()
        Path(temporary_name).chmod(0o644)
        Path(temporary_name).replace(path)
    finally:
        if temporary_name is not None:
            temporary_path = Path(temporary_name)
            if temporary_path.exists():
                temporary_path.unlink()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def readme(base_url: str, shard_count: int) -> bytes:
    text = f"""# Pinned checkpoint metadata

These fixtures contain metadata only; no model tensor payloads are included.

- Model: `{MODEL}`
- Revision: `{REVISION}`
- Source: `{base_url}`
- SafeTensors files: {shard_count}

The files in `headers/` contain the original 8-byte little-endian SafeTensors
header length followed by the complete JSON header. Regenerate this directory with
`python3 tools/fetch_checkpoint_metadata.py` from the repository root.
"""
    return text.encode("utf-8")


def fetch_checkpoint_metadata(base_url: str, output_dir: Path) -> None:
    """Fetch, validate, and write all metadata for the pinned checkpoint."""
    downloaded: dict[str, bytes] = {}

    for filename in SMALL_FILES:
        print(f"fetching {filename}")
        data = request(url_for(base_url, filename), max_bytes=MAX_SMALL_FILE_SIZE)
        parse_json_object(data, filename)
        downloaded[filename] = data

    index = parse_json_object(downloaded[INDEX_FILE], INDEX_FILE)
    shards = shard_names(index)
    headers: dict[str, bytes] = {}
    header_sizes: dict[str, int] = {}

    for position, shard_name in enumerate(shards, start=1):
        print(f"fetching header {position}/{len(shards)}: {shard_name}")
        fixture, header_size = fetch_safetensors_header(base_url, shard_name)
        headers[shard_name] = fixture
        header_sizes[shard_name] = header_size

    validate_index_headers(index, headers)

    for filename, data in downloaded.items():
        write_atomic(output_dir / filename, data)
    for shard_name, data in headers.items():
        write_atomic(output_dir / "headers" / shard_name, data)

    header_dir = output_dir / "headers"
    for existing in header_dir.glob("*.safetensors"):
        if existing.name not in headers:
            existing.unlink()

    manifest = {
        "model": MODEL,
        "revision": REVISION,
        "base_url": base_url,
        "files": {
            name: {"size": len(data), "sha256": sha256(data)}
            for name, data in sorted(downloaded.items())
        },
        "shards": [
            {
                "name": name,
                "header_size": header_sizes[name],
                "fixture_size": len(headers[name]),
                "sha256": sha256(headers[name]),
            }
            for name in shards
        ],
    }
    manifest_data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    write_atomic(output_dir / MANIFEST_FILE, manifest_data)
    write_atomic(output_dir / README_FILE, readme(base_url, len(shards)))

    print(f"wrote metadata for {len(shards)} SafeTensors files to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=BASE_URL,
        help="checkpoint directory URL (defaults to the pinned Hugging Face revision)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"fixture output directory (default: {OUTPUT_DIR})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        fetch_checkpoint_metadata(args.base_url, args.output_dir)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
