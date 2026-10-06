# Pinned checkpoint metadata

These fixtures contain metadata only; no model tensor payloads are included.

- Model: `thinkingmachines/Inkling-Small-NVFP4`
- Revision: `b6a99534467840620d411e4cd4ad5819b2610d9c`
- Source: `https://huggingface.co/thinkingmachines/Inkling-Small-NVFP4/resolve/b6a99534467840620d411e4cd4ad5819b2610d9c/`
- SafeTensors files: 10

The files in `headers/` contain the original 8-byte little-endian SafeTensors
header length followed by the complete JSON header. Regenerate this directory with
`python3 tools/fetch_checkpoint_metadata.py` from the repository root.

Run `bin/inkling tests/fixtures/checkpoint --verify-model --metadata-only --with-mtp`
after `make` for the complete offline census. This mode trusts the pinned original
payload extents, not the fixture lengths. `make test` also creates sparse local
shards to exercise actual-file bounds; no network or tensor payload is required.
