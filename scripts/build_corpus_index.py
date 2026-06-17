"""CLI: build the EEGWindowDataset sidecar index from a corpus manifest.

Usage:
    uv run python scripts/build_corpus_index.py \
        data/processed/open_corpus/manifest.json

Writes ``index.json`` next to the manifest (i.e. alongside the shards),
because the dataset resolves shard paths relative to the index's parent.
"""

import argparse
import json
from pathlib import Path

from neurostream.data.corpus_index import build_corpus_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Path to manifest.json")
    parser.add_argument(
        "--window-samples", type=int, default=1000, help="Window length (informational)"
    )
    parser.add_argument(
        "--sampling-rate-hz", type=int, default=None, help="Sampling rate to stamp"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default: index.json next to the manifest)",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    index = build_corpus_index(
        manifest,
        window_samples=args.window_samples,
        sampling_rate_hz=args.sampling_rate_hz,
    )
    out = args.out or args.manifest.parent / "index.json"
    out.write_text(json.dumps(index, indent=2))

    n_segments = sum(len(v) for v in index["segments_by_source"].values())
    print(f"Wrote {out}")
    print(f"  shards:   {len(index['shards'])}")
    print(f"  sources:  {len(index['segments_by_source'])}")
    print(f"  segments: {n_segments}")


if __name__ == "__main__":
    main()
