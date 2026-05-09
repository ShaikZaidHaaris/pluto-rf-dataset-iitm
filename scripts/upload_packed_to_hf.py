"""
Upload a consolidated Hub bundle (output of consolidate_pluto_capture_hub.py).

Expects in --src:
  iq_windows.npy, windows_meta.json.gz, dataset_header.json

Usage:

    hf auth login
    python scripts/upload_packed_to_hf.py \\
        --src ./pluto_iitm_hub_bundle \\
        --repo-id BlackWhite123/pluto-rf-dataset-iitm-packed \\
        --create
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Packed bundle directory.")
    parser.add_argument("--repo-id", required=True, help="HF dataset repo id.")
    parser.add_argument("--create", action="store_true", help="create_repo(..., exist_ok=True)")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("ERROR: pip install huggingface_hub", file=sys.stderr)
        return 2

    src = Path(args.src).expanduser().resolve()
    if not src.is_dir():
        print(f"ERROR: not a directory: {src}", file=sys.stderr)
        return 2

    need = ("iq_windows.npy", "windows_meta.json.gz", "dataset_header.json")
    for name in need:
        if not (src / name).is_file():
            print(f"ERROR: bundle missing {name} under {src}", file=sys.stderr)
            return 2

    api = HfApi()
    if args.create:
        create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )

    print(f"[hf] upload_large_folder {src} -> {args.repo_id}")
    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(src),
        ignore_patterns=[".DS_Store", "*.bak", ".cache"],
        num_workers=args.workers,
        print_report=True,
        print_report_every=60,
    )
    print(f"[hf] Done: https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
