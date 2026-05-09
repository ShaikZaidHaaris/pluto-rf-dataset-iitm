"""
upload_to_hf.py - one-shot uploader for pluto-rf-dataset-iitm.

Pushes the full ``data/pluto_capture/`` capture tree (manifest + derived
per-window IQ npy files + packed tensors) to a Hugging Face dataset repo.

Usage:

    pip install huggingface_hub
    huggingface-cli login        # paste a write token from https://huggingface.co/settings/tokens

    python scripts/upload_to_hf.py \
        --src "/Users/zaidhaaris/Downloads/theGreatProject/Radar/video_radar/libiio_v026/scripts/data/pluto_capture" \
        --repo-id "BlackWhite123/pluto-rf-dataset-iitm" \
        --create

The first run creates the dataset repo (``--create``); subsequent runs without
``--create`` will sync incrementally.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        required=True,
        help="Path to local pluto_capture directory (the one with manifest_phase1.json + derived/).",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face dataset repo id, e.g. `myuser/pluto-rf-dataset-iitm`.",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create the dataset repo on the Hub if it does not exist.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="If creating, mark the dataset as private.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload pluto_capture dataset",
        help="Reserved for smaller uploads; upload_large_folder uses incremental commits.",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=[".DS_Store", "*.bak"],
        help="Glob(s) to skip. Pass multiple times to add more.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel workers for upload_large_folder (default: Hub chooses).",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("ERROR: huggingface_hub is not installed.", file=sys.stderr)
        print("       Install with:  pip install huggingface_hub", file=sys.stderr)
        return 2

    src = Path(args.src).expanduser().resolve()
    if not src.is_dir():
        print(f"ERROR: --src is not a directory: {src}", file=sys.stderr)
        return 2

    manifest = src / "manifest_phase1.json"
    if not manifest.exists():
        print(
            f"ERROR: expected manifest_phase1.json under {src} but did not find it.",
            file=sys.stderr,
        )
        return 2

    api = HfApi()

    if args.create:
        print(f"[hf] Creating dataset repo {args.repo_id} (private={args.private}) ...")
        create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )

    # Use upload_large_folder for multi‑GB trees (resumable, parallel uploads).
    # Plain upload_folder warns / may fail on ~GB datasets — see Hub docs.
    print(f"[hf] Upload-large-folder {src}  ->  {args.repo_id} (dataset)")
    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(src),
        ignore_patterns=list(args.ignore_pattern),
        num_workers=args.workers,
        print_report=True,
        print_report_every=60,
    )

    url = f"https://huggingface.co/datasets/{args.repo_id}"
    print(f"[hf] Done. Dataset is live at: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
