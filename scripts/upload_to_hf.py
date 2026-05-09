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
        help="Commit message for the upload.",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=[".DS_Store", "*.bak"],
        help="Glob(s) to skip. Pass multiple times to add more.",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, create_repo, upload_folder
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

    print(f"[hf] Uploading folder {src}  ->  {args.repo_id}")
    upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(src),
        path_in_repo=".",
        commit_message=args.commit_message,
        ignore_patterns=list(args.ignore_pattern),
    )

    url = f"https://huggingface.co/datasets/{args.repo_id}"
    print(f"[hf] Done. Dataset is live at: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
