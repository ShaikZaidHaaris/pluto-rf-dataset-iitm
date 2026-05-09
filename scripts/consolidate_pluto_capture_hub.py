#!/usr/bin/env python3
"""
Pack Phase-1 Pluto capture (manifest + derived/*.npy) into a few Hub-friendly files.

Typical layout before:
  manifest_phase1.json + tens of thousands of derived/*.npy

After (--out-dir):
  iq_windows.npy       float32 (N, 2, 2048) — stack order == manifest ``windows`` order
  windows_meta.json.gz gzipped JSON list — one object per row (no per-file paths)

Row i always corresponds to manifest ``windows[i]`` after validating shapes.

Large derived artifacts like packed_hybrid_* are NOT copied — regenerate locally if needed.

Usage:
  python3 scripts/consolidate_pluto_capture_hub.py \\
    --dataset-dir path/to/pluto_capture \\
    --out-dir ./pluto_iitm_hub_bundle

Upload (recommended):
  hf upload-large-folder BlackWhite123/pluto-rf-dataset-iitm-packed \\
    ./pluto_iitm_hub_bundle --repo-type dataset --num-workers 8
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _strip_window_for_hub(w: Dict[str, Any]) -> Dict[str, Any]:
    """Drop bulky / obsolete keys; keep training provenance."""
    out = {k: v for k, v in w.items() if k != "derived_npy_rel"}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset-dir",
        required=True,
        help="pluto_capture root containing manifest_phase1.json and derived/",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Output directory (created).",
    )
    ap.add_argument(
        "--manifest-name",
        default="manifest_phase1.json",
        help="Manifest filename under dataset-dir.",
    )
    ap.add_argument(
        "--max-windows",
        type=int,
        default=0,
        help="If >0, only stack the first N windows (smoke test).",
    )
    args = ap.parse_args()

    ds = Path(args.dataset_dir).expanduser().resolve()
    out = Path(args.out_dir).expanduser().resolve()
    man_path = ds / args.manifest_name

    if not man_path.is_file():
        print(f"ERROR: missing manifest {man_path}", file=sys.stderr)
        return 2

    out.mkdir(parents=True, exist_ok=True)

    with open(man_path, encoding="utf-8") as f:
        manifest = json.load(f)

    windows: List[Dict[str, Any]] = manifest.get("windows") or []
    if args.max_windows > 0:
        windows = windows[: args.max_windows]
    n = len(windows)
    if n == 0:
        print("ERROR: manifest has no windows", file=sys.stderr)
        return 2

    ws = int(manifest.get("window_samples") or 0)
    if ws <= 0:
        print("ERROR: manifest.window_samples invalid", file=sys.stderr)
        return 2

    stack = np.empty((n, 2, ws), dtype=np.float32)
    meta_rows: List[Dict[str, Any]] = []

    print(f"[pack] {n} windows  shape (N, 2, {ws})  dtype=float32")
    print(f"[pack] reading from {ds}")

    for i, w in enumerate(windows):
        rel = w.get("derived_npy_rel")
        if not rel:
            print(f"ERROR: window[{i}] missing derived_npy_rel", file=sys.stderr)
            return 2
        p = ds / rel
        if not p.is_file():
            print(f"ERROR: missing file for window[{i}]: {p}", file=sys.stderr)
            return 2

        arr = np.load(p)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        if arr.shape != (2, ws):
            print(
                f"ERROR: window[{i}] bad shape {arr.shape}; expected (2, {ws}) path={p}",
                file=sys.stderr,
            )
            return 2

        stack[i] = arr
        meta_rows.append(_strip_window_for_hub(w))

        if (i + 1) % 2000 == 0 or i + 1 == n:
            print(f"[pack] stacked {i + 1}/{n}")

    iq_path = out / "iq_windows.npy"
    meta_path = out / "windows_meta.json.gz"
    header_path = out / "dataset_header.json"

    np.save(iq_path, stack)

    with gzip.open(meta_path, "wt", encoding="utf-8") as gz:
        json.dump(meta_rows, gz, separators=(",", ":"))

    header = {
        "schema": "pluto_phase1_hub_bundle_v1",
        "source_manifest": str(man_path.name),
        "n_windows": n,
        "iq_tensor_file": iq_path.name,
        "iq_shape": list(stack.shape),
        "iq_dtype": str(stack.dtype),
        "meta_file": meta_path.name,
        "manifest_global_fields": {
            k: manifest[k]
            for k in (
                "schema_version",
                "phase",
                "source",
                "bands_policy",
                "window_samples",
                "hop_samples",
                "target_sample_rate_hz",
                "label_names",
                "notes",
            )
            if k in manifest
        },
        "note": "Row index i matches manifest windows[i] and iq_windows[i]. derived_npy_rel omitted from meta.",
    }
    header_path.write_text(json.dumps(header, indent=2), encoding="utf-8")

    mb = iq_path.stat().st_size / (1024 * 1024)
    mb_meta = meta_path.stat().st_size / (1024 * 1024)
    print(f"[done] {iq_path}  ({mb:.1f} MiB)")
    print(f"[done] {meta_path}  ({mb_meta:.2f} MiB)")
    print(f"[done] {header_path}")
    print(f"[hint] Upload only these three (+ README); skip derived/ and packed_hybrid_*/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
