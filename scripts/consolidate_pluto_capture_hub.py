#!/usr/bin/env python3
"""
Pack Phase-1 Pluto capture (manifest + derived/*.npy) into a few Hub-friendly files.

Typical layout before:
  manifest_phase1.json + tens of thousands of derived/*.npy

After (--out-dir):
  iq_windows.npy       float32 (N, 2, 2048) — stack order == manifest ``windows`` order
  windows_meta.json.gz gzipped JSON list — one object per row (no per-file paths)

Row i always corresponds to manifest ``windows[i]`` after validating shapes.

**Labels:** Captures encode class in each derived filename / window ``id``:

  - ``_nd_`` → not-drone (checked **before** ``_d_`` so tokens do not collide)
  - ``_d_`` → drone

By default the packer **overwrites** manifest ``label`` / ``label_index`` using those tokens
(the GUI manifest is often wrong). Pass ``--use-manifest-labels`` to keep manifest labels.

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
from typing import Any, Dict, List, Tuple

import numpy as np


def _label_from_derived_filename(
    derived_rel: str,
    *,
    not_drone_name: str,
    drone_name: str,
    label_names: List[str],
) -> Tuple[str, int]:
    """
    Decode drone vs not-drone from Phase-1 naming convention (matches collector exports).

    Rules (same order as ``fix_manifest_labels_from_id``):
      - basename contains ``_nd_`` → not-drone class
      - else basename contains ``_d_`` → drone class
    """
    stem = Path(derived_rel).name.lower()
    if "_nd_" in stem:
        lab = not_drone_name
    elif "_d_" in stem:
        lab = drone_name
    else:
        raise ValueError(
            f"derived filename has neither '_nd_' nor '_d_' token (cannot infer label): {derived_rel!r}"
        )
    if lab not in label_names:
        raise ValueError(f"Inferred label {lab!r} not in manifest label_names {label_names!r}")
    return lab, int(label_names.index(lab))


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
    ap.add_argument(
        "--use-manifest-labels",
        action="store_true",
        help="Keep manifest label fields (default: overwrite from _nd_/_d_ filename tokens).",
    )
    ap.add_argument("--drone-name", default="drone", help="Label string for drone class.")
    ap.add_argument(
        "--not-drone-name",
        default="not_drone",
        help="Label string for not-drone class.",
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

    label_names = list(manifest.get("label_names") or [])
    if args.drone_name not in label_names or args.not_drone_name not in label_names:
        print(
            "ERROR: manifest.label_names must contain both "
            f"{args.not_drone_name!r} and {args.drone_name!r}; got {label_names!r}",
            file=sys.stderr,
        )
        return 2

    stack = np.empty((n, 2, ws), dtype=np.float32)
    meta_rows: List[Dict[str, Any]] = []
    relabeled = 0

    print(f"[pack] {n} windows  shape (N, 2, {ws})  dtype=float32")
    print(f"[pack] reading from {ds}")
    if args.use_manifest_labels:
        print("[pack] labels: using manifest (GUI) fields (--use-manifest-labels)")
    else:
        print("[pack] labels: from filename tokens _nd_ → not_drone, _d_ → drone")

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

        row = dict(w)
        if not args.use_manifest_labels:
            fn_lab, fn_idx = _label_from_derived_filename(
                rel,
                not_drone_name=args.not_drone_name,
                drone_name=args.drone_name,
                label_names=label_names,
            )
            if row.get("label") != fn_lab or int(row.get("label_index", -1)) != fn_idx:
                relabeled += 1
            row["label"] = fn_lab
            row["label_index"] = fn_idx

        meta_rows.append(_strip_window_for_hub(row))

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
        "note": (
            "Row index i matches manifest windows[i] and iq_windows[i]. derived_npy_rel omitted from meta."
        ),
        "label_provenance": (
            "manifest_gui_fields"
            if args.use_manifest_labels
            else (
                "filename_tokens: '_nd_' -> not_drone, '_d_' -> drone "
                "(manifest labels overwritten; check labels_relabeled_count)."
            )
        ),
        "labels_relabeled_count": relabeled if not args.use_manifest_labels else 0,
    }
    header_path.write_text(json.dumps(header, indent=2), encoding="utf-8")

    mb = iq_path.stat().st_size / (1024 * 1024)
    mb_meta = meta_path.stat().st_size / (1024 * 1024)
    print(f"[done] {iq_path}  ({mb:.1f} MiB)")
    print(f"[done] {meta_path}  ({mb_meta:.2f} MiB)")
    print(f"[done] {header_path}")
    if not args.use_manifest_labels:
        print(f"[done] relabeled {relabeled}/{n} rows where manifest disagreed with filename tokens")
    print(f"[hint] Upload only these three (+ README); skip derived/ and packed_hybrid_*/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
