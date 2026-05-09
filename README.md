# pluto-rf-dataset-iitm

> Drone vs not-drone RF dataset captured on the IIT Madras campus using ADALM-Pluto SDR.

This repository holds **documentation**, **small metadata copies**, and **scripts** to rebuild or upload the dataset. The heavy IQ tensors live on the Hugging Face Hub.

---

## Canonical dataset (recommended)

**Packed Phase‑1 IQ windows** — few files, ~570 MiB IQ tensor, no tens of thousands of per-window `.npy` files on the Hub:

**[BlackWhite123/pluto-rf-dataset-iitm-packed](https://huggingface.co/datasets/BlackWhite123/pluto-rf-dataset-iitm-packed)**

| Property | Value |
|----------|-------|
| Windows | **36,468** |
| Labels (from filenames, not GUI manifest) | `not_drone` **18,216**, `drone` **18,252** — token **`_nd_`** → not-drone, **`_d_`** → drone (`_nd_` is checked before `_d_`). Raw `manifest_phase1.json` had **14,147** wrong rows vs these tokens; **`consolidate_pluto_capture_hub.py` overwrites labels by default** (`--use-manifest-labels` keeps GUI fields). |
| Manifest split field | all rows marked `train` in Phase‑1 export — **hold out your own val/test** (e.g. stratified by label / segment) |
| Tensor | `iq_windows.npy` — `float32` **`(N, 2, 2048)`** (channel 0 = I, channel 1 = Q) |
| Sample rate | 25 MHz (`target_sample_rate_hz` in `dataset_header.json`) |
| Meta | `windows_meta.json.gz` — one JSON object per row, same order as `iq_windows[i]` |
| Provenance | IIT Madras campus; ADALM-Pluto; collector export described in `manifest_global_fields.notes` |

This repo includes a copy of **`data/dataset_header.json`** describing that bundle (schema `pluto_phase1_hub_bundle_v1`, includes `label_provenance` and `labels_relabeled_count`).

### Download (HF)

```bash
pip install huggingface_hub

hf download BlackWhite123/pluto-rf-dataset-iitm-packed \
  --repo-type dataset \
  --local-dir ./pluto_iitm_packed
```

Or:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="BlackWhite123/pluto-rf-dataset-iitm-packed",
    repo_type="dataset",
    local_dir="./pluto_iitm_packed",
)
```

### Load one window

```python
import gzip, json
import numpy as np

iq = np.load("pluto_iitm_packed/iq_windows.npy", mmap_mode="r")
with gzip.open("pluto_iitm_packed/windows_meta.json.gz", "rt", encoding="utf-8") as f:
    meta = json.load(f)

i = 0
x = iq[i]  # (2, 2048)
label = meta[i]["label"]
segment = meta[i]["segment_pair_id"]
```

---

## Pack locally (from raw `derived/*.npy`)

If you have the full capture tree (`manifest_phase1.json` + `derived/`), produce the Hub bundle:

```bash
pip install -r requirements.txt

python3 scripts/consolidate_pluto_capture_hub.py \
  --dataset-dir /path/to/pluto_capture \
  --out-dir ./pluto_iitm_hub_bundle
```

Add a short `README.md` into `./pluto_iitm_hub_bundle` if you want a richer dataset card on the Hub (optional).

---

## Upload to Hugging Face

After `hf auth login`:

**Packed bundle (recommended)**

```bash
hf upload-large-folder BlackWhite123/pluto-rf-dataset-iitm-packed \
  ./pluto_iitm_hub_bundle \
  --repo-type dataset \
  --num-workers 8
```

Or via this repo’s helper:

```bash
python3 scripts/upload_packed_to_hf.py \
  --src ./pluto_iitm_hub_bundle \
  --repo-id BlackWhite123/pluto-rf-dataset-iitm-packed \
  --create
```

**Whole `pluto_capture` tree** (many files / multi‑GB — not recommended):

```bash
python3 scripts/upload_to_hf.py \
  --src /path/to/pluto_capture \
  --repo-id YOUR_USER/your-repo-name \
  --create
```

---

## Legacy files in this repo

| Path | Role |
|------|------|
| `data/manifest_phase1.json` | Smaller **7,327‑window** Phase‑1 manifest (train/val splits, IITM indoor/outdoor segments). Useful for older experiments — **not** the same export as the 36k packed Hub corpus. |
| [BlackWhite123/pluto-rf-dataset-iitm](https://huggingface.co/datasets/BlackWhite123/pluto-rf-dataset-iitm) | Earlier Hub experiment / layout; prefer **`pluto-rf-dataset-iitm-packed`** for the full IITM IQ stack in few files. |

---

## Repository layout

```
pluto-rf-dataset-iitm/
  README.md
  LICENSE                     CC-BY-4.0
  requirements.txt
  data/
    dataset_header.json       snapshot describing the packed Hub bundle (36k windows)
    manifest_phase1.json      legacy 7k-window Phase-1 manifest
  scripts/
    consolidate_pluto_capture_hub.py   manifest + derived/ → iq_windows.npy + meta gzip
    upload_packed_to_hf.py            upload that bundle
    upload_to_hf.py                   upload_large_folder for a raw pluto_capture tree
```

---

## Citation

```bibtex
@misc{pluto_rf_dataset_iitm_2026,
  title  = {Pluto RF IITM: drone vs not-drone IQ windows (IIT Madras campus)},
  author = {Shaik Zaid Haaris},
  year   = {2026},
  note   = {Packed release: Hugging Face datasets BlackWhite123/pluto-rf-dataset-iitm-packed},
  url    = {https://huggingface.co/datasets/BlackWhite123/pluto-rf-dataset-iitm-packed},
}
```

---

## License

[Creative Commons Attribution 4.0 International (CC-BY-4.0)](LICENSE).
