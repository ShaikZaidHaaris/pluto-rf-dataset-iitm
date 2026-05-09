# pluto-rf-dataset-iitm

> Drone vs not-drone RF dataset captured on the IIT Madras campus using ADALM-Pluto SDR.

This repository is the **lightweight index** for an RF I/Q dataset recorded on the
IIT Madras campus with an [ADALM-Pluto](https://www.analog.com/en/resources/evaluation-hardware-and-software/evaluation-boards-kits/adalm-pluto.html)
SDR. The full I/Q tensors (~2.8 GB) live on the Hugging Face Hub:

**Dataset on Hugging Face:** `https://huggingface.co/datasets/BlackWhite123/pluto-rf-dataset-iitm`

The GitHub repo only hosts the metadata, loader code, and documentation so the
download/clone is fast.

---

## Dataset at a glance

| Property | Value |
|----------|-------|
| Total windows | **7,327** |
| Train / Val | 5,449 / 1,878 |
| Classes | `not_drone` (3,712), `drone` (3,615) |
| Window length | 2,048 IQ samples (non-overlapping; hop = 2,048) |
| Sample rate | 25 MHz |
| Sample dtype | `complex64` (real-passband I/Q) |
| Recording bands | logical band `H` (Pluto-supported drone control bands) |
| Recording sites | IIT Madras campus — indoor and outdoor segments |
| Recording device | ADALM-Pluto SDR |

### Recording segments

| Segment | Drone | Not-drone | Notes |
|---------|------:|----------:|-------|
| `pluto_indoors_001_6f5762e60039_train` | 0 | 1034 | indoor ambient |
| `pluto_indoors_001_d_c5edfad2e9be_train` | 904 | 0 | indoor, drone present |
| `pluto_indoors_002_bluetooth_390c7f28569e_train` | 0 | 824 | indoor, Bluetooth interference (no drone) |
| `pluto_indoors_2_e1f4175823db_train` | 903 | 0 | indoor, drone present |
| `pluto_outdoors_001_d_f3c849950296_train` | 905 | 0 | outdoor, drone present |
| `pluto_outdoors_1_6247843463dc_train` | 0 | 879 | outdoor ambient |
| `pluto_outdoors_2_d_aee1697c0a75_val` | 903 | 0 | outdoor, drone present (val) |
| `pluto_outdoors_2_val_bcf4d3abf4da_val` | 0 | 975 | outdoor ambient (val) |

The `bluetooth` segment is included as a deliberate negative-class hard case
(non-drone RF activity in the same band).

---

## Repository layout

```
pluto-rf-dataset-iitm/
  README.md                  this file
  LICENSE                    CC-BY-4.0
  requirements.txt           Python deps for the loader / uploader
  data/
    manifest_phase1.json     index of every window (label, split, segment, npy path)
  scripts/
    upload_to_hf.py          one-shot Hugging Face upload script
```

The actual per-window arrays (`derived/*.npy`) and the packed tensors
(`packed/`, `packed_hybrid*/`) are **not** in this repo. They are on the Hugging
Face dataset page above.

---

## Manifest schema (`data/manifest_phase1.json`)

```json
{
  "schema_version": 1,
  "phase": 1,
  "source": "pluto_capture",
  "window_samples": 2048,
  "hop_samples": 2048,
  "target_sample_rate_hz": 25000000.0,
  "label_names": ["not_drone", "drone"],
  "windows": [
    {
      "id": "indoors_001_6f5762e60039_00000",
      "segment_pair_id": "pluto_indoors_001_6f5762e60039_train",
      "band": "H",
      "split": "train",
      "label": "not_drone",
      "label_index": 0,
      "start_sample": 0,
      "end_sample": 2048,
      "n_samples": 2048,
      "source_native_fs_hz": 25000000.0,
      "target_sample_rate_hz": 25000000.0,
      "derived_npy_rel": "derived/indoors_001_6f5762e60039_00000.npy"
    }
  ]
}
```

`derived_npy_rel` is the path of the per-window IQ array **inside the Hugging
Face dataset** (and inside the original `data/pluto_capture/` capture tree).

---

## Quickstart

### 1. Install deps

```bash
pip install -r requirements.txt
```

### 2. Pull the data from Hugging Face

```python
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="BlackWhite123/pluto-rf-dataset-iitm",
    repo_type="dataset",
    local_dir="./pluto_capture",
)
print("dataset at:", local_dir)
```

### 3. Load a single window

```python
import json, numpy as np

with open("data/manifest_phase1.json") as f:
    manifest = json.load(f)

w = manifest["windows"][0]
iq = np.load(f"./pluto_capture/{w['derived_npy_rel']}").astype(np.complex64)
print(w["split"], w["label"], iq.shape, iq.dtype)
```

---

## Provenance and reproducibility

- Capture stack: `data_collector_gui.py` / `classifier_collector_gui.py` from
  the `RFML Drone Detection` toolchain (`scripts/drone_classifier_opensource/`).
- Hardware: ADALM-Pluto SDR (stock profile, 70 MHz–6 GHz only on hacked units).
- Sample rate: 25 MS/s → 2,048-sample windows ≈ **81.92 µs per window**.
- Bands: logical `H` band (drone control bands at 2.4 GHz / 5.8 GHz, depending
  on tile selection — see `layer_hardware.LOGICAL_DRONE_BANDS`).
- Format on disk: per-window `.npy` arrays of `complex64` IQ (real passband),
  shape `(2048,)`.

---

## Citation

If you use this dataset, please cite it as:

```bibtex
@misc{pluto_rf_dataset_iitm_2026,
  title  = {pluto-rf-dataset-iitm: Drone vs not-drone RF dataset captured on IIT Madras campus using ADALM-Pluto SDR},
  author = {Shaik Zaid Haaris},
  year   = {2026},
  url    = {https://huggingface.co/datasets/BlackWhite123/pluto-rf-dataset-iitm},
}
```

---

## License

Released under [Creative Commons Attribution 4.0 International (CC-BY-4.0)](LICENSE).
You are free to share and adapt the dataset for any purpose, including
commercial use, as long as you give appropriate credit.
