# pluto-rf-dataset-iitm

Drone vs not-drone RF IQ windows recorded at IIT Madras with an ADALM-Pluto SDR.  
**Download the files from Hugging Face** (not this repo):

**https://huggingface.co/datasets/BlackWhite123/pluto-rf-dataset-iitm-packed**

| | |
|--|--|
| Windows | 36,468 |
| Tensor | `iq_windows.npy` — `float32` shape `(N, 2, 2048)` (I/Q) |
| Meta | `windows_meta.json.gz` — same row order as the tensor |
| Rate | 25 MHz |

Labels follow the window id / filename: `_nd_` → not_drone, `_d_` → drone (`_nd_` matched first). See `data/dataset_header.json` for bundle metadata.

## Download

```bash
pip install huggingface_hub

hf download BlackWhite123/pluto-rf-dataset-iitm-packed \
  --repo-type dataset \
  --local-dir ./pluto_iitm_packed
```

## Load

```python
import gzip, json
import numpy as np

iq = np.load("pluto_iitm_packed/iq_windows.npy", mmap_mode="r")
with gzip.open("pluto_iitm_packed/windows_meta.json.gz", "rt", encoding="utf-8") as f:
    meta = json.load(f)
x, label = iq[0], meta[0]["label"]
```

## Citation

```bibtex
@misc{pluto_rf_dataset_iitm_2026,
  title  = {Pluto RF IITM: drone vs not-drone IQ windows},
  author = {Shaik Zaid Haaris},
  year   = {2026},
  url    = {https://huggingface.co/datasets/BlackWhite123/pluto-rf-dataset-iitm-packed},
}
```

## License

[CC-BY-4.0](LICENSE)
