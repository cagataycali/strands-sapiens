# Checkpoints

Strands Sapiens looks for Sapiens2 weights on disk. It doesn't download them for you (model licensing varies - see [MODEL_ZOO.md](https://github.com/facebookresearch/sapiens2/blob/main/MODEL_ZOO.md) upstream).

## Checkpoint root

The root directory is controlled by the `SAPIENS_CHECKPOINT_ROOT` env var.
Default: `~/sapiens2_host`.

```bash
export SAPIENS_CHECKPOINT_ROOT=/data/sapiens2_host
```

## Expected layout

```
$SAPIENS_CHECKPOINT_ROOT/
├── pretrain/
│   ├── sapiens2_0.1b_pretrain.safetensors
│   ├── sapiens2_0.4b_pretrain.safetensors
│   ├── sapiens2_0.8b_pretrain.safetensors
│   ├── sapiens2_1b_pretrain.safetensors
│   ├── sapiens2_1b_4k_pretrain.safetensors
│   └── sapiens2_5b_pretrain.safetensors
├── seg/
│   ├── sapiens2_0.4b_seg.safetensors
│   ├── sapiens2_0.8b_seg.safetensors
│   ├── sapiens2_1b_seg.safetensors
│   └── sapiens2_5b_seg.safetensors
├── normal/    (same sizes as seg)
├── albedo/    (same sizes as seg)
├── pointmap/  (same sizes as seg)
├── pose/      (same sizes as seg)
└── detector/
    └── detr-resnet-101-dc5/              # required for pose (HuggingFace model dir)
```

## Supported sizes per task

| Task       | Sizes |
|------------|-------|
| `pretrain` | `0.1b` · `0.4b` · `0.8b` · `1b` · `1b_4k` · `5b` |
| `seg`      | `0.4b` · `0.8b` · `1b` · `5b` |
| `normal`   | `0.4b` · `0.8b` · `1b` · `5b` |
| `albedo`   | `0.4b` · `0.8b` · `1b` · `5b` |
| `pointmap` | `0.4b` · `0.8b` · `1b` · `5b` |
| `pose`     | `0.4b` · `0.8b` · `1b` · `5b` |

The `1b_4k` variant uses the 1B-parameter backbone at a 4096×3072 input resolution.

## Discover what you have

```python
from strands_sapiens import sapiens_info
print(sapiens_info())
```

```python
{
  "status": "success",
  "checkpoint_root": "/data/sapiens2_host",
  "checkpoint_root_exists": true,
  "available": {
    "pretrain": ["0.1b"],
    "seg":      ["0.4b"]
  },
  "detector_present": true,
  "cuda": {"available": true, "device_count": 1, "device_name": "Orin"},
  "sapiens_package": true
}
```

## Download sources

- **Model weights**: see Sapiens2 [MODEL_ZOO.md](https://github.com/facebookresearch/sapiens2/blob/main/MODEL_ZOO.md) - typically hosted on Hugging Face under `facebook/sapiens2-*`.
- **Person detector** (DETR): `huggingface-cli download facebook/detr-resnet-101-dc5 --local-dir $SAPIENS_CHECKPOINT_ROOT/detector/detr-resnet-101-dc5`

## Size vs. VRAM cheat-sheet

| Size    | Params | Min VRAM (fp16 inference) | Notes |
|---------|--------|----------------------------|-------|
| 0.1b    | 100M   | ~1 GB                      | Pretrain only |
| 0.4b    | 400M   | ~2 GB                      | Good default for Jetson |
| 0.8b    | 800M   | ~4 GB                      | |
| 1b      | 1B     | ~6 GB                      | |
| 1b_4k   | 1B     | ~16 GB                     | 4096×3072 resolution |
| 5b      | 5B     | ~24 GB                     | Best quality |

(Values approximate - actual usage depends on batch size & precision.)
