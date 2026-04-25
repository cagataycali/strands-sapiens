# API Reference

## Module import

```python
import strands_sapiens as ss
from strands_sapiens import TOOLS   # list of @tool, ready for Agent(tools=TOOLS)
```

## Tools

### `sapiens_info`

Report available checkpoints, CUDA state, and whether `sapiens` is importable.

```python
sapiens_info() -> dict
```

**Returns**

```python
{
  "status": "success",
  "checkpoint_root": "/Users/.../sapiens2_host",
  "checkpoint_root_exists": bool,
  "available": {"pretrain": ["0.1b"], "seg": ["0.4b"]},
  "detector_present": bool,
  "cuda": {"available": bool, "device_count": int, "device_name": str|None},
  "sapiens_package": bool
}
```

---

### `sapiens_backbone`

Raw Sapiens2 pretrain-backbone features from an image.

```python
sapiens_backbone(
    image_path:       str,
    model_size:       str  = "0.1b",     # 0.1b|0.4b|0.8b|1b|1b_4k|5b
    img_h:            int  = 1024,
    img_w:            int  = 768,
    device:           str  = "cuda:0",
    save_features_to: str|None = None,
    overwrite:        bool = False,
) -> dict
```

**Returns** `{"status", "message", "feature_shape", "checkpoint", "saved_to"}`

---

### `sapiens_seg`

29-class body-part segmentation.

```python
sapiens_seg(
    input_path: str,               # file OR directory
    output_dir: str,
    model_size: str   = "0.4b",    # 0.4b|0.8b|1b|5b
    device:     str   = "cuda:0",
    save_pred:  bool  = True,      # also write _seg.npy
) -> dict
```

Output per image: `out/<name>.<ext>` side-by-side viz and `out/<name>_seg.npy`.

---

### `sapiens_normal`

Per-pixel surface-normal estimation.

```python
sapiens_normal(input_path, output_dir, model_size="0.4b",
               device="cuda:0", save_pred=True) -> dict
```

`_normal.npy` = `(3, H, W)` float.

---

### `sapiens_albedo`

Intrinsic albedo (illumination-invariant color).

```python
sapiens_albedo(input_path, output_dir, model_size="0.4b",
               device="cuda:0", save_pred=True) -> dict
```

`_albedo.npy` = `(3, H, W)` float.

---

### `sapiens_pointmap`

Per-pixel 3D pointmap in camera space.

```python
sapiens_pointmap(input_path, output_dir, model_size="0.4b",
                 device="cuda:0", save_pred=True) -> dict
```

`_pointmap.npy` = `(3, H, W)` float, channels = (X, Y, Z).

---

### `sapiens_pose`

308-keypoint 2D pose estimation (face 274 + body + hands + feet).

```python
sapiens_pose(
    input_path:     str,
    output_dir:     str,
    model_size:     str   = "0.4b",
    device:         str   = "cuda:0",
    kpt_thres:      float = 0.3,
    line_thickness: int   = 2,
    radius:         int   = 3,
) -> dict
```

Requires `$SAPIENS_CHECKPOINT_ROOT/detector/rtmdet_m.pth`.

Output per image: `out/<name>` overlay + `out/<stem>.json` instances.

---

## Public helpers (`strands_sapiens._common`)

These aren't `@tool`s but are useful for scripts and tests.

```python
checkpoint_root() -> Path
checkpoint_path(task: str, size: str) -> Path
validate_size(task: str, size: str) -> str
arch_name(size: str) -> str
resolve_input(path: str, recursive: bool = False) -> tuple[Path, list[Path]]
ensure_output(dir: str) -> Path
ensure_checkpoint_root() -> tuple[Path, bool]
ok(message: str, **extra) -> dict
err(message: str, **extra) -> dict
TASK_SIZES: dict[str, tuple[str, ...]]
```

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `SAPIENS_CHECKPOINT_ROOT` | `~/sapiens2_host` | Where checkpoints live. |

## Response shape

All tools return a dict with at minimum `status` + `message`. Error responses include `traceback`. Success responses include task-specific keys documented per tool.
