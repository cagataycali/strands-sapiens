# API Reference

## Module import

```python
import strands_sapiens as ss
from strands_sapiens import TOOLS   # list of @tool, ready for Agent(tools=TOOLS)
```

## Response format

All tools return the standard [Strands `ToolResult`](https://github.com/strands-agents/sdk-python) format:

```python
{
    "status": "success",          # or "error"
    "content": [
        {"text": "...summary..."},                                         # always present
        {"image": {"format": "jpeg", "source": {"bytes": b"..."}}},       # inline vis (up to 5)
        {"json": {"task": "...", "outputs": [...], ...}}                   # structured data
    ]
}
```

On error, `content` contains a text message and optionally a `json` block with `traceback`.

---

## Tools

### `sapiens_info`

Report available checkpoints, CUDA state, and whether `sapiens` is importable.

```python
sapiens_info() -> dict
```

**JSON block contains:**

| Field | Type | Description |
|-------|------|-------------|
| `checkpoint_root` | `str` | Resolved checkpoint root path |
| `checkpoint_root_exists` | `bool` | Whether the root dir exists |
| `available` | `dict` | Map of `task → [sizes_present]` |
| `detector_present` | `bool` | Whether any pose detector is found |
| `detector_type` | `str` | `"detr-resnet-101-dc5"`, `"rtmdet_m"`, or `"none"` |
| `cuda` | `dict` | `{available, device_count, device_name}` |
| `sapiens_package` | `bool` | Whether `import sapiens` succeeds |

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

**JSON block:** `feature_shape`, `checkpoint`, `saved_to`

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

`_albedo.npy` = `(3, H, W)` float, clamped to `[0, 1]`.

---

### `sapiens_pointmap`

Per-pixel 3D pointmap in camera space (metric scale).

```python
sapiens_pointmap(input_path, output_dir, model_size="0.4b",
                 device="cuda:0", save_pred=True) -> dict
```

`_pointmap.npy` = `(3, H, W)` float, channels = (X, Y, Z).

When `open3d` is installed, also exports `.ply` point clouds.

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

Requires `$SAPIENS_CHECKPOINT_ROOT/detector/detr-resnet-101-dc5/` (HuggingFace `facebook/detr-resnet-101-dc5`). Falls back to legacy `rtmdet_m.pth` if present.

Output per image: `out/<name>` overlay + `out/<stem>.json` instances.

---

### `sapiens_video`

Process a video frame-by-frame through any dense task.

```python
sapiens_video(
    video_path:   str,
    output_dir:   str,
    task:         str   = "seg",       # seg|normal|albedo|pointmap
    model_size:   str   = "0.4b",
    device:       str   = "cuda:0",
    fps:          float = 0,           # 0 = source FPS
    max_frames:   int   = 0,           # 0 = all
    save_pred:    bool  = False,
    save_frames:  bool  = True,
    reassemble:   bool  = True,        # create output MP4
) -> dict
```

**JSON block:** `video_input`, `output_video`, `frames_processed`, `source_fps`, `target_fps`, `frame_outputs`

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
ok_with_images(message: str, image_paths: list = None, **extra) -> dict
err(message: str, **extra) -> dict
TASK_SIZES: dict[str, tuple[str, ...]]
```

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `SAPIENS_CHECKPOINT_ROOT` | `~/sapiens2_host` | Where checkpoints live. |
