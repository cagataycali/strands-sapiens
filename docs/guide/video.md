# Video Processing

Process videos frame-by-frame through any Sapiens2 dense task — segmentation, normals, albedo, or pointmaps.

!!! info "New in v0.1.0"
    `sapiens_video` extracts frames, runs inference per-frame, and reassembles the results into an output video.

## Quick example

```python
sapiens_video(
    video_path="dance.mp4",
    output_dir="out/",
    task="seg",
    model_size="0.4b",
)
```

This will:

1. Extract every frame from `dance.mp4`
2. Run 29-class body-part segmentation on each frame
3. Write per-frame visualizations to `out/vis/`
4. Reassemble a side-by-side video at `out/dance_seg.mp4`

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `video_path` | *(required)* | Input video file (mp4, avi, mov, webm) |
| `output_dir` | *(required)* | Where to write output frames & video |
| `task` | `"seg"` | Dense task: `seg`, `normal`, `albedo`, `pointmap` |
| `model_size` | `"0.4b"` | Model size: `0.4b`, `0.8b`, `1b`, `5b` |
| `device` | `"cuda:0"` | Torch device |
| `fps` | `0` | Target FPS (0 = use source FPS) |
| `max_frames` | `0` | Max frames to process (0 = all) |
| `save_pred` | `False` | Save raw `.npy` predictions per frame |
| `save_frames` | `True` | Keep individual frame images |
| `reassemble` | `True` | Create output MP4 from processed frames |

## Use cases

### Normals video

```python
sapiens_video(
    video_path="walk.mp4",
    output_dir="out/normals/",
    task="normal",
    model_size="1b",
    fps=15,           # subsample to 15 fps for speed
    max_frames=300,   # cap at 300 frames
)
```

### Pointmap depth video

```python
sapiens_video(
    video_path="scene.mp4",
    output_dir="out/depth/",
    task="pointmap",
    save_pred=True,   # also export .npy + .ply per frame
)
```

### Agent-driven video analysis

```python
from strands import Agent
from strands_sapiens import TOOLS

agent = Agent(tools=TOOLS)
agent("Segment every person in /data/video.mp4, save to /data/out, use the 0.4b model")
agent("Run surface normals on /data/dance.mp4 at 10fps, max 200 frames")
```

## Tips

- **FPS subsampling**: For long videos, set `fps=10` or `fps=15` to skip frames and speed up processing.
- **Max frames**: Use `max_frames` to process only the first N frames for quick previews.
- **Memory**: Each frame loads independently — no extra VRAM beyond single-image inference.
- **Output format**: The reassembled video uses `mp4v` codec. For web-friendly H.264, re-encode with ffmpeg:

```bash
ffmpeg -i out/dance_seg.mp4 -c:v libx264 -crf 23 out/dance_seg_h264.mp4
```
