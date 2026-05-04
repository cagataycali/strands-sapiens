# Pose (308 keypoints)

Top-down 2D pose estimation with **308 keypoints**: 274 face points + body + hands + feet.

Uses the DETR-ResNet-101-DC5 person detector for bbox proposals, then Sapiens2 pose heads per crop.

## Signature

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

## Requirements

You need **two** checkpoints for this tool:

1. `$SAPIENS_CHECKPOINT_ROOT/pose/sapiens2_<size>_pose.safetensors`
2. `$SAPIENS_CHECKPOINT_ROOT/detector/detr-resnet-101-dc5/` (HuggingFace model dir)

Run `sapiens_info()` to confirm both are present:

```python
result = sapiens_info()
# check result["content"][1]["json"]
info = result["content"][1]["json"]
assert "pose" in info["available"]
assert info["detector_present"]
```

Download the detector:

```bash
huggingface-cli download facebook/detr-resnet-101-dc5 \
    --local-dir $SAPIENS_CHECKPOINT_ROOT/detector/detr-resnet-101-dc5
```

## Example

```python
from strands_sapiens import sapiens_pose

sapiens_pose(
    input_path="dance.jpg",
    output_dir="out/",
    model_size="0.4b",
    kpt_thres=0.3,
)
```

Output per image:

- `out/dance.jpg` - skeleton + keypoints overlay
- `out/dance.json` - structured detections

Example JSON shape (may vary slightly by upstream version):

```json
{
  "instances": [
    {
      "bbox":      [x1, y1, x2, y2],
      "score":     0.97,
      "keypoints": [[x, y], ...],        // 308 points
      "keypoint_scores": [0.91, ...]      // 308 confidences
    }
  ]
}
```

## Compatibility strategy

`sapiens_pose` tries three integration paths in order, so it keeps working across upstream Sapiens2 refactors:

1. `sapiens.pose.inference.Inferencer` (high-level API).
2. `sapiens.pose.models.init_pose_model` + `PoseVisualizer`.
3. If neither works, returns an error pointing you at the upstream shell script with the correct paths.

You can see which path ran from the response `"api"` field.

## Tips

- **`kpt_thres`**: lower it (0.1–0.2) if you're post-processing; raise to 0.5+ for crisp overlays.
- **Face-only use**: the 274 face keypoints form a dense landmark grid - useful for head-pose tracking, gaze, expression.
- **Performance**: RTMDet-m dominates latency for small crops; consider running detection once per video frame and pose across frames.

## Related

- [Segmentation →](segmentation.md) - combine seg + pose for per-limb attention.

<figure markdown="span">
  ![Pose pipeline](../assets/diagrams/pose-pipeline.svg){ width="100%" loading="lazy" }
  <figcaption>Pose estimation data flow: DETR detector → per-person crop → 308-keypoint heatmaps</figcaption>
</figure>
