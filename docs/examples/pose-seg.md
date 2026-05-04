# Pose + seg pipeline

Combine 308-keypoint pose with body-part segmentation to get *per-limb attention weights* - "how much of this limb is visible and confidently detected?".

This is a building block for action recognition, injury analysis, or clothing fit.

## Code

```python
import json
import numpy as np

from strands_sapiens import sapiens_seg, sapiens_pose

IMG = "dance.jpg"
OUT = "out/"

# 1) Run both heads
sapiens_seg(input_path=IMG, output_dir=OUT, model_size="0.4b")
sapiens_pose(input_path=IMG, output_dir=OUT, model_size="0.4b", kpt_thres=0.3)

# 2) Load raw outputs
seg    = np.load(f"{OUT}/dance_seg.npy")                # (H, W) class indices
pose   = json.load(open(f"{OUT}/dance.json"))           # {"instances": [...]}

# 3) For each detected person, compute per-limb coverage
# (rough mapping - check your palette for exact ids)
LIMB_CLASSES = {
    "head":       [3, 4, 5],
    "torso":      [1, 2],
    "left_arm":   [6, 7, 10],
    "right_arm":  [8, 9, 11],
    "left_leg":   [12, 13, 16],
    "right_leg":  [14, 15, 17],
}

H, W = seg.shape
for person in pose["instances"]:
    kpts   = np.array(person["keypoints"])          # (308, 2)
    scores = np.array(person["keypoint_scores"])    # (308,)
    x1, y1, x2, y2 = map(int, person["bbox"])

    crop = seg[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]
    crop_area = crop.size or 1

    print(f"Person bbox=({x1},{y1})-({x2},{y2}) avg_kpt_score={scores.mean():.2f}")
    for limb, ids in LIMB_CLASSES.items():
        share = np.isin(crop, ids).mean()
        print(f"   {limb:10s}: {share:.1%} of bbox pixels")
```

## Sample output

```
Person bbox=(120,48)-(540,820) avg_kpt_score=0.78
   head      : 4.2% of bbox pixels
   torso     : 18.6% of bbox pixels
   left_arm  : 6.3% of bbox pixels
   right_arm : 7.1% of bbox pixels
   left_leg  : 11.5% of bbox pixels
   right_leg : 12.0% of bbox pixels
```

## Going further

- **Occlusion detection**: if `left_arm` has low seg share *and* low keypoint score, the arm is probably occluded.
- **Self-healing pipeline**: if pose confidence < 0.3 in a region, fall back to seg-only analysis.
- **Action features**: per-limb (seg_area × pose_confidence) is a dense, pose-aware descriptor.

## Related

- [Pose guide →](../guide/pose.md)
- [Segmentation guide →](../guide/segmentation.md)
