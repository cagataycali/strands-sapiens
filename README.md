# strands-sapiens

**Strands Agents `@tool` wrappers around Meta [Sapiens2](https://github.com/facebookresearch/sapiens2)** —
a family of high-resolution transformers pretrained on 1B human images, for
human-centric vision tasks.

Exposes every Sapiens2 capability as a Python-callable, Strands-registered tool:

| Tool | Task | Backing script |
|------|------|----------------|
| `sapiens_seg`      | 29-class body-part segmentation | `sapiens/dense/tools/vis/vis_seg.py` |
| `sapiens_normal`   | Surface-normal estimation       | `sapiens/dense/tools/vis/vis_normal.py` |
| `sapiens_albedo`   | Albedo estimation               | `sapiens/dense/tools/vis/vis_albedo.py` |
| `sapiens_pointmap` | Pointmap / 3D lift              | `sapiens/dense/tools/vis/vis_pointmap.py` |
| `sapiens_pose`     | 308-keypoint 2D pose            | `sapiens/pose/tools/vis/...`            |
| `sapiens_backbone` | Raw pretrained backbone features | `sapiens.backbones.standalone.sapiens2` |
| `sapiens_info`     | Inspect local checkpoints / env | — |

## Install

```bash
# 1) Install CUDA-enabled torch first (platform-specific):
#    e.g. pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2) Install sapiens2 (backbone + dense/pose modules)
pip install git+https://github.com/facebookresearch/sapiens2.git

# 3) Install this package
pip install git+ssh://git@github.com/cagataycali/strands-sapiens.git
#   or from a local clone:
#   pip install -e .

# 4) Download checkpoints (see MODEL_ZOO in sapiens2 repo)
#    Default location: $SAPIENS_CHECKPOINT_ROOT (fallback: ~/sapiens2_host)
```

Checkpoint layout expected (matches upstream):

```
~/sapiens2_host/
├── pretrain/ sapiens2_{0.1b,0.4b,0.8b,1b,5b}_pretrain.safetensors
├── seg/      sapiens2_{0.4b,0.8b,1b,5b}_seg.safetensors
├── normal/   sapiens2_{0.4b,0.8b,1b,5b}_normal.safetensors
├── pose/     sapiens2_{0.4b,0.8b,1b,5b}_pose.safetensors
├── pointmap/ sapiens2_{0.4b,0.8b,1b,5b}_pointmap.safetensors
└── detector/ rtmdet_m.pth
```

## Use from a Strands agent

```python
from strands import Agent
from strands_sapiens import TOOLS  # list[@tool]

agent = Agent(tools=TOOLS)
agent("Segment every person in /data/photos and save to /data/out")
```

Or cherry-pick individual tools:

```python
from strands_sapiens import sapiens_seg, sapiens_pose, sapiens_backbone

sapiens_seg(input_path="/tmp/sapiens2_test/input",
            output_dir="/tmp/sapiens2_test/output",
            model_size="0.4b")
```

## Use directly (no agent)

Every tool is a regular Python function too:

```python
from strands_sapiens import sapiens_seg
result = sapiens_seg(
    input_path="human.jpg",
    output_dir="./out",
    model_size="0.4b",
    save_pred=True,
)
print(result["status"], result["outputs"])
```

## Verified environment

Tested on NVIDIA Thor / JetPack with:
- `sapiens2_0.1b_pretrain.safetensors`
- `sapiens2_0.4b_seg.safetensors`
- CUDA PyTorch 2.7+

End-to-end seg inference on a real human image succeeded:
`/tmp/sapiens2_test/output/{human.jpg, human_seg.npy, human_vis_compressed.jpg}`.

## License

Wrapper code: proprietary (this repo is private).
Upstream Sapiens2 model & code: governed by its own
[Sapiens2 License](https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md).
