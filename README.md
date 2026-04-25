<div align="center">
  <img src="strands-sapiens-logo.svg" alt="Strands Sapiens" width="180">
  <h1>strands-sapiens</h1>
  <p><strong>Give your agent a body.</strong> Pixel-perfect human understanding, as tools.</p>
  <p>
    <a href="https://cagataycali.github.io/strands-sapiens/"><img alt="Docs" src="https://img.shields.io/badge/docs-latest-F97316?style=flat-square"></a>
    <a href="https://github.com/cagataycali/strands-sapiens"><img alt="GitHub" src="https://img.shields.io/badge/github-cagataycali%2Fstrands--sapiens-0866FF?style=flat-square&logo=github"></a>
  </p>
</div>

---

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

> This repo is private. The `pip install git+ssh://...` commands below assume
> you have SSH access.

```bash
# 1) Install a CUDA-enabled PyTorch first (platform-specific):
#    e.g. pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2) Install sapiens2 (backbone + dense/pose modules)
pip install git+https://github.com/facebookresearch/sapiens2.git

# 3) Install this wrapper
pip install git+ssh://git@github.com/cagataycali/strands-sapiens.git
#   or from a local clone:
#   pip install -e .

# 4) Download checkpoints (see MODEL_ZOO in the sapiens2 repo).
#    Default location: $SAPIENS_CHECKPOINT_ROOT (fallback: ~/sapiens2_host)
```

Checkpoint layout expected (matches upstream):

```
~/sapiens2_host/
├── pretrain/ sapiens2_{0.1b,0.4b,0.8b,1b,1b_4k,5b}_pretrain.safetensors
├── seg/      sapiens2_{0.4b,0.8b,1b,5b}_seg.safetensors
├── normal/   sapiens2_{0.4b,0.8b,1b,5b}_normal.safetensors
├── albedo/   sapiens2_{0.4b,0.8b,1b,5b}_albedo.safetensors
├── pose/     sapiens2_{0.4b,0.8b,1b,5b}_pose.safetensors
├── pointmap/ sapiens2_{0.4b,0.8b,1b,5b}_pointmap.safetensors
└── detector/ rtmdet_m.pth
```

Override the root with:

```bash
export SAPIENS_CHECKPOINT_ROOT=/data/sapiens2_host
```

## Supported model sizes

| Task       | Sizes |
|------------|-------|
| `pretrain` | `0.1b · 0.4b · 0.8b · 1b · 1b_4k · 5b` |
| `seg`      | `0.4b · 0.8b · 1b · 5b` |
| `normal`   | `0.4b · 0.8b · 1b · 5b` |
| `albedo`   | `0.4b · 0.8b · 1b · 5b` |
| `pointmap` | `0.4b · 0.8b · 1b · 5b` |
| `pose`     | `0.4b · 0.8b · 1b · 5b` |

The `1b_4k` pretrain variant uses the 1B-parameter backbone with a 4096×3072
input resolution (see the upstream MODEL_ZOO).

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

sapiens_seg(
    input_path="/tmp/sapiens2_test/input",
    output_dir="/tmp/sapiens2_test/output",
    model_size="0.4b",
)
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

## Response shape

Every tool returns a dict of the form:

```python
{"status": "success" | "error",
 "message": "...",
 # task-specific extras:
 "outputs": [...],
 "checkpoint": "...",
 ...}
```

On error you also get `traceback` for debugging.

## Verified environment

Tested on NVIDIA Thor (JetPack 6 aarch64) with:
- `sapiens2_0.1b_pretrain.safetensors`
- `sapiens2_0.4b_seg.safetensors`
- CUDA PyTorch 2.7+

End-to-end seg inference on a real human image succeeded:
`/tmp/sapiens2_test/output/{human.jpg, human_seg.npy, human_vis_compressed.jpg}`.

> JetPack 6 ships Python 3.10 by default. This package targets
> `requires-python >= 3.10`; if you're on a newer interpreter, nothing changes.

## Development

```bash
pip install -e '.[dev]'
pytest -q
```

The smoke tests do **not** require CUDA, GPU, or any Sapiens2 checkpoints.

## Troubleshooting

- `Missing checkpoint: ...` → your `$SAPIENS_CHECKPOINT_ROOT` doesn't contain
  the file the tool expects. Run `sapiens_info()` to see what's present.
- `No config found for task=...` → the installed `sapiens` package version
  doesn't match the config path this wrapper expects. The wrapper falls back
  to `rglob` of `configs/<task>/**/sapiens2_<size>_<task>*.py` — if that
  still fails, open an issue with your `pip show sapiens` output.
- `sapiens.pose high-level API not available` → your installed sapiens2
  build doesn't expose `sapiens.pose.inference.Inferencer` or
  `sapiens.pose.models.init_pose_model`. The error message tells you how
  to invoke the upstream CLI script directly with the right paths.

## License

- Wrapper code: **proprietary** (this repo is private).
- Upstream Sapiens2 model & code: governed by its own
  [Sapiens2 License](https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md).