<div class="ss-hero" markdown>
<img src="strands-sapiens-logo.svg" alt="Strands Sapiens">

# Strands Sapiens

<p class="ss-sub"><strong>Give your agent a body.</strong> Pixel-perfect human understanding, as tools.</p>
</div>

**Strands Sapiens** exposes Meta's [Sapiens2](https://github.com/facebookresearch/sapiens2) — a family of high-resolution transformers pretrained on **1B human images** — as idiomatic [Strands Agents](https://strandsagents.com) tools.

One Python import, every human-centric vision head: body-part segmentation, surface normals, intrinsic albedo, 3D pointmaps, 308-keypoint pose, plus raw pretrained backbone features.

---

## What it gives you

<div class="grid cards" markdown>

-   **🧩 29-class body-part segmentation**

    ```python
    sapiens_seg(input_path="people/", output_dir="out/", model_size="0.4b")
    ```

    → [Guide](guide/segmentation.md)

-   **🧭 Surface normals & albedo**

    ```python
    sapiens_normal(input_path="person.jpg", output_dir="out/")
    sapiens_albedo(input_path="person.jpg", output_dir="out/")
    ```

    → [Normals](guide/normals.md) · [Albedo](guide/albedo.md)

-   **🌌 3D pointmap (per-pixel 3D)**

    ```python
    sapiens_pointmap(input_path="person.jpg", output_dir="out/")
    ```

    → [Pointmap guide](guide/pointmap.md)

-   **🦴 308-keypoint 2D pose**

    Face 274 + body + hands + feet.

    ```python
    sapiens_pose(input_path="person.jpg", output_dir="out/")
    ```

    → [Pose guide](guide/pose.md)

-   **🧠 Pretrain backbone features**

    Drop-in dense features for RAG / downstream heads.

    ```python
    sapiens_backbone(image_path="person.jpg", model_size="0.1b")
    ```

    → [Backbone guide](guide/backbone.md)

-   **🔍 Checkpoint / env discovery**

    ```python
    sapiens_info()   # → what's available locally
    ```

    → [Checkpoints](getting-started/checkpoints.md)

</div>

---

## Why

Sapiens2 is the current state-of-the-art for human perception at native high resolution (up to **4096×3072**). It's one of the highest-signal models ever open-sourced for human-centric understanding — but it ships as a research codebase with CLI scripts, config gymnastics, and manual checkpoint wiring.

**Strands Sapiens turns each head into a one-line, agent-callable tool**, with structured responses, defensive fallbacks, and compatibility across the upstream's breaking API shuffles.

---

## 60-second quickstart

```bash
# 1) CUDA PyTorch (platform-specific; e.g. CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2) Sapiens2
pip install git+https://github.com/facebookresearch/sapiens2.git

# 3) This wrapper
pip install git+ssh://git@github.com/cagataycali/strands-sapiens.git

# 4) Drop a checkpoint into ~/sapiens2_host/seg/sapiens2_0.4b_seg.safetensors
export SAPIENS_CHECKPOINT_ROOT=~/sapiens2_host

# 5) Go
python -c "from strands_sapiens import sapiens_info; print(sapiens_info())"
```

[:rocket: Full installation →](getting-started/installation.md){ .md-button .md-button--primary }
[:book: API reference →](api-reference.md){ .md-button }

---

## Use from a Strands agent

```python
from strands import Agent
from strands_sapiens import TOOLS   # list of @tool

agent = Agent(tools=TOOLS)
agent("Segment every person in /data/photos and save to /data/out")
agent("Run 308-kpt pose on /data/photos/jump.jpg")
agent("What Sapiens2 checkpoints do I have available locally?")
```

Every tool returns a structured dict:

```python
{
  "status":   "success" | "error",
  "message":  "...",
  "outputs":  [...],      # per-image entries
  "checkpoint": "...",
  # task-specific keys
}
```

---

## Verified environment

Tested on **NVIDIA Thor** (JetPack 6, aarch64) with CUDA PyTorch 2.7+ and the
`sapiens2_0.4b_seg` / `sapiens2_0.1b_pretrain` checkpoints.

Python **3.10+** (Thor's default 3.10 works; newer also fine).

---

<p style="text-align:center; margin-top: 3rem;">
  <em>Built on top of Meta's <a href="https://github.com/facebookresearch/sapiens2">Sapiens2</a>, powered by <a href="https://strandsagents.com">Strands Agents</a>.</em>
</p>
