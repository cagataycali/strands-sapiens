# Quickstart

You should already have [installed](installation.md) the package and [downloaded at least one checkpoint](checkpoints.md).

## Single-image segmentation

```python
from strands_sapiens import sapiens_seg

result = sapiens_seg(
    input_path="/path/to/person.jpg",
    output_dir="./out",
    model_size="0.4b",
    save_pred=True,
)
print(result["status"])  # "success"
```

You'll get, per image:

- `./out/person.jpg` — side-by-side input + segmentation viz
- `./out/person_seg.npy` — raw `H×W` class-index map (29 classes)

<figure markdown="span">
  ![Segmentation example output](../assets/images/examples/sapiens_sample_seg_vis.jpg){ width="560" loading="lazy" }
  <figcaption>Real output: input (left) vs. segmentation visualization (right) — 0.4b model</figcaption>
</figure>

## Batch over a folder

```python
sapiens_seg(input_path="./photos", output_dir="./out")
```

Picks up `.jpg/.jpeg/.png/.webp/.bmp/.tif/.tiff` from the folder.

## Use from a Strands agent

```python
from strands import Agent
from strands_sapiens import TOOLS

agent = Agent(tools=TOOLS)
agent("Segment every person in ./photos and save to ./out")
```

The agent will:

1. Call `sapiens_info` to see what's available.
2. Pick the best size it has a checkpoint for.
3. Call `sapiens_seg` with the right paths.
4. Return a summary of the output files.

## Chain multiple heads

```python
from strands_sapiens import sapiens_seg, sapiens_normal, sapiens_pose

for tool in (sapiens_seg, sapiens_normal, sapiens_pose):
    tool(input_path="person.jpg", output_dir=f"out/{tool.__name__}")
```

## Structured response

Every tool returns the standard Strands `ToolResult` format:

```python
{
    "status": "success",          # or "error"
    "content": [
        {"text": "seg complete on 1 image(s)"},          # summary
        {"image": {"format": "jpeg", "source": {"bytes": b"..."}}},  # inline vis
        {"json": {                                        # structured data
            "task": "seg",
            "model_size": "0.4b",
            "checkpoint": "/.../sapiens2_0.4b_seg.safetensors",
            "output_dir": "./out",
            "outputs": [
                {"input": "person.jpg", "vis": "./out/person.jpg", "pred": "./out/person_seg.npy"}
            ]
        }}
    ]
}
```

The agent can **read** the text summary, **see** inline visualizations, and **parse** structured JSON for downstream chaining.

On error, `content` contains a text message and optionally a `json` block with `traceback`.

## Next

- [Learn about each head →](../guide/segmentation.md)
- [See example pipelines →](../examples/overview.md)
- [API reference →](../api-reference.md)
