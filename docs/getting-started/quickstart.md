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
print(result["status"], result["output_dir"])
```

You'll get, per image:

- `./out/person.jpg` — side-by-side input + segmentation viz
- `./out/person_seg.npy` — raw `HxW` class-index map (29 classes)

## Batch over a folder

```python
sapiens_seg(input_path="./photos", output_dir="./out")
```

Recurses the folder picking up `.jpg/.jpeg/.png/.webp/.bmp/.tif/.tiff`.

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

Every tool returns:

```python
{
  "status":   "success" | "error",
  "message":  "...",
  "outputs":  [                         # per-image
    {"input": "person.jpg",
     "vis":   "./out/person.jpg",
     "pred":  "./out/person_seg.npy"}
  ],
  "checkpoint":  "...",                 # path used
  "model_size":  "0.4b",
  "task":        "seg",
  "output_dir":  "./out"
}
```

On error you additionally get a `traceback` string — great for agent self-healing loops.

## Next

- [Learn about each head →](../guide/segmentation.md)
- [See example pipelines →](../examples/overview.md)
- [API reference →](../api-reference.md)
