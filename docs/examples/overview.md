# Examples

Practical, runnable examples that compose multiple Sapiens heads.

<figure markdown="span">
  ![All heads on one image](../assets/images/examples/hero_triptych_sm.jpg){ width="100%" loading="lazy" }
  <figcaption>Input → Segmentation → Surface Normals (real output, 0.4b model)</figcaption>
</figure>

<div class="grid cards" markdown>

-   **📁 Batch segmentation on a folder**

    Run 29-class segmentation on every image in a directory and summarize class coverage.

    → [Read](batch-seg.md)

-   **🧍 Pose + seg pipeline**

    Combine pose + segmentation to get *per-limb attention weights* on a dance photo.

    → [Read](pose-seg.md)

-   **🧠 Backbone features for RAG**

    Embed a library of human images with the 0.1B pretrain backbone and retrieve with cosine similarity.

    → [Read](backbone-features.md)

</div>

## Common patterns

### Conditional execution

Only run a head if the checkpoint is available:

```python
from strands_sapiens import sapiens_info, sapiens_seg

info = sapiens_info()
data = info["content"][1]["json"]   # the structured JSON block
if "seg" in data["available"]:
    sapiens_seg(input_path="...", output_dir="...")
else:
    print("No seg checkpoint - download from MODEL_ZOO")
```

### Agent-driven pipeline

Let the agent decide which heads to run:

```python
from strands import Agent
from strands_sapiens import TOOLS

agent = Agent(tools=TOOLS)
agent("""
  For every photo in ./input:
  1. Run seg.
  2. If the subject is in frame, also run pose.
  3. Save results to ./output.
  Summarize at the end.
""")
```

### Error-surface for retry loops

Every tool returns `status: error` with a `traceback` in the JSON block rather than raising.
That makes it trivial to wrap tools in retry logic or self-healing loops:

```python
for attempt in range(3):
    result = sapiens_seg(input_path="bad.jpg", output_dir="out/")
    if result["status"] == "success":
        break
    print(f"Attempt {attempt+1} failed: {result['content'][0]['text']}")
```
