# Backbone features for RAG

Build a retrieval index over a library of human images using Sapiens2's pretrain backbone as a dense visual encoder. No fine-tuning required — Sapiens2 was pretrained on 1B human images and its features are already discriminative for human-centric similarity.

## The plan

1. Walk a folder of images.
2. For each, call `sapiens_backbone` and pool the feature map to a fixed-size vector.
3. Store vectors + paths.
4. At query time, embed a query image the same way and rank by cosine similarity.

## Code

```python
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from strands_sapiens import sapiens_backbone

# 1) Embed the corpus
corpus = list(Path("./library").glob("*.jpg"))
index  = []          # list[(path, vector)]

for path in corpus:
    result = sapiens_backbone(
        image_path=str(path),
        model_size="0.1b",
        save_features_to=f"/tmp/feats/{path.stem}.pt",
        overwrite=True,
    )
    if result["status"] != "success":
        continue
    feats = torch.load(f"/tmp/feats/{path.stem}.pt")   # (1, C, H', W')
    # Global average pool → (C,)
    vec = feats.mean(dim=(0, 2, 3))
    vec = F.normalize(vec, dim=0)
    index.append((str(path), vec))

print(f"Indexed {len(index)} images")

# 2) Query
def embed(path):
    sapiens_backbone(image_path=path, model_size="0.1b",
                     save_features_to="/tmp/query.pt", overwrite=True)
    f = torch.load("/tmp/query.pt")
    return F.normalize(f.mean(dim=(0, 2, 3)), dim=0)

q = embed("./query.jpg")
scores = [(p, float((v * q).sum())) for p, v in index]
scores.sort(key=lambda x: -x[1])

# Top 5
for path, score in scores[:5]:
    print(f"{score:.3f}  {path}")
```

## Sample output

```
Indexed 1,024 images
0.987  ./library/person_0042.jpg
0.974  ./library/person_0310.jpg
0.971  ./library/person_0127.jpg
0.968  ./library/person_0989.jpg
0.963  ./library/person_0003.jpg
```

## Why this is a good idea

- **No labels, no fine-tuning** — Sapiens2's SSL objective already separates "same person" / "same pose" / "same clothing" signals.
- **Cheap**: `0.1b` runs at ~50ms/image on a modern GPU.
- **Compose with seg**: mask out background before pooling to get identity-focused vectors.

## Scale it

- Swap the pure-Python argmax loop for **FAISS**: `faiss.IndexFlatIP(d)`.
- **Quantize** to int8 for 4× memory savings with <1% recall drop.
- **Per-region features**: don't pool globally — keep spatial tokens and do masked attention retrieval.

## Related

- [Pretrain backbone guide →](../guide/backbone.md)
