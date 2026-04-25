# Pretrain Backbone

Raw dense feature extraction from any Sapiens2 pretrain checkpoint. Useful for downstream heads, RAG, or custom fine-tuning.

## Signature

```python
sapiens_backbone(
    image_path:        str,
    model_size:        str           = "0.1b",   # 0.1b | 0.4b | 0.8b | 1b | 1b_4k | 5b
    img_h:             int           = 1024,
    img_w:             int           = 768,
    device:            str           = "cuda:0",
    save_features_to:  str | None    = None,     # optional .pt dump
    overwrite:         bool          = False,
) -> dict
```

## Example

```python
from strands_sapiens import sapiens_backbone

result = sapiens_backbone(
    image_path="person.jpg",
    model_size="0.1b",
    save_features_to="features/person.pt",
)
print(result["feature_shape"])   # e.g. [1, C, H', W']
```

## 4k variant

For the `1b_4k` pretrain checkpoint, pass its native resolution:

```python
sapiens_backbone(
    image_path="runway.jpg",
    model_size="1b_4k",
    img_h=4096, img_w=3072,
)
```

## Load features back

```python
import torch
feats = torch.load("features/person.pt")  # CPU tensor
```

## Use-cases

- **RAG over human images**: embed once with backbone, index with FAISS, retrieve with cosine similarity.
- **Custom task heads**: freeze backbone, train a small head for your vertical (e.g. clothing brand classification, medical posture).
- **Feature matching across frames**: for tracking or re-identification.

## Output shape handling

Upstream `Sapiens2.forward(...)` can return:

- a single tensor,
- a list/tuple of stage features (multi-scale),
- a dict.

The wrapper always returns the *final-stage* feature map and reports its shape in the response. The raw tensor is only written if you pass `save_features_to`.

## Normalization

The wrapper does BGR→RGB and applies ImageNet mean/std on-device. If you're feeding a pre-processed tensor, skip `sapiens_backbone` and call `Sapiens2` directly — this tool is a "start from a JPG" convenience.

## Related

- [Architecture →](../architecture.md)
- [Backbone features for RAG example →](../examples/backbone-features.md)
