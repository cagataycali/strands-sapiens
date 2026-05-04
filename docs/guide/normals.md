# Surface Normals

Per-pixel surface-normal estimation. Output channels are the (x, y, z) components of the normal vector in camera space.

<figure markdown="span">
  ![Normal estimation output](../assets/images/examples/normal_overlay.jpg){ width="480" loading="lazy" }
  <figcaption>Real <code>sapiens_normal</code> output (0.4b) — surface normals mapped to RGB, blended with input</figcaption>
</figure>

## Signature

```python
sapiens_normal(
    input_path:  str,
    output_dir:  str,
    model_size:  str = "0.4b",
    device:      str = "cuda:0",
    save_pred:   bool = True,
) -> dict
```

## Example

```python
from strands_sapiens import sapiens_normal

sapiens_normal(
    input_path="person.jpg",
    output_dir="out/",
    model_size="0.4b",
)
```

Output:

- `out/person.jpg` - side-by-side input vs. remapped RGB normals
- `out/person_normal.npy` - `3×H×W` float array, unit-normalized

<div class="grid" markdown>

<figure markdown="span">
  ![Input image](../assets/images/examples/input.jpg){ width="240" loading="lazy" }
  <figcaption>Input</figcaption>
</figure>

<figure markdown="span">
  ![Normal visualization](../assets/images/examples/normal_vis.jpg){ width="240" loading="lazy" }
  <figcaption>Surface normals (RGB-mapped)</figcaption>
</figure>

<figure markdown="span">
  ![Normal overlay](../assets/images/examples/normal_overlay.jpg){ width="240" loading="lazy" }
  <figcaption>Blended overlay</figcaption>
</figure>

</div>

## Visualization

The viz maps each normal `(x, y, z) ∈ [-1, 1]³` to RGB via

```
rgb = (normal * 0.5 + 0.5) * 255
```

Red channel = x (horizontal), Green = y (vertical), Blue = z (depth / toward camera).

Flat walls facing the camera show up as `~(128, 128, 255)`.

## Use-cases

- **Relighting / portrait studio**: combine with `albedo` → relight photos after the fact.
- **Clothing wrinkle analysis**: normals are sensitive to fine fabric detail.
- **AR filter preprocessing**: drive shading in real time without a depth sensor.

## Consume the raw .npy

```python
import numpy as np
n = np.load("out/person_normal.npy")  # (3, H, W)
# unit-normalize (already close but re-do for safety)
norm = np.linalg.norm(n, axis=0, keepdims=True)
n = n / np.clip(norm, 1e-6, None)
```

## Related

- [Albedo →](albedo.md) · [Pointmap →](pointmap.md)

<figure markdown="span">
  ![Normals pipeline](../assets/diagrams/normals-pipeline.svg){ width="100%" loading="lazy" }
  <figcaption>Normals data flow</figcaption>
</figure>
