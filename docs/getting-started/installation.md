# Installation

Strands Sapiens has three install steps because Sapiens2 depends on a CUDA-enabled PyTorch that's platform-specific.

## 1. CUDA PyTorch

Install a CUDA-enabled PyTorch matching your hardware. Some examples:

=== "CUDA 12.4 (Linux x86_64)"

    ```bash
    pip install torch torchvision \
      --index-url https://download.pytorch.org/whl/cu124
    ```

=== "JetPack 6 / Thor (aarch64)"

    ```bash
    # Use the NVIDIA-provided PyTorch wheels for JetPack.
    # See: https://developer.nvidia.com/embedded/downloads
    pip install --no-cache $TORCH_WHEEL_URL
    ```

=== "CPU-only (no GPU)"

    ```bash
    pip install torch torchvision
    # Note: Sapiens2 inference on CPU will be slow but functional.
    ```

## 2. Sapiens2 (upstream)

```bash
pip install git+https://github.com/facebookresearch/sapiens2.git
```

The `sapiens` package provides:

- `sapiens.backbones.standalone.sapiens2.Sapiens2` - raw backbone
- `sapiens.dense` - seg / normal / albedo / pointmap
- `sapiens.pose` - 308-keypoint top-down pose

## 3. Strands Sapiens

=== "From GitHub"

    ```bash
    pip install git+ssh://git@github.com/cagataycali/strands-sapiens.git
    ```

=== "From a local clone"

    ```bash
    git clone git@github.com:cagataycali/strands-sapiens.git
    cd strands-sapiens
    pip install -e .
    ```

=== "Dev install"

    ```bash
    pip install -e '.[dev]'
    pytest -q
    ```

## 4. Verify

```bash
python -c "from strands_sapiens import sapiens_info; import json; print(json.dumps(sapiens_info(), indent=2))"
```

You should see:

```json
{
  "status": "success",
  "content": [
    {"text": "sapiens info"},
    {"json": {
      "checkpoint_root": "/home/you/sapiens2_host",
      "checkpoint_root_exists": false,
      "available": {},
      "detector_present": false,
      "detector_type": "none",
      "cuda": { "available": true, "device_count": 1, "device_name": "NVIDIA ..." },
      "sapiens_package": true
    }}
  ]
}
```

If `sapiens_package` is `false`, go back to step 2. If `cuda.available` is `false`, go back to step 1.

The next thing to do is [download your first checkpoint →](checkpoints.md)
