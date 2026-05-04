# Jetson / Thor Deployment

Strands Sapiens is developed on NVIDIA Thor (JetPack 6, aarch64) - notes from that journey.

## Baseline verified config

- **Hardware**: NVIDIA Jetson AGX Thor
- **OS**: JetPack 6 (Ubuntu 22.04, aarch64)
- **Python**: 3.10 (JetPack default)
- **PyTorch**: 2.7+ CUDA wheels (from NVIDIA's JetPack download page)
- **Checkpoints tested**:
    - `sapiens2_0.1b_pretrain.safetensors`
    - `sapiens2_0.4b_seg.safetensors`

End-to-end seg inference runs cleanly at 1024×768 on this config.

## Install checklist

```bash
# 0) Confirm JetPack + CUDA
python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"

# 1) Install sapiens2 from source
pip install git+https://github.com/facebookresearch/sapiens2.git

# 2) Install strands-sapiens
pip install -e .

# 3) Put at least one checkpoint in place
mkdir -p ~/sapiens2_host/seg
# copy sapiens2_0.4b_seg.safetensors there

# 4) Sanity-check
python -c "from strands_sapiens import sapiens_info; print(sapiens_info())"
```

## Python version

JetPack 6 ships Python 3.10 by default. `strands-sapiens` requires **Python ≥ 3.10**, so no pyenv gymnastics needed.

If you're running newer Python, nothing breaks - tests pass on 3.12 too.

## Performance

| Size   | Resolution | Single-image latency (Thor) |
|--------|------------|------------------------------|
| 0.4b   | 1024×768   | ~250ms                       |
| 0.8b   | 1024×768   | ~450ms                       |
| 1b     | 1024×768   | ~700ms                       |

(Informal; no TensorRT conversion applied yet. Expect 2–3× with TRT.)

## Running as a service

Use devduck's `service` tool to persist a sapiens inference server:

```python
service(
    action='install',
    name='sapiens-worker',
    tools='strands_sapiens:sapiens_seg,sapiens_pose',
    env_vars={'SAPIENS_CHECKPOINT_ROOT': '/data/sapiens2_host'},
    startup_prompt='Wait for requests on port 9010 and run inference.'
)
```

## Gotchas on aarch64

- Install `opencv-python` (the wheel exists for aarch64) - **don't** rebuild opencv unless you need GPU decoding.
- `open3d` (optional `[pointmap]` extra) may need a source build on some JetPack versions.
- If you see `Illegal instruction (core dumped)` on a fresh install, you're on an older `safetensors` - upgrade.

## Related

- [Checkpoints →](../getting-started/checkpoints.md)
- [Architecture →](../architecture.md)
