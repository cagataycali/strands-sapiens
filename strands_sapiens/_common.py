"""Shared helpers for strands-sapiens tools."""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINT_ROOT = Path.home() / "sapiens2_host"

VALID_SIZES: tuple[str, ...] = ("0.1b", "0.4b", "0.8b", "1b", "1b_4k", "5b")

# Which sizes ship for each head (matches upstream MODEL_ZOO).
TASK_SIZES = {
    "pretrain": ("0.1b", "0.4b", "0.8b", "1b", "1b_4k", "5b"),
    "seg":      ("0.4b", "0.8b", "1b", "5b"),
    "normal":   ("0.4b", "0.8b", "1b", "5b"),
    "albedo":   ("0.4b", "0.8b", "1b", "5b"),
    "pointmap": ("0.4b", "0.8b", "1b", "5b"),
    "pose":     ("0.4b", "0.8b", "1b", "5b"),
}


def checkpoint_root() -> Path:
    """Return the active checkpoint root (env var or default)."""
    return Path(os.environ.get("SAPIENS_CHECKPOINT_ROOT", str(DEFAULT_CHECKPOINT_ROOT)))


def checkpoint_path(task: str, size: str) -> Path:
    """Expected absolute path for a checkpoint (may or may not exist)."""
    size = size.lower()
    if task == "pretrain":
        fname = f"sapiens2_{size}_pretrain.safetensors"
    else:
        fname = f"sapiens2_{size}_{task}.safetensors"
    return checkpoint_root() / task / fname


def validate_size(task: str, size: str) -> str:
    """Normalise + validate that a size exists for a given task."""
    size = size.lower()
    if size not in TASK_SIZES.get(task, ()):
        raise ValueError(
            f"Invalid model_size={size!r} for task={task!r}. "
            f"Valid sizes: {', '.join(TASK_SIZES.get(task, ()))}"
        )
    return size


def arch_name(size: str) -> str:
    """Convert size token to the upstream Sapiens2 arch key.

    Upstream arch keys strip the dot:
        0.1b -> sapiens2_01b
        0.4b -> sapiens2_04b
        0.8b -> sapiens2_08b
        1b   -> sapiens2_1b
        5b   -> sapiens2_5b
        1b_4k -> sapiens2_1b   (same arch, different input resolution)
    """
    size = size.lower()
    if size == "1b_4k":
        return "sapiens2_1b"
    return "sapiens2_" + size.replace(".", "")


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


def resolve_input(input_path: str, recursive: bool = False) -> tuple[Path, list[Path]]:
    """Return ``(input_dir, images)``.

    Accepts either a directory (all images inside) or a single image file
    (wrapped in a list).

    Args:
        input_path: file or directory path.
        recursive:  if True and input is a directory, also recurse into subdirs.
    """
    p = Path(input_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {p}")
    if p.is_dir():
        iterator = p.rglob("*") if recursive else p.iterdir()
        images = sorted(
            [x for x in iterator if x.is_file() and x.suffix.lower() in IMG_EXTS]
        )
        return p, images
    if p.suffix.lower() in IMG_EXTS:
        return p.parent, [p]
    raise ValueError(f"Unsupported input type: {p} (expected one of {IMG_EXTS})")


def ensure_output(output_dir: str) -> Path:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def ensure_checkpoint_root() -> tuple[Path, bool]:
    """Return (path, exists) for the checkpoint root."""
    root = checkpoint_root()
    return root, root.exists()


# ---------------------------------------------------------------------------
# Response formatting
# ---------------------------------------------------------------------------

def ok(message: str, **extra) -> dict:
    return {"status": "success", "message": message, **extra}


def err(message: str, **extra) -> dict:
    return {"status": "error", "message": message, **extra}
