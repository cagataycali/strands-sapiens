"""Shared helpers for strands-sapiens tools."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Tuple

# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINT_ROOT = Path.home() / "sapiens2_host"

VALID_SIZES: Tuple[str, ...] = ("0.1b", "0.4b", "0.8b", "1b", "1b_4k", "5b")

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


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

IMG_EXTS = (".jpg", ".jpeg", ".png")


def resolve_input(input_path: str) -> Tuple[Path, List[Path]]:
    """Return ``(input_dir, images)``.

    Accepts either a directory (all images inside, non-recursive) or a single
    image file (wrapped in a list).
    """
    p = Path(input_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {p}")
    if p.is_dir():
        images = sorted(
            [x for x in p.iterdir() if x.suffix.lower() in IMG_EXTS]
        )
        return p, images
    if p.suffix.lower() in IMG_EXTS:
        return p.parent, [p]
    raise ValueError(f"Unsupported input type: {p}")


def ensure_output(output_dir: str) -> Path:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Response formatting
# ---------------------------------------------------------------------------

def ok(message: str, **extra) -> dict:
    return {"status": "success", "message": message, **extra}


def err(message: str, **extra) -> dict:
    return {"status": "error", "message": message, **extra}
