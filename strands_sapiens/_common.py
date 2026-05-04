"""Shared helpers for strands-sapiens tools."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any


# Checkpoint discovery


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



# Input handling


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



# Response formatting - Strands ToolResult convention
#
# Strands tools MUST return:
#   {"status": "success"|"error", "content": [{"text": ...}, {"image": ...}, ...]}
#
# Image content block format (matches Converse API / strands_tools.image_reader):
#   {"image": {"format": "png"|"jpeg"|"webp"|"gif", "source": {"bytes": <bytes>}}}
#
# JSON content block (structured data the model can reason over):
#   {"json": {...}}


def ok(message: str, **extra) -> dict:
    """Build a successful Strands ToolResult.

    Text content is the message. Any extra kwargs are included as a JSON
    content block so the model can inspect structured data.
    """
    content: list[dict[str, Any]] = [{"text": message}]
    if extra:
        content.append({"json": extra})
    return {"status": "success", "content": content}


def err(message: str, **extra) -> dict:
    """Build an error Strands ToolResult."""
    content: list[dict[str, Any]] = [{"text": message}]
    if extra:
        content.append({"json": extra})
    return {"status": "error", "content": content}


def ok_with_images(
    message: str,
    image_paths: list[str | Path] | None = None,
    **extra,
) -> dict:
    """Build a successful ToolResult that includes output images.

    Each image is read from disk and included as an inline image content block
    (same format as strands_tools.image_reader). This allows the model to
    visually inspect tool output.

    Args:
        message: Summary text.
        image_paths: Optional list of output image file paths to include.
        **extra: Additional structured data (becomes a JSON content block).
    """
    content: list[dict[str, Any]] = [{"text": message}]

    if image_paths:
        for img_path in image_paths:
            img_block = _read_image_block(img_path)
            if img_block:
                content.append(img_block)

    if extra:
        content.append({"json": extra})

    return {"status": "success", "content": content}


def _read_image_block(path: str | Path) -> dict | None:
    """Read an image file and return a Strands image content block, or None."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        return None

    suffix = p.suffix.lower()
    format_map = {
        ".png": "png",
        ".jpg": "jpeg",
        ".jpeg": "jpeg",
        ".webp": "webp",
        ".gif": "gif",
    }
    fmt = format_map.get(suffix, "png")

    try:
        with open(p, "rb") as f:
            img_bytes = f.read()
        return {"image": {"format": fmt, "source": {"bytes": img_bytes}}}
    except Exception:
        return None
