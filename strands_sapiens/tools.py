"""Strands ``@tool`` wrappers exposing every Sapiens2 capability.

Each tool is self-contained, imports heavy deps (``torch``, ``sapiens``, ``cv2``)
lazily, and returns a structured ``{"status": ..., ...}`` dict so it composes
cleanly with LLM agents.

Tested on NVIDIA Thor with:
    - sapiens2_0.1b_pretrain.safetensors
    - sapiens2_0.4b_seg.safetensors
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import List, Optional

from strands import tool

from ._common import (
    TASK_SIZES,
    checkpoint_path,
    checkpoint_root,
    ensure_output,
    err,
    ok,
    resolve_input,
    validate_size,
)

# ---------------------------------------------------------------------------
# sapiens_info — environment / checkpoint discovery
# ---------------------------------------------------------------------------

@tool
def sapiens_info() -> dict:
    """Report which Sapiens2 checkpoints are available locally and whether
    CUDA / the ``sapiens`` package are importable.

    Returns:
        A dict with keys:
          - ``checkpoint_root``: resolved path
          - ``available``: mapping of ``task -> [sizes_present]``
          - ``cuda``: {available, device_count, device_name}
          - ``sapiens_package``: bool (can ``import sapiens``)
    """
    root = checkpoint_root()
    available: dict[str, list[str]] = {}
    for task, sizes in TASK_SIZES.items():
        present = [s for s in sizes if checkpoint_path(task, s).exists()]
        if present:
            available[task] = present

    # detector is special
    detector = (root / "detector" / "rtmdet_m.pth").exists()

    try:
        import torch  # type: ignore
        cuda = {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        }
    except Exception as e:  # noqa: BLE001
        cuda = {"available": False, "error": str(e)}

    try:
        import sapiens  # type: ignore  # noqa: F401
        sapiens_ok = True
    except Exception:  # noqa: BLE001
        sapiens_ok = False

    return ok(
        "sapiens info",
        checkpoint_root=str(root),
        available=available,
        detector_present=detector,
        cuda=cuda,
        sapiens_package=sapiens_ok,
    )


# ---------------------------------------------------------------------------
# sapiens_backbone — raw pretrained features
# ---------------------------------------------------------------------------

@tool
def sapiens_backbone(
    image_path: str,
    model_size: str = "0.1b",
    img_h: int = 1024,
    img_w: int = 768,
    device: str = "cuda:0",
    save_features_to: Optional[str] = None,
) -> dict:
    """Run a forward pass through a pretrained Sapiens2 backbone and return
    dense features for a single image.

    Args:
        image_path:         path to an RGB image (jpg/png).
        model_size:         one of ``0.1b | 0.4b | 0.8b | 1b | 1b_4k | 5b``.
        img_h, img_w:       target input size (H, W). Use 4096x3072 for ``1b_4k``.
        device:             torch device string.
        save_features_to:   optional ``.pt`` path to dump the feature tensor.

    Returns:
        dict with feature shape + optional file path.
    """
    try:
        size = validate_size("pretrain", model_size)
        ckpt = checkpoint_path("pretrain", size)
        if not ckpt.exists():
            return err(f"Missing checkpoint: {ckpt}")

        import cv2, numpy as np, torch  # type: ignore
        from safetensors.torch import load_file  # type: ignore
        from sapiens.backbones.standalone.sapiens2 import Sapiens2  # type: ignore

        img_path = Path(image_path).expanduser().resolve()
        if not img_path.is_file():
            return err(f"Image not found: {img_path}")

        arch = f"sapiens2_{size.replace('_', '')}" if size != "1b_4k" else "sapiens2_1b"
        model = Sapiens2(arch=arch, img_size=(img_h, img_w), patch_size=16).eval().to(device)
        model.load_state_dict(load_file(str(ckpt)))

        img = cv2.imread(str(img_path))[:, :, ::-1]  # BGR->RGB
        img = cv2.resize(img, (img_w, img_h))
        x = torch.from_numpy(img.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = ((x - mean) / std).to(device)

        with torch.no_grad():
            features = model(x)[0]

        shape = list(features.shape)
        saved = None
        if save_features_to:
            saved = str(Path(save_features_to).expanduser().resolve())
            Path(saved).parent.mkdir(parents=True, exist_ok=True)
            torch.save(features.cpu(), saved)

        return ok(
            f"Backbone forward pass complete ({arch})",
            feature_shape=shape,
            checkpoint=str(ckpt),
            saved_to=saved,
        )
    except Exception as e:  # noqa: BLE001
        return err(f"sapiens_backbone failed: {e}", traceback=traceback.format_exc())


# ---------------------------------------------------------------------------
# Generic "dense task" runner (seg / normal / albedo / pointmap)
# ---------------------------------------------------------------------------

_DENSE_TASKS = {
    "seg":      ("configs/seg/shutterstock_goliath",      "sapiens2_{size}_seg_shutterstock_goliath-1024x768"),
    "normal":   ("configs/normal/metasim_render_people",   "sapiens2_{size}_normal_metasim_render_people-1024x768"),
    "albedo":   ("configs/albedo/metasim_render_people",   "sapiens2_{size}_albedo_metasim_render_people-1024x768"),
    "pointmap": ("configs/pointmap/metasim_render_people", "sapiens2_{size}_pointmap_metasim_render_people-1024x768"),
}


def _run_dense_task(
    task: str,
    input_path: str,
    output_dir: str,
    model_size: str,
    device: str,
    save_pred: bool,
) -> dict:
    """Core loop for seg/normal/albedo/pointmap — mirrors vis_seg.py semantics."""
    try:
        size = validate_size(task, model_size)
        ckpt = checkpoint_path(task, size)
        if not ckpt.exists():
            return err(f"Missing checkpoint: {ckpt}")

        import cv2, numpy as np, torch  # type: ignore
        import torch.nn.functional as F  # type: ignore

        # Lazy imports from sapiens.dense
        from sapiens.dense.models import init_model  # type: ignore

        # Find config file (packaged inside `sapiens.dense`)
        import sapiens.dense as _dense  # type: ignore
        dense_root = Path(_dense.__file__).parent
        cfg_dir, cfg_tmpl = _DENSE_TASKS[task]
        cfg_file = dense_root / cfg_dir / f"{cfg_tmpl.format(size=size)}.py"
        if not cfg_file.exists():
            return err(f"Missing config file: {cfg_file}")

        model = init_model(str(cfg_file), str(ckpt), device=device)

        in_dir, images = resolve_input(input_path)
        if not images:
            return err(f"No images found in {in_dir}")
        out_dir = ensure_output(output_dir)

        outputs: List[dict] = []
        visualizer = None
        if task == "seg":
            from sapiens.dense.visualizers import SegVisualizer  # type: ignore
            visualizer = SegVisualizer(class_palette_type="dome29", with_labels=False)
        elif task == "normal":
            try:
                from sapiens.dense.visualizers import NormalVisualizer  # type: ignore
                visualizer = NormalVisualizer()
            except Exception:
                visualizer = None
        elif task == "albedo":
            try:
                from sapiens.dense.visualizers import AlbedoVisualizer  # type: ignore
                visualizer = AlbedoVisualizer()
            except Exception:
                visualizer = None

        for img_path in images:
            image = cv2.imread(str(img_path))
            if image is None:
                outputs.append({"input": str(img_path), "status": "skipped_unreadable"})
                continue

            data = model.pipeline(dict(img=image))
            data = model.data_preprocessor(data)
            inputs = data["inputs"]

            with torch.no_grad():
                logits = model(inputs)

            logits = F.interpolate(logits, size=image.shape[:2], mode="bilinear")
            base = out_dir / img_path.stem
            ext = img_path.suffix.lstrip(".")

            if task == "seg":
                pred_labels = logits.argmax(dim=1).cpu().numpy().squeeze(0)
                vis_seg = visualizer._visualize_segmentation(image, pred_labels)
                vis_image = np.concatenate([image, vis_seg], axis=1)
                vis_path = base.with_suffix(f".{ext}")
                cv2.imwrite(str(vis_path), vis_image)
                entry = {"input": str(img_path), "vis": str(vis_path)}
                if save_pred:
                    npy_path = base.parent / f"{base.name}_seg.npy"
                    np.save(str(npy_path), pred_labels)
                    entry["pred"] = str(npy_path)
                outputs.append(entry)
            else:
                pred = logits.cpu().numpy().squeeze(0)  # C x H x W
                npy_path = base.parent / f"{base.name}_{task}.npy"
                if save_pred:
                    np.save(str(npy_path), pred)
                entry = {"input": str(img_path), "pred": str(npy_path) if save_pred else None}
                # Best-effort visualization
                if visualizer is not None and hasattr(visualizer, "_visualize"):
                    try:
                        vis = visualizer._visualize(image, pred)  # type: ignore
                        vis_path = base.with_suffix(f".{ext}")
                        cv2.imwrite(str(vis_path), np.concatenate([image, vis], axis=1))
                        entry["vis"] = str(vis_path)
                    except Exception:
                        pass
                outputs.append(entry)

        return ok(
            f"{task} complete on {len(outputs)} image(s)",
            task=task,
            model_size=size,
            checkpoint=str(ckpt),
            output_dir=str(out_dir),
            outputs=outputs,
        )
    except Exception as e:  # noqa: BLE001
        return err(f"sapiens_{task} failed: {e}", traceback=traceback.format_exc())


# ---------------------------------------------------------------------------
# Public dense tools
# ---------------------------------------------------------------------------

@tool
def sapiens_seg(
    input_path: str,
    output_dir: str,
    model_size: str = "0.4b",
    device: str = "cuda:0",
    save_pred: bool = True,
) -> dict:
    """29-class body-part segmentation.

    Classes (see ``docs/SEG.md``): background, apparel, eyeglass, face_neck,
    hair, feet/hands/arms/legs (L/R), torso, upper/lower clothing, shoes, socks,
    lips, teeth, tongue.

    Args:
        input_path:  image file OR directory of images.
        output_dir:  where to save visualizations (and ``*_seg.npy`` if ``save_pred``).
        model_size:  ``0.4b | 0.8b | 1b | 5b`` (requires matching checkpoint).
        device:      torch device string.
        save_pred:   write raw predicted label maps to ``.npy``.
    """
    return _run_dense_task("seg", input_path, output_dir, model_size, device, save_pred)


@tool
def sapiens_normal(
    input_path: str,
    output_dir: str,
    model_size: str = "0.4b",
    device: str = "cuda:0",
    save_pred: bool = True,
) -> dict:
    """Per-pixel surface-normal estimation.

    Output channels represent the (x, y, z) normal vector in camera space.
    """
    return _run_dense_task("normal", input_path, output_dir, model_size, device, save_pred)


@tool
def sapiens_albedo(
    input_path: str,
    output_dir: str,
    model_size: str = "0.4b",
    device: str = "cuda:0",
    save_pred: bool = True,
) -> dict:
    """Per-pixel albedo (intrinsic color, illumination-invariant) estimation."""
    return _run_dense_task("albedo", input_path, output_dir, model_size, device, save_pred)


@tool
def sapiens_pointmap(
    input_path: str,
    output_dir: str,
    model_size: str = "0.4b",
    device: str = "cuda:0",
    save_pred: bool = True,
) -> dict:
    """3D pointmap estimation — lifts each pixel to a 3D point in camera space.

    Install the optional ``[pointmap]`` extra for ``open3d``-based visualization.
    """
    return _run_dense_task("pointmap", input_path, output_dir, model_size, device, save_pred)


# ---------------------------------------------------------------------------
# sapiens_pose — top-down 308-keypoint pose (requires RTMDet detector)
# ---------------------------------------------------------------------------

@tool
def sapiens_pose(
    input_path: str,
    output_dir: str,
    model_size: str = "0.4b",
    device: str = "cuda:0",
    kpt_thres: float = 0.3,
    line_thickness: int = 2,
    radius: int = 3,
) -> dict:
    """308-keypoint 2D pose estimation (face 274 + body + hands + feet).

    Top-down: uses RTMDet person detector checkpoint at
    ``$SAPIENS_CHECKPOINT_ROOT/detector/rtmdet_m.pth``.

    Args:
        input_path:       image file OR directory.
        output_dir:       where to write visualized images + per-image JSONs.
        model_size:       ``0.4b | 0.8b | 1b | 5b``.
        device:           torch device.
        kpt_thres:        keypoint confidence threshold for visualization.
        line_thickness:   skeleton line thickness (px).
        radius:           keypoint dot radius (px).
    """
    try:
        size = validate_size("pose", model_size)
        ckpt = checkpoint_path("pose", size)
        if not ckpt.exists():
            return err(f"Missing checkpoint: {ckpt}")

        detector_ckpt = checkpoint_root() / "detector" / "rtmdet_m.pth"
        if not detector_ckpt.exists():
            return err(
                f"Missing person detector: {detector_ckpt}. Download from "
                "huggingface.co/facebook/sapiens-pose-bbox-detector"
            )

        import cv2, json, numpy as np, torch  # type: ignore
        import sapiens.pose as _pose  # type: ignore

        pose_root = Path(_pose.__file__).parent
        cfg_file = (
            pose_root
            / "configs"
            / "keypoints308"
            / f"sapiens2_{size}_keypoints308-1024x768.py"
        )
        if not cfg_file.exists():
            # fallback search
            candidates = list((pose_root / "configs").rglob(f"*{size}*keypoints308*.py"))
            if not candidates:
                return err(f"Pose config not found for size={size} under {pose_root}")
            cfg_file = candidates[0]

        # Prefer upstream high-level API if exposed; otherwise fall back to script call.
        try:
            from sapiens.pose.models import init_pose_model  # type: ignore
            from sapiens.pose.visualizers import PoseVisualizer  # type: ignore

            model = init_pose_model(str(cfg_file), str(ckpt), device=device)
            vis = PoseVisualizer(
                kpt_thres=kpt_thres, line_thickness=line_thickness, radius=radius
            )
            in_dir, images = resolve_input(input_path)
            out_dir = ensure_output(output_dir)
            outputs = []
            for img_path in images:
                image = cv2.imread(str(img_path))
                if image is None:
                    outputs.append({"input": str(img_path), "status": "skipped"})
                    continue
                result = model.infer(image, detector_checkpoint=str(detector_ckpt))
                vis_img = vis.draw(image, result)
                out_img = out_dir / img_path.name
                cv2.imwrite(str(out_img), vis_img)
                out_json = out_dir / f"{img_path.stem}.json"
                out_json.write_text(json.dumps(result, default=str))
                outputs.append({"input": str(img_path), "vis": str(out_img), "kpts": str(out_json)})
            return ok(
                f"pose complete on {len(outputs)} image(s)",
                task="pose",
                model_size=size,
                outputs=outputs,
            )
        except ImportError:
            return err(
                "sapiens.pose high-level API not available in installed version. "
                "Use the upstream script directly: "
                f"`cd $SAPIENS_ROOT/sapiens/pose && ./scripts/demo/keypoints308.sh` "
                f"with CHECKPOINT={ckpt}"
            )
    except Exception as e:  # noqa: BLE001
        return err(f"sapiens_pose failed: {e}", traceback=traceback.format_exc())
