"""Strands ``@tool`` wrappers exposing every Sapiens2 capability.

Each tool is self-contained, imports heavy deps (``torch``, ``sapiens``, ``cv2``)
lazily, and returns a structured ``{"status": ..., ...}`` dict so it composes
cleanly with LLM agents.

Tested on NVIDIA Thor with:
    - sapiens2_0.1b_pretrain.safetensors
    - sapiens2_0.4b_seg.safetensors
"""

from __future__ import annotations

import traceback
from pathlib import Path

from strands import tool

from ._common import (
    TASK_SIZES,
    arch_name,
    checkpoint_path,
    checkpoint_root,
    ensure_checkpoint_root,
    ensure_output,
    err,
    ok,
    ok_with_images,
    resolve_input,
    validate_size,
)


# sapiens_info - environment / checkpoint discovery


@tool
def sapiens_info() -> dict:
    """Report which Sapiens2 checkpoints are available locally and whether
    CUDA / the ``sapiens`` package are importable.

    Returns:
        A dict with keys:
          - ``checkpoint_root``: resolved path
          - ``checkpoint_root_exists``: bool
          - ``available``: mapping of ``task -> [sizes_present]``
          - ``detector_present``: bool (rtmdet_m.pth)
          - ``cuda``: {available, device_count, device_name}
          - ``sapiens_package``: bool (can ``import sapiens``)
    """
    root, root_exists = ensure_checkpoint_root()
    available: dict[str, list[str]] = {}
    if root_exists:
        for task, sizes in TASK_SIZES.items():
            present = [s for s in sizes if checkpoint_path(task, s).exists()]
            if present:
                available[task] = present

    # Sapiens2 uses DETR-ResNet-101-DC5 (HuggingFace) for person detection.
    # Legacy sapiens v1 used rtmdet_m.pth — check both for compatibility.
    detector_detr = (root / "detector" / "detr-resnet-101-dc5").is_dir() if root_exists else False
    detector_rtmdet = (root / "detector" / "rtmdet_m.pth").exists() if root_exists else False
    detector = detector_detr or detector_rtmdet

    cuda: dict = {"available": False, "device_count": 0, "device_name": None}
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            cuda = {
                "available": True,
                "device_count": int(torch.cuda.device_count()),
                "device_name": torch.cuda.get_device_name(0),
            }
    except Exception as e:  # noqa: BLE001
        cuda = {"available": False, "device_count": 0, "device_name": None, "error": str(e)}

    try:
        import sapiens  # type: ignore  # noqa: F401
        sapiens_ok = True
    except Exception:  # noqa: BLE001
        sapiens_ok = False

    return ok(
        "sapiens info",
        checkpoint_root=str(root),
        checkpoint_root_exists=root_exists,
        available=available,
        detector_present=detector,
        detector_type="detr-resnet-101-dc5" if detector_detr else ("rtmdet_m" if detector_rtmdet else "none"),
        cuda=cuda,
        sapiens_package=sapiens_ok,
    )



# sapiens_backbone - raw pretrained features


@tool
def sapiens_backbone(
    image_path: str,
    model_size: str = "0.1b",
    img_h: int = 1024,
    img_w: int = 768,
    device: str = "cuda:0",
    save_features_to: str | None = None,
    overwrite: bool = False,
) -> dict:
    """Run a forward pass through a pretrained Sapiens2 backbone and return
    dense features for a single image.

    Args:
        image_path:         path to an RGB image (jpg/png/webp/bmp/tif).
        model_size:         one of ``0.1b | 0.4b | 0.8b | 1b | 1b_4k | 5b``.
        img_h, img_w:       target input size (H, W). Use 4096x3072 for ``1b_4k``.
        device:             torch device string.
        save_features_to:   optional ``.pt`` path to dump the feature tensor.
        overwrite:          if False (default), error instead of overwriting
                            an existing ``save_features_to`` file.

    Returns:
        dict with feature shape + optional file path.
    """
    try:
        size = validate_size("pretrain", model_size)
        ckpt = checkpoint_path("pretrain", size)
        if not ckpt.exists():
            return err(f"Missing checkpoint: {ckpt}")

        import cv2  # type: ignore
        import torch  # type: ignore
        from safetensors.torch import load_file  # type: ignore
        from sapiens.backbones.standalone.sapiens2 import Sapiens2  # type: ignore

        img_path = Path(image_path).expanduser().resolve()
        if not img_path.is_file():
            return err(f"Image not found: {img_path}")

        if save_features_to and not overwrite:
            if Path(save_features_to).expanduser().resolve().exists():
                return err(
                    f"Refusing to overwrite {save_features_to} "
                    "(pass overwrite=True)"
                )

        arch = arch_name(size)
        model = (
            Sapiens2(arch=arch, img_size=(img_h, img_w), patch_size=16)
            .eval()
            .to(device)
        )
        model.load_state_dict(load_file(str(ckpt)))

        # BGR->RGB via explicit conversion (avoids stride/view issues).
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return err(f"cv2.imread returned None for {img_path}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_w, img_h))
        x = (
            torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device)
            / 255.0
        )
        # ImageNet normalization (on-device).
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        x = (x - mean) / std

        with torch.no_grad():
            out = model(x)

        # Sapiens2.forward may return a tensor, a list/tuple of stage features,
        # or a dict. Prefer the final stage for downstream use.
        if isinstance(out, (list, tuple)):
            features = out[-1]
        elif isinstance(out, dict):
            features = out.get("features", out.get("out", next(iter(out.values()))))
        else:
            features = out

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



# Inline visualizers (don't rely on upstream private APIs)


def _normal_to_rgb(normal_chw):
    """Map (3,H,W) normals in [-1,1] to a BGR uint8 image for display."""
    import numpy as np
    n = normal_chw
    # l2-normalize across channel axis; guard against zeros.
    denom = np.linalg.norm(n, axis=0, keepdims=True)
    denom = np.where(denom < 1e-6, 1.0, denom)
    n = n / denom
    rgb = ((n.transpose(1, 2, 0) * 0.5 + 0.5) * 255.0).clip(0, 255).astype("uint8")
    # cv2 writes BGR
    return rgb[:, :, ::-1]


def _albedo_to_rgb(albedo_chw):
    """Map (3,H,W) albedo in [0,1] (or arbitrary float) to BGR uint8."""
    import numpy as np
    a = albedo_chw.transpose(1, 2, 0)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-6:
        a_norm = np.zeros_like(a)
    else:
        a_norm = (a - lo) / (hi - lo)
    rgb = (a_norm * 255.0).clip(0, 255).astype("uint8")
    return rgb[:, :, ::-1]


def _pointmap_to_rgb(pm_chw):
    """Visualize a (3,H,W) pointmap as a false-color depth image (from z)."""
    import cv2  # type: ignore
    import numpy as np
    z = pm_chw[2]
    lo, hi = float(np.percentile(z, 2)), float(np.percentile(z, 98))
    if hi - lo < 1e-6:
        z_norm = np.zeros_like(z, dtype="uint8")
    else:
        z_norm = ((z - lo) / (hi - lo) * 255.0).clip(0, 255).astype("uint8")
    return cv2.applyColorMap(z_norm, cv2.COLORMAP_TURBO)



# Generic "dense task" runner (seg / normal / albedo / pointmap)


_DENSE_TASKS = {
    "seg":      ("configs/seg/shutterstock_goliath",       "sapiens2_{size}_seg_shutterstock_goliath-1024x768"),
    "normal":   ("configs/normal/metasim_render_people",   "sapiens2_{size}_normal_metasim_render_people-1024x768"),
    "albedo":   ("configs/albedo/render_people",           "sapiens2_{size}_albedo_render_people-1024x768"),
    "pointmap": ("configs/pointmap/render_people",         "sapiens2_{size}_pointmap_render_people-1024x768"),
}


def _find_dense_config(task: str, size: str):
    """Resolve the config file for a dense task, with a robust rglob fallback.

    Returns:
        (cfg_file: Path, dense_root: Path) on success.
    Raises:
        FileNotFoundError if no plausible config found.
    """
    import sapiens.dense as _dense  # type: ignore
    dense_root = Path(_dense.__file__).parent
    cfg_dir, cfg_tmpl = _DENSE_TASKS[task]
    cfg_file = dense_root / cfg_dir / f"{cfg_tmpl.format(size=size)}.py"
    if cfg_file.exists():
        return cfg_file, dense_root
    # Fallback: glob for anything matching `sapiens2_{size}_{task}*.py` under the
    # task's config tree. Handles upstream dataset renames.
    task_dir = dense_root / "configs" / task
    if task_dir.is_dir():
        candidates = sorted(task_dir.rglob(f"sapiens2_{size}_{task}*.py"))
        if candidates:
            return candidates[0], dense_root
    raise FileNotFoundError(
        f"No config found for task={task} size={size} under {dense_root}. "
        f"Tried {cfg_file}."
    )


def _run_dense_task(
    task: str,
    input_path: str,
    output_dir: str,
    model_size: str,
    device: str,
    save_pred: bool,
) -> dict:
    """Core loop for seg/normal/albedo/pointmap - mirrors upstream vis scripts."""
    try:
        size = validate_size(task, model_size)
        ckpt = checkpoint_path(task, size)
        if not ckpt.exists():
            return err(f"Missing checkpoint: {ckpt}")

        import cv2  # type: ignore
        import numpy as np  # type: ignore
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore
        from sapiens.dense.models import init_model  # type: ignore

        try:
            cfg_file, _dense_root = _find_dense_config(task, size)
        except FileNotFoundError as e:
            return err(str(e))

        model = init_model(str(cfg_file), str(ckpt), device=device)

        in_dir, images = resolve_input(input_path)
        if not images:
            return err(f"No images found in {in_dir}")
        out_dir = ensure_output(output_dir)

        # Optional seg visualizer from upstream (public API if present).
        seg_visualizer = None
        if task == "seg":
            try:
                from sapiens.dense.visualizers import SegVisualizer  # type: ignore
                seg_visualizer = SegVisualizer(
                    class_palette_type="dome29", with_labels=False
                )
            except Exception:  # noqa: BLE001
                seg_visualizer = None

        outputs: list[dict] = []
        for img_path in images:
            image = cv2.imread(str(img_path))
            if image is None:
                outputs.append({"input": str(img_path), "status": "skipped_unreadable"})
                continue

            # Upstream mm*-style pipeline: build a data dict, preprocess, forward.
            try:
                data = model.pipeline(dict(img=image))
                data = model.data_preprocessor(data)
            except AttributeError:
                outputs.append({
                    "input": str(img_path),
                    "status": "skipped_api_mismatch",
                    "note": "installed sapiens.dense init_model() lacks "
                            ".pipeline/.data_preprocessor; wrapper needs update",
                })
                continue

            inputs = data["inputs"]
            data_samples = data.get("data_samples")

            with torch.no_grad():
                try:
                    raw_output = model(inputs)
                except TypeError:
                    raw_output = model(inputs, data_samples)

            # -----------------------------------------------------------
            # Task-specific post-processing (matching upstream vis scripts)
            # -----------------------------------------------------------
            base = out_dir / img_path.stem
            ext = img_path.suffix.lstrip(".") or "jpg"

            if task == "pointmap":
                # Upstream vis_pointmap.py: model returns (pointmap, scale)
                if isinstance(raw_output, (list, tuple)) and len(raw_output) == 2:
                    logits, scale = raw_output
                    logits = logits / scale  # convert to metric pointmap
                else:
                    logits = raw_output

                # Remove padding added by pipeline (upstream does this)
                if data_samples and "meta" in data_samples:
                    pad = data_samples["meta"].get("padding_size")
                    if pad is not None:
                        pad_left, pad_right, pad_top, pad_bottom = pad
                        logits = logits[
                            :, :,
                            pad_top : inputs.shape[2] - pad_bottom,
                            pad_left : inputs.shape[3] - pad_right,
                        ]

                logits = F.interpolate(logits, size=image.shape[:2], mode="bilinear")
                pred = logits.cpu().numpy().squeeze(0)  # 3 x H x W
                entry = {"input": str(img_path)}
                if save_pred:
                    npy_path = base.parent / f"{base.name}_pointmap.npy"
                    np.save(str(npy_path), pred)
                    entry["pred"] = str(npy_path)

                    # Export .ply point cloud (upstream does this via open3d)
                    try:
                        import open3d as o3d  # type: ignore
                        pm_hwc = pred.transpose(1, 2, 0)  # H x W x 3
                        mask = np.ones(image.shape[:2], dtype=bool)
                        points = pm_hwc[mask].reshape(-1, 3)
                        colors = image[mask] / 255.0
                        colors = colors[:, [2, 1, 0]]  # BGR -> RGB
                        pc = o3d.geometry.PointCloud()
                        pc.points = o3d.utility.Vector3dVector(points)
                        pc.colors = o3d.utility.Vector3dVector(colors)
                        ply_path = str(base.parent / f"{base.name}.ply")
                        o3d.io.write_point_cloud(ply_path, pc)
                        entry["ply"] = ply_path
                    except ImportError:
                        pass  # open3d optional
                    except Exception as ply_err:
                        entry["ply_error"] = str(ply_err)

                # Depth visualization
                try:
                    vis = _pointmap_to_rgb(pred)
                    if vis.shape[:2] != image.shape[:2]:
                        vis = cv2.resize(vis, (image.shape[1], image.shape[0]))
                    vis_path = base.with_suffix(f".{ext}")
                    cv2.imwrite(str(vis_path), np.concatenate([image, vis], axis=1))
                    entry["vis"] = str(vis_path)
                except Exception as ve:
                    entry["vis_error"] = str(ve)
                outputs.append(entry)

            elif task == "albedo":
                logits = raw_output
                logits = logits.clamp(0, 1)  # upstream clamps to [0,1]

                # Remove padding (upstream does this)
                if data_samples and "meta" in data_samples:
                    pad = data_samples["meta"].get("padding_size")
                    if pad is not None:
                        pad_left, pad_right, pad_top, pad_bottom = pad
                        logits = logits[
                            :, :,
                            pad_top : inputs.shape[2] - pad_bottom,
                            pad_left : inputs.shape[3] - pad_right,
                        ]

                logits = F.interpolate(logits, size=image.shape[:2], mode="bilinear")
                pred = logits.cpu().numpy().squeeze(0)  # 3 x H x W
                entry = {"input": str(img_path)}
                if save_pred:
                    npy_path = base.parent / f"{base.name}_albedo.npy"
                    np.save(str(npy_path), pred)
                    entry["pred"] = str(npy_path)

                try:
                    vis = _albedo_to_rgb(pred)
                    if vis.shape[:2] != image.shape[:2]:
                        vis = cv2.resize(vis, (image.shape[1], image.shape[0]))
                    vis_path = base.with_suffix(f".{ext}")
                    cv2.imwrite(str(vis_path), np.concatenate([image, vis], axis=1))
                    entry["vis"] = str(vis_path)
                except Exception as ve:
                    entry["vis_error"] = str(ve)
                outputs.append(entry)

            elif task == "seg":
                logits = raw_output
                logits = F.interpolate(logits, size=image.shape[:2], mode="bilinear")
                pred_labels = logits.argmax(dim=1).cpu().numpy().squeeze(0)
                entry = {"input": str(img_path)}
                if seg_visualizer is not None:
                    try:
                        if hasattr(seg_visualizer, "visualize"):
                            vis_seg = seg_visualizer.visualize(image, pred_labels)
                        elif callable(seg_visualizer):
                            vis_seg = seg_visualizer(image, pred_labels)
                        else:
                            vis_seg = seg_visualizer._visualize_segmentation(
                                image, pred_labels
                            )
                        if vis_seg.shape[:2] != image.shape[:2]:
                            vis_seg = cv2.resize(
                                vis_seg, (image.shape[1], image.shape[0])
                            )
                        vis_image = np.concatenate([image, vis_seg], axis=1)
                        vis_path = base.with_suffix(f".{ext}")
                        cv2.imwrite(str(vis_path), vis_image)
                        entry["vis"] = str(vis_path)
                    except Exception as ve:
                        entry["vis_error"] = str(ve)
                if save_pred:
                    npy_path = base.parent / f"{base.name}_seg.npy"
                    np.save(str(npy_path), pred_labels)
                    entry["pred"] = str(npy_path)
                outputs.append(entry)

            else:
                # normal (and any future dense tasks)
                logits = raw_output
                logits = F.interpolate(logits, size=image.shape[:2], mode="bilinear")
                pred = logits.cpu().numpy().squeeze(0)  # C x H x W
                entry = {"input": str(img_path)}
                if save_pred:
                    npy_path = base.parent / f"{base.name}_{task}.npy"
                    np.save(str(npy_path), pred)
                    entry["pred"] = str(npy_path)

                try:
                    if task == "normal":
                        vis = _normal_to_rgb(pred)
                    else:
                        vis = None
                    if vis is not None:
                        if vis.shape[:2] != image.shape[:2]:
                            vis = cv2.resize(vis, (image.shape[1], image.shape[0]))
                        vis_path = base.with_suffix(f".{ext}")
                        cv2.imwrite(
                            str(vis_path), np.concatenate([image, vis], axis=1)
                        )
                        entry["vis"] = str(vis_path)
                except Exception as ve:
                    entry["vis_error"] = str(ve)
                outputs.append(entry)

        # Collect visualization image paths for inline display
        vis_paths = [o["vis"] for o in outputs if "vis" in o]

        return ok_with_images(
            f"{task} complete on {len(outputs)} image(s)",
            image_paths=vis_paths[:5],
            task=task,
            model_size=size,
            checkpoint=str(ckpt),
            output_dir=str(out_dir),
            outputs=outputs,
        )
    except Exception as e:  # noqa: BLE001
        return err(f"sapiens_{task} failed: {e}", traceback=traceback.format_exc())



# Public dense tools


@tool
def sapiens_seg(
    input_path: str,
    output_dir: str,
    model_size: str = "0.4b",
    device: str = "cuda:0",
    save_pred: bool = True,
) -> dict:
    """29-class body-part segmentation.

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
    Visualization remaps each normal from [-1,1] to RGB [0,255].
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
    """3D pointmap estimation - lifts each pixel to a 3D point in camera space.

    Visualization shows a colormap of the z-channel. Install the optional
    ``[pointmap]`` extra for ``open3d``-based downstream use on the raw ``.npy``.
    """
    return _run_dense_task("pointmap", input_path, output_dir, model_size, device, save_pred)



# sapiens_pose - top-down 308-keypoint pose (requires RTMDet detector)


def _find_pose_config(size: str):
    import sapiens.pose as _pose  # type: ignore
    pose_root = Path(_pose.__file__).parent
    cfg_file = (
        pose_root
        / "configs"
        / "keypoints308"
        / f"sapiens2_{size}_keypoints308-1024x768.py"
    )
    if cfg_file.exists():
        return cfg_file, pose_root
    # fallback search
    candidates = sorted(
        (pose_root / "configs").rglob(f"*{size}*keypoints308*.py")
    )
    if not candidates:
        raise FileNotFoundError(
            f"Pose config not found for size={size} under {pose_root}"
        )
    return candidates[0], pose_root


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

    Top-down: uses DETR-ResNet-101-DC5 person detector from
    ``$SAPIENS_CHECKPOINT_ROOT/detector/detr-resnet-101-dc5/``
    (HuggingFace ``facebook/detr-resnet-101-dc5``).

    Falls back to legacy RTMDet detector at
    ``$SAPIENS_CHECKPOINT_ROOT/detector/rtmdet_m.pth`` if present.

    This wrapper tries three integration paths, in order:

    1.  ``sapiens.pose.inference.inferencer`` / high-level inference API
    2.  ``sapiens.pose.models.init_pose_model`` + ``PoseVisualizer``
        (best-effort; ctor kwargs forwarded only if supported)
    3.  Fallback: return an error telling the user how to run the upstream
        CLI script directly.

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

        # Sapiens2 uses DETR detector; fall back to legacy rtmdet
        detector_detr_dir = checkpoint_root() / "detector" / "detr-resnet-101-dc5"
        detector_rtmdet = checkpoint_root() / "detector" / "rtmdet_m.pth"

        if detector_detr_dir.is_dir():
            detector_ckpt = detector_detr_dir
            detector_type = "detr"
        elif detector_rtmdet.exists():
            detector_ckpt = detector_rtmdet
            detector_type = "rtmdet"
        else:
            return err(
                f"Missing person detector. Download DETR:\n"
                f"  huggingface-cli download facebook/detr-resnet-101-dc5 "
                f"--local-dir {detector_detr_dir}\n"
                f"Or legacy RTMDet: {detector_rtmdet}"
            )

        import json

        import cv2  # type: ignore

        try:
            cfg_file, _pose_root = _find_pose_config(size)
        except FileNotFoundError as e:
            return err(str(e))

        # Try high-level inferencer API
        try:
            from sapiens.pose.inference import Inferencer  # type: ignore
            inferencer = Inferencer(
                config=str(cfg_file),
                checkpoint=str(ckpt),
                detector_checkpoint=str(detector_ckpt),
                device=device,
            )
            in_dir, images = resolve_input(input_path)
            out_dir = ensure_output(output_dir)
            outputs = []
            for img_path in images:
                result = inferencer.infer(str(img_path))
                vis = inferencer.visualize(
                    str(img_path),
                    result,
                    kpt_thres=kpt_thres,
                    line_thickness=line_thickness,
                    radius=radius,
                )
                out_img = out_dir / img_path.name
                cv2.imwrite(str(out_img), vis)
                out_json = out_dir / f"{img_path.stem}.json"
                out_json.write_text(json.dumps(result, default=str))
                outputs.append({
                    "input": str(img_path),
                    "vis": str(out_img),
                    "kpts": str(out_json),
                })
            return ok_with_images(
                f"pose complete on {len(outputs)} image(s)",
                image_paths=[o["vis"] for o in outputs if "vis" in o][:5],
                task="pose",
                model_size=size,
                outputs=outputs,
                api="sapiens.pose.inference.Inferencer",
            )
        except ImportError:
            pass

        # Try init_pose_model + PoseVisualizer
        try:
            import inspect

            from sapiens.pose.models import init_pose_model  # type: ignore
            from sapiens.pose.visualizers import PoseVisualizer  # type: ignore

            model = init_pose_model(str(cfg_file), str(ckpt), device=device)

            # Only pass visualizer kwargs that its ctor actually accepts.
            sig = inspect.signature(PoseVisualizer)
            vis_kwargs = {
                k: v for k, v in {
                    "kpt_thres": kpt_thres,
                    "line_thickness": line_thickness,
                    "radius": radius,
                }.items() if k in sig.parameters
            }
            vis = PoseVisualizer(**vis_kwargs)

            in_dir, images = resolve_input(input_path)
            out_dir = ensure_output(output_dir)
            outputs = []
            for img_path in images:
                image = cv2.imread(str(img_path))
                if image is None:
                    outputs.append({"input": str(img_path), "status": "skipped"})
                    continue
                # Different sapiens versions expose different infer signatures.
                try:
                    result = model.infer(image, detector_checkpoint=str(detector_ckpt))
                except TypeError:
                    result = model.infer(image)  # type: ignore[call-arg]

                if hasattr(vis, "draw"):
                    vis_img = vis.draw(image, result)
                elif hasattr(vis, "visualize"):
                    vis_img = vis.visualize(image, result)
                else:
                    vis_img = image  # give up on vis, still save JSON

                out_img = out_dir / img_path.name
                cv2.imwrite(str(out_img), vis_img)
                out_json = out_dir / f"{img_path.stem}.json"
                out_json.write_text(json.dumps(result, default=str))
                outputs.append({
                    "input": str(img_path),
                    "vis": str(out_img),
                    "kpts": str(out_json),
                })
            return ok_with_images(
                f"pose complete on {len(outputs)} image(s)",
                image_paths=[o["vis"] for o in outputs if "vis" in o][:5],
                task="pose",
                model_size=size,
                outputs=outputs,
                api="sapiens.pose.models.init_pose_model",
            )
        except ImportError:
            pass

        # Fallback: point the user at the upstream script
        return err(
            "sapiens.pose high-level API not available in installed version. "
            "Run upstream script directly: "
            f"`cd $SAPIENS_ROOT/sapiens/pose && ./scripts/demo/keypoints308.sh` "
            f"with CHECKPOINT={ckpt} DETECTOR={detector_ckpt} CONFIG={cfg_file}",
            checkpoint=str(ckpt),
            detector=str(detector_ckpt),
            config=str(cfg_file),
        )
    except Exception as e:  # noqa: BLE001
        return err(f"sapiens_pose failed: {e}", traceback=traceback.format_exc())


# sapiens_video - frame-by-frame video processing


@tool
def sapiens_video(
    video_path: str,
    output_dir: str,
    task: str = "seg",
    model_size: str = "0.4b",
    device: str = "cuda:0",
    fps: float = 0,
    max_frames: int = 0,
    save_pred: bool = False,
    save_frames: bool = True,
    reassemble: bool = True,
) -> dict:
    """Process a video file frame-by-frame through any Sapiens2 dense task.

    Extracts frames from the video, runs the specified task (seg, normal,
    albedo, pointmap) on each frame, and optionally reassembles the
    visualizations into an output video.

    Args:
        video_path:     path to input video (mp4/avi/mov/webm).
        output_dir:     where to save output frames and video.
        task:           one of ``seg | normal | albedo | pointmap``.
        model_size:     ``0.4b | 0.8b | 1b | 5b``.
        device:         torch device string.
        fps:            target FPS for frame extraction (0 = use source FPS).
        max_frames:     max frames to process (0 = all frames).
        save_pred:      write raw predictions (.npy) per frame.
        save_frames:    save individual frame visualizations.
        reassemble:     create output video from processed frames.

    Returns:
        dict with output video path, frame count, and per-frame entries.
    """
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        vid_path = Path(video_path).expanduser().resolve()
        if not vid_path.is_file():
            return err(f"Video not found: {vid_path}")

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            return err(f"Cannot open video: {vid_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        target_fps = fps if fps > 0 else src_fps
        frame_interval = max(1, int(round(src_fps / target_fps)))

        # Extract frames to temp dir
        out = ensure_output(output_dir)
        frames_dir = out / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        frame_paths: list[Path] = []
        frame_idx = 0
        extracted = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                fname = frames_dir / f"frame_{extracted:06d}.jpg"
                cv2.imwrite(str(fname), frame)
                frame_paths.append(fname)
                extracted += 1
                if max_frames > 0 and extracted >= max_frames:
                    break
            frame_idx += 1

        cap.release()

        if not frame_paths:
            return err("No frames extracted from video")

        # Run dense task on extracted frames directory
        result = _run_dense_task(
            task=task,
            input_path=str(frames_dir),
            output_dir=str(out / "vis"),
            model_size=model_size,
            device=device,
            save_pred=save_pred,
        )

        if result.get("status") != "success":
            return result

        # Reassemble into video
        output_video_path = None
        if reassemble:
            vis_dir = out / "vis"
            vis_frames = sorted(vis_dir.glob("frame_*.jpg")) + sorted(vis_dir.glob("frame_*.png"))
            if vis_frames:
                # Read first frame to get dimensions
                sample = cv2.imread(str(vis_frames[0]))
                if sample is not None:
                    vh, vw = sample.shape[:2]
                    output_video_path = str(out / f"{vid_path.stem}_{task}.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(output_video_path, fourcc, target_fps, (vw, vh))
                    for vf in vis_frames:
                        img = cv2.imread(str(vf))
                        if img is not None:
                            if img.shape[:2] != (vh, vw):
                                img = cv2.resize(img, (vw, vh))
                            writer.write(img)
                    writer.release()

        # Clean up extracted frames if not saving
        if not save_frames:
            import shutil
            shutil.rmtree(str(frames_dir), ignore_errors=True)

        # Build response
        inner_outputs = result.get("content", [{}])
        json_block = {}
        for block in inner_outputs:
            if isinstance(block, dict) and "json" in block:
                json_block = block["json"]
                break

        return ok(
            f"Video {task} complete: {extracted} frames processed",
            task=task,
            model_size=model_size,
            video_input=str(vid_path),
            output_video=output_video_path,
            frames_processed=extracted,
            source_fps=src_fps,
            target_fps=target_fps,
            output_dir=str(out),
            frame_outputs=json_block.get("outputs", []),
        )
    except Exception as e:  # noqa: BLE001
        return err(f"sapiens_video failed: {e}", traceback=traceback.format_exc())
        return err(f"sapiens_pose failed: {e}", traceback=traceback.format_exc())
