# Changelog

## 0.1.1 (bug-bash)

### Fixed
- **`sapiens_backbone` arch name**: `0.1b / 0.4b / 0.8b` previously produced
  `sapiens2_0.1b`-style keys (kept the dot). Now produces `sapiens2_01b`
  via a dedicated `arch_name()` helper and keeps `sapiens2_1b` for `1b_4k`.
- **`sapiens_pose` dead-end**: the tool used to always fall through to an
  `ImportError` path. Now tries three integration paths in order
  (`sapiens.pose.inference.Inferencer` → `init_pose_model` + `PoseVisualizer`
  → scripted fallback) and filters visualizer ctor kwargs so it doesn't
  TypeError on unknown arguments.
- **Pose visualizer kwargs**: `kpt_thres` / `line_thickness` / `radius` are
  now only forwarded if the installed `PoseVisualizer` actually accepts them.
- **Normal / albedo / pointmap visualization**: replaced speculative
  `NormalVisualizer` / `AlbedoVisualizer` imports with inline numpy-based
  renderers so vis files are always written.
- **Seg visualizer**: prefer public `.visualize()` / `.__call__()` before
  falling back to `_visualize_segmentation`; resize vis to match input
  before horizontal concat (was crashing on shape mismatch).
- **Dense model forward**: tolerate both `model(inputs)` and
  `model(inputs, data_samples)` signatures.
- **Dense config lookup**: if the hard-coded config path is missing, fall
  back to `rglob("sapiens2_<size>_<task>*.py")` under the task config tree.
- **`sapiens_backbone` forward output**: handle tensor / list / tuple / dict
  return types — picks the final stage feature map.
- **BGR→RGB**: switched from negative-stride slice to
  `cv2.cvtColor(..., COLOR_BGR2RGB)` for stability.
- **ImageNet normalization**: mean/std now built on-device (matters at 4k).
- **`save_features_to`** now has an explicit `overwrite=False` guard.
- **`cv2.imread` failure** no longer crashes — returns a structured error.
- **`sapiens_info`**: `cuda` dict always includes `available`, `device_count`,
  `device_name`; added `checkpoint_root_exists`.
- **`resolve_input`**: now accepts `.webp`, `.bmp`, `.tif`, `.tiff` and has
  an explicit `recursive=True` mode.

### Added
- `strands_sapiens._common.arch_name()` helper.
- `strands_sapiens._common.ensure_checkpoint_root()` helper.
- New smoke tests covering `arch_name`, `validate_size`, `checkpoint_path`,
  `resolve_input` (file/dir/missing/unsupported/recursive), `sapiens_info`,
  and the `ok/err` response shape.
- `[tool.pytest.ini_options]` + `[tool.ruff]` in `pyproject.toml`.
- `.gitignore` (models, caches, build output).
- README: supported-size table, troubleshooting section, Python 3.10 note
  for JetPack compatibility.

### Changed
- `requires-python` dropped from `>=3.12` to `>=3.10` to match Thor/JetPack.
- Package version bumped to `0.1.1`.
