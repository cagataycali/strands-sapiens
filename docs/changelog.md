# Changelog

## 0.1.0

🧬 *Give your agent a body.*

### Added
- **`sapiens_video`** tool — frame-by-frame video processing through any dense task (seg/normal/albedo/pointmap) with FPS subsampling, frame cap, and MP4 reassembly.
- **DETR-ResNet-101-DC5** person detector support (upstream sapiens2 standard). Falls back to legacy RTMDet if present.
- **Pointmap `.ply` export** — when `open3d` is installed, pointmap tool exports colored point clouds alongside `.npy`.
- **Inline image return** — all dense tools return output visualizations as inline image content blocks (Converse API compatible).
- Video processing guide at `docs/guide/video.md`.

### Fixed
- **Albedo config path**: `configs/albedo/metasim_render_people` → `configs/albedo/render_people` (matching upstream).
- **Pointmap config path**: `configs/pointmap/metasim_render_people` → `configs/pointmap/render_people` (matching upstream).
- **Pointmap scale normalization**: upstream divides by returned scale factor for metric coordinates — now we do too.
- **Pointmap/Albedo padding removal**: upstream removes pipeline padding before resize — now we do too.
- **Albedo clamping**: upstream clamps output to `[0, 1]` — now we do too.
- **Pose detector**: updated from RTMDet (`rtmdet_m.pth`) to DETR (`detr-resnet-101-dc5/`), matching sapiens2.
- **`sapiens_info` detector reporting**: now reports `detector_type` field (`"detr-resnet-101-dc5"`, `"rtmdet_m"`, or `"none"`).
- **Docs**: all response format examples corrected to match actual Strands `ToolResult` format (`status` + `content` list with `text`/`json`/`image` blocks).
- **Docs**: all RTMDet references updated to DETR throughout guides, API reference, and architecture.

### Changed
- Version: `0.1.0` (first public release).
- All 8 tools exported in `TOOLS` list.
