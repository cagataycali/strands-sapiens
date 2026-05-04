"""strands-sapiens - Strands @tool wrappers around Meta Sapiens2."""

from .tools import (
    sapiens_albedo,
    sapiens_backbone,
    sapiens_info,
    sapiens_normal,
    sapiens_pointmap,
    sapiens_pose,
    sapiens_seg,
    sapiens_video,
)

__all__ = [
    "TOOLS",
    "sapiens_info",
    "sapiens_backbone",
    "sapiens_seg",
    "sapiens_normal",
    "sapiens_albedo",
    "sapiens_pointmap",
    "sapiens_pose",
    "sapiens_video",
]

#: Convenience: every sapiens-backed tool, ready to pass to ``Agent(tools=TOOLS)``.
TOOLS = [
    sapiens_info,
    sapiens_backbone,
    sapiens_seg,
    sapiens_normal,
    sapiens_albedo,
    sapiens_pointmap,
    sapiens_pose,
    sapiens_video,
]

__version__ = "0.1.0"
