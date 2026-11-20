"""Interface between the generic pipeline and the customer-specific mappers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pyarrow as pa

from robot_ingest.config import RobotSpec, SourceSpec
from robot_ingest.sources.lerobot import LeRobotRepo, VideoSegment


class SourceDataError(Exception):
    """The source episode itself is unusable (as opposed to a bug or a network error). It gets rejected."""


def make_episode_id(source_id: str, source_episode_index: int) -> str:
    """Deterministic and human-readable, so re-runs overwrite rather than duplicate."""
    return f"{source_id}-{source_episode_index:06d}"


def commercial_use_allowed(license_id: str | None) -> bool | None:
    """Coarse license gate for training-set selection. None = unknown, a human has to decide."""
    if not license_id:
        return None
    license_id = license_id.lower()
    if "-nc" in license_id or "noncommercial" in license_id:
        return False
    if license_id in {"apache-2.0", "mit", "cc-by-4.0", "cc-by-sa-4.0", "cc0-1.0"}:
        return True
    return None


@dataclass
class CameraSource:
    """A camera stream with real footage in this episode, and where to read it."""

    camera_id: str
    role: str  # "wrist" | "external"
    source_video_key: str
    segment: VideoSegment
    serial: str | None = None
    extrinsics_pos: list[float] | None = None
    extrinsics_quat: list[float] | None = None


@dataclass
class NormalizedEpisode:
    """One episode in canonical form, before videos are transcoded and the bundle is written."""

    episode_id: str
    source_episode_index: int
    frames: pa.Table  # canonical FRAMES table
    raw_frames: pa.Table  # the source rows exactly as read (kept as the bronze copy)
    cameras: list[CameraSource]
    metadata: dict  # mapper-owned columns of the episodes table (task, success, license, ...)
    warnings: list[str] = field(default_factory=list)


class Mapper(ABC):
    """Knows what one customer's columns mean and converts them to the canonical schema."""

    # Bump when a change to a mapper alters what it produces: its episodes are then re-normalized on the
    # next run. Their videos are reused unless the video segments it points at changed too.
    version = "1"

    def __init__(self, spec: SourceSpec, robot: RobotSpec, repo: LeRobotRepo):
        self.spec = spec
        self.robot = robot
        self.repo = repo

    @property
    def source_format(self) -> str:
        return f"lerobot_{self.repo.version}"

    def episode_indices(self) -> list[int]:
        return sorted(self.repo.episodes)

    def snapshot_files(self) -> list[str]:
        """Source metadata files copied into the raw (bronze) zone for provenance."""
        return ["meta/info.json"]

    @abstractmethod
    def normalize(self, episode_index: int) -> NormalizedEpisode: ...
