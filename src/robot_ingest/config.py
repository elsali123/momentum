"""Typed view of configs/sources.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RobotSpec:
    """One embodiment (arm + gripper). Shared by every episode recorded on it."""

    robot_id: str
    arm_model: str
    gripper_model: str
    joint_names: tuple[str, ...]
    gripper_max_width_m: float
    has_ft_sensor: bool
    kinematics: str | None = None  # DH chain in kinematics.CHAINS, enables the pose-vs-joints check
    ee_offset_from_flange_m: float | None = None  # where the reported EE point sits along the tool z-axis

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)


@dataclass(frozen=True)
class SourceSpec:
    """One customer dataset, pinned to an exact revision."""

    source_id: str
    customer_id: str
    mapper: str  # which customer mapper interprets the columns ("rh20t", "droid")
    repo_id: str
    revision: str
    robot_id: str
    num_episodes: int
    seed: int
    episodes: tuple[int, ...] = ()  # explicit source episode indices; overrides random sampling
    license: str | None = None  # default license; a mapper may refine it per episode


@dataclass(frozen=True)
class PipelineSettings:
    min_episode_frames: int = 10
    video_crf: int = 20
    video_keyframe_interval: int = 10


@dataclass(frozen=True)
class Config:
    pipeline: PipelineSettings
    robots: dict[str, RobotSpec]
    sources: tuple[SourceSpec, ...]

    def source(self, source_id: str) -> SourceSpec:
        for s in self.sources:
            if s.source_id == source_id:
                return s
        raise KeyError(f"unknown source {source_id!r}; known: {[s.source_id for s in self.sources]}")


def load_config(path: str | Path) -> Config:
    raw = tomllib.loads(Path(path).read_text())

    robots = {
        robot_id: RobotSpec(robot_id=robot_id, **{**spec, "joint_names": tuple(spec["joint_names"])})
        for robot_id, spec in raw.get("robots", {}).items()
    }
    sources = tuple(
        SourceSpec(**{**s, "episodes": tuple(s.get("episodes", ()))}) for s in raw.get("sources", [])
    )

    ids = [s.source_id for s in sources]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate source_id in {path}: {ids}")
    for s in sources:
        if s.robot_id not in robots:
            raise ValueError(f"source {s.source_id!r} uses undefined robot {s.robot_id!r}")

    return Config(
        pipeline=PipelineSettings(**raw.get("pipeline", {})),
        robots=robots,
        sources=sources,
    )
