"""Checks on a canonical episode. Errors reject the episode (quarantined with reasons); warnings are stored on it.

The thresholds are deliberately loose physical sanity bounds: they catch unit and convention
mistakes in a mapper (mm vs m, degrees vs radians, wrong column) rather than judging data quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa
from scipy.spatial.transform import Rotation

from robot_ingest.config import RobotSpec
from robot_ingest.kinematics import CHAINS, forward_kinematics, orientation_residual_deg
from robot_ingest.schema import column_matrix

MAX_ABS_POSITION_M = 2.0  # nothing in a tabletop workcell is 2 m from the robot base: catches mm vs m
MAX_ABS_JOINT_RAD = 2 * np.pi + 0.1  # beyond one turn: catches degrees vs radians
MAX_EE_STEP_M = 0.10  # per-frame jump; 1-1.5 m/s at 10-15 fps is a glitch, not motion
QUAT_NORM_TOLERANCE = 1e-3
MAX_FK_RESIDUAL_DEG = 1.0  # correct conventions fit to <0.25 deg on all three sources; wrong ones by 2-80 deg
BLACK_LUMA = 3.0  # mean brightness (0-255) under which a camera stream is considered dead


@dataclass
class Validation:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_frames(frames: pa.Table, robot: RobotSpec, fps: float, min_frames: int) -> Validation:
    v = Validation()
    n = frames.num_rows
    if n < min_frames:
        v.errors.append(f"only {n} frames (minimum {min_frames})")
    if n == 0:
        return v

    frame_index = frames["frame_index"].to_numpy()
    if not np.array_equal(frame_index, np.arange(n)):
        v.errors.append("frame_index is not 0..N-1 without gaps")
    ts = frames["timestamp_s"].to_numpy()
    if np.any(np.diff(ts) <= 0):
        v.errors.append("timestamps are not strictly increasing")
    elif np.max(np.abs(ts - np.arange(n) / fps)) > 0.5 / fps:
        v.errors.append("timestamps drift more than half a frame from frame_index / fps")

    for name in ("obs_ee_pos", "obs_ee_quat", "obs_joint_pos", "obs_gripper_closure", "obs_wrench",
                 "act_ee_pos", "act_ee_quat", "act_gripper_closure", "act_joint_pos"):
        if frames[name].null_count == n:
            continue  # optional column not provided by this source
        if not np.all(np.isfinite(column_matrix(frames, name))):
            v.errors.append(f"{name} contains NaN or inf")

    for name in ("obs_joint_pos", "act_joint_pos"):
        if frames[name].null_count == n:
            continue
        joints = column_matrix(frames, name)
        if joints.shape[1] != robot.num_joints:
            v.errors.append(f"{name} has {joints.shape[1]} values, {robot.robot_id} has {robot.num_joints} joints")
        if np.abs(joints).max() > MAX_ABS_JOINT_RAD:
            v.errors.append(f"{name} exceeds one full turn: degrees instead of radians?")

    for name in ("obs_ee_pos", "act_ee_pos"):
        if np.abs(column_matrix(frames, name)).max() > MAX_ABS_POSITION_M:
            v.errors.append(f"{name} is more than {MAX_ABS_POSITION_M} m from the base: millimetres instead of metres?")
    for name in ("obs_ee_quat", "act_ee_quat"):
        if np.abs(np.linalg.norm(column_matrix(frames, name), axis=1) - 1).max() > QUAT_NORM_TOLERANCE:
            v.errors.append(f"{name} is not unit-norm")
    for name in ("obs_gripper_closure", "act_gripper_closure"):
        values = column_matrix(frames, name)
        if values.min() < 0 or values.max() > 1:
            v.errors.append(f"{name} outside [0, 1]")

    jumps = ee_jumps(frames)
    if jumps:
        frame, distance = max(jumps, key=lambda j: j[1])
        v.warnings.append(f"end effector jumps {distance:.2f} m between frames {frame - 1} and {frame}")
    if not v.errors:
        v.warnings += check_orientation_vs_joints(frames, robot)
    return v


def ee_jumps(frames: pa.Table) -> list[tuple[int, float]]:
    """(frame, distance) for every frame the end effector reached by moving over MAX_EE_STEP_M since the one before.

    Shared by validation (warns about the largest) and the viewer (marks them all on the plots).
    """
    if frames.num_rows < 2:
        return []
    step = np.linalg.norm(np.diff(column_matrix(frames, "obs_ee_pos"), axis=0), axis=1)
    return [(int(i) + 1, float(step[i])) for i in np.flatnonzero(step > MAX_EE_STEP_M)]


def check_orientation_vs_joints(frames: pa.Table, robot: RobotSpec) -> list[str]:
    """The reported orientation must equal FK(joints) up to constant base/tool rotations.

    A wrong quaternion order or Euler convention breaks that, by 2-80 degrees on this data.
    (Frame-level mistakes, like cfg7's base rotated 180 deg, need positions too and are checked
    once per source at onboarding by scripts/check_orientation.py.)
    """
    if robot.kinematics is None or frames.num_rows < 20:
        return []
    fk = forward_kinematics(CHAINS[robot.kinematics], column_matrix(frames, "obs_joint_pos"))[:, :3, :3]
    measured = Rotation.from_quat(column_matrix(frames, "obs_ee_quat")).as_matrix()
    residual = float(np.median(orientation_residual_deg(measured, fk)))
    if residual > MAX_FK_RESIDUAL_DEG:
        return [f"ee orientation disagrees with the joint angles by {residual:.1f} deg (median): "
                f"wrong quaternion order or Euler convention?"]
    return []


def validate_camera(camera_id: str, mean_luma: float) -> list[str]:
    """Warnings for one transcoded camera stream."""
    if mean_luma < BLACK_LUMA:
        return [f"camera {camera_id} is black (mean brightness {mean_luma:.1f}) although the source lists it as recording"]
    return []
