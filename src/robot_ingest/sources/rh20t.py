"""Customer A: RH20T configs 3 and 7, as reformatted to LeRobot v3.0 (robot-lev/rh20t_cfg*).

Meaning of the source columns (each verified against the data; see DESIGN.md, "Source survey"):
  observation.state.ee_pose  [x y z qx qy qz qw]: m, robot base frame. The port forces qx >= 0,
                             which flips the quaternion sign thousands of times; we re-canonicalize.
  observation.state.joint    rad, in RH20T's robot_joint_sequence order
  observation.state.gripper  opening width in mm (WSG-50: 0-110, Robotiq 2F-85: 0-85)
  observation.force/torque   ATI wrist force/torque sensor, N and N*m  -> obs_wrench
  observation.robot_ft       the arm's own F/T estimate (far noisier); not canonical, kept in raw
  action                     [ee_pose at t+1 (7), commanded gripper width in mm (1)]
  meta/rh20t_episodes.json   per episode: cameras that actually recorded, rating, scene, user,
                             native capture rate (frames were resampled to 10 fps)
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from robot_ingest.schema import column_matrix, frames_table
from robot_ingest.sources.base import CameraSource, Mapper, NormalizedEpisode, SourceDataError, make_episode_id
from robot_ingest.transforms import align_quat_to, clip_unit, make_quat_continuous, normalize_quat, width_to_closure

# RH20T is dual-licensed by scene (dataset card): scenes 1-5 CC BY-SA 4.0, scenes 6-10 CC BY-NC 4.0.
NON_COMMERCIAL_SCENES = range(6, 11)

# RH20T rating: 0 = robot failure, 1 = task failure, 2-9 = completion quality, -1 = not rated.
MIN_SUCCESS_RATING = 2


@dataclass(frozen=True)
class OrientationFix:
    """How to turn the port's stored quaternion into the canonical orientation: base @ R(stored) @ tool."""

    stored_order: str  # component order actually used in the file
    base: Rotation  # left-multiplied: re-expresses the orientation in the positions' base frame
    tool: Rotation  # right-multiplied: makes the tool z-axis the gripper's approach direction


# The port labels every quaternion "xyzw". Fitting the poses against forward kinematics of the
# recorded joint angles (scripts/check_orientation.py; DESIGN.md D6) shows:
#   cfg 3: the components are really w, x, y, z. Otherwise consistent with the positions.
#   cfg 7: x, y, z, w is right, but the orientation is in a base frame turned 180 deg about z
#          relative to the positions, and its tool z-axis points back into the arm.
ORIENTATION_FIXES = {
    3: OrientationFix("wxyz", Rotation.identity(), Rotation.identity()),
    7: OrientationFix("xyzw", Rotation.from_euler("z", np.pi), Rotation.from_euler("y", np.pi)),
}
_TO_XYZW = {"xyzw": [0, 1, 2, 3], "wxyz": [1, 2, 3, 0]}


class Rh20tMapper(Mapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.port_config = json.loads(self.repo.fetch("meta/rh20t_config.json").read_text())
        self.port_episodes = json.loads(self.repo.fetch("meta/rh20t_episodes.json").read_text())
        self._check_port_conventions()

    def snapshot_files(self) -> list[str]:
        return super().snapshot_files() + ["meta/rh20t_config.json", "meta/rh20t_episodes.json"]

    def _check_port_conventions(self) -> None:
        """Fail fast if the source stops matching the assumptions this mapper is built on."""
        cfg, robot = self.port_config, self.robot
        problems = []
        if cfg["quat_order"] != "xyzw":
            problems.append(f"quat_order is {cfg['quat_order']!r}, expected 'xyzw'")
        if cfg["dof"] != robot.num_joints:
            problems.append(f"dof {cfg['dof']} != {robot.num_joints} joints of {robot.robot_id}")
        if abs(cfg["gripper_max_mm"] / 1000 - robot.gripper_max_width_m) > 1e-6:
            problems.append(f"gripper_max_mm {cfg['gripper_max_mm']} != robot spec {robot.gripper_max_width_m} m")
        if abs(cfg["fps"] - self.repo.fps) > 1e-6:
            problems.append(f"fps {cfg['fps']} != info.json fps {self.repo.fps}")
        if cfg["cfg"] not in ORIENTATION_FIXES:
            problems.append(f"no verified orientation convention for RH20T cfg {cfg['cfg']}: run scripts/check_orientation.py")
        if problems:
            raise ValueError(f"{self.spec.source_id}: source conventions changed: " + "; ".join(problems))
        self.orientation_fix = ORIENTATION_FIXES[cfg["cfg"]]

    def _orientation(self, stored: np.ndarray) -> tuple[np.ndarray, float]:
        """Stored quaternions (N, 4) -> canonical xyzw quaternions, plus the input's max norm error."""
        fix = self.orientation_fix
        quat, norm_error = normalize_quat(stored[:, _TO_XYZW[fix.stored_order]])
        return (fix.base * Rotation.from_quat(quat) * fix.tool).as_quat(), norm_error

    def normalize(self, episode_index: int) -> NormalizedEpisode:
        episode_id = make_episode_id(self.spec.source_id, episode_index)
        raw = self.repo.read_frames(episode_index)
        meta = self.port_episodes[str(episode_index)]
        expected = self.repo.episodes[episode_index]["length"]
        if raw.num_rows != expected:
            raise SourceDataError(f"{raw.num_rows} data rows but the episode index says {expected}")
        warnings: list[str] = []

        # End-effector pose: fixed orientation convention, unit quaternions, continuous in time,
        # and action quats in the same hemisphere as the state.
        ee = column_matrix(raw, "observation.state.ee_pose")
        obs_quat, norm_error = self._orientation(ee[:, 3:7])
        if norm_error > 1e-3:
            warnings.append(f"ee quaternion norms off by up to {norm_error:.3g}; renormalized")
        obs_quat = make_quat_continuous(obs_quat)
        action = column_matrix(raw, "action")
        act_quat = align_quat_to(self._orientation(action[:, 3:7])[0], obs_quat)

        # Gripper: width in mm -> closure in [0, 1].
        max_width = self.robot.gripper_max_width_m
        obs_grip, obs_out = clip_unit(width_to_closure(column_matrix(raw, "observation.state.gripper") / 1000, max_width))
        act_grip, act_out = clip_unit(width_to_closure(action[:, 7] / 1000, max_width))
        if obs_out or act_out:
            warnings.append(f"gripper width outside 0-{max_width * 1000:.0f} mm "
                            f"on {obs_out} obs / {act_out} action frames; clipped")

        frames = frames_table(episode_id, {
            "frame_index": raw["frame_index"].to_numpy(),
            "timestamp_s": np.round(raw["timestamp"].to_numpy().astype(np.float64), 6),  # source is float32
            "obs_ee_pos": ee[:, :3],
            "obs_ee_quat": obs_quat,
            "obs_joint_pos": column_matrix(raw, "observation.state.joint"),
            "obs_gripper_closure": obs_grip,
            "obs_wrench": np.hstack([column_matrix(raw, "observation.force"), column_matrix(raw, "observation.torque")]),
            "act_ee_pos": action[:, :3],
            "act_ee_quat": act_quat,
            "act_gripper_closure": act_grip,
            "act_joint_pos": None,  # RH20T has no joint-space action
        })

        rating = int(meta["rating"])
        if set(raw["meta.rating"].to_pylist()) != {rating}:
            warnings.append("meta.rating column disagrees with rh20t_episodes.json; using the json")
        native_hz = float(meta["native_hz"])
        if native_hz < 0.5 * self.repo.fps:
            warnings.append(f"captured at {native_hz:.1f} Hz and resampled to {self.repo.fps:.0f} fps: most frames are repeats")

        scene = int(meta["scene_id"])
        license_ = "CC-BY-NC-4.0" if scene in NON_COMMERCIAL_SCENES else "CC-BY-SA-4.0"
        task_index = raw["task_index"][0].as_py()
        metadata = {
            "task": self.repo.tasks.get(task_index),  # the port only kept "task <id>"; no natural language
            "language_instruction": None,
            "language_instructions": [],
            "success": None if rating < 0 else rating >= MIN_SUCCESS_RATING,
            "quality_rating": rating if rating >= 0 else None,
            "native_capture_hz": native_hz,
            "action_ee_semantics": "next_observed_state",
            "operator_id": f"rh20t_user_{meta['user_id']:04d}",
            "scene_id": f"rh20t_scene_{scene:04d}",
            "license": license_,
            "source_extras": {
                "folder": meta["folder"],
                "rh20t_task_id": meta["task_id"],
                "master_camera": meta["master_camera"],
                "rh20t_cfg": meta["cfg"],
            },
        }
        return NormalizedEpisode(
            episode_id=episode_id,
            source_episode_index=episode_index,
            frames=frames,
            raw_frames=raw,
            cameras=self._cameras(episode_index, meta, warnings),
            metadata=metadata,
            warnings=warnings,
        )

    def _cameras(self, episode_index: int, meta: dict, warnings: list[str]) -> list[CameraSource]:
        """Only cameras that actually recorded: the other streams in the file are black filler frames."""
        in_hand = set(self.port_config["inhand"])
        cameras = []
        for serial in sorted(meta["cameras_present"], key=lambda s: (s not in in_hand, s)):  # wrist first
            key = f"observation.images.cam_{serial}"
            if key not in self.repo.video_keys:
                warnings.append(f"camera {serial} listed as present but has no video stream; skipped")
                continue
            cameras.append(CameraSource(
                camera_id=f"cam_{serial}",
                role="wrist" if serial in in_hand else "external",
                serial=serial,
                source_video_key=key,
                segment=self.repo.video_segment(episode_index, key),
            ))
        if not cameras:
            raise SourceDataError("no camera recorded footage in this episode")
        return cameras
