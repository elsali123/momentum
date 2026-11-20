"""Customer B: DROID 1.0.1, as converted to LeRobot v2.1 (cadene/droid_1.0.1).

Meaning of the source columns (each verified against the data; see DESIGN.md, "Source survey"):
  observation.state.cartesian_position  [x y z roll pitch yaw]: m / rad, extrinsic-xyz Euler, base frame
  observation.state.joint_position      7 joints, rad (the dataset labels them "x..yaw": mislabeled)
  observation.state.gripper_position    Robotiq closure, 0 = open .. 1 = closed (already canonical)
  action.cartesian_position             commanded absolute EE target from VR teleop (leads the measured pose)
  action.joint_position / .gripper_position   commanded joint target / gripper closure
  action.original                       duplicate of action.cartesian_position + gripper; ignored
  camera_extrinsics.<camera>            [x y z roll pitch yaw] in the base frame, constant per episode
  language_instruction{,_2,_3}          up to three paraphrases; often empty
  building, collector_id                per-episode metadata
  task_category, date                   mis-mapped by the port (task_category == building and
                                        date == collector_id in every episode checked): ignored, kept in raw
"""

from __future__ import annotations

import numpy as np

from robot_ingest.schema import column_matrix, frames_table
from robot_ingest.sources.base import CameraSource, Mapper, NormalizedEpisode, SourceDataError, make_episode_id
from robot_ingest.transforms import align_quat_to, clip_unit, euler_xyz_to_quat, make_quat_continuous

# (source camera key, role), wrist first
CAMERAS = (("wrist_left", "wrist"), ("exterior_1_left", "external"), ("exterior_2_left", "external"))
LANGUAGE_COLUMNS = ("language_instruction", "language_instruction_2", "language_instruction_3")
PER_EPISODE_COLUMNS = (*LANGUAGE_COLUMNS, "is_episode_successful", "building", "collector_id")


class DroidMapper(Mapper):
    def normalize(self, episode_index: int) -> NormalizedEpisode:
        episode_id = make_episode_id(self.spec.source_id, episode_index)
        raw = self.repo.read_frames(episode_index)
        expected = self.repo.episodes[episode_index]["length"]
        if raw.num_rows != expected:
            raise SourceDataError(f"{raw.num_rows} data rows but the episode index says {expected}")
        if raw.num_rows == 0:
            raise SourceDataError("episode has no frames")
        warnings: list[str] = []

        state = column_matrix(raw, "observation.state.cartesian_position")
        obs_quat = make_quat_continuous(euler_xyz_to_quat(state[:, 3:6]))
        target = column_matrix(raw, "action.cartesian_position")
        act_quat = align_quat_to(euler_xyz_to_quat(target[:, 3:6]), obs_quat)

        obs_grip, obs_out = clip_unit(column_matrix(raw, "observation.state.gripper_position"))
        act_grip, act_out = clip_unit(column_matrix(raw, "action.gripper_position"))
        if obs_out or act_out:
            warnings.append(f"gripper position outside [0, 1] on {obs_out} obs / {act_out} action frames; clipped")

        frames = frames_table(episode_id, {
            "frame_index": raw["frame_index"].to_numpy(),
            "timestamp_s": np.round(raw["timestamp"].to_numpy().astype(np.float64), 6),
            "obs_ee_pos": state[:, :3],
            "obs_ee_quat": obs_quat,
            "obs_joint_pos": column_matrix(raw, "observation.state.joint_position"),
            "obs_gripper_closure": obs_grip,
            "obs_wrench": None,  # DROID's Franka setup records no wrist force/torque
            "act_ee_pos": target[:, :3],
            "act_ee_quat": act_quat,
            "act_gripper_closure": act_grip,
            "act_joint_pos": column_matrix(raw, "action.joint_position"),
        })

        # Episode-level fields are repeated on every row; take the first and flag any that vary.
        for column in PER_EPISODE_COLUMNS:
            if len(raw[column].unique()) > 1:
                warnings.append(f"{column} varies within the episode; using the first value")
        first = raw.slice(0, 1).to_pylist()[0]
        instructions = list(dict.fromkeys(
            text.strip() for text in (first[c] for c in LANGUAGE_COLUMNS) if text and text.strip()
        ))
        if not instructions:
            warnings.append("no language instruction")

        metadata = {
            "task": instructions[0] if instructions else None,
            "language_instruction": instructions[0] if instructions else None,
            "language_instructions": instructions,
            "success": bool(first["is_episode_successful"]),
            "quality_rating": None,
            "native_capture_hz": None,
            "action_ee_semantics": "commanded_target",
            "operator_id": first["collector_id"] or None,
            "scene_id": first["building"] or None,
            "license": self.spec.license,
            "source_extras": {},
        }
        return NormalizedEpisode(
            episode_id=episode_id,
            source_episode_index=episode_index,
            frames=frames,
            raw_frames=raw,
            cameras=self._cameras(episode_index, raw, warnings),
            metadata=metadata,
            warnings=warnings,
        )

    def _cameras(self, episode_index: int, raw, warnings: list[str]) -> list[CameraSource]:
        cameras = []
        for name, role in CAMERAS:
            pos = quat = None
            if role == "external":
                # Fixed cameras: one pose per episode. The wrist camera's stored pose is also constant
                # even though it moves with the arm, so it is not a usable extrinsic; we leave it null.
                ext = column_matrix(raw, f"camera_extrinsics.{name}")
                if np.abs(ext - ext[0]).max() > 1e-5:
                    warnings.append(f"{name} extrinsics vary within the episode; using the first frame")
                pos = ext[0, :3].tolist()
                quat = euler_xyz_to_quat(ext[0, 3:6]).tolist()
            key = f"observation.images.{name}"
            if key not in self.repo.video_keys:
                raise SourceDataError(f"missing video stream {key}")
            cameras.append(CameraSource(
                camera_id=name,
                role=role,
                source_video_key=key,
                segment=self.repo.video_segment(episode_index, key),
                extrinsics_pos=pos,
                extrinsics_quat=quat,
            ))
        return cameras
