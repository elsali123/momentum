"""The canonical schema (v1): every table, column, type and unit in one place.

    robots    one row per embodiment           <store>/robots.parquet
    episodes  one row per episode              <store>/episodes.parquet
    cameras   one row per (episode, camera)    <store>/cameras.parquet
    frames    one row per (episode, frame)     <store>/episodes/<episode_id>/frames.parquet
    videos    <store>/episodes/<episode_id>/videos/<camera_id>.mp4   (video frame i == frames row i)

Units and docs are attached to each Arrow field as metadata, so they travel inside the
Parquet files and are visible to anyone who opens them.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

SCHEMA_VERSION = "1.0"

VEC3 = pa.list_(pa.float32(), 3)
QUAT = pa.list_(pa.float32(), 4)
VEC6 = pa.list_(pa.float32(), 6)
FLOATS = pa.list_(pa.float32())
STRINGS = pa.list_(pa.string())


def _field(name: str, type_: pa.DataType, doc: str, unit: str | None = None, *, required: bool = False) -> pa.Field:
    metadata = {"doc": doc} | ({"unit": unit} if unit else {})
    return pa.field(name, type_, nullable=not required, metadata=metadata)


FRAMES = pa.schema([
    _field("episode_id", pa.string(), "episode this frame belongs to", required=True),
    _field("frame_index", pa.int32(), "0-based; equals the frame index in every camera video of the episode", required=True),
    _field("timestamp_s", pa.float64(), "time since episode start", "s", required=True),
    # observation (measured)
    _field("obs_ee_pos", VEC3, "measured end-effector (TCP) position, robot base frame", "m", required=True),
    _field("obs_ee_quat", QUAT, "measured end-effector orientation, robot base frame; unit quaternion xyzw, "
           "sign-continuous; tool z-axis = gripper approach direction", required=True),
    _field("obs_joint_pos", FLOATS, "measured joint positions in robots.joint_names order", "rad", required=True),
    _field("obs_gripper_closure", pa.float32(), "measured gripper closure: 0 = fully open, 1 = fully closed", required=True),
    _field("obs_wrench", VEC6, "wrist force/torque [fx fy fz tx ty tz], sensor frame; null if no sensor", "N | N*m"),
    # action (what the policy should output)
    _field("act_ee_pos", VEC3, "end-effector position target, robot base frame (see episodes.action_ee_semantics)",
           "m", required=True),
    _field("act_ee_quat", QUAT, "end-effector orientation target, xyzw, same hemisphere as obs_ee_quat", required=True),
    _field("act_gripper_closure", pa.float32(), "commanded gripper closure: 0 = open, 1 = closed", required=True),
    _field("act_joint_pos", FLOATS, "commanded joint positions; null if the source has no joint-space action", "rad"),
])

EPISODES = pa.schema([
    _field("episode_id", pa.string(), "globally unique and deterministic: <source_id>-<source_episode_index:06d>", required=True),
    _field("customer_id", pa.string(), "customer that owns the data", required=True),
    _field("source_id", pa.string(), "dataset within the customer (configs/sources.toml)", required=True),
    _field("source_repo", pa.string(), "Hugging Face dataset repo the episode was read from", required=True),
    _field("source_revision", pa.string(), "exact repo commit the episode was read from", required=True),
    _field("source_format", pa.string(), "container format of the source, e.g. lerobot_v3.0", required=True),
    _field("source_episode_index", pa.int64(), "episode index inside the source dataset", required=True),
    _field("robot_id", pa.string(), "embodiment, joins robots.robot_id", required=True),
    _field("task", pa.string(), "task label as given by the source"),
    _field("language_instruction", pa.string(), "primary natural-language instruction; null if the source has none"),
    _field("language_instructions", STRINGS, "all distinct non-empty instructions (DROID has up to 3 paraphrases)"),
    _field("num_frames", pa.int32(), "rows in frames and frames in every video", required=True),
    _field("fps", pa.float32(), "frame rate of frames and videos", "Hz", required=True),
    _field("duration_s", pa.float32(), "num_frames / fps", "s", required=True),
    _field("success", pa.bool_(), "task success; null if unknown"),
    _field("quality_rating", pa.int32(), "source quality score (RH20T: 0 robot failure, 1 task failure, 2-9 quality)"),
    _field("native_capture_hz", pa.float32(), "rate the source originally captured at, when it was resampled to fps", "Hz"),
    _field("action_ee_semantics", pa.string(),
           "'commanded_target' (teleop command) or 'next_observed_state' (measured pose at t+1)", required=True),
    _field("operator_id", pa.string(), "person who collected the episode"),
    _field("scene_id", pa.string(), "scene / location identifier"),
    _field("license", pa.string(), "license the episode is used under"),
    _field("commercial_use_allowed", pa.bool_(), "false for non-commercial licenses: exclude from commercial training sets"),
    _field("camera_ids", STRINGS, "cameras with real footage, wrist first", required=True),
    _field("has_wrench", pa.bool_(), "obs_wrench is populated", required=True),
    _field("validation_warnings", STRINGS, "non-fatal issues found at ingest", required=True),
    _field("source_extras", pa.string(), "JSON object of source-specific fields with no canonical column"),
    _field("schema_version", pa.string(), "canonical schema version", required=True),
    _field("pipeline_version", pa.string(), "ingest code version", required=True),
    _field("ingested_at", pa.timestamp("us", tz="UTC"), "when the episode was written", required=True),
])

CAMERAS = pa.schema([
    _field("episode_id", pa.string(), "joins episodes.episode_id", required=True),
    _field("camera_id", pa.string(), "stable camera name within the source (serial-based for RH20T)", required=True),
    _field("role", pa.string(), "'wrist' (moves with the gripper) or 'external' (fixed)", required=True),
    _field("serial", pa.string(), "hardware serial number, if known"),
    _field("width", pa.int32(), "pixels", required=True),
    _field("height", pa.int32(), "pixels", required=True),
    _field("fps", pa.float32(), "frame rate", "Hz", required=True),
    _field("num_frames", pa.int32(), "frames in the video; always equals episodes.num_frames", required=True),
    _field("video_path", pa.string(), "MP4 path relative to the store root", required=True),
    _field("thumbnail_path", pa.string(), "JPEG path relative to the store root", required=True),
    _field("codec", pa.string(), "codec of video_path", required=True),
    _field("source_video_key", pa.string(), "feature name in the source dataset", required=True),
    _field("source_codec", pa.string(), "codec of the source video", required=True),
    _field("extrinsics_pos", VEC3, "camera position in the robot base frame; static cameras only", "m"),
    _field("extrinsics_quat", QUAT, "camera orientation in the robot base frame, xyzw; static cameras only"),
    _field("mean_luma", pa.float32(), "average brightness (0-255) of sampled frames; ~0 means a dead stream", required=True),
])

ROBOTS = pa.schema([
    _field("robot_id", pa.string(), "embodiment id", required=True),
    _field("arm_model", pa.string(), "arm make and model", required=True),
    _field("gripper_model", pa.string(), "gripper make and model", required=True),
    _field("num_joints", pa.int32(), "length of obs_joint_pos", required=True),
    _field("joint_names", STRINGS, "order of obs_joint_pos / act_joint_pos", required=True),
    _field("gripper_max_width_m", pa.float32(), "opening at closure = 0; width = (1 - closure) * this", "m", required=True),
    _field("has_ft_sensor", pa.bool_(), "a wrist force/torque sensor is fitted", required=True),
    _field("kinematics", pa.string(), "DH chain used to check poses against joint angles"),
    _field("ee_offset_from_flange_m", pa.float32(), "reported EE point = flange + this offset along the tool z-axis "
           "(0 = flange, DROID; ~0.23-0.25 = fingertip TCP, RH20T)", "m"),
])


def frames_table(episode_id: str, columns: dict[str, np.ndarray | None]) -> pa.Table:
    """Build a FRAMES table from numpy arrays, checking each column's shape against the schema.

    `columns` maps column name -> array (N,) or (N, k). Nullable columns may be omitted or None.
    """
    n = len(columns["frame_index"])
    arrays = []
    for field in FRAMES:
        if field.name == "episode_id":
            arrays.append(pa.array([episode_id] * n, pa.string()))
            continue
        value = columns.get(field.name)
        if value is None:
            if not field.nullable:
                raise ValueError(f"frames column {field.name!r} is required")
            arrays.append(pa.nulls(n, field.type))
            continue
        arrays.append(_to_arrow(field, np.asarray(value), n))
    return pa.Table.from_arrays(arrays, schema=FRAMES)


def _to_arrow(field: pa.Field, value: np.ndarray, n: int) -> pa.Array:
    t = field.type
    if len(value) != n:
        raise ValueError(f"{field.name}: expected {n} rows, got {len(value)}")
    if pa.types.is_fixed_size_list(t):
        if value.shape != (n, t.list_size):
            raise ValueError(f"{field.name}: expected shape ({n}, {t.list_size}), got {value.shape}")
        return pa.FixedSizeListArray.from_arrays(pa.array(value.reshape(-1).astype(np.float32)), t.list_size)
    if pa.types.is_list(t):
        if value.ndim != 2:
            raise ValueError(f"{field.name}: expected a 2-D array, got shape {value.shape}")
        offsets = pa.array(np.arange(n + 1, dtype=np.int32) * value.shape[1])
        return pa.ListArray.from_arrays(offsets, pa.array(value.reshape(-1).astype(np.float32)))
    return pa.array(value.astype(t.to_pandas_dtype()), type=t)


def column_matrix(table: pa.Table, name: str) -> np.ndarray:
    """A list<float> column of a source table as a dense (N, k) float64 array (fast path, no Python objects)."""
    col = table[name].combine_chunks()
    n = len(col)
    if pa.types.is_list(col.type) or pa.types.is_fixed_size_list(col.type) or pa.types.is_large_list(col.type):
        if col.null_count:
            raise ValueError(f"source column {name!r} has {col.null_count} null rows")
        flat = col.flatten().to_numpy(zero_copy_only=False).astype(np.float64)
        if n and flat.size % n:
            raise ValueError(f"source column {name!r} is ragged")
        return flat.reshape(n, -1)
    return col.to_numpy(zero_copy_only=False).astype(np.float64)
