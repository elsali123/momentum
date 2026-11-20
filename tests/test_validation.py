import numpy as np
import pytest

from robot_ingest.config import RobotSpec
from robot_ingest.pipeline import select_candidates
from robot_ingest.schema import frames_table
from robot_ingest.validate import validate_frames

ROBOT = RobotSpec("test_arm", "Arm", "Gripper", tuple(f"j{i}" for i in range(6)), 0.1, True)
FPS = 10.0


def make_columns(n=30, **overrides):
    t = np.arange(n)
    pos = np.column_stack([0.5 + 0.001 * t, np.zeros(n), np.full(n, 0.3)])
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
    cols = {
        "frame_index": t,
        "timestamp_s": t / FPS,
        "obs_ee_pos": pos,
        "obs_ee_quat": quat,
        "obs_joint_pos": np.zeros((n, 6)),
        "obs_gripper_closure": np.linspace(0, 1, n),
        "act_ee_pos": pos,
        "act_ee_quat": quat,
        "act_gripper_closure": np.linspace(0, 1, n),
    }
    return cols | overrides


def validate(cols):
    return validate_frames(frames_table("ep", cols), ROBOT, FPS, min_frames=10)


def test_clean_episode_passes():
    v = validate(make_columns())
    assert v.errors == [] and v.warnings == []


def test_too_short_is_rejected():
    assert any("only 5 frames" in e for e in validate(make_columns(n=5)).errors)


def test_non_monotonic_timestamps_are_rejected():
    cols = make_columns()
    cols["timestamp_s"][5] = cols["timestamp_s"][4]
    assert any("strictly increasing" in e for e in validate(cols).errors)


def test_nan_is_rejected():
    cols = make_columns()
    cols["obs_ee_pos"][3, 1] = np.nan
    assert any("NaN" in e for e in validate(cols).errors)


def test_unit_mistakes_are_rejected():
    cols = make_columns()
    assert any("millimetres" in e for e in validate(cols | {"obs_ee_pos": cols["obs_ee_pos"] * 1000}).errors)
    assert any("degrees" in e for e in validate(cols | {"obs_joint_pos": np.full((30, 6), 90.0)}).errors)


def test_wrong_joint_count_is_rejected():
    assert any("7 values" in e for e in validate(make_columns(obs_joint_pos=np.zeros((30, 7)))).errors)


def test_teleport_is_a_warning_not_an_error():
    cols = make_columns()
    cols["obs_ee_pos"][10:, 0] += 0.5
    v = validate(cols)
    assert v.errors == [] and any("jumps" in w for w in v.warnings)


def test_frames_table_enforces_shapes():
    with pytest.raises(ValueError, match="obs_ee_quat"):
        frames_table("ep", make_columns(obs_ee_quat=np.zeros((30, 3))))


def test_candidate_selection_is_deterministic_and_seeded():
    available = list(range(1000))
    assert select_candidates(available, seed=0) == select_candidates(list(reversed(available)), seed=0)
    assert select_candidates(available, seed=0)[:10] != select_candidates(available, seed=1)[:10]
    assert sorted(select_candidates(available, seed=0)) == available  # a permutation: rejects get replaced
    assert select_candidates(available, seed=0, explicit=(5, 3)) == [5, 3]
    with pytest.raises(ValueError):
        select_candidates(available, seed=0, explicit=(5000,))
