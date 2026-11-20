import numpy as np
from scipy.spatial.transform import Rotation

from robot_ingest.transforms import (
    align_quat_to,
    clip_unit,
    euler_xyz_to_quat,
    make_quat_continuous,
    normalize_quat,
    width_to_closure,
)


def test_euler_identity_is_unit_quaternion_xyzw():
    np.testing.assert_allclose(euler_xyz_to_quat(np.zeros((1, 3))), [[0, 0, 0, 1]], atol=1e-12)


def test_euler_is_extrinsic_xyz_roll_then_pitch_then_yaw():
    roll, pitch, yaw = 0.3, -0.7, 1.9
    q = euler_xyz_to_quat(np.array([[roll, pitch, yaw]]))[0]

    def rx(a):
        return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])

    def ry(a):
        return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])

    def rz(a):
        return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])

    # extrinsic x, then y, then z  ==  R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    np.testing.assert_allclose(Rotation.from_quat(q).as_matrix(), rz(yaw) @ ry(pitch) @ rx(roll), atol=1e-12)


def test_quat_continuity_removes_sign_flips_and_keeps_rotations():
    rng = np.random.default_rng(0)
    smooth = Rotation.from_rotvec(np.linspace([0, 0, 2.5], [0.4, -0.3, 3.6], 50)).as_quat()
    flipped = smooth * rng.choice([-1.0, 1.0], size=(50, 1))  # what the RH20T port does to its quats

    fixed = make_quat_continuous(flipped)

    assert np.all(np.sum(fixed[1:] * fixed[:-1], axis=1) > 0), "no hemisphere jumps left"
    assert fixed[0, 3] >= 0, "deterministic start: w >= 0"
    np.testing.assert_allclose(np.abs(np.sum(fixed * flipped, axis=1)), 1.0, atol=1e-12)  # same rotations


def test_align_quat_to_reference_hemisphere():
    ref = np.array([[0, 0, 0, 1.0], [0, 0, 0, 1.0]])
    q = np.array([[0, 0, 0.1, -0.995], [0, 0, 0.1, 0.995]])
    np.testing.assert_allclose(align_quat_to(q, ref)[:, 3], [0.995, 0.995])


def test_normalize_quat_reports_deviation():
    q, deviation = normalize_quat(np.array([[0, 0, 0, 2.0]]))
    np.testing.assert_allclose(q, [[0, 0, 0, 1]])
    assert deviation == 1.0


def test_gripper_width_to_closure():
    np.testing.assert_allclose(width_to_closure(np.array([0.0, 0.055, 0.110]), 0.110), [1.0, 0.5, 0.0])
    clipped, outside = clip_unit(np.array([-0.1, 0.5, 1.2]))
    np.testing.assert_allclose(clipped, [0.0, 0.5, 1.0])
    assert outside == 2
