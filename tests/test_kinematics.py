import numpy as np
from scipy.spatial.transform import Rotation

from robot_ingest.kinematics import CHAINS, fit_pose_to_fk, forward_kinematics, orientation_residual_deg


def test_known_zero_pose_flange_positions():
    # Published zero-configuration flange positions of each arm.
    franka = forward_kinematics(CHAINS["franka_panda"], np.zeros((1, 7)))[0, :3, 3]
    ur5 = forward_kinematics(CHAINS["ur5"], np.zeros((1, 6)))[0, :3, 3]
    np.testing.assert_allclose(franka, [0.088, 0, 0.926], atol=1e-6)
    np.testing.assert_allclose(ur5, [-0.81725, -0.19145, -0.005491], atol=1e-6)


def random_joint_path(dof, n=200, seed=0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0, 0.03, size=(n, dof)), axis=0) + rng.uniform(-1, 1, dof)


def test_residual_is_zero_for_constant_frame_offsets_and_large_for_wrong_quaternion_order():
    fk = forward_kinematics(CHAINS["kuka_iiwa14"], random_joint_path(7))[:, :3, :3]
    base, tool = Rotation.random(random_state=1).as_matrix(), Rotation.random(random_state=2).as_matrix()
    measured = base @ fk @ tool
    assert np.median(orientation_residual_deg(measured, fk)) < 1e-6

    quat = Rotation.from_matrix(measured).as_quat()
    misread = Rotation.from_quat(quat[:, [3, 0, 1, 2]]).as_matrix()  # wxyz data read as xyzw
    assert np.median(orientation_residual_deg(misread, fk)) > 2


def test_pose_fit_recovers_tool_offset_and_detects_base_frame_mismatch():
    joints = random_joint_path(6)
    fk = forward_kinematics(CHAINS["ur5"], joints)
    rot, pos = fk[:, :3, :3], fk[:, :3, 3]
    tip = pos + rot @ np.array([0, 0, 0.25])  # a TCP 25 cm along the flange z-axis

    consistent = fit_pose_to_fk(CHAINS["ur5"], joints, tip, Rotation.from_matrix(rot).as_quat())
    assert consistent.base_mismatch_deg < 0.5 and consistent.tool_z_alignment > 0.99
    np.testing.assert_allclose(consistent.ee_offset_m, [0, 0, 0.25], atol=1e-3)

    turned = Rotation.from_euler("z", np.pi).as_matrix() @ rot  # orientation in a base frame turned 180 deg
    assert fit_pose_to_fk(CHAINS["ur5"], joints, tip, Rotation.from_matrix(turned).as_quat()).base_mismatch_deg > 170
