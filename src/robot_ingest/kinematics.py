"""Minimal forward kinematics (Denavit-Hartenberg), used to check and repair end-effector orientation.

Why this exists: the RH20T port declares its quaternions as xyzw, but checking them against the
orientation implied by the recorded joint angles shows cfg3 is really wxyz and cfg7's stored
orientation doesn't match the kinematics in any standard reading (DESIGN.md, D6). A reported pose
must agree with FK(joints) up to two *constant* rotations (base mounting, tool mounting), so
fitting those constants and looking at the residual catches convention bugs automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

PI = np.pi


@dataclass(frozen=True)
class DHChain:
    convention: str  # "standard" (T = Rz Tz Tx Rx) or "modified"/Craig (T = Rx Tx Rz Tz)
    a: tuple[float, ...]
    d: tuple[float, ...]
    alpha: tuple[float, ...]


# Nominal parameters from the manufacturers' published DH tables.
CHAINS = {
    "ur5": DHChain("standard", a=(0, -0.425, -0.39225, 0, 0, 0), d=(0.089159, 0, 0, 0.10915, 0.09465, 0.0823),
                   alpha=(PI / 2, 0, 0, PI / 2, -PI / 2, 0)),
    "kuka_iiwa14": DHChain("standard", a=(0,) * 7, d=(0.36, 0, 0.42, 0, 0.40, 0, 0.126),
                           alpha=(-PI / 2, PI / 2, PI / 2, -PI / 2, -PI / 2, PI / 2, 0)),
    "franka_panda": DHChain("modified", a=(0, 0, 0, 0.0825, -0.0825, 0, 0.088), d=(0.333, 0, 0.316, 0, 0.384, 0, 0.107),
                            alpha=(0, -PI / 2, PI / 2, PI / 2, -PI / 2, PI / 2, PI / 2)),
}


def forward_kinematics(chain: DHChain, joints: np.ndarray) -> np.ndarray:
    """Flange poses (N, 4, 4) in the arm's base frame for joint angles (N, dof)."""
    joints = np.asarray(joints, dtype=np.float64)
    n = len(joints)
    pose = np.tile(np.eye(4), (n, 1, 1))
    for i in range(joints.shape[1]):
        ct, st = np.cos(joints[:, i]), np.sin(joints[:, i])
        ca, sa = np.cos(chain.alpha[i]), np.sin(chain.alpha[i])
        a, d = chain.a[i], chain.d[i]
        T = np.zeros((n, 4, 4))
        if chain.convention == "standard":
            T[:, 0] = np.stack([ct, -st * ca, st * sa, a * ct], 1)
            T[:, 1] = np.stack([st, ct * ca, -ct * sa, a * st], 1)
            T[:, 2] = np.array([0, sa, ca, d])
        else:
            T[:, 0] = np.stack([ct, -st, np.zeros(n), np.full(n, a)], 1)
            T[:, 1] = np.stack([st * ca, ct * ca, np.full(n, -sa), np.full(n, -sa * d)], 1)
            T[:, 2] = np.stack([st * sa, ct * sa, np.full(n, ca), np.full(n, ca * d)], 1)
        T[:, 3, 3] = 1
        pose = pose @ T
    return pose


def _closest_rotation(m: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(m)
    return u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt


def _angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation angle (degrees) between stacks of rotation matrices."""
    return np.degrees(Rotation.from_matrix(np.einsum("...ji,...jk->...ik", a, b)).magnitude())


def fit_rotation_frames(measured: np.ndarray, fk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Constant rotations (A, B) with measured_t ~= A @ fk_t @ B (A: base mounting, B: tool mounting).

    B comes first, from body-frame relative rotations where A cancels:
    rotvec(Rm_t^T Rm_t+k) = B^T rotvec(Rf_t^T Rf_t+k). Then A is one Procrustes fit.
    Both steps are closed-form, so there are no local minima to get stuck in.
    """
    k = max(1, len(measured) // 10)
    dm = Rotation.from_matrix(np.einsum("nji,njk->nik", measured[:-k], measured[k:])).as_rotvec()
    df = Rotation.from_matrix(np.einsum("nji,njk->nik", fk[:-k], fk[k:])).as_rotvec()
    B = _closest_rotation(df.T @ dm)  # dm ~= df @ B
    A = _closest_rotation(np.einsum("nij,kj,nlk->il", measured, B, fk))  # sum_t Rm_t B^T Rf_t^T
    return A, B


def orientation_residual_deg(measured: np.ndarray, fk: np.ndarray) -> np.ndarray:
    """Per-frame angle (degrees) between measured rotations and the best constant-frame fit of FK."""
    A, B = fit_rotation_frames(measured, fk)
    return _angle_deg(A @ fk @ B, measured)


@dataclass(frozen=True)
class PoseFit:
    """How well reported EE poses match FK(joints). Used at onboarding (scripts/check_orientation.py)."""

    orientation_residual_deg: float  # median; free constant base and tool rotations
    position_residual_mm: float  # median; fitted base rotation/offset and tool offset
    base_mismatch_deg: float  # base frame implied by orientations vs by positions (should be ~0)
    tool_z_alignment: float  # cosine between the reported tool z-axis and the flange z-axis (1 = approach)
    ee_offset_m: np.ndarray  # reported EE point in the flange frame


def fit_pose_to_fk(chain: DHChain, joints: np.ndarray, position: np.ndarray, quat_xyzw: np.ndarray) -> PoseFit:
    fk = forward_kinematics(chain, joints)
    rot_fk, pos_fk = fk[:, :3, :3], fk[:, :3, 3]
    measured = Rotation.from_quat(quat_xyzw).as_matrix()
    base_rot, tool_rot = fit_rotation_frames(measured, rot_fk)

    # position = A (pos_fk + rot_fk @ offset) + c. Alternate: Kabsch for A, least squares for (offset, c).
    n = len(position)
    base_pos, offset, c = np.eye(3), np.zeros(3), np.zeros(3)
    for _ in range(50):
        tip = pos_fk + rot_fk @ offset
        base_pos = _closest_rotation((position - position.mean(0)).T @ (tip - tip.mean(0)))
        lhs = np.concatenate([base_pos @ rot_fk, np.broadcast_to(np.eye(3), (n, 3, 3))], axis=2).reshape(-1, 6)
        sol, *_ = np.linalg.lstsq(lhs, (position - pos_fk @ base_pos.T).reshape(-1), rcond=None)
        offset, c = sol[:3], sol[3:]
    pos_error = np.linalg.norm(position - (pos_fk + rot_fk @ offset) @ base_pos.T - c, axis=1)

    return PoseFit(
        orientation_residual_deg=float(np.median(_angle_deg(base_rot @ rot_fk @ tool_rot, measured))),
        position_residual_mm=float(1000 * np.median(pos_error)),
        base_mismatch_deg=float(_angle_deg(base_pos, base_rot)),
        tool_z_alignment=float(tool_rot[2, 2]),
        ee_offset_m=offset,
    )
