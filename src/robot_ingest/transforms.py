"""Pure math used by the mappers: rotation conventions and gripper normalization.

No I/O here, so every convention decision is unit-tested in isolation (tests/test_transforms.py).
Canonical conventions (see DESIGN.md):
  * orientation = unit quaternion in **xyzw** order (scipy / ROS order)
  * quaternion sign is chosen for continuity over time, starting with w >= 0
  * gripper = closure in [0, 1]: 0 = fully open, 1 = fully closed
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def euler_xyz_to_quat(euler: np.ndarray) -> np.ndarray:
    """Roll/pitch/yaw (extrinsic x-y-z, radians) -> quaternion xyzw.

    Extrinsic "xyz" is scipy's lowercase convention and what DROID uses to produce its
    cartesian poses (droid/misc/transformations.py: R.from_euler("xyz", ...)).
    """
    return Rotation.from_euler("xyz", np.asarray(euler, dtype=np.float64)).as_quat()


def normalize_quat(q: np.ndarray) -> tuple[np.ndarray, float]:
    """Return unit quaternions plus the largest deviation of the input norms from 1 (a data-quality signal)."""
    q = np.asarray(q, dtype=np.float64)
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero-norm quaternion")
    return q / norms, float(np.max(np.abs(norms - 1.0)))


def make_quat_continuous(q: np.ndarray) -> np.ndarray:
    """Choose each quaternion's sign so consecutive frames sit in the same hemisphere.

    q and -q are the same rotation, but a sign flip is a jump of length ~2 in the vector a
    network regresses. The RH20T port forces qx >= 0, which creates thousands of such flips.
    We fix the first frame to w >= 0 (deterministic) and then propagate continuity:
    s_t = s_{t-1} * sign(<q_t, q_{t-1}>), i.e. a cumulative product of pairwise signs.
    """
    q = np.asarray(q, dtype=np.float64)
    if len(q) == 0:
        return q.copy()
    pair_sign = np.where(np.sum(q[1:] * q[:-1], axis=-1) < 0, -1.0, 1.0)
    first_sign = -1.0 if q[0, 3] < 0 else 1.0
    signs = first_sign * np.cumprod(np.concatenate([[1.0], pair_sign]))
    return q * signs[:, None]


def align_quat_to(q: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Flip q frame-by-frame into the hemisphere of `reference` (keeps action quats consistent with state quats)."""
    q = np.asarray(q, dtype=np.float64)
    flip = np.sum(q * reference, axis=-1) < 0
    out = q.copy()
    out[flip] *= -1.0
    return out


def width_to_closure(width_m: np.ndarray, max_width_m: float) -> np.ndarray:
    """Parallel-jaw opening width -> closure in [0, 1] (0 = fully open, 1 = fully closed). Not clipped."""
    return 1.0 - np.asarray(width_m, dtype=np.float64) / max_width_m


def clip_unit(x: np.ndarray) -> tuple[np.ndarray, int]:
    """Clip to [0, 1] and report how many values were outside (sensor overshoot, calibration offsets)."""
    x = np.asarray(x, dtype=np.float64)
    outside = int(np.sum((x < 0.0) | (x > 1.0)))
    return np.clip(x, 0.0, 1.0), outside
