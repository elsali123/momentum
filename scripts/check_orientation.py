"""Onboarding check: are a source's end-effector poses consistent with its own joint angles?

Run once when a source is onboarded (or its revision changes). It normalizes a sample of episodes
with the source's mapper and fits the reported poses to forward kinematics of the joints:

  orientation residual  quaternion order / Euler convention is right        (~0 deg)
  base mismatch         orientation and position use the same base frame    (~0 deg)
  tool z alignment      tool z-axis is the gripper approach direction       (+1)
  EE offset             where the reported EE point sits on the tool axis   (configs: ee_offset_from_flange_m)

    uv run python scripts/check_orientation.py                  # canonical output of every source
    uv run python scripts/check_orientation.py --as-declared    # RH20T as the port labels it (shows the bugs)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from robot_ingest.config import load_config
from robot_ingest.kinematics import CHAINS, fit_pose_to_fk
from robot_ingest.pipeline import select_candidates
from robot_ingest.schema import column_matrix
from robot_ingest.sources import MAPPERS
from robot_ingest.sources.lerobot import LeRobotRepo
from robot_ingest.sources.rh20t import OrientationFix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="configs/sources.toml")
    parser.add_argument("--cache-dir", default="data/hf_cache")
    parser.add_argument("--source", action="append", help="source_id (repeatable; default all)")
    parser.add_argument("--episodes", type=int, default=20, help="episodes sampled per source")
    parser.add_argument("--as-declared", action="store_true", help="RH20T: use the port's declared xyzw without fixes")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    config = load_config(args.config)
    print(f"{'source':12s} {'orient. resid':>13s} {'base mismatch':>13s} {'tool z align':>12s} "
          f"{'EE offset (flange frame)':>26s} {'pos. resid':>10s}  verdict")
    ok = True
    for spec in [config.source(s) for s in args.source] if args.source else config.sources:
        robot = config.robots[spec.robot_id]
        mapper = MAPPERS[spec.mapper](spec, robot, LeRobotRepo(spec.repo_id, spec.revision, Path(args.cache_dir)))
        if args.as_declared and hasattr(mapper, "orientation_fix"):
            mapper.orientation_fix = OrientationFix("xyzw", Rotation.identity(), Rotation.identity())

        joints, pos, quat = [], [], []
        for index in select_candidates(mapper.episode_indices(), seed=123)[:args.episodes]:
            frames = mapper.normalize(index).frames
            joints.append(column_matrix(frames, "obs_joint_pos"))
            pos.append(column_matrix(frames, "obs_ee_pos"))
            quat.append(column_matrix(frames, "obs_ee_quat"))
        fit = fit_pose_to_fk(CHAINS[robot.kinematics], *map(np.concatenate, (joints, pos, quat)))

        good = fit.orientation_residual_deg < 1 and fit.base_mismatch_deg < 2 and fit.tool_z_alignment > 0.99
        ok &= good
        print(f"{spec.source_id:12s} {fit.orientation_residual_deg:11.2f} deg {fit.base_mismatch_deg:9.1f} deg "
              f"{fit.tool_z_alignment:12.2f} {np.array2string(fit.ee_offset_m, precision=3, suppress_small=True):>24s} m "
              f"{fit.position_residual_mm:7.1f} mm  {'OK' if good else 'INCONSISTENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
