"""Why are the ankle pitch motors loaded? Where the centre of mass sits against the ankle axes, in the robot's heading frame, in the
default pose and through recorded takes. The static torque one ankle holds is weight x offset / 2. Also how far each take's reference
leans, so a bow in the motion isn't read as a fall.

  SM_ROBOT_CFG=train/supermimic_robot.json PYTHONPATH=<gwbc-echo> python train/ankle_load.py <take.npz> [...] [--ref motion.pkl]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import mujoco
import numpy as np

from gear_sonic.supermimic.assets import build_x2


def offset(m, d, joint):
    """CoM x ahead of the mean ankle pitch axis, in the pelvis's heading frame, m."""
    mujoco.mj_forward(m, d)
    com = d.subtree_com[1]
    ank = 0.5 * sum(d.xanchor[m.joint(f'{s}_ankle_pitch').id] for s in 'LR')
    w, x, y, z = d.qpos[3:7]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return (com - ank)[:2] @ np.array([np.cos(yaw), np.sin(yaw)])


def lean_of(q):
    """Lean from upright of xyzw quaternions, rad."""
    return np.arccos(np.clip(1 - 2 * (q[:, 0] ** 2 + q[:, 1] ** 2), -1, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('takes', nargs='*', type=Path)
    ap.add_argument('--ref', type=Path, default=None)
    a = ap.parse_args()
    x2 = build_x2()
    m = x2.model
    d = mujoco.MjData(m)
    W = mujoco.mj_getTotalmass(m) * 9.81
    d.qpos[:] = 0
    d.qpos[2], d.qpos[3] = 1.0, 1.0
    o = offset(m, d, None)
    print(f'zero pose: CoM {o * 1000:+.0f} mm ahead of the ankle axes, {W * o / 2:+.1f} N m per ankle')
    d.qpos[7:] = x2.default_q
    o = offset(m, d, None)
    print(f'default pose: CoM {o * 1000:+.0f} mm, {W * o / 2:+.1f} N m per ankle')
    ref = joblib.load(a.ref) if a.ref else {}
    for f in a.takes:
        t = np.load(f)
        os_ = []
        for r, q, g in zip(t['root_pos'][::5], t['root_quat_wxyz'][::5], t['q'][::5]):
            d.qpos[:3], d.qpos[3:7], d.qpos[7:] = r, q, g
            os_.append(offset(m, d, None))
        os_ = np.array(os_)
        w = t['root_quat_wxyz']
        lean = np.degrees(np.arccos(np.clip(1 - 2 * (w[:, 1] ** 2 + w[:, 2] ** 2), -1, 1)))
        s = f'{f.stem[:44]:46s}CoM ahead {os_.mean() * 1000:+4.0f} mm mean ({W * os_.mean() / 2:+5.1f} N m per ankle), pelvis lean {lean.mean():4.1f} mean {lean.max():4.1f} max'
        if f.stem in ref:
            s += f', reference lean max {np.degrees(lean_of(ref[f.stem]["root_rot"])).max():.1f}'
        print(s)


if __name__ == '__main__':
    main()
