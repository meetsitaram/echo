"""Zero-action stand with the trainer's plant, on whichever robot SM_ROBOT_CFG selects (unset: the X2, as a control), for a few
default poses: where the centre of mass sits over the feet, and how long the robot stays up.

  SM_ROBOT_CFG=... PYTHONPATH=<gwbc-echo> python train/stand_control.py [hip knee ankle] ...   (echo-style triples to try)
"""
from __future__ import annotations

import sys

import mujoco
import numpy as np

from gear_sonic.supermimic.assets import SIM_DT, build_x2


def run(x, q0, secs=4.0):
    m = x.model
    d = mujoco.MjData(m)
    d.qpos[3:7], d.qpos[7:] = (1, 0, 0, 0), q0
    d.qpos[2] = 1.0
    mujoco.mj_kinematics(m, d)
    # the lowest collision point on a foot, over all colliding geoms on non-world bodies
    box = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
    lo, xs = [], []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == 0 or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        R = d.geom_xmat[g].reshape(3, 3)
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            c = d.geom_xpos[g] + (box * m.geom_size[g]) @ R.T
            lo.append(c[:, 2].min())
            xs += list(c[c[:, 2] < c[:, 2].min() + 0.01, 0])
        else:
            lo.append(d.geom_xpos[g][2] - m.geom_size[g][0])
            xs.append(d.geom_xpos[g][0])
    low = min(lo)
    d.qpos[2] -= low
    mujoco.mj_forward(m, d)
    com = d.subtree_com[1].copy()
    z0 = d.qpos[2]
    feet_x = (min(x_ for x_ in xs), max(x_ for x_ in xs))
    d.ctrl[:] = q0
    fell = None
    tq = []
    for s in range(int(secs / SIM_DT)):
        mujoco.mj_step(m, d)
        tq.append(np.abs(d.actuator_force).copy())
        if fell is None and (d.xmat[1].reshape(3, 3)[2, 2] < 0.8 or d.qpos[2] < z0 - 0.25):
            fell = d.time
    t = np.array(tq)[: int(0.5 / SIM_DT)].mean(0) / x.effort
    i = int(np.argmax(t))
    return com, feet_x, z0, fell, d.qpos[2], x.joint_names[i], float(t[i])


def main():
    x = build_x2()
    names = x.joint_names
    print(f'{x.plant.name}  mass {x.model.body_mass.sum():.1f} kg  pelvis start, com x, support x range, time to fall')
    def show(label, q):
        com, fx, z0, fell, z, j, t = run(x, q)
        print(f'{label:26s} pelvis {z0:.3f}  com x {com[0]:+.3f}  support [{fx[0]:+.3f} {fx[1]:+.3f}]  falls {"never in 4 s" if fell is None else f"at {fell:.2f} s"}  end z {z:.3f}  worst {j} {t:.2f} of peak')
    show('default pose', x.default_q)
    args = [float(a) for a in sys.argv[1:]]
    for k in range(0, len(args), 3):
        h, n, a = args[k:k + 3]
        q = x.default_q.copy()
        for i, name in enumerate(names):
            if 'hip_pitch' in name:
                q[i] = h
            elif 'knee' in name:
                q[i] = n
            elif 'ankle_pitch' in name:
                q[i] = a
        show(f'hip {h} knee {n} ankle {a}', q)


if __name__ == '__main__':
    main()
