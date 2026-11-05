"""Before training echo: is it really our robot, and does it stand? (LESSONS: a reused recipe can drop a robot at every reset.)

  SM_ROBOT_CFG=train/supermimic_robot.json PYTHONPATH=<gwbc-echo> python train/pre_run_check.py

Prints the compiled model's source, mass and each joint's values against spec/actuators.json, then, with the trainer's own
5 ms physics and PD servos, sets the default pose on the floor, holds it with zero actions for 2 s and reads the numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.supermimic.assets import DECIMATION, MJCF_PATH, SIM_DT, build_x2

ROOT = Path(__file__).resolve().parents[1]


def main():
    x = build_x2()
    m = x.model
    spec = json.loads((ROOT / 'spec' / 'robot.json').read_text(encoding='utf-8'))
    cap = {j['name']: j['capacity'] for j in spec['joints']}
    print(f'model {MJCF_PATH}\nplant {x.plant.name}  mass {m.body_mass.sum():.2f} kg  nq {m.nq} nu {m.nu} ngeom {m.ngeom} dt {m.opt.timestep} x {DECIMATION}')
    print(f"{'joint':18s}{'kp':>7s}{'kd':>7s}{'effort':>8s}{'peak':>7s}{'rated':>7s}{'noload':>8s}{'arm':>9s}{'alpha':>7s}{'q0':>7s}{'lo':>7s}{'hi':>7s}")
    bad = []
    for i, n in enumerate(x.joint_names):
        c = cap[n]
        a = spec['actuators'][c['model']]
        peak, rated, speed = a['peak_nm'] * c['count'] * c['ratio'], a['rated_nm'] * c['count'] * c['ratio'], a['noload_rads'] / c['ratio']
        if abs(peak - x.effort[i]) > 0.5:
            bad.append(f'{n}: effort {x.effort[i]} vs parts {peak}')
        if not x.q_lo[i] <= x.default_q[i] <= x.q_hi[i]:
            bad.append(f'{n}: default {x.default_q[i]} outside limits')
        print(f'{n:18s}{x.kp[i]:7.1f}{x.kd[i]:7.2f}{x.effort[i]:8.1f}{peak:7.1f}{rated:7.1f}{speed:8.1f}{x.armature[i]:9.5f}{x.action_scale[i]:7.3f}{x.default_q[i]:7.2f}{x.q_lo[i]:7.2f}{x.q_hi[i]:7.2f}')
    print('problems:', bad or 'none')

    d = mujoco.MjData(m)
    d.qpos[3:7], d.qpos[7:] = (1, 0, 0, 0), x.default_q
    d.qpos[2] = 1.0
    mujoco.mj_kinematics(m, d)
    sole = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in ('L_foot_sole', 'R_foot_sole')]
    corners = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
    low = min((d.geom_xpos[g][2] + (corners * m.geom_size[g] @ d.geom_xmat[g].reshape(3, 3)[2]).min()) for g in sole)
    d.qpos[2] -= low
    z0 = d.qpos[2]
    d.ctrl[:] = x.default_q
    forces = []
    for s in range(int(2.0 / SIM_DT)):
        mujoco.mj_step(m, d)
        forces.append(np.abs(d.actuator_force).copy())
        if s % 100 == 99:
            up = d.xmat[1].reshape(3, 3)[2, 2]
            fz = sum(mujoco.mj_contactForce(m, d, c, np.zeros(6)) or 0 for c in [])
            print(f'  t {d.time:4.1f} s  pelvis z {d.qpos[2]:.3f} (start {z0:.3f})  tilt {np.degrees(np.arccos(np.clip(up, -1, 1))):5.2f} deg  contacts {d.ncon}')
    f = np.array(forces)[-100:].mean(0)
    w = m.body_mass.sum() * 9.81
    fn = 0.0
    r = np.zeros(6)
    for c in range(d.ncon):
        mujoco.mj_contactForce(m, d, c, r)
        fn += r[0]
    print(f'floor carries {fn:.0f} N of {w:.0f} N weight')
    worst = np.argsort(-f / x.effort)[:5]
    print('holding torque, worst:', [(x.joint_names[i], round(float(f[i]), 1), round(float(f[i] / x.effort[i]), 2)) for i in worst])


if __name__ == '__main__':
    main()
