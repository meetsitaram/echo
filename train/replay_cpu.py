"""Second opinion on the GPU sim: runs the trained policy closed loop in plain CPU MuJoCo on the same model, and checks the floor
carries the robot. The GPU sim (MuJoCo Warp) only builds the observations here; every physics step is mujoco.mj_step.

  SM_ROBOT_CFG=train/supermimic_robot.json PYTHONPATH=<gwbc-echo> python train/replay_cpu.py <ckpt.pt> <motion.pkl> [--only regex] [--exclude regex] [--clips N] [--seconds S] [--out f.json]

Per clip: seconds run, whether it stayed upright, mean pelvis error from the reference, and mean floor force / weight.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import mujoco
import numpy as np
import torch

from gear_sonic.supermimic.assets import CONTROL_DT, build_x2
from gear_sonic.supermimic.kinematics import X2Kinematics
from gear_sonic.supermimic.mdp import MDP
from gear_sonic.supermimic.motion import MotionLib
from gear_sonic.supermimic.networks import Policy
from gear_sonic.supermimic.robot_cfg import CFG
from gear_sonic.supermimic.sim import TrackingSim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('motion')
    ap.add_argument('--only', default='.')
    ap.add_argument('--exclude', default='')
    ap.add_argument('--clips', type=int, default=24)
    ap.add_argument('--seconds', type=float, default=10.0)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    dev = 'cuda'
    data = joblib.load(a.motion)
    keys = sorted(k for k in data if re.search(a.only, k, re.I) and not (a.exclude and re.search(a.exclude, k, re.I)))[:a.clips]
    x2 = build_x2()
    motion = MotionLib({k: data[k] for k in keys}, X2Kinematics(x2.model, dev), dev)
    f32 = lambda v: torch.tensor(v, device=dev, dtype=torch.float32)  # noqa: E731
    mdp = MDP(motion, f32(x2.default_q), f32(x2.q_lo), f32(x2.q_hi))
    N = len(keys)
    sim = TrackingSim(N, motion, mdp, x2=x2, eval_mode=True, seed=1)
    policy = Policy().to(dev)
    policy.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=False)['pi'])
    policy.eval()
    starts, ends = motion.clip_start[:N].clone(), motion.clip_end[:N].clone()
    sim.reset_all(starts.clone())
    m = x2.model
    assert m.nu == len(x2.joint_names)
    for i in range(m.ngeom):
        if m.geom_bodyid[i] == 0:
            m.geom_friction[i, 0] = 1.0  # as the GPU evaluation's floor
    effort = np.array(x2.effort, dtype=float)
    noload = sim.noload[0].cpu().numpy()
    knee = CFG['torque_speed_knee']
    ds = [mujoco.MjData(m) for _ in range(N)]
    for i, d in enumerate(ds):
        d.qpos[:], d.qvel[:] = sim.qpos[i].cpu().numpy(), sim.qvel[i].cpu().numpy()
        mujoco.mj_forward(m, d)
    sub = round(CONTROL_DT / m.opt.timestep)
    weight = mujoco.mj_getTotalmass(m) * 9.81
    alive = np.ones(N, bool)
    fell = np.full(N, -1.0)
    err, floor, steps = np.zeros(N), np.zeros(N), np.zeros(N)
    T = min(int((ends - starts).max()), int(a.seconds / CONTROL_DT))
    for t in range(T):
        with torch.no_grad():
            act = policy(sim.policy_input()).clamp(-20.0, 20.0)
        ctrl = (sim.default_q + sim.xi[:, 1:1 + len(x2.joint_names)] + sim.alpha * act).cpu().numpy()
        f1 = sim.f + 1
        ref = motion.root_p[f1].cpu().numpy()
        for i, d in enumerate(ds):
            if not alive[i]:
                continue
            d.ctrl[:] = ctrl[i]
            fz = 0.0
            for _ in range(sub):
                u = d.qvel[6:] / noload
                m.actuator_forcerange[:, 1] = effort * np.clip((1 - u) / (1 - knee), 0, 1)
                m.actuator_forcerange[:, 0] = -effort * np.clip((1 + u) / (1 - knee), 0, 1)
                mujoco.mj_step(m, d)
                w = np.zeros(6)
                for c in range(d.ncon):
                    mujoco.mj_contactForce(m, d, c, w)
                    fz += w[0] * abs(d.contact[c].frame[2])
            lean = np.arccos(np.clip(1 - 2 * (d.qpos[4] ** 2 + d.qpos[5] ** 2), -1, 1))
            if lean > 1.0 or ref[i, 2] - d.qpos[2] > 0.35:
                alive[i], fell[i] = False, t * CONTROL_DT
            err[i] += np.linalg.norm(d.qpos[:2] - ref[i, :2])
            floor[i] += fz / sub
            steps[i] += 1
            sim.qpos[i] = torch.as_tensor(d.qpos, device=dev, dtype=torch.float32)
            sim.qvel[i] = torch.as_tensor(d.qvel, device=dev, dtype=torch.float32)
            if int(f1[i]) >= int(ends[i]) - 1:
                alive[i] = False
        sim.f = torch.minimum(f1, ends - 1)
        sim.a_prev = act.clone()
        sim.obs_hist = torch.cat((sim.obs_hist[:, 1:], sim._obs().unsqueeze(1)), 1)
        if not alive.any():
            break
    per = lambda v: v / np.maximum(steps, 1)  # noqa: E731
    res = {k: dict(seconds=round(float(steps[i]) * CONTROL_DT, 2), fell_at_s=float(fell[i]), xy_err_m=round(float(per(err)[i]), 3), floor_over_weight=round(float(per(floor)[i] / weight), 3)) for i, k in enumerate(keys)}
    for k, v in res.items():
        print(k[:48].ljust(50), v)
    print(f'{N} clips in CPU MuJoCo, {steps.sum() * CONTROL_DT:.0f} s, fell {int((fell >= 0).sum())}, xy err median {np.median(per(err)):.3f} m, floor/weight median {np.median(per(floor) / weight):.3f}')
    if a.out:
        a.out.write_text(json.dumps(res, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
