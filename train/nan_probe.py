"""Where does echo's physics blow up? Runs the training sim with noisy actions on a motion set and, at the first non-finite or
huge state, prints the step, the worlds, their context (armature and friction scale, friction) and the worst joint.

  SM_ROBOT_CFG=train/supermimic_robot.json PYTHONPATH=<gwbc-echo> python train/nan_probe.py <motion.pkl> [steps] [action noise]
"""
from __future__ import annotations

import sys

import joblib
import torch

from gear_sonic.supermimic.assets import build_x2
from gear_sonic.supermimic.kinematics import X2Kinematics
from gear_sonic.supermimic.mdp import MDP, XI_ACT, XI_MASS
from gear_sonic.supermimic.motion import MotionLib
from gear_sonic.supermimic.sim import TrackingSim


def main():
    path, steps, noise = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3000, float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    dev = 'cuda'
    x2 = build_x2()
    motion = MotionLib(joblib.load(path), X2Kinematics(x2.model, dev), dev)
    f32 = lambda v: torch.tensor(v, device=dev, dtype=torch.float32)  # noqa: E731
    mdp = MDP(motion, f32(x2.default_q), f32(x2.q_lo), f32(x2.q_hi))
    N = 2048
    sim = TrackingSim(N, motion, mdp, x2=x2, seed=0, dr='sonic2')
    sim.reset_all()
    big = 0
    for t in range(steps):
        a = noise * torch.randn(N, len(x2.joint_names), device=dev)
        pre = sim.state()
        q0, qd0 = pre[4].clone(), pre[5].clone()
        p0 = pre[0].clone()
        xi = sim.xi.clone()
        f = sim.f.clone()
        # step without the trainer's reset hiding the state: read physics right after
        sim._physics(sim.default_q + xi[:, 1:30] + sim.alpha * a)
        bad = ~torch.isfinite(sim.qpos).all(1) | ~torch.isfinite(sim.qvel).all(1) | (sim.qvel.abs().max(1).values > 300)
        if bad.any():
            ids = torch.nonzero(bad).flatten()[:6]
            print(f'step {t}: {int(bad.sum())} of {N} worlds')
            for i in ids.tolist():
                j = int(qd0[i].abs().argmax())
                v = sim.qvel[i]
                k = int(torch.nan_to_num(v[6:], nan=1e9).abs().argmax())
                print(f'  world {i}: arm x{xi[i, XI_ACT.start]:.2f} fric x{xi[i, XI_ACT.start + 1]:.2f} floor mu {xi[i, 0]:.2f} mass {xi[i, XI_MASS].tolist()} clip {motion.names[motion.clip_id[f[i]]]}'
                      f'\n    before: root z {p0[i, 2]:.2f} v_root {qd0[i].abs().max():.1f}, fastest joint {x2.joint_names[j]} {qd0[i, j]:.1f}; after: worst {x2.joint_names[k]} {v[6 + k]:.1f}, root {v[:6].tolist()}')
            big += 1
            if big >= 3:
                return
            sim.reset(torch.nonzero(bad).flatten())
            continue
        sim.f = sim.f  # frame kept: only the physics is probed
        if t % 40 == 39:
            sim.reset(torch.arange(N, device=dev)[torch.rand(N, device=dev) < 0.5])
    print('no blow-up in', steps, 'steps')


if __name__ == '__main__':
    main()
