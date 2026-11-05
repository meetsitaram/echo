"""Move the reference motions' arms clear of the hips: 95 % of the retargeted clips have a hand in the hip or thigh, because
echo's hips are wider than the human's (at the zero pose the hand's inner edge is 2 cm inside the thigh's outer one).
Per clip, adjust the arm joints only, as little and as smoothly as will clear the selfcol_* spheres.

  SM_ROBOT_CFG=train/supermimic_robot.json PYTHONPATH=<gwbc-echo> python train/clear_arms.py <in.pkl> <out.pkl> [--margin 0.01]
"""
from __future__ import annotations

import argparse
import pathlib
import time

import joblib
import mujoco
import numpy as np
import torch

from gear_sonic.supermimic.kinematics import X2Kinematics
from gear_sonic.supermimic.self_clear import SelfClear

XML = pathlib.Path(__file__).resolve().parents[1] / 'urdf' / 'echo.xml'
ARM = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist')
# cost of moving each joint: the roll swings the arm out with the least change to the motion's look
WEIGHT = {'shoulder_roll': 0.2, 'shoulder_pitch': 1.0, 'shoulder_yaw': 1.0, 'elbow': 1.0, 'wrist': 2.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('out')
    ap.add_argument('--margin', type=float, default=0.01)
    ap.add_argument('--steps', type=int, default=150)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = mujoco.MjModel.from_xml_path(str(XML))
    kin, sc = X2Kinematics(m, dev), SelfClear(m, dev)
    names = [m.joint(i).name for i in range(1, m.njnt)]
    idx = [i for i, n in enumerate(names) if any(k in n for k in ARM)]
    w = torch.tensor([next(v for k, v in WEIGHT.items() if k in names[i]) for i in idx], device=dev, dtype=torch.float32)
    lo, hi = (torch.tensor(m.jnt_range[1:, k], device=dev, dtype=torch.float32)[idx] for k in (0, 1))
    clips, out = joblib.load(a.src), {}
    before, after, moved, peak = [], [], [], []
    t0 = time.time()
    for n, (k, d) in enumerate(clips.items()):
        q0 = torch.tensor(d['dof'], device=dev, dtype=torch.float32)
        T = q0.shape[0]
        z, eye = torch.zeros(T, 3, device=dev), torch.eye(3, device=dev).expand(T, 3, 3)
        qd0 = torch.zeros_like(q0)

        def depth(q, margin):
            P, R = kin.forward(z, eye, z, z, q, qd0)[:2]
            return sc.depth(P, R, margin)

        dq = torch.zeros(T, len(idx), device=dev, requires_grad=True)
        opt = torch.optim.Adam([dq], lr=0.01)
        for _ in range(a.steps):
            q = q0.clone()
            q[:, idx] = torch.max(torch.min(q0[:, idx] + dq, hi), lo)
            loss = 30.0 * depth(q, a.margin).mean() + 0.02 * (w * dq ** 2).sum(-1).mean()
            if T > 2:
                dt = 1.0 / d['fps']
                loss = loss + 2e-3 * (((dq[1:] - dq[:-1]) / dt) ** 2).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            q = q0.clone()
            q[:, idx] = torch.max(torch.min(q0[:, idx] + dq, hi), lo)
            before.append(float(depth(q0, 0.0).max()))
            after.append(float(depth(q, 0.0).max()))
            moved.append(float((q - q0)[:, idx].abs().mean()))
            peak.append(float((q - q0)[:, idx].abs().max()))
        out[k] = {**d, 'dof': q.cpu().numpy().astype(np.float32)}
        if n % 100 == 0:
            print(f'{n}/{len(clips)} {time.time() - t0:.0f} s', flush=True)
    joblib.dump(out, a.out, compress=3)
    b, f = np.array(before), np.array(after)
    print(f'{len(clips)} clips: summed overlap median {np.median(b) * 100:.1f} -> {np.median(f) * 100:.1f} cm, max {b.max() * 100:.1f} -> {f.max() * 100:.1f} cm; clear at 1 mm {(f < 0.001).mean() * 100:.0f}%; arm joint change mean {np.degrees(np.mean(moved)):.1f} deg, peak {np.degrees(np.max(peak)):.1f} deg')


if __name__ == '__main__':
    main()
