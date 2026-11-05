"""Move the reference motions clear of the contacts the sim and reward now have: the self contacts of
gear_sonic.supermimic.self_clear, and the floor under hands, knees and the rest (floorcol_* geoms). The retargeted motions put a
hand in the hip or thigh, because echo's hips are wider than the human's, cross their legs, and in crawling and kneeling clips
put hands and knees under the floor. Per clip, adjust as little and as smoothly as will clear them: the arm joints; the leg
joints only in clips whose legs touch or go under the floor; the root's height only in clips under the floor. Feet are held
where they were, hardest while planted, so a swing foot moves rather than a stance foot sliding. Touching the floor stays
allowed: only depth under it is removed.

  PYTHONPATH=<gwbc-v6> python train/clear_arms.py <in.pkl> <out.pkl> [--margin 0.01]

Refuses to overwrite <out.pkl>.
"""
from __future__ import annotations

import argparse
import pathlib
import time

import joblib
import numpy as np
import torch

from self_clearance import TOUCH, Scan, report

ARM = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist')
LEG = ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')
# cost of moving each joint: the shoulder roll swings the arm out, and the hip roll the leg, with the least change to the look
WEIGHT = {'shoulder_roll': 0.2, 'shoulder_pitch': 1.0, 'shoulder_yaw': 1.0, 'elbow': 1.0, 'wrist': 2.0,
          'hip_roll': 0.5, 'hip_yaw': 0.5, 'hip_pitch': 1.0, 'knee': 1.0, 'ankle_pitch': 1.0, 'ankle_roll': 1.0}
FOOT_HOLD, FOOT_PLANTED = 50.0, 1000.0  # against squared foot movement in m, swinging and planted
PLANT_Z, PLANT_V = 0.03, 0.4  # a foot within this of the lowest it gets (m), moving slower than this (m/s), is planted
FLOOR_TOL = 0.0005  # m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('out')
    ap.add_argument('--margin', type=float, default=0.01)
    ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--xml', default=None)
    a = ap.parse_args()
    out_path = pathlib.Path(a.out)
    assert not out_path.exists(), f'{out_path} exists; this tool never overwrites a corpus'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    s = Scan(a.xml) if a.xml else Scan()
    m, sc, kin = s.m, s.sc, s.kin
    names = s.names
    arm = [i for i, n in enumerate(names) if any(k in n for k in ARM)]
    leg = [i for i, n in enumerate(names) if any(k in n for k in LEG)]
    wgt = torch.tensor([next((v for k, v in WEIGHT.items() if k in n), 1.0) for n in names], dtype=torch.float32)
    lo, hi = (torch.tensor(m.jnt_range[1:, k], dtype=torch.float32) for k in (0, 1))
    feet = kin.body_index(['L_foot', 'R_foot'])
    clips, out = joblib.load(a.src), {}
    moved, peak, planted_peak, foot_peak, lift = [], [], [], [], []
    legs_changed, roots_changed = 0, 0
    t0 = time.time()
    for n, (k, d) in enumerate(clips.items()):
        q0 = torch.tensor(d['dof'], dtype=torch.float32)
        pos0 = torch.tensor(d['root_trans_offset'], dtype=torch.float32)
        quat = torch.tensor(d['root_rot'][:, [3, 0, 1, 2]], dtype=torch.float32)
        T = q0.shape[0]
        P0, R0 = s.world(q0, pos0, quat)
        d0 = sc.depths(P0, R0, a.margin)
        floor0, sole0 = s.below_floor(P0, R0)
        under = bool(floor0.max() > FLOOR_TOL)
        if not under and not any(v.max() > 0 for v in d0.values()):
            out[k] = d
            continue
        legs_on = under or bool(d0['leg'].max() > 0)
        idx = arm + leg if legs_on else arm
        legs_changed += legs_on
        roots_changed += under
        ids = torch.tensor(idx)
        dt = 1.0 / d['fps']
        foot0 = P0[:, feet]
        speed = torch.zeros(T, 2)
        speed[1:] = (foot0[1:] - foot0[:-1]).norm(dim=-1) / dt
        planted = (foot0[:, :, 2] < foot0[:, :, 2].amin(0, keepdim=True) + PLANT_Z) & (speed < PLANT_V)
        hold = torch.where(planted, FOOT_PLANTED, FOOT_HOLD)

        def apply(dq, dz):
            q = q0.clone()
            q[:, ids] = torch.max(torch.min(q0[:, ids] + dq, hi[ids]), lo[ids])
            pos = pos0.clone()
            pos[:, 2] = pos0[:, 2] + dz
            return q, pos

        dq = torch.zeros(T, len(idx), requires_grad=True)
        dz = torch.zeros(T, requires_grad=under)
        opt = torch.optim.Adam([dq, dz] if under else [dq], lr=0.01)
        for _ in range(a.steps):
            q, pos = apply(dq, dz)
            P, R = s.world(q, pos, quat)
            depth = sc.depth(P, R, a.margin)
            floor, sole = s.below_floor(P, R)
            floor = floor + torch.relu(sole - sole0)  # a sole may stay as deep as the retarget left it, no deeper
            if float(depth.detach().max()) == 0.0 and float(floor.detach().max()) < FLOOR_TOL:
                break
            loss = 30.0 * depth.mean() + 30.0 * floor.mean() + 0.02 * (wgt[ids] * dq ** 2).sum(-1).mean()
            if legs_on:
                loss = loss + (hold * ((P[:, feet] - foot0) ** 2).sum(-1)).sum(-1).mean()
            if under:
                loss = loss + 0.2 * (dz ** 2).mean()
            if T > 2:
                loss = loss + 2e-3 * (((dq[1:] - dq[:-1]) / dt) ** 2).sum(-1).mean()
                if under:
                    loss = loss + 2e-3 * (((dz[1:] - dz[:-1]) / dt) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            q, pos = apply(dq, dz)
            P = s.world(q, pos, quat)[0]
            moved.append(float((q - q0)[:, ids].abs().mean()))
            peak.append(float((q - q0)[:, ids].abs().max()))
            f = (P[:, feet] - foot0).norm(dim=-1)
            foot_peak.append(float(f.max()))
            planted_peak.append(float(f[planted].max()) if planted.any() else 0.0)
            lift.append(float((pos - pos0)[:, 2].max()))
        out[k] = {**d, 'dof': q.numpy().astype(np.float32), 'root_trans_offset': pos.numpy().astype(d['root_trans_offset'].dtype)}
        if n % 20 == 0:
            print(f'{n}/{len(clips)} {time.time() - t0:.0f} s', flush=True)
    joblib.dump(out, out_path, compress=3)
    fr = lambda D: {k: (v['root_trans_offset'], v['root_rot'][:, [3, 0, 1, 2]], v['dof']) for k, v in D.items()}  # noqa: E731
    print(f'touching over {TOUCH * 1e3:.0f} mm, every frame. before:')
    report(s, fr(clips), step=1)
    print('after:')
    report(s, fr(out), step=1)
    print(f'{len(moved)} of {len(clips)} clips changed, legs in {legs_changed}, root height in {roots_changed} (up to {max(lift) * 100:.1f} cm); joint change mean {np.degrees(np.mean(moved)):.1f} deg, peak {np.degrees(np.max(peak)):.1f} deg; '
          f'foot moved: planted, clip peak median {np.median(planted_peak) * 100:.1f} cm, max {np.max(planted_peak) * 100:.1f} cm; any, max {np.max(foot_peak) * 100:.1f} cm')


if __name__ == '__main__':
    main()
