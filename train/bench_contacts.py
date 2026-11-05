"""What floor contact for the whole body costs in MuJoCo Warp: steps per second, contacts and GPU memory, against a model
without it, turn about so a training run on the same GPU loads both alike. Brief by design.

  SM_ROBOT_CFG=train/supermimic_robot_v6.json PYTHONPATH=<gwbc-v6> python train/bench_contacts.py <baseline.xml> <urdf/echo.xml> [--nworld 300]

<baseline.xml>: the same XML without the floorcol_* geoms. Scenes: standing, put back on its pose every 10 control steps as it
has no policy to hold it, and fallen (dropped from 0.5 m in random orientations with the servos holding the default pose, the worst
contact load). Prints each model's control steps per second over the rounds, the most contacts and constraint rows any world
used, and memory by nconmax. Also a kneeling hold on CPU: a kneeling clip's first frame on each model, servos at that pose.
"""
from __future__ import annotations

import argparse
import pathlib
import time

import joblib
import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from gear_sonic.supermimic.assets import DECIMATION, build_x2

KNEEL = 'kneeling_stop_002__A039'


def make(xml, nworld, nconmax, njmax, scene, seed=0):
    x2 = build_x2(mjcf_path=pathlib.Path(xml))
    m = x2.model
    d = mujoco.MjData(m)
    d.qpos[2], d.qpos[3], d.qpos[7:] = 0.88, 1.0, x2.default_q
    d.ctrl[:] = x2.default_q
    mujoco.mj_forward(m, d)
    used0 = wp.get_mempool_used_mem_current('cuda:0')
    mm = mjw.put_model(m)
    dd = mjw.put_data(m, d, nworld=nworld, nconmax=nconmax, njmax=njmax)
    used = wp.get_mempool_used_mem_current('cuda:0') - used0
    if scene == 'fallen':
        rng = np.random.default_rng(seed)
        q = rng.normal(size=(nworld, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        qpos = wp.to_torch(dd.qpos)
        qpos[:, 2] = 0.5
        qpos[:, 3:7] = torch.tensor(q, dtype=qpos.dtype, device=qpos.device)
    return m, mm, dd, used


def run(mm, dd, settle, steps, hold_every=0):
    """hold_every: put each world back on its start pose this often, so a robot with no policy stays on its feet."""
    qpos, qvel = wp.to_torch(dd.qpos), wp.to_torch(dd.qvel)
    qpos0, qvel0 = qpos.clone(), qvel.clone()
    def sub():
        for _ in range(DECIMATION):
            mjw.step(mm, dd)
    sub()
    with wp.ScopedCapture() as cap:
        sub()
    g = cap.graph
    worst_con, worst_efc = 0, 0

    def tally():
        nonlocal worst_con, worst_efc
        n = int(dd.nacon.numpy()[0])
        w = dd.contact.worldid.numpy()[:n]
        worst_con = max(worst_con, int(np.bincount(w).max()) if n else 0)
        worst_efc = max(worst_efc, int(dd.nefc.numpy().max()))

    for i in range(settle):
        wp.capture_launch(g)
        if hold_every and (i + 1) % hold_every == 0:
            wp.synchronize()
            qpos[:] = qpos0
            qvel[:] = qvel0
        if i % 10 == 0:
            wp.synchronize()
            tally()
    wp.synchronize()
    t0 = time.perf_counter()
    for i in range(steps):
        wp.capture_launch(g)
        if hold_every and (i + 1) % hold_every == 0:
            wp.synchronize()
            qpos[:] = qpos0
            qvel[:] = qvel0
    wp.synchronize()
    rate = dd.nworld * steps / (time.perf_counter() - t0)
    tally()
    return rate, worst_con, worst_efc, dd.qpos.numpy()[:, 2]


def kneel(xml, motion):
    d0 = joblib.load(motion)[KNEEL]
    frame = len(d0['dof']) // 2
    x2 = build_x2(mjcf_path=pathlib.Path(xml))
    m = x2.model
    d = mujoco.MjData(m)
    d.qpos[:3], d.qpos[3:7], d.qpos[7:] = d0['root_trans_offset'][frame], d0['root_rot'][frame][[3, 0, 1, 2]], d0['dof'][frame]
    d.ctrl[:] = d0['dof'][frame]
    z0 = float(d.qpos[2])
    mujoco.mj_forward(m, d)
    low = float(d.geom_xpos[:, 2].min())
    tr = []
    for i in range(int(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        if (i + 1) % int(0.25 / m.opt.timestep) == 0:
            tr.append(round(float(d.qpos[2]), 3))
    names = sorted({m.geom(g).name for c in d.contact[:d.ncon] for g in (c.geom1, c.geom2) if m.geom(g).name != 'floor'})
    return z0, tr, d.ncon, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('baseline')
    ap.add_argument('new')
    ap.add_argument('--nworld', type=int, default=300)
    ap.add_argument('--steps', type=int, default=150)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--nconmax', type=int, default=64)
    ap.add_argument('--njmax', type=int, default=300)
    ap.add_argument('--motion', default='C:/Users/PC/projects/bones-seed/echo_soma/v4/test.pkl')
    a = ap.parse_args()
    for scene, settle in (('standing', 30), ('fallen', 150)):
        res = {'baseline': [], 'new': []}
        for r in range(a.rounds):
            for name, xml in (('baseline', a.baseline), ('new', a.new)):
                with wp.ScopedDevice('cuda:0'):
                    _, mm, dd, used = make(xml, a.nworld, a.nconmax, a.njmax, scene, seed=r)
                    rate, con, efc, z = run(mm, dd, settle, a.steps, 10 if scene == 'standing' else 0)
                res[name].append(rate)
                print(f'{scene:9s} round {r} {name:9s} {rate / 1e3:6.1f}k steps/s  worst world {con} contacts, {efc} rows; pelvis z median {np.median(z):.2f}, nan {int(np.isnan(z).sum())}; arrays {used / 2**20:.0f} MB', flush=True)
                del mm, dd
        b, n = np.median(res['baseline']), np.median(res['new'])
        print(f'== {scene}: baseline {b / 1e3:.1f}k, new {n / 1e3:.1f}k steps/s ({(n / b - 1) * 100:+.0f}%)', flush=True)
    for nc in (64, 128):
        with wp.ScopedDevice('cuda:0'):
            _, mm, dd, used = make(a.new, a.nworld, nc, a.njmax, 'standing')
            print(f'nconmax {nc} x {a.nworld} worlds: {used / 2**20:.0f} MB')
            del mm, dd
    print('kneeling hold, servos at the clip\'s pose, pelvis z at 0.25 s steps:')
    for name, xml in (('baseline', a.baseline), ('new', a.new)):
        z0, tr, nc, names = kneel(xml, a.motion)
        print(f'  {name:9s} reference z {z0:.3f}, then {tr}; {nc} contacts on {names}')


if __name__ == '__main__':
    main()
