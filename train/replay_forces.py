"""One clip, trained policy, closed loop in plain CPU MuJoCo: every physics step is mujoco.mj_step (the GPU sim only builds the
observations), and the floor's contact forces and the motors' torques are recorded from that solver. train/render_forces.py draws them.

  SM_ROBOT_CFG=train/supermimic_robot_v4.json PYTHONPATH=<gwbc-echo> python train/replay_forces.py <ckpt.pt> <motion.pkl> <clip> --out f.npz [--seconds 15]
"""
from __future__ import annotations

import argparse

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


MAX_CONTACTS = 24


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('motion')
    ap.add_argument('clip')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seconds', type=float, default=15.0)
    a = ap.parse_args()
    dev = 'cuda'
    data = joblib.load(a.motion)
    x2 = build_x2()
    motion = MotionLib({a.clip: data[a.clip]}, X2Kinematics(x2.model, dev), dev)
    f32 = lambda v: torch.tensor(v, device=dev, dtype=torch.float32)  # noqa: E731
    mdp = MDP(motion, f32(x2.default_q), f32(x2.q_lo), f32(x2.q_hi))
    sim = TrackingSim(1, motion, mdp, x2=x2, eval_mode=True, seed=1)
    policy = Policy().to(dev)
    policy.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=False)['pi'])
    policy.eval()
    start, end = motion.clip_start[:1].clone(), motion.clip_end[:1].clone()
    sim.reset_all(start.clone())
    m = x2.model
    for i in range(m.ngeom):
        if m.geom_bodyid[i] == 0:
            m.geom_friction[i, 0] = 1.0  # as the GPU evaluation's floor
    effort = np.array(x2.effort, dtype=float)
    noload = sim.noload[0].cpu().numpy()
    knee = CFG['torque_speed_knee']
    d = mujoco.MjData(m)
    d.qpos[:], d.qvel[:] = sim.qpos[0].cpu().numpy(), sim.qvel[0].cpu().numpy()
    mujoco.mj_forward(m, d)
    sub = round(CONTROL_DT / m.opt.timestep)
    side = {}
    for b in range(m.nbody):
        n = m.body(b).name
        side[b] = 0 if n.startswith('L_') else 1 if n.startswith('R_') else 2
    names = [m.joint(i).name for i in range(1, m.njnt)]
    rec = {k: [] for k in ('qpos', 'force', 'cop', 'torque', 'slip', 'other_n')}
    CP, CF = [], []
    T = min(int((end - start).item()), int(a.seconds / CONTROL_DT))
    fell_at = -1.0
    for t in range(T):
        with torch.no_grad():
            act = policy(sim.policy_input()).clamp(-20.0, 20.0)
        d.ctrl[:] = (sim.default_q + sim.xi[:, 1:1 + len(names)] + sim.alpha * act)[0].cpu().numpy()
        f1 = sim.f + 1
        F, C, N = np.zeros((2, 3)), np.zeros((2, 3)), np.zeros(2)
        tau = np.zeros(m.nu)
        other = 0.0
        for _ in range(sub):
            u = d.qvel[6:] / noload
            m.actuator_forcerange[:, 1] = effort * np.clip((1 - u) / (1 - knee), 0, 1)
            m.actuator_forcerange[:, 0] = -effort * np.clip((1 + u) / (1 - knee), 0, 1)
            mujoco.mj_step(m, d)
            tau += d.actuator_force
            w = np.zeros(6)
            for c in range(d.ncon):
                con = d.contact[c]
                b1, b2 = m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]
                if 0 not in (b1, b2):
                    continue  # robot on itself: not the floor
                mujoco.mj_contactForce(m, d, c, w)
                # contact frame rows are world axes: x the normal, force[0] the normal force. Force on the robot, so up is positive.
                f = con.frame.reshape(3, 3).T @ w[:3]
                if f[2] < 0:
                    f = -f
                s = side[b2 if b1 == 0 else b1]
                if s == 2:
                    other += f[2]
                    continue
                F[s] += f
                C[s] += f[2] * con.pos
                N[s] += f[2]
        rec['qpos'].append(d.qpos.copy())
        rec['force'].append(F / sub)
        rec['cop'].append(np.divide(C, N[:, None], out=np.full((2, 3), np.nan), where=N[:, None] > 1e-6))
        rec['torque'].append(tau / sub)
        rec['other_n'].append(other / sub)
        cp, cf = np.full((MAX_CONTACTS, 3), np.nan), np.full((MAX_CONTACTS, 3), np.nan)  # the last substep's floor contacts
        n, w = 0, np.zeros(6)
        for c in range(d.ncon):
            con = d.contact[c]
            if 0 not in (m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]) or n == MAX_CONTACTS:
                continue
            mujoco.mj_contactForce(m, d, c, w)
            f = con.frame.reshape(3, 3).T @ w[:3]
            cp[n], cf[n] = con.pos, f if f[2] >= 0 else -f
            n += 1
        CP.append(cp)
        CF.append(cf)
        lean = np.arccos(np.clip(1 - 2 * (d.qpos[4] ** 2 + d.qpos[5] ** 2), -1, 1))
        if lean > 1.0 and fell_at < 0:
            fell_at = t * CONTROL_DT
            break
        sim.qpos[0] = torch.as_tensor(d.qpos, device=dev, dtype=torch.float32)
        sim.qvel[0] = torch.as_tensor(d.qvel, device=dev, dtype=torch.float32)
        sim.f = torch.minimum(f1, end - 1)
        sim.a_prev = act.clone()
        sim.obs_hist = torch.cat((sim.obs_hist[:, 1:], sim._obs().unsqueeze(1)), 1)
    q = np.array(rec['qpos'])
    F = np.array(rec['force'])
    weight = mujoco.mj_getTotalmass(m) * 9.81
    np.savez(a.out, t=np.arange(len(q)) * CONTROL_DT, q=q[:, 7:], joint_names=np.array(names), root_pos=q[:, :3], root_quat_wxyz=q[:, 3:7],
             force=F, cop=np.array(rec['cop']), torque=np.array(rec['torque']), other_n=np.array(rec['other_n']), weight=weight, fell_at=fell_at, clip=a.clip, contact_pos=np.array(CP), contact_force=np.array(CF))
    fz = F[..., 2]
    print(f'{a.clip}: {len(q) * CONTROL_DT:.1f} s, fell_at {fell_at}; weight {weight:.0f} N; floor force mean {fz.sum(1).mean():.0f} N ({fz.sum(1).mean() / weight:.3f} of weight), peak foot {fz.max():.0f} N; '
          f'both feet off {(fz.sum(1) < 1).mean() * 100:.1f}% of steps; one foot only {((fz > 1).sum(1) == 1).mean() * 100:.1f}%; both on {((fz > 1).sum(1) == 2).mean() * 100:.1f}%; floor force on other bodies mean {np.mean(rec["other_n"]):.1f} N')
    tq = np.array(rec['torque'])
    for j in ('L_ankle_pitch', 'R_ankle_pitch', 'L_ankle_roll', 'R_ankle_roll', 'L_knee', 'R_knee'):
        x = tq[:, names.index(j)]
        print(f'  {j:14s} real torque RMS {np.sqrt((x ** 2).mean()):5.1f} N m, peak {np.abs(x).max():5.1f}')


if __name__ == '__main__':
    main()
