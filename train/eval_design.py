"""Does the design hold up? Rolls a trained SuperMimic policy out on motions with nominal physics and reads each joint against
its parts. Sim only: every number below inherits the sim's estimates (armature, friction, the torque-speed curve).

  SM_ROBOT_CFG=train/supermimic_robot.json PYTHONPATH=<gwbc-echo> python train/eval_design.py <ckpt.pt> <motion.pkl> [--out dir] [--clips N]

Writes eval.json and motion/<clip>.npz: per clip whether it finished, tracking error, and per joint RMS and peak torque against
rated and peak, peak speed against no-load, and time at a limit. Per robot, in the world frame, as each clip is its own test.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import torch
import warp as wp

from gear_sonic.supermimic.assets import CONTROL_DT, build_x2
from gear_sonic.supermimic.kinematics import X2Kinematics
from gear_sonic.supermimic.mdp import MDP, rot_angle_sq
from gear_sonic.supermimic.motion import MotionLib
from gear_sonic.supermimic.networks import Policy
from gear_sonic.supermimic.sim import TrackingSim

ROOT = Path(__file__).resolve().parents[1]


def parts(names):
    spec = json.loads((ROOT / 'spec' / 'robot.json').read_text(encoding='utf-8'))
    cap = {j['name']: j['capacity'] for j in spec['joints']}
    out = {}
    for n in names:
        c = cap[n]
        a = spec['actuators'][c['model']]
        k = c['count'] * c['ratio']
        out[n] = dict(rated=a['rated_nm'] * k, peak=a['peak_nm'] * k, noload=a['noload_rads'] / c['ratio'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('motion')
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--clips', type=int, default=256)
    ap.add_argument('--push', type=float, default=0.0, help='m/s added to the pelvis in a random direction')
    ap.add_argument('--push-every', type=float, default=2.0)
    ap.add_argument('--only', default='.', help='regex on clip names')
    ap.add_argument('--exclude', default='', help='regex on clip names')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = joblib.load(a.motion)
    keys = sorted(k for k in data if re.search(a.only, k, re.I) and not (a.exclude and re.search(a.exclude, k, re.I)))[:a.clips]
    x2 = build_x2()
    kin = X2Kinematics(x2.model, dev)
    motion = MotionLib({k: data[k] for k in keys}, kin, dev)
    f32 = lambda v: torch.tensor(v, device=dev, dtype=torch.float32)  # noqa: E731
    mdp = MDP(motion, f32(x2.default_q), f32(x2.q_lo), f32(x2.q_hi))
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    N = len(keys)
    sim = TrackingSim(N, motion, mdp, x2=x2, eval_mode=True, seed=1)
    policy = Policy().to(dev)
    policy.load_state_dict(ck['pi'])
    policy.eval()
    force = wp.to_torch(sim.dd.actuator_force)
    starts = motion.clip_start[: N].clone()
    ends = motion.clip_end[: N].clone()
    sim.done_eval[:] = False
    sim.failed_eval[:] = False
    sim.reset_all(starts.clone())
    T = int((ends - starts).max()) + 1
    names = x2.joint_names
    tq2, tq_peak, spd_peak = torch.zeros(N, len(names), device=dev), torch.zeros(N, len(names), device=dev), torch.zeros(N, len(names), device=dev)
    lim = torch.zeros(N, len(names), device=dev)
    steps = torch.zeros(N, device=dev)
    fell = sim.failed_eval
    pos_err = torch.zeros(N, device=dev)
    strict = torch.zeros(N, dtype=torch.bool, device=dev)
    path, ref_path = torch.zeros(N, device=dev), torch.zeros(N, device=dev)
    rec = [[] for _ in range(N)]
    # Falls, measured apart from the sim's lenient termination: pelvis well below the reference, or tilted further from it than a recovery allows
    fell_real = torch.zeros(N, dtype=torch.bool, device=dev)
    fell_t = torch.full((N,), -1.0, device=dev)
    max_dz, max_tilt, max_foot, max_lean = torch.zeros(N, device=dev), torch.zeros(N, device=dev), torch.zeros(N, device=dev), torch.zeros(N, device=dev)
    sat = torch.zeros(N, len(names), device=dev)
    why = torch.zeros(N, 4, device=dev)  # steps in violation: pelvis height, pelvis tilt, ankle height, ankle position
    # Torque over 0.2 s: the part that heats a motor and that the design must carry, without the controller's chatter
    lp = torch.zeros(N, len(names), device=dev)
    lp2 = torch.zeros(N, len(names), device=dev)
    lp_a = CONTROL_DT / 0.2
    pushes = torch.zeros(N, device=dev)
    for t in range(T):
        alive = ~sim.done_eval.clone()
        if not alive.any():
            break
        with torch.no_grad():
            act = policy(sim.policy_input()).clamp(-20.0, 20.0)
        f0 = sim.f.clone()
        sim.step(act)
        S = sim.state()
        qd, q = S[5], S[4]
        tau = force
        m = alive.unsqueeze(-1).float()
        tq2 += m * tau ** 2
        tq_peak = torch.maximum(tq_peak, m * tau.abs())
        spd_peak = torch.maximum(spd_peak, m * qd.abs())
        lim += m * ((q < sim.q_lo + 0.01) | (q > sim.q_hi - 0.01)).float()
        steps += alive.float()
        f1 = f0 + 1
        pos_err += alive.float() * (motion.root_p[f1][:, :2] - S[0][:, :2]).norm(dim=-1)
        # SONIC's strict criterion: pelvis height, pelvis tilt and both ankles, against the reference anchored on the robot
        ref_z = motion.root_p[f1][:, 2]
        thr = torch.where(ref_z < 0.5, 0.75, 0.15)
        bp, ref_bp = mdp.robot_bodies(S)[0], mdp._ref_anchored(S, f1)[0]
        viol = ((ref_z - S[0][:, 2]).abs() > thr) | (rot_angle_sq(motion.root_R[f1], S[1]) > 0.2) \
            | ((ref_bp[:, [3, 6], 2] - bp[:, [3, 6], 2]).abs().amax(-1) > thr) | ((ref_bp[:, [3, 6]] - bp[:, [3, 6]]).norm(dim=-1).amax(-1) > 0.2)
        strict |= alive & viol
        dz = (ref_z - S[0][:, 2])
        tilt = rot_angle_sq(motion.root_R[f1], S[1]).clamp(min=0).sqrt()  # against the reference, heading included: a missed spin counts
        lean = S[1][:, 2, 2].clamp(-1, 1).acos()  # from upright, against nothing
        foot = (ref_bp[:, [3, 6]] - bp[:, [3, 6]]).norm(dim=-1).amax(-1)
        max_dz = torch.maximum(max_dz, alive.float() * dz.abs())
        max_tilt = torch.maximum(max_tilt, alive.float() * tilt)
        max_lean = torch.maximum(max_lean, alive.float() * lean)
        max_foot = torch.maximum(max_foot, alive.float() * foot)
        why[:, 0] += (alive & ((ref_z - S[0][:, 2]).abs() > thr)).float()
        why[:, 1] += (alive & (tilt ** 2 > 0.2)).float()
        why[:, 2] += (alive & ((ref_bp[:, [3, 6], 2] - bp[:, [3, 6], 2]).abs().amax(-1) > thr)).float()
        why[:, 3] += (alive & (foot > 0.2)).float()
        down = alive & ((dz > 0.35) | (lean > 1.0))
        fell_t = torch.where(down & (fell_t < 0), t * CONTROL_DT * torch.ones(N, device=dev), fell_t)
        fell_real |= down
        sat += m * (tau.abs() > 0.98 * sim.effort).float()
        lp += lp_a * (tau - lp)
        lp2 += m * lp ** 2
        path += alive.float() * S[2][:, :2].norm(dim=-1) * CONTROL_DT
        ref_path += alive.float() * (motion.root_p[f1][:, :2] - motion.root_p[f0][:, :2]).norm(dim=-1)
        if a.push and t % int(a.push_every / CONTROL_DT) == int(a.push_every / CONTROL_DT) - 1:
            d = torch.randn(N, 2, device=dev)
            sim.qvel[:, :2] += a.push * d / d.norm(dim=-1, keepdim=True)
            pushes += alive.float()
        for i in torch.nonzero(alive).flatten().tolist():
            rec[i].append(np.concatenate((S[0][i].cpu().numpy(), sim.qpos[i, 3:7].cpu().numpy(), q[i].cpu().numpy(), sim.ctrl[i].cpu().numpy())))
    P = parts(names)
    per = {}
    steps_c = steps.clamp(min=1)
    rms = (tq2 / steps_c.unsqueeze(-1)).sqrt()
    rms_lp = (lp2 / steps_c.unsqueeze(-1)).sqrt()
    for i, n in enumerate(names):
        p = P[n]
        r, pk, sp, li = rms[:, i], tq_peak[:, i], spd_peak[:, i], lim[:, i] / steps_c
        per[n] = dict(rated=p['rated'], peak=p['peak'], noload=p['noload'], rms_median=round(float(r.median()), 1), rms_max=round(float(r.max()), 1), rms_over_rated_clips_pct=round(100 * float((r > p['rated']).float().mean()), 1), saturated_pct_median=round(100 * float((sat[:, i] / steps_c).median()), 1), saturated_pct_max=round(100 * float((sat[:, i] / steps_c).max()), 1), rms_lp_median=round(float(rms_lp[:, i].median()), 1), rms_lp_max=round(float(rms_lp[:, i].max()), 1), rms_lp_over_rated_clips_pct=round(100 * float((rms_lp[:, i] > p['rated']).float().mean()), 1),
                      peak_p95=round(float(pk.quantile(0.95)), 1), peak_max=round(float(pk.max()), 1), speed_p95=round(float(sp.quantile(0.95)), 2), speed_max=round(float(sp.max()), 2), at_limit_pct_median=round(100 * float(li.median()), 1))
    done = (~fell).float()
    per_clip = {k: dict(finished=bool(~fell[i]), strict_pass=bool(~strict[i]), seconds=round(float(steps[i]) * CONTROL_DT, 1), of=round(float(ends[i] - starts[i]) * CONTROL_DT, 1), xy_err_m=round(float(pos_err[i] / steps_c[i]), 3),
                        path_m=round(float(path[i]), 2), ref_path_m=round(float(ref_path[i]), 2), fell=bool(fell_real[i]), fell_at_s=round(float(fell_t[i]), 2), max_pelvis_dz_m=round(float(max_dz[i]), 3), max_tilt_deg=round(float(max_tilt[i]) * 57.3, 1), max_lean_deg=round(float(max_lean[i]) * 57.3, 1), max_ankle_err_m=round(float(max_foot[i]), 3),
                        strict_steps=dict(zip(('pelvis_z', 'tilt', 'ankle_z', 'ankle_pos'), why[i].int().tolist())), pushes=int(pushes[i])) for i, k in enumerate(keys)}
    prog = float((steps / (ends - starts).float().clamp(min=1)).clamp(max=1).mean())
    res = dict(ckpt=a.ckpt, motion=a.motion, clips=N, push_mps=a.push, finished_pct=round(100 * float(done.mean()), 1), fell_pct=round(100 * float(fell_real.float().mean()), 1), blown=int(sim.blown_count), seconds=round(float(steps.sum()) * CONTROL_DT, 1), pushes=int(pushes.sum()), strict_pass_pct=round(100 * float((~strict).float().mean()), 1), progress_pct=round(100 * prog, 1),
               path_ratio_median=round(float((path / ref_path.clamp(min=0.5)).median()), 2),
               xy_err_m_median=round(float((pos_err / steps_c).median()), 3), joints=per, per_clip=per_clip)
    out_dir = a.out or Path(a.ckpt).parent / 'eval'
    (out_dir / 'motion').mkdir(parents=True, exist_ok=True)
    (out_dir / 'eval.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    for i, k in enumerate(keys):
        if rec[i]:
            r = np.array(rec[i])
            np.savez_compressed(out_dir / 'motion' / f'{k}.npz', t=np.arange(len(r)) * CONTROL_DT, q=r[:, 7:7 + len(names)], ctrl=r[:, 7 + len(names):], joint_names=np.array(names), root_pos=r[:, :3], root_quat_wxyz=r[:, 3:7], clip=k)
    print(json.dumps({k: v for k, v in res.items() if k not in ('joints', 'per_clip')}))
    print(f"{'joint':18s}{'rated':>7s}{'rms med':>9s}{'rms max':>9s}{'>rated %':>9s}{'lp med':>8s}{'lp max':>8s}{'lp>r %':>8s}{'sat %':>7s}{'peak':>7s}{'p95':>7s}{'max':>7s}{'noload':>8s}{'spd p95':>9s}{'spd max':>9s}{'limit %':>9s}")
    for n, d in per.items():
        print(f"{n:18s}{d['rated']:7.1f}{d['rms_median']:9.1f}{d['rms_max']:9.1f}{d['rms_over_rated_clips_pct']:9.1f}{d['rms_lp_median']:8.1f}{d['rms_lp_max']:8.1f}{d['rms_lp_over_rated_clips_pct']:8.1f}{d['saturated_pct_median']:7.1f}{d['peak']:7.1f}{d['peak_p95']:7.1f}{d['peak_max']:7.1f}{d['noload']:8.1f}{d['speed_p95']:9.2f}{d['speed_max']:9.2f}{d['at_limit_pct_median']:9.1f}")


if __name__ == '__main__':
    main()
