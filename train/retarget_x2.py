"""An X2 motion (SONIC pkl: root_trans_offset, root_rot xyzw, dof 31, fps) -> echo (root pose + 29 joints), for review.

Echo's joint axes match the X2's, but not its proportions (thigh x 1.21, shin x 1.60), so copying the leg angles would
slide the feet. Instead the pelvis and both ankles follow the X2's paths scaled by standing leg length (1.33 here: pelvis
to sole 0.90 m against 0.675 m), the soles stay at floor height, and IK finds the leg angles. The rest are copied:
waist yaw, head, and arms (echo's wrist has pitch and roll in the other order: its pitch axis is the X2's roll axis).
Echo has no waist pitch or roll, so the X2's are dropped.

  python train/retarget_x2.py <clip.pkl> [--clip NAME] [--out train/review/<name>.npz]

Writes the take tools/render_motion.py reads, and prints what the design had to give up: joint limits hit, foot error,
foot slide in stance, sole height and the peak joint speeds against the actuators'. Kinematic only: no forces yet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mink
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ECHO_XML = ROOT / 'urdf' / 'echo.xml'
X2_XML = ROOT.parent / 'gwbc-supermimic/gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml'
FPS = 50

X2_SOLE = 0.073         # m below the X2's ankle_roll_link origin
ECHO_SOLE = 0.085       # m below echo's L_foot origin: urdf/echo.xml, sole box at -0.07, half height 0.015
SCALE = 0.90 / 0.675    # standing pelvis to sole, echo / X2
STANCE_Z = 0.015        # m: sole this close to the floor counts as stance in the report
LEGS = ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')
ARMS = {'shoulder_pitch': 'shoulder_pitch', 'shoulder_roll': 'shoulder_roll', 'shoulder_yaw': 'shoulder_yaw', 'elbow': 'elbow',
        'wrist_yaw': 'wrist_yaw', 'wrist_pitch': 'wrist_roll', 'wrist_roll': 'wrist_pitch'}   # echo -> X2


def names(m):
    return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)]


def copied_joints(x2_names):
    """Echo joint -> X2 dof index, for the joints copied or used as IK's rest posture."""
    idx = {n: i for i, n in enumerate(x2_names)}
    out = {'waist_yaw': idx['waist_yaw_joint'], 'neck_yaw': idx['head_yaw_joint'], 'neck_pitch': idx['head_pitch_joint']}
    for S, s in (('L', 'left'), ('R', 'right')):
        out.update({f'{S}_{j}': idx[f'{s}_{j}_joint'] for j in LEGS})
        out.update({f'{S}_{e}': idx[f'{s}_{x}_joint'] for e, x in ARMS.items()})
    return out


def x2_feet(m, p, xyzw, dof):
    """World position and wxyz orientation of the X2's two ankle_roll_links, per frame."""
    d = mujoco.MjData(m)
    L, R = m.body('left_ankle_roll_link').id, m.body('right_ankle_roll_link').id
    out = np.zeros((len(p), 2, 3))
    quat = np.zeros((len(p), 2, 4))
    for i in range(len(p)):
        d.qpos[:3], d.qpos[3:7], d.qpos[7:] = p[i], [xyzw[i, 3], *xyzw[i, :3]], dof[i]
        mujoco.mj_kinematics(m, d)
        out[i] = d.xpos[[L, R]]
        quat[i] = d.xquat[[L, R]]
    return out, quat


def retarget(clip, iters=4, damping=0.1):
    p, xyzw, dof = (np.asarray(clip[k], dtype=float) for k in ('root_trans_offset', 'root_rot', 'dof'))
    assert float(clip['fps']) == FPS, 'resample to 50 Hz first'
    mx, me = mujoco.MjModel.from_xml_path(str(X2_XML)), mujoco.MjModel.from_xml_path(str(ECHO_XML))
    jn = names(me)
    jid = [me.joint(n).id for n in jn]
    adr = np.array([me.jnt_qposadr[j] for j in jid])
    vadr = np.array([me.jnt_dofadr[j] for j in jid])
    lo, hi = me.jnt_range[jid].T
    copy = copied_joints(names(mx))
    src = np.array([copy[n] for n in jn])
    q_ref = dof[:, src]
    feet, fquat = x2_feet(mx, p, xyzw, dof)

    # target paths: everything scaled about the world origin, soles referred to the floor
    root = p * SCALE
    tfeet = feet * SCALE
    tfeet[..., 2] = np.maximum(feet[..., 2] - X2_SOLE, 0.0) * SCALE + ECHO_SOLE

    pelvis = mink.FrameTask('pelvis', 'body', position_cost=50.0, orientation_cost=50.0, lm_damping=1.0)
    foot = [mink.FrameTask(b, 'body', position_cost=100.0, orientation_cost=20.0, lm_damping=1.0) for b in ('L_foot', 'R_foot')]
    posture = mink.PostureTask(me, cost=0.2)
    tasks = [pelvis, *foot, posture]
    limits = [mink.ConfigurationLimit(me)]
    cfg = mink.Configuration(me)
    key = me.qpos0.copy()
    out_q = np.zeros((len(p), len(jn)))
    out_root = np.zeros((len(p), 7))
    err = np.zeros((len(p), 2))
    wanted = np.clip(q_ref, lo, hi)
    for i in range(len(p)):
        rq = np.array([xyzw[i, 3], *xyzw[i, :3]])
        key[:3], key[3:7] = root[i], rq
        key[adr] = wanted[i]
        if i == 0:
            cfg.update(key)
        posture.set_target(key)
        pelvis.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rq), root[i]))
        for k in range(2):
            foot[k].set_target(mink.SE3.from_rotation_and_translation(mink.SO3(fquat[i, k]), tfeet[i, k]))
        for _ in range(iters if i else 60):
            v = mink.solve_ik(cfg, tasks, 1.0 / FPS, 'daqp', damping, limits=limits)
            cfg.integrate_inplace(v, 1.0 / FPS)
        out_q[i], out_root[i, :3], out_root[i, 3:] = cfg.q[adr], cfg.q[:3], cfg.q[3:7]
        err[i] = [np.linalg.norm(cfg.data.body(b).xpos - tfeet[i, k]) for k, b in enumerate(('L_foot', 'R_foot'))]
    return dict(jn=jn, q=out_q, root=out_root, q_ref=q_ref, err=err, lo=lo, hi=hi, me=me, adr=adr, vadr=vadr)


def report(r, spec):
    """What the design gave up on this clip, as numbers."""
    jn, q, q_ref, lo, hi, me = r['jn'], r['q'], r['q_ref'], r['lo'], r['hi'], r['me']
    out = {'frames': len(q), 'foot_err_mm': {'rms': float(1e3 * np.sqrt((r['err'] ** 2).mean())), 'max': float(1e3 * r['err'].max())}}
    want = [(n, 100 * float(((q_ref[:, i] < lo[i]) | (q_ref[:, i] > hi[i])).mean()), float(q_ref[:, i].min()), float(q_ref[:, i].max()), float(lo[i]), float(hi[i]))
            for i, n in enumerate(jn)]
    at = [(n, 100 * float(((q[:, i] < lo[i] + 0.01) | (q[:, i] > hi[i] - 0.01)).mean())) for i, n in enumerate(jn)]
    out['source_outside_limits_pct'] = {n: {'pct': round(p, 1), 'src': [round(a, 2), round(b, 2)], 'limit': [round(c, 2), round(e, 2)]} for n, p, a, b, c, e in want if p > 0}
    out['at_limit_pct'] = {n: round(p, 1) for n, p in at if p > 0.5}
    qd = np.gradient(q, 1.0 / FPS, axis=0)
    cap = {j['name']: j['capacity'] for j in spec['joints']}
    speed = {}
    for i, n in enumerate(jn):
        c = cap[n]
        limit = spec['actuators'][c['model']]['noload_rads'] / c['ratio']
        speed[n] = {'peak': round(float(np.abs(qd[:, i]).max()), 2), 'noload': round(limit, 1), 'frac': round(float(np.abs(qd[:, i]).max() / limit), 2)}
    out['joint_speed_rads'] = dict(sorted(speed.items(), key=lambda kv: -kv[1]['frac'])[:8])
    # the feet on echo, by forward kinematics of the solved motion
    d = mujoco.MjData(me)
    sole = np.zeros((len(q), 2))
    xy = np.zeros((len(q), 2, 2))
    ids = [me.geom('L_foot_sole').id, me.geom('R_foot_sole').id]
    for f in range(len(q)):
        d.qpos[:7], d.qpos[r['adr']] = r['root'][f], q[f]
        mujoco.mj_kinematics(me, d)
        for k, g in enumerate(ids):
            corners = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]) * me.geom_size[g]
            sole[f, k] = d.geom_xpos[g][2] + (corners @ d.geom_xmat[g].reshape(3, 3)[2]).min()
            xy[f, k] = d.geom_xpos[g][:2]
    v = np.linalg.norm(np.gradient(xy, 1.0 / FPS, axis=0), axis=-1)
    stance = sole < STANCE_Z
    out['sole'] = {'min_mm': round(float(1e3 * sole.min()), 1), 'both_up_pct': round(100 * float((~stance).all(1).mean()), 1),
                   'stance_slide_mps_mean': round(float(v[stance].mean()), 3), 'stance_slide_mps_p95': round(float(np.percentile(v[stance], 95)), 3)}
    out['root_speed_mps'] = round(float(np.linalg.norm(np.gradient(r['root'][:, :2], 1.0 / FPS, axis=0), axis=1).mean()), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pkl', type=Path)
    ap.add_argument('--clip', default=None)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    data = joblib.load(a.pkl)
    name = a.clip or next(iter(data))
    r = retarget(data[name])
    out = a.out or ROOT / 'train' / 'review' / f'{name}.npz'
    out.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(len(r['q'])) / FPS
    np.savez_compressed(out, t=t, q=r['q'], joint_names=np.array(r['jn']), root_pos=r['root'][:, :3], root_quat_wxyz=r['root'][:, 3:], clip=name)
    spec = json.loads((ROOT / 'spec' / 'robot.json').read_text(encoding='utf-8'))
    rep = report(r, spec)
    out.with_suffix('.json').write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(out)
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
