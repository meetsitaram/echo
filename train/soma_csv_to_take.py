"""A soma-retargeter echo CSV (120 Hz; cm, degrees) -> the take tools/render_motion.py and train/render_review.py read, at
50 Hz, with the same report as train/retarget_x2.py: limits hit, sole height, foot slide, joint speeds.

  python train/soma_csv_to_take.py <clip.csv> [--fps 120] [--skip 0] [--out train/review/<name>.npz]

--skip drops the retargeter's lead-in frames (its initialization pose and settling) from the start.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R, Slerp

ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / 'urdf' / 'echo.xml'
FPS = 50
STANCE_Z = 0.015     # m: sole this close to the floor counts as stance in the report


def load(path, fps, skip):
    with open(path, encoding='utf-8') as f:
        head = f.readline().strip().split(',')
    d = np.loadtxt(path, delimiter=',', skiprows=1)[skip:]
    names = [h[:-4] for h in head[7:]]
    t0 = np.arange(len(d)) / fps
    t = np.arange(0.0, t0[-1], 1.0 / FPS)
    p = np.stack([np.interp(t, t0, d[:, 1 + i]) for i in range(3)], 1) / 100.0
    rot = Slerp(t0, R.from_euler('xyz', d[:, 4:7], degrees=True))(t)
    q = np.stack([np.interp(t, t0, d[:, 7 + i]) for i in range(len(names))], 1)
    xyzw = rot.as_quat()
    return t, names, np.deg2rad(q), p, np.column_stack((xyzw[:, 3], xyzw[:, :3]))


def ground(names, q, p, wxyz, sigma_s=0.06):
    """Put the lowest sole on the floor by moving the root, smoothed (LESSONS: planted feet sink): a box sole's corner dips
    wherever the human foot rolls. Returns the new root positions and the largest shift."""
    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    adr = [m.jnt_qposadr[m.joint(n).id] for n in names]
    ids = [m.geom('L_foot_sole').id, m.geom('R_foot_sole').id]
    box = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
    low = np.zeros(len(q))
    for f in range(len(q)):
        d.qpos[:3], d.qpos[3:7], d.qpos[adr] = p[f], wxyz[f], q[f]
        mujoco.mj_kinematics(m, d)
        low[f] = min(d.geom_xpos[g][2] + ((box * m.geom_size[g]) @ d.geom_xmat[g].reshape(3, 3)[2]).min() for g in ids)
    shift = -gaussian_filter1d(low, sigma_s * FPS, mode='nearest')
    out = p.copy()
    out[:, 2] += shift
    return out, float(np.abs(shift).max())


def report(names, q, p, wxyz, spec):
    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    jid = [m.joint(n).id for n in names]
    adr = [m.jnt_qposadr[j] for j in jid]
    lo, hi = m.jnt_range[jid].T
    out = {'frames': len(q), 'at_limit_pct': {n: round(100 * float(((q[:, i] < lo[i] + 0.01) | (q[:, i] > hi[i] - 0.01)).mean()), 1)
                                              for i, n in enumerate(names) if ((q[:, i] < lo[i] + 0.01) | (q[:, i] > hi[i] - 0.01)).mean() > 0.005}}
    qd = np.gradient(q, 1.0 / FPS, axis=0)
    cap = {j['name']: j['capacity'] for j in spec['joints']}
    speed = {}
    for i, n in enumerate(names):
        limit = spec['actuators'][cap[n]['model']]['noload_rads'] / cap[n]['ratio']
        peak = float(np.abs(qd[:, i]).max())
        speed[n] = {'peak': round(peak, 2), 'noload': round(limit, 1), 'frac': round(peak / limit, 2)}
    out['joint_speed_rads'] = dict(sorted(speed.items(), key=lambda kv: -kv[1]['frac'])[:6])
    out['smoothness_rms_jerk_rads3_legs'] = round(float(np.sqrt((np.gradient(np.gradient(qd, 1.0 / FPS, axis=0), 1.0 / FPS, axis=0)[:, [i for i, n in enumerate(names) if 'hip' in n or 'knee' in n or 'ankle' in n]] ** 2).mean())), 1)
    ids = [m.geom('L_foot_sole').id, m.geom('R_foot_sole').id]
    box = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
    sole = np.zeros((len(q), 2))
    xy = np.zeros((len(q), 2, 2))
    for f in range(len(q)):
        d.qpos[:3], d.qpos[3:7], d.qpos[adr] = p[f], wxyz[f], q[f]
        mujoco.mj_kinematics(m, d)
        for k, g in enumerate(ids):
            sole[f, k] = d.geom_xpos[g][2] + ((box * m.geom_size[g]) @ d.geom_xmat[g].reshape(3, 3)[2]).min()
            xy[f, k] = d.geom_xpos[g][:2]
    v = np.linalg.norm(np.gradient(xy, 1.0 / FPS, axis=0), axis=-1)
    st = sole < STANCE_Z
    out['sole'] = {'min_mm': round(float(1e3 * sole.min()), 1), 'median_lowest_mm': round(float(1e3 * np.median(sole.min(1))), 1),
                   'both_up_pct': round(100 * float((~st).all(1).mean()), 1),
                   'stance_slide_mps_median': round(float(np.median(v[st])), 3) if st.any() else None,
                   'stance_slide_mps_p95': round(float(np.percentile(v[st], 95)), 3) if st.any() else None}
    out['root'] = {'height_m': [round(float(p[:, 2].min()), 3), round(float(p[:, 2].max()), 3)],
                   'speed_mps': round(float(np.linalg.norm(np.gradient(p[:, :2], 1.0 / FPS, axis=0), axis=1).mean()), 2)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv', type=Path)
    ap.add_argument('--fps', type=float, default=120.0)
    ap.add_argument('--skip', type=int, default=0)
    ap.add_argument('--clip', default=None)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--pkl', type=Path, default=None, help='also write the clip as a SONIC motion pkl, for SuperMimic')
    a = ap.parse_args()
    t, names, q, p, wxyz = load(a.csv, a.fps, a.skip)
    p, shift = ground(names, q, p, wxyz)
    print(f'grounded: largest root shift {1e3 * shift:.0f} mm')
    out = a.out or ROOT / 'train' / 'review' / f'{a.csv.stem}.npz'
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, t=t, q=q, joint_names=np.array(names), root_pos=p, root_quat_wxyz=wxyz, clip=a.clip or a.csv.stem)
    if a.pkl:
        m = mujoco.MjModel.from_xml_path(str(XML))
        order = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)]
        assert names == order, 'CSV joints are not in MJCF order'
        xyzw = np.column_stack((wxyz[:, 1:], wxyz[:, :1]))
        joblib.dump({a.clip or a.csv.stem: dict(root_trans_offset=p.astype(np.float32), root_rot=xyzw.astype(np.float32), dof=q.astype(np.float32), fps=FPS)}, a.pkl)
    rep = report(names, q, p, wxyz, json.loads((ROOT / 'spec' / 'robot.json').read_text(encoding='utf-8')))
    out.with_suffix('.json').write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(out)
    print(json.dumps(rep))


if __name__ == '__main__':
    main()
