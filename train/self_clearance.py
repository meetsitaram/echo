"""How far the arms sit inside the pelvis and upper legs (selfcol_* geoms of urdf/echo.xml), through a motion file or takes.

  python train/self_clearance.py <motion.pkl | take.npz> [...] [--step 5] [--bad] [--list N]

--bad: rolls the arms into the hips on purpose; the test must see it (exit 1). Prints the share of clips and frames in
contact and the depth; exits 1 if any frame is deeper than 1 mm.
"""
import argparse
import collections
import pathlib

import joblib
import mujoco
import numpy as np

XML = pathlib.Path(__file__).resolve().parents[1] / 'urdf' / 'echo.xml'


def frames(path):
    """{clip: (root_pos, root_quat_wxyz, q)}"""
    path = pathlib.Path(path)
    if path.suffix == '.npz':
        t = np.load(path, allow_pickle=True)
        return {path.stem: (t['root_pos'], t['root_quat_wxyz'], t['q'])}
    out = {}
    for k, d in joblib.load(path).items():
        out[k] = (d['root_trans_offset'], d['root_rot'][:, [3, 0, 1, 2]], d['dof'])
    return out


def scan(m, d, pos, quat, q, step, bad=None):
    depth = collections.defaultdict(list)
    n = 0
    for i in range(0, len(q), step):
        d.qpos[:3], d.qpos[3:7], d.qpos[7:] = pos[i], quat[i], q[i]
        if bad:
            for j, v in bad:
                d.qpos[j] = v
        mujoco.mj_forward(m, d)
        n += 1
        worst = {}
        for c in d.contact[:d.ncon]:
            a, b = m.geom(c.geom1).name, m.geom(c.geom2).name
            if 'selfcol' in a and 'selfcol' in b and c.dist < 0:
                k = tuple(sorted(x.replace('selfcol_arm_', '').replace('selfcol_body_', '').rsplit('_', 1)[0] for x in (a, b)))
                worst[k] = max(worst.get(k, 0.0), -c.dist)
        for k, v in worst.items():
            depth[k].append(v)
    return n, depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('motions', nargs='+')
    ap.add_argument('--step', type=int, default=5)
    ap.add_argument('--bad', action='store_true')
    ap.add_argument('--list', type=int, default=0)
    a = ap.parse_args()
    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    assert m.opt.timestep > 0
    bad = None
    if a.bad:  # shoulders rolled hard inward, elbows straight
        bad = [(m.jnt_qposadr[m.joint(f'{s}_shoulder_roll').id], m.jnt_range[m.joint(f'{s}_shoulder_roll').id][0 if s == 'L' else 1]) for s in 'LR']
    found = False
    for p in a.motions:
        clips = frames(p)
        total, hit, per_pair, per_clip = 0, 0, collections.defaultdict(lambda: [0, 0.0]), []
        for k, (pos, quat, q) in clips.items():
            n, dep = scan(m, d, pos, quat, q, a.step, bad)
            total += n
            frames_hit = max((len(v) for v in dep.values()), default=0)
            hit += frames_hit
            w = max((max(v) for v in dep.values()), default=0.0)
            per_clip.append((w, frames_hit / max(n, 1), k))
            for pair, v in dep.items():
                per_pair[pair][0] += len(v)
                per_pair[pair][1] = max(per_pair[pair][1], max(v))
        w = np.array([x[0] for x in per_clip])
        found |= bool((w > 0.001).any())
        print(f'{pathlib.Path(p).name}: {len(clips)} clips, {(w > 0.001).mean() * 100:.0f}% touch, {(w > 0.01).mean() * 100:.0f}% over 1 cm, {(w > 0.03).mean() * 100:.0f}% over 3 cm; frames in contact {hit / max(total, 1) * 100:.0f}%; median clip worst {np.median(w) * 100:.1f} cm, max {w.max() * 100:.1f} cm')
        for pair, (n, v) in sorted(per_pair.items(), key=lambda kv: -kv[1][0])[:10]:
            print(f'   {pair[0]} / {pair[1]}: {n / max(total, 1) * 100:.1f}% of frames, worst {v * 100:.1f} cm')
        for w_, f_, k in sorted(per_clip, reverse=True)[:a.list]:
            print(f'   {k[:50]:52s}{w_ * 100:5.1f} cm, {f_ * 100:3.0f}% of frames')
    for pose, name in ((np.zeros(m.nv - 6), 'zero pose'),):
        d.qpos[:3], d.qpos[3:7], d.qpos[7:] = (0, 0, 1), (1, 0, 0, 0), pose
        mujoco.mj_forward(m, d)
        print(name, 'selfcol contacts:', sorted((m.geom(c.geom1).name[8:], m.geom(c.geom2).name[8:], round(-c.dist * 100, 1)) for c in d.contact[:d.ncon] if 'selfcol' in m.geom(c.geom1).name and c.dist < 0)[:12])
    raise SystemExit(1 if found else 0)


if __name__ == '__main__':
    main()
