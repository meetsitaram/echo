"""Where the body sits inside itself or the floor (selfcol_* and floorcol_* geoms of urdf/echo.xml), through a motion file or
takes, measured with the class the reward uses (gear_sonic.supermimic.self_clear).

  PYTHONPATH=<gwbc-v6> python train/self_clearance.py <motion.pkl | take.npz> [...] [--step 5] [--list N]
  PYTHONPATH=<gwbc-v6> python train/self_clearance.py --bad

Prints, per group (arm: hand and forearm against the body; leg: leg against leg; floor: anything but a sole under the floor),
the clips and frames touching and the depth; exits 1 if any frame is deeper than 1 mm.
--bad: poses that must be flagged (crossed legs, a hand inside a thigh, ...) and poses that must not (zero pose, standing).
Exits 1 when it sees every bad pose and passes every good one, as a check on a broken design should.
"""
import argparse
import pathlib

import joblib
import mujoco
import numpy as np
import torch
import yaml

from gear_sonic.supermimic.kinematics import X2Kinematics, quat_wxyz_to_mat
from gear_sonic.supermimic.self_clear import SelfClear

HERE = pathlib.Path(__file__).resolve().parent
XML = HERE.parents[0] / 'urdf' / 'echo.xml'
PLANT = HERE / 'echo_plant_v6.yaml'
TOUCH = 0.001  # m
MARGIN = 0.005  # the training margin (self_pen_margin): a good pose must be clear of it too

BAD = {
    'legs crossed at the thigh': {'L_hip_roll': -0.35, 'R_hip_roll': 0.35},
    'one thigh across the other': {'L_hip_pitch': -0.8, 'L_hip_roll': -0.35, 'R_hip_roll': 0.35, 'R_hip_pitch': -0.8},
    'shank on shank, knees in': {'L_hip_roll': -0.3, 'R_hip_roll': 0.3, 'L_hip_yaw': 0.5, 'R_hip_yaw': -0.5, 'L_knee': 0.5, 'R_knee': 0.5},
    'toes through each other': {'L_hip_yaw': -1.2, 'R_hip_yaw': 1.2, 'L_shoulder_roll': 0.8, 'R_shoulder_roll': -0.8},
    'hand inside the thigh': {'L_hip_roll': 0.6, 'L_shoulder_roll': -0.05},
    'hand inside the belly': {'L_shoulder_yaw': -1.57, 'L_elbow': -1.6},
    'forearm through the chest': {'R_shoulder_yaw': 1.57, 'R_elbow': -1.9, 'R_shoulder_pitch': -0.3},
}
GOOD = {
    'zero pose': {},
    'wide stance, arms clear': {'L_hip_roll': 0.3, 'R_hip_roll': -0.3, 'L_shoulder_roll': 0.5, 'R_shoulder_roll': -0.5},
    'arms out': {'L_shoulder_roll': 1.5, 'R_shoulder_roll': -1.5},
    'deep squat, arms forward': {'L_hip_pitch': -1.5, 'R_hip_pitch': -1.5, 'L_knee': 2.0, 'R_knee': 2.0, 'L_ankle_pitch': -0.5, 'R_ankle_pitch': -0.5, 'L_shoulder_pitch': -1.5, 'R_shoulder_pitch': -1.5},
}


class Scan:
    def __init__(self, xml=XML):
        self.m = mujoco.MjModel.from_xml_path(str(xml))
        self.d = mujoco.MjData(self.m)
        self.kin, self.sc = X2Kinematics(self.m, 'cpu'), SelfClear(self.m, 'cpu')
        self.names = [self.m.joint(i).name for i in range(1, self.m.njnt)]
        self.floor = {g for g in range(self.m.ngeom) if (self.m.geom(g).name or '').startswith('floorcol_')}

    def pose(self, q):
        """q (T, nj) -> P, R of every body with the root at the origin (self contact does not depend on it)."""
        q = q if torch.is_tensor(q) else torch.as_tensor(np.asarray(q), dtype=torch.float32)
        z, eye = torch.zeros(len(q), 3), torch.eye(3).expand(len(q), 3, 3)
        return self.kin.forward(z, eye, z, z, q, torch.zeros_like(q))[:2]

    def world(self, q, pos, quat):
        """Bodies in the world: P (T, nb, 3), R (T, nb, 3, 3). pos and quat (wxyz) tensors."""
        q = q if torch.is_tensor(q) else torch.as_tensor(np.asarray(q), dtype=torch.float32)
        z = torch.zeros(len(q), 3)
        return self.kin.forward(pos, quat_wxyz_to_mat(quat), z, z, q, torch.zeros_like(q))[:2]

    def below_floor(self, P, R):
        """(T,) summed depth under the floor over the floorcol_* geoms, and over the soles; differentiable. The mj_collision
        reading in floor_depth checks it. Capsule ends, sphere centres, box corners."""
        if not hasattr(self, '_pts'):
            m, b, p, r, sole = self.m, [], [], [], []
            for g in range(m.ngeom):
                n = m.geom(g).name or ''
                if not (n.startswith('floorcol_') or n.endswith('_sole')):
                    continue
                t, size, rot = m.geom_type[g], m.geom_size[g], np.zeros(9)
                mujoco.mju_quat2Mat(rot, m.geom_quat[g])
                rot = rot.reshape(3, 3)
                if t == mujoco.mjtGeom.mjGEOM_SPHERE:
                    loc, rad = [np.zeros(3)], size[0]
                elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    loc, rad = [np.array([0, 0, s_ * size[1]]) for s_ in (-1, 1)], size[0]
                else:
                    loc, rad = [np.array([x, y, z]) * size for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], 0.0
                for v in loc:
                    b.append(m.geom_bodyid[g] - 1), p.append(m.geom_pos[g] + rot @ v), r.append(rad), sole.append(n.endswith('_sole'))
            self._pts = torch.tensor(b), torch.tensor(np.array(p), dtype=torch.float32), torch.tensor(r, dtype=torch.float32), torch.tensor(sole)
        b, p, r, sole = self._pts
        depth = torch.relu(r - P[:, b, 2] - (R[:, b, 2, :] * p).sum(-1))
        return depth[:, ~sole].sum(-1), depth[:, sole].sum(-1)

    def self_depth(self, q, margin=0.0):
        """{group: (T,) summed overlap in m}."""
        P, R = self.pose(q)
        return {k: v.numpy() for k, v in self.sc.depths(P, R, margin).items()}

    def floor_depth(self, pos, quat, q):
        """(T,) deepest floorcol_* geom under the floor in m, and {geom: frames}."""
        m, d = self.m, self.d
        out, where = np.zeros(len(q)), {}
        for i in range(len(q)):
            d.qpos[:3], d.qpos[3:7], d.qpos[7:] = pos[i], quat[i], q[i]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_collision(m, d)
            for c in d.contact[:d.ncon]:
                g = c.geom1 if c.geom1 in self.floor else c.geom2 if c.geom2 in self.floor else -1
                if g >= 0 and c.dist < 0:
                    out[i] = max(out[i], -c.dist)
                    if -c.dist > TOUCH:
                        where[m.geom(g).name] = where.get(m.geom(g).name, 0) + 1
        return out, where


def frames(path):
    """{clip: (root_pos, root_quat_wxyz, q)}"""
    path = pathlib.Path(path)
    if path.suffix == '.npz':
        t = np.load(path, allow_pickle=True)
        return {path.stem: (t['root_pos'], t['root_quat_wxyz'], t['q'])}
    return {k: (d['root_trans_offset'], d['root_rot'][:, [3, 0, 1, 2]], d['dof']) for k, d in joblib.load(path).items()}


def report(s, clips, step=5):
    """Prints the touching clips and frames per group; returns {group: (clips touching, frames touching, worst m), 'total': ...}."""
    res, n_total = {}, 0
    floor_where = {}
    for pos, quat, q in clips.values():
        i = np.arange(0, len(q), step)
        n_total += len(i)
        dep = s.self_depth(q[i])
        dep['floor'], w = s.floor_depth(pos[i], quat[i], q[i])
        for k, v in w.items():
            floor_where[k] = floor_where.get(k, 0) + v
        for k, v in dep.items():
            r = res.setdefault(k, [0, 0, 0.0])
            r[0] += int((v > TOUCH).any())
            r[1] += int((v > TOUCH).sum())
            r[2] = max(r[2], float(v.max()))
    for k, (c, f, w) in res.items():
        print(f'  {k:6s} {c}/{len(clips)} clips, {f}/{n_total} frames ({f / max(n_total, 1) * 100:.1f}%), worst {w * 100:.1f} cm')
    if floor_where:
        print('  floor, frames per geom:', dict(sorted(floor_where.items(), key=lambda kv: -kv[1])))
    res['frames'] = n_total
    return res


def bad_test(s):
    plant = yaml.safe_load(PLANT.read_text(encoding='utf-8'))['default_joint_pos']
    default = {n: next((v for k, v in plant.items() if k in n), 0.0) for n in s.names}
    good = dict(GOOD, **{'standing (the plant default)': default})
    ok = True

    def depth(p):
        q = torch.zeros(1, len(s.names))
        for k, v in p.items():
            q[0, s.names.index(k)] = v
        P, R = s.pose(q)
        return s.sc, P[0], R[0]

    print(f'must be flagged (over {TOUCH * 1e3:.0f} mm, no margin; the leg cases in the leg group, with the arms out of the way where they say so):')
    for name, p in BAD.items():
        sc, P, R = depth(p)
        over = sc.touching(P, R)
        total = sum(x[2] for x in over)
        ok &= total > TOUCH
        w = over[0] if over else ('-', '-', 0.0)
        groups = {k: round(float(v) * 100, 1) for k, v in sc.depths(P, R).items()}
        print(f'  {"seen  " if total > TOUCH else "MISSED"} {name:30s} cm summed {groups}; deepest {w[0][8:]} / {w[1][8:]} {w[2] * 100:.1f} cm')
    print(f'must be clear (even with the {MARGIN * 1e3:.0f} mm training margin):')
    for name, p in good.items():
        sc, P, R = depth(p)
        over = sc.touching(P, R, MARGIN)
        ok &= not over
        print(f'  {"clear " if not over else "FALSE "} {name:30s} {[(a[8:], b[8:], round(v * 100, 1)) for a, b, v in over[:3]]}')
        # the closest any sphere pair comes, so a pass near the line reads as one
        d = min(float((torch.linalg.norm(a.world(P, R)[:, None] - b.world(P, R)[None, :], dim=-1) - a.r[:, None] - b.r[None, :]).min()) for _, a, b in sc.blocks)
        print(f'         closest pair {d * 1000:.0f} mm apart')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('motions', nargs='*')
    ap.add_argument('--step', type=int, default=5)
    ap.add_argument('--bad', action='store_true')
    ap.add_argument('--xml', default=str(XML))
    a = ap.parse_args()
    s = Scan(a.xml)
    if a.bad:
        ok = bad_test(s)
        print('the test sees the broken poses and passes the good ones' if ok else 'THE TEST IS WRONG: see above')
        raise SystemExit(1 if ok else 0)
    found = False
    for p in a.motions:
        print(pathlib.Path(p).name)
        res = report(s, frames(p), a.step)
        found |= any(v[1] for k, v in res.items() if k != 'frames')
    raise SystemExit(1 if found else 0)


if __name__ == '__main__':
    main()
