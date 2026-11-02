"""PICO full-body capture -> echo (root pose + 29 joints), for kinematic playback.

Ported from the X2 retargeter in agibot-x2-record-and-replay (src/x2_recorder/pico_mocap/x2_ik.py, itself from
gwbc-supermimic and GMR; both only read): the PICO 24-joint body is morphed to the robot's segment lengths along the
human's directions, then a two-stage mink IK follows it (a frame task per body, the elbow flex as a posture task),
warm-started from the previous frame. For echo only the body names, the elbow range and two reference bodies differ:
'pelvis_ik' on the hip line (our pelvis frame is 3 cm under it; the X2's is on it) and a toe body in each foot. The IK
table is tools/ik/pico_to_echo.json.

The IK table and the solve loop are adapted from GMR (github.com/YanjieZe/GMR, MIT licence: its notice is in
THIRD_PARTY_NOTICES.md); the rest is the author's own code.

  python tools/retarget_pico.py <raw tape .npz> [--start S] [--dur D] [--out reports/motion/<name>.npz]

The take: t, q (29 joints, MuJoCo order), joint_names, root_pos, root_quat_wxyz at 50 Hz, smoothed (sigma 40 ms), plus
the human skeleton at robot scale (skel, 24 x 3 per frame) for side-by-side playback. Plain python (mink and MuJoCo
live in the user site-packages); the tape is read with allow_pickle=False.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as sRot, Slerp

ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / 'urdf' / 'echo.xml'
IK_CONFIG = ROOT / 'tools' / 'ik' / 'pico_to_echo.json'

JOINT_NAMES = [
    'Pelvis', 'Left_Hip', 'Right_Hip', 'Spine1', 'Left_Knee', 'Right_Knee',
    'Spine2', 'Left_Ankle', 'Right_Ankle', 'Spine3', 'Left_Foot',
    'Right_Foot', 'Neck', 'Left_Collar', 'Right_Collar', 'Head',
    'Left_Shoulder', 'Right_Shoulder', 'Left_Elbow', 'Right_Elbow',
    'Left_Wrist', 'Right_Wrist', 'Left_Hand', 'Right_Hand',
]
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

# Unity (left-handed Y-up) -> right-handed Z-up (XRobotStreamer convention)
_M_UNITY = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
_Q_UNITY = sRot.from_matrix(_M_UNITY)

TOE_X = 0.13                   # toe body ahead of the ankle axis (the Pico Foot joint is at the ball of the foot)
FOOT_TARGET_Z = 0.018          # toe body height above the sole when standing flat
FOOT_ROLL_MAX = np.sin(np.radians(10))   # foot out of the knee plane the ankle roll can still give
FOOT_JOINT_H = 0.03            # Pico Foot joint above the floor, foot flat
LEG_TO_PELVIS_H = 0.965        # standing pelvis height / (pelvis-hip + thigh + shin), Pico, 16 tapes (X2 repo)
FLAT_BAND = 0.08               # ankle this high above its flat height -> human foot pitch
MOCAP_TIMESTEP = 0.004         # mink integrates by opt.timestep; 1 ms lags fast folds
FPS = 50
SMOOTH_SIGMA_S = 0.04

MORPH_CHAIN = [
    ('Spine3', 'Pelvis', 'torso'), ('Head', 'Pelvis', 'head'),
    ('Left_Shoulder', 'Spine3', 'left_clavicle'), ('Right_Shoulder', 'Spine3', 'right_clavicle'),
    ('Left_Elbow', 'Left_Shoulder', 'left_upper'), ('Right_Elbow', 'Right_Shoulder', 'right_upper'),
    ('Left_Wrist', 'Left_Elbow', 'left_fore'), ('Right_Wrist', 'Right_Elbow', 'right_fore'),
]


# ---------------------------------------------------------------- model

def build_mocap_model(xml: Path = XML) -> mujoco.MjModel:
    """Our MJCF plus 'pelvis_ik' on the hip line and a toe body in each foot, coarser timestep, for kinematics."""
    m0 = mujoco.MjModel.from_xml_path(str(xml))
    d0 = mujoco.MjData(m0)
    mujoco.mj_forward(m0, d0)
    x = lambda n: d0.xpos[m0.body(n).id].copy()
    # a fresh spec, compiled once: adding bodies to a spec compiled before put the right toe on the ankle pitch
    # link instead of the foot (2026-10-31), so the right ankle roll never moved
    spec = mujoco.MjSpec.from_file(str(xml))
    spec.option.timestep = MOCAP_TIMESTEP
    spec.body('pelvis').add_body(name='pelvis_ik', pos=[0, 0, x('L_hip_roll_link')[2] - x('pelvis')[2]])
    for s in 'LR':
        spec.body(f'{s}_foot').add_body(name=f'{s}_toe', pos=[TOE_X, 0, FOOT_TARGET_Z - x(f'{s}_foot')[2]])
    m = spec.compile()
    for child, parent in (('pelvis_ik', 'pelvis'), ('L_toe', 'L_foot'), ('R_toe', 'R_foot')):
        got = m.body(m.body_parentid[m.body(child).id]).name
        assert got == parent, f'{child} landed on {got}, not {parent}'
    return m


def measure_segments(model: mujoco.MjModel) -> dict:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    p = lambda n: data.xpos[model.body(n).id].copy()
    seg = {}
    pel = p('pelvis_ik')
    for S, s in (('L', 'left'), ('R', 'right')):
        seg[f'{s}_hip_off'] = p(f'{S}_hip_roll_link') - pel
        seg[f'{s}_hip_yaw_along'] = np.linalg.norm(p(f'{S}_hip_yaw_link') - p(f'{S}_hip_roll_link'))
        seg[f'{s}_thigh'] = np.linalg.norm(p(f'{S}_knee_link') - p(f'{S}_hip_roll_link'))
        seg[f'{s}_shin'] = np.linalg.norm(p(f'{S}_ankle_pitch_link') - p(f'{S}_knee_link'))
        seg[f'{s}_foot'] = np.linalg.norm(p(f'{S}_toe') - p(f'{S}_ankle_pitch_link'))
        # the arm's top is the shoulder centre, where the roll axis crosses the pitch axis
        seg[f'{s}_upper'] = np.linalg.norm(p(f'{S}_elbow_link') - p(f'{S}_shoulder_roll_link'))
        seg[f'{s}_fore'] = np.linalg.norm(p(f'{S}_wrist_pitch_link') - p(f'{S}_elbow_link'))
        seg[f'{s}_shoulder_off'] = p(f'{S}_shoulder_roll_link') - pel
    seg['torso'] = np.linalg.norm(p('torso') - pel)
    seg['head'] = np.linalg.norm(p('head') - pel)
    seg['pelvis_h'] = pel[2] - min(p('L_toe')[2], p('R_toe')[2])
    foot = p('L_toe') - p('L_ankle_pitch_link')
    seg['foot_pitch'] = float(np.arcsin(-foot[2] / np.linalg.norm(foot)))   # ankle -> toe, sole flat
    seg['ankle_h'] = float(p('L_ankle_pitch_link')[2] - p('L_toe')[2]) + FOOT_TARGET_Z
    seg['torso_off'] = p('torso') - pel
    return seg


# ---------------------------------------------------------------- human frames (X2 repo, unchanged)

def body_to_frame(body: np.ndarray) -> dict:
    """(24, 7) Unity [x y z qx qy qz qw] -> {joint: [pos, quat_wxyz]} in Z-up."""
    frame = {}
    for j, name in enumerate(JOINT_NAMES):
        pos = _M_UNITY @ body[j, :3].astype(np.float64)
        quat = (_Q_UNITY * sRot.from_quat(body[j, [3, 4, 5, 6]].astype(np.float64))).as_quat(scalar_first=True)
        frame[name] = [pos, quat]
    return frame


def arm_plane_frames(S, E, W, lat_ref):
    """(upper-arm, forearm) frames as wxyz quats: x = segment, y = arm-plane normal (kept on lat_ref's side)."""
    u = (E - S) / max(np.linalg.norm(E - S), 1e-9)
    w = (W - E) / max(np.linalg.norm(W - E), 1e-9)
    n = np.cross(u, w)
    if np.linalg.norm(n) < 0.15:
        n = lat_ref - u * np.dot(lat_ref, u)
        if np.linalg.norm(n) < 1e-6:
            n = np.array([0.0, 1.0, 0.0])
    n = n / np.linalg.norm(n)
    if np.dot(n, lat_ref) < 0:
        n = -n

    def frame(x):
        z = np.cross(x, n)
        z = z / max(np.linalg.norm(z), 1e-9)
        return sRot.from_matrix(np.stack([x, n, z], axis=1)).as_quat(scalar_first=True)

    return frame(u), frame(w), n


def _trunk_rotation(frame: dict) -> np.ndarray:
    up = np.asarray(frame['Neck'][0]) - np.asarray(frame['Pelvis'][0])
    up = up / max(np.linalg.norm(up), 1e-9)
    lat = np.asarray(frame['Left_Shoulder'][0]) - np.asarray(frame['Right_Shoulder'][0])
    lat = lat - up * np.dot(lat, up)
    lat = lat / max(np.linalg.norm(lat), 1e-9)
    return np.stack([np.cross(lat, up), lat, up], axis=1)


def _morph_leg(frame: dict, seg: dict, out: dict, side: str, stance_scale: float | None = None) -> None:
    """Hip pivot on the human hip line; thigh, shin, foot along the human directions with robot lengths."""
    s = side.lower()
    lat = np.asarray(frame['Left_Hip'][0]) - np.asarray(frame['Right_Hip'][0])
    lat = lat / max(np.linalg.norm(lat), 1e-9)
    pivot = out['Pelvis'][0] + lat * seg[f'{s}_hip_off'][1]

    def unit(a, b):
        d = np.asarray(frame[b][0]) - np.asarray(frame[a][0])
        n = np.linalg.norm(d)
        return d / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])

    thigh = unit(f'{side}_Hip', f'{side}_Knee')
    knee = pivot + thigh * seg[f'{s}_thigh']
    ankle = knee + unit(f'{side}_Knee', f'{side}_Ankle') * seg[f'{s}_shin']
    if stance_scale is not None:
        want = np.dot(np.asarray(frame[f'{side}_Ankle'][0]) - np.asarray(frame['Pelvis'][0]), lat) * stance_scale
        shift = lat * (want - np.dot(ankle - out['Pelvis'][0], lat))
        knee, ankle = knee + shift / 2, ankle + shift
    out[f'{side}_Hip'] = [pivot + thigh * seg[f'{s}_hip_yaw_along'], frame[f'{side}_Hip'][1]]
    out[f'{side}_Knee'] = [knee, frame[f'{side}_Knee'][1]]
    out[f'{side}_Ankle'] = [ankle, frame[f'{side}_Ankle'][1]]
    foot = unit(f'{side}_Ankle', f'{side}_Foot')
    out[f'{side}_Foot'] = [ankle + foot * seg[f'{s}_foot'], frame[f'{side}_Foot'][1]]
    shin = unit(f'{side}_Knee', f'{side}_Ankle')
    hinge = np.cross(thigh, shin)
    k = np.clip(np.linalg.norm(hinge) / np.sin(np.radians(20)), 0.0, 1.0)
    hinge = k * hinge / max(np.linalg.norm(hinge), 1e-9) * np.sign(np.dot(hinge, lat) + 1e-9) + (1 - k) * lat
    hinge = hinge - shin * np.dot(hinge, shin)
    hinge = hinge / max(np.linalg.norm(hinge), 1e-9)
    dn = np.dot(foot, hinge)
    lifted = foot - hinge * (dn - np.clip(dn, -FOOT_ROLL_MAX, FOOT_ROLL_MAX))
    out[f'{side}_FootLifted'] = [lifted / max(np.linalg.norm(lifted), 1e-9), None]


def morph_frame(frame: dict, seg: dict, height_scale: float, prev_normals: dict, human_stance: bool = False) -> dict:
    """Robot segment lengths along human directions; arm rotations rebuilt from geometry."""
    out = {'Pelvis': [np.asarray(frame['Pelvis'][0]) * height_scale, frame['Pelvis'][1]]}
    for side in ('Left', 'Right'):
        _morph_leg(frame, seg, out, side, height_scale if human_stance else None)
    trunk = _trunk_rotation(frame)
    for joint, parent, key in MORPH_CHAIN:
        if joint in ('Left_Shoulder', 'Right_Shoulder'):
            off = seg[f"{joint.split('_')[0].lower()}_shoulder_off"]
            out[joint] = [out['Pelvis'][0] + trunk @ off, frame[joint][1]]
            continue
        d = np.asarray(frame[joint][0]) - np.asarray(frame[parent][0])
        n = np.linalg.norm(d)
        d = d / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
        out[joint] = [out[parent][0] + d * seg[key], frame[joint][1]]
    lat = np.asarray(frame['Left_Shoulder'][0]) - np.asarray(frame['Right_Shoulder'][0])
    for side, sgn in (('Left', -1.0), ('Right', 1.0)):
        ref = prev_normals.get(side, lat * sgn)
        q_up, q_fore, n_used = arm_plane_frames(*(np.asarray(frame[f'{side}_{j}'][0])
                                                  for j in ('Shoulder', 'Elbow', 'Wrist')), ref)
        prev_normals[side] = n_used
        out[f'{side}_Shoulder'][1] = q_up
        out[f'{side}_Elbow'][1] = q_fore
    return out


def standing_problem(frame: dict) -> str | None:
    feet = min(frame['Left_Foot'][0][2], frame['Right_Foot'][0][2])
    if frame['Pelvis'][0][2] - feet < 0.5 * (frame['Head'][0][2] - feet):
        return 'stand up straight to calibrate'
    for side in ('Left', 'Right'):
        h, k, f = (np.asarray(frame[f'{side}_{j}'][0]) for j in ('Hip', 'Knee', 'Foot'))
        u, v = h - k, f - k
        if np.degrees(np.arccos(np.clip(np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-9), -1, 1))) < 150:
            return 'straighten your legs to calibrate'
    return None


def is_rest_pose(frame: dict, slack: float = 0.25) -> bool:
    return max(frame['Left_Wrist'][0][2], frame['Right_Wrist'][0][2]) < frame['Pelvis'][0][2] + slack


# ---------------------------------------------------------------- retargeter

class EchoRetargeter:
    """Stateful per-frame solver (warm-started from the previous frame); X2Retargeter with echo names."""

    def __init__(self, config_path: Path = IK_CONFIG, iters: int = 2, damping: float = 0.5):
        self.model = build_mocap_model()
        self.seg = measure_segments(self.model)
        self.base_cfg = json.loads(Path(config_path).read_text(encoding='utf-8'))
        self.iters = iters
        self.damping = damping
        self.height_scale: float | None = None
        self.ground_z: float | None = None
        self._normals: dict = {}
        self.human_stance = True     # feet as far apart as the human's, scaled (offline retargeting)
        self._build(self.base_cfg)

    @property
    def calibrated(self) -> bool:
        return self.height_scale is not None

    def _build(self, cfg: dict) -> None:
        self.cfg = cfg
        self.configuration = mink.Configuration(self.model)
        self.limits = [mink.ConfigurationLimit(self.model)]
        self.tasks: list[list] = [[], []]
        self.task_of: list[dict] = [{}, {}]
        self.pos_off: dict = {}
        self.rot_off: dict = {}
        for k, table in enumerate(('ik_match_table1', 'ik_match_table2')):
            for link, (human, pw, rw, pos_off, rot_off) in cfg[table].items():
                if pw == 0 and rw == 0:
                    continue
                task = mink.FrameTask(frame_name=link, frame_type='body', position_cost=pw, orientation_cost=rw,
                                      lm_damping=1)
                self.tasks[k].append(task)
                self.task_of[k][human] = task
                if k == 0:
                    self.pos_off[human] = np.asarray(pos_off, dtype=np.float64)
                    self.rot_off[human] = sRot.from_quat(rot_off, scalar_first=True)
        cost = np.zeros(self.model.nv)
        self._elbow = {}
        for s in ('Left', 'Right'):
            j = self.model.joint(f'{s[0]}_elbow').id
            self._elbow[s] = (self.model.jnt_qposadr[j], self.model.jnt_range[j])
            cost[self.model.jnt_dofadr[j]] = 8.0
        self.posture = mink.PostureTask(self.model, cost=cost)
        for tasks in self.tasks:
            tasks.append(self.posture)
        self.scale_keys = set(cfg['human_scale_table'])

    def calibrate(self, body: np.ndarray, warmup: int = 20) -> None:
        """Rest sample (standing, arms hanging) -> height ratio + arm and head rotation offsets."""
        frame = body_to_frame(body)
        hum_pelvis_h = frame['Pelvis'][0][2] - min(frame['Left_Foot'][0][2], frame['Right_Foot'][0][2])
        if hum_pelvis_h < 0.4:
            raise ValueError(f'implausible pelvis height {hum_pelvis_h:.2f} m')
        self.height_scale = float(self.seg['pelvis_h'] / hum_pelvis_h)
        self.ground_z = None
        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        yaw = sRot.from_quat(frame['Pelvis'][1], scalar_first=True).as_euler('zyx')[0]
        r_yaw = sRot.from_euler('z', yaw)
        cfg = copy.deepcopy(self.base_cfg)

        def set_offset(link: str, human_q) -> None:
            r_link = sRot.from_quat(data.xquat[self.model.body(link).id], scalar_first=True)
            off = (sRot.from_quat(human_q, scalar_first=True).inv() * r_yaw * r_link).as_quat(scalar_first=True)
            for tbl in ('ik_match_table1', 'ik_match_table2'):
                if link in cfg[tbl]:
                    cfg[tbl][link][4] = [float(v) for v in off]

        lat = frame['Left_Shoulder'][0] - frame['Right_Shoulder'][0]
        for side, sgn in (('Left', -1.0), ('Right', 1.0)):
            q_up, q_fore, _ = arm_plane_frames(*(frame[f'{side}_{j}'][0] for j in ('Shoulder', 'Elbow', 'Wrist')),
                                               lat * sgn)
            set_offset(f'{side[0]}_shoulder_yaw_link', q_up)
            set_offset(f'{side[0]}_elbow_link', q_fore)
            set_offset(f'{side[0]}_wrist_pitch_link', frame[f'{side}_Wrist'][1])
        set_offset('head', frame['Head'][1])
        self._normals = {}
        self._build(cfg)
        for _ in range(warmup):
            self.step(body)

    def _targets(self, frame: dict) -> dict:
        root_pos = frame['Pelvis'][0]
        out = {}
        for name in self.scale_keys:
            pos, quat = frame[name]
            s = self.cfg['human_scale_table'][name]
            p = s * root_pos if name == 'Pelvis' else \
                (pos - root_pos) * s + root_pos * self.cfg['human_scale_table']['Pelvis']
            if name in self.rot_off:
                r = sRot.from_quat(quat, scalar_first=True) * self.rot_off[name]
                quat = r.as_quat(scalar_first=True)
                p = p + r.apply(self.pos_off[name])
            out[name] = [p, quat]
        if self.ground_z is None:     # the floor, once (from the calibration frame): jumps and lifted legs stay
            self.ground_z = min(v[0][2] for k, v in out.items() if 'Foot' in k) - FOOT_TARGET_Z
        for v in out.values():
            v[0] = v[0] - np.array([0.0, 0.0, self.ground_z])
        for side in ('Left', 'Right'):
            foot = out[f'{side}_Foot']
            foot[1] = self._plant_foot(out[f'{side}_Ankle'][0], foot[0], foot[1], frame[f'{side}_FootLifted'][0],
                                       side.lower())
        return out

    def _plant_foot(self, ankle, toe, quat, lifted, side):
        """Sole flat near the floor, the human's foot (in the knee plane) once lifted; nothing below the floor."""
        ankle[2] = max(ankle[2], self.seg['ankle_h'])
        w = np.clip((ankle[2] - self.seg['ankle_h']) / FLAT_BAND, 0.0, 1.0)
        d = toe - ankle
        horiz = np.array([d[0], d[1], 0.0])
        nh = np.linalg.norm(horiz)
        d = lifted
        if nh > 1e-6:
            p = self.seg['foot_pitch']
            flat = horiz / nh * np.cos(p) - np.array([0.0, 0.0, np.sin(p)])
            d = w * lifted + (1 - w) * flat
            d = d / max(np.linalg.norm(d), 1e-9)
        toe[:] = ankle + d * self.seg[f'{side}_foot']
        toe[2] = max(toe[2], FOOT_TARGET_Z)
        r = sRot.from_quat(quat, scalar_first=True)
        r_flat = sRot.from_euler('z', r.as_euler('zyx')[0])
        return Slerp([0.0, 1.0], sRot.concatenate([r_flat, r]))(w).as_quat(scalar_first=True)

    def step(self, body: np.ndarray) -> np.ndarray:
        """One PICO body sample (24, 7, Unity) -> qpos (7 root + 29 joints)."""
        return self.solve(*self.prepare(body))

    def prepare(self, body: np.ndarray):
        """The morphed targets and elbow angles for one sample, without solving (so a whole take's targets can be
        edited first: lock_feet)."""
        assert self.calibrated, 'call calibrate() with a rest frame first'
        frame = morph_frame(body_to_frame(body), self.seg, self.height_scale, self._normals, self.human_stance)
        elbows = {}
        for side in ('Left', 'Right'):
            S, E, W = (np.asarray(frame[f'{side}_{j}'][0]) for j in ('Shoulder', 'Elbow', 'Wrist'))
            u, v = S - E, W - E
            c = np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-9)
            adr, (lo, hi) = self._elbow[side]
            elbows[adr] = np.clip(-(np.pi - np.arccos(np.clip(c, -1, 1))), lo, hi)
        return self._targets(frame), elbows

    def solve(self, targets: dict, elbows: dict) -> np.ndarray:
        q = self.configuration.q.copy()
        for adr, v in elbows.items():
            q[adr] = v
        self.posture.set_target(q)
        for k in range(2):
            for human, task in self.task_of[k].items():
                pos, quat = targets[human]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(quat), pos))
        dt = self.model.opt.timestep
        for _ in range(self.iters):
            for tasks in self.tasks:
                err = self._error(tasks)
                for _ in range(10):
                    vel = mink.solve_ik(self.configuration, tasks, dt, 'daqp', self.damping, limits=self.limits)
                    self.configuration.integrate_inplace(vel, dt)
                    new = self._error(tasks)
                    if err - new <= 1e-3:
                        break
                    err = new
        return self.configuration.q.copy()

    def _error(self, tasks) -> float:
        return float(np.linalg.norm(np.concatenate([t.compute_error(self.configuration) for t in tasks])))

    def skeleton(self, body: np.ndarray) -> np.ndarray:
        """The human's 24 joints at robot scale, on the robot's floor (for side-by-side playback)."""
        f = body_to_frame(body)
        return np.array([f[n][0] for n in JOINT_NAMES]) * self.height_scale - np.array([0.0, 0.0, self.ground_z])


# ---------------------------------------------------------------- tapes and takes

def load_tape(path):
    d = np.load(path, allow_pickle=False)
    t = (d['stamp_ns'] - d['stamp_ns'][0]) * 1e-9
    ok = d['body_ok'] if 'body_ok' in d.files else np.ones(len(t), bool)
    return t, d['body'], ok


def smooth(q: np.ndarray, sigma_s: float = SMOOTH_SIGMA_S, fps: int = FPS) -> np.ndarray:
    sig = sigma_s * fps
    k = np.arange(-int(3 * sig), int(3 * sig) + 1)
    w = np.exp(-0.5 * (k / sig) ** 2)
    w /= w.sum()
    pad = np.pad(q, ((len(k) // 2, len(k) // 2), (0, 0)), mode='edge')
    out = np.stack([np.convolve(pad[:, j], w, mode='valid') for j in range(q.shape[1])], axis=1)
    if out.shape[1] >= 7:
        out[:, 3:7] /= np.linalg.norm(out[:, 3:7], axis=1, keepdims=True)
    return out


STANCE_Z = 0.015      # toe target this close to its flat height ...
STANCE_V = 0.30       # ... and slower than this (m/s) -> the foot is planted
STANCE_MIN_S = 0.10   # shorter plants are not locked
LOCK_BLEND = 3        # frames to blend a lock in and out (no pops)


def lock_feet(pre: list, fps: int = FPS) -> dict:
    """Pin each planted foot: during a plant the toe and ankle targets move to the plant's median toe position (the
    knee half as far). The human foot drifts 5-10 cm per step in the PICO data (no foot trackers' contact), which
    played as the robot's stance foot sliding 9 cm/s (2026-10-31, seg30). Edits the targets in place."""
    stats = {}
    for side in ('Left', 'Right'):
        toe = np.array([p[0][f'{side}_Foot'][0] for p in pre])
        v = np.r_[0.0, np.linalg.norm(np.diff(toe[:, :2], axis=0), axis=1) * fps]
        v = smooth(v[:, None], 0.03, fps)[:, 0]
        st = (toe[:, 2] < FOOT_TARGET_Z + STANCE_Z) & (v < STANCE_V)
        n, i, plants = len(pre), 0, 0
        while i < n:
            if not st[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and st[j + 1]:
                j += 1
            if j - i + 1 >= STANCE_MIN_S * fps:
                plants += 1
                anchor = np.median(toe[i:j + 1, :2], axis=0)
                for k in range(i, j + 1):
                    w = min(1.0, (k - i + 1) / LOCK_BLEND, (j - k + 1) / LOCK_BLEND)
                    delta = (anchor - toe[k, :2]) * w
                    for key, f in ((f'{side}_Foot', 1.0), (f'{side}_Ankle', 1.0), (f'{side}_Knee', 0.5)):
                        pre[k][0][key][0][:2] += delta * f
            i = j + 1
        stats[side] = (plants, float(st.mean()))
    return stats


def ground(model: mujoco.MjModel, q: np.ndarray, sigma_s: float = 0.06) -> np.ndarray:
    """Lift or lower the root each frame so the lowest sole corner touches the floor (smoothed): walking has no
    flight phase, and the IK left the soles up to 2.6 cm in the floor at heel and toe."""
    data = mujoco.MjData(model)
    boxes = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
             and model.body(model.geom_bodyid[g]).name in ('L_foot', 'R_foot')]
    low = np.zeros(len(q))
    for i in range(len(q)):
        data.qpos[:] = q[i]
        mujoco.mj_kinematics(model, data)
        low[i] = min((data.geom_xpos[g] + data.geom_xmat[g].reshape(3, 3) @ (np.array([sx, sy, -1]) * model.geom_size[g]))[2]
                     for g in boxes for sx in (-1, 1) for sy in (-1, 1))
    q = q.copy()
    q[:, 2] -= smooth(low[:, None], sigma_s)[:, 0]
    return q


PIN_V = 0.25          # a foot this slow (m/s) ...
PIN_Z = 0.012         # ... and this close to its lowest height is planted
PIN_TAU_S = 4.0       # the root's correction leaks back to the captured path over this long


def pin_feet(model: mujoco.MjModel, q: np.ndarray, fps: int = FPS) -> tuple[np.ndarray, float]:
    """Move the root horizontally so a planted foot stays exactly still (the slower foot when both are down); the
    correction leaks back toward the captured path. The legs are untouched. Video captures slid the stance foot
    ~5 cm/s even after lock_feet, the IK following its targets loosely (2026-11-01). Returns (q, planted fraction)."""
    data = mujoco.MjData(model)
    feet = {s: [] for s in 'LR'}
    for i in range(len(q)):
        data.qpos[:] = q[i]
        mujoco.mj_kinematics(model, data)
        for s in 'LR':
            feet[s].append(data.xpos[model.body(f'{s}_foot').id].copy())
    F = {s: np.array(v) for s, v in feet.items()}
    V = {s: np.r_[np.zeros((1, 2)), np.diff(F[s][:, :2], axis=0)] for s in 'LR'}
    low = {s: F[s][:, 2] < np.percentile(F[s][:, 2], 5) + PIN_Z for s in 'LR'}
    corr, out, pinned = np.zeros(2), q.copy(), 0
    for i in range(1, len(q)):
        moving = [s for s in 'LR' if low[s][i] and np.linalg.norm(V[s][i]) * fps < PIN_V]
        if moving:
            s = min(moving, key=lambda k: np.linalg.norm(V[k][i]))
            corr -= V[s][i]                               # cancel that foot's slide
            pinned += 1
        corr -= corr / (PIN_TAU_S * fps)
        out[i, :2] += corr
    return out, pinned / max(len(q) - 1, 1)


def retarget(tape, start=0.0, dur=None, log=print, lock=True, hold=(), pin=False):
    t, body, ok = load_tape(tape)
    rt = EchoRetargeter()
    cal = next((i for i in range(len(t)) if ok[i] and standing_problem(body_to_frame(body[i])) is None
                and is_rest_pose(body_to_frame(body[i]))), None)
    if cal is None:
        raise SystemExit('no frame standing straight with the arms hanging to calibrate from')
    rt.calibrate(body[cal])
    q_cal = rt.configuration.q.copy()
    log(f'calibrated on t = {t[cal]:.2f} s: height ratio {rt.height_scale:.3f}')
    t1 = t[-1] if dur is None else min(t[-1], start + dur)
    tu = np.arange(start, t1, 1.0 / FPS)
    idx = [int(np.clip(np.searchsorted(t, tt, side='right') - 1, 0, len(t) - 1)) for tt in tu]
    pre = [rt.prepare(body[i]) for i in idx]                      # pass 1: every frame's targets
    skel = np.array([rt.skeleton(body[i]) for i in idx])
    if lock:
        for side, (plants, frac) in lock_feet(pre).items():
            log(f'  {side} foot: {plants} plants locked ({frac * 100:.0f} % of frames)')
    rt.configuration.update(q_cal)                                 # pass 2: solve, warm-started frame to frame
    q, t0 = [], time.time()
    for k, p in enumerate(pre):
        q.append(rt.solve(*p))
        if k % 500 == 0:
            log(f'  {tu[k]:6.1f} s  ({(time.time() - t0) / max(k, 1) * 1000:.0f} ms/frame)')
    names = [rt.model.joint(j).name for j in range(1, rt.model.njnt)]
    q = np.array(q)
    for n in hold:            # joints the capture can't see (a video's head and wrists): held straight
        q[:, 7 + names.index(n)] = 0.0
    q = ground(rt.model, smooth(q))
    if pin:
        q, frac = pin_feet(rt.model, q)
        log(f'  feet pinned in {frac * 100:.0f} % of frames')
    return dict(t=tu - tu[0], q=q[:, 7:], joint_names=np.array(names), root_pos=q[:, :3], root_quat_wxyz=q[:, 3:7],
                skel=skel, height_ratio=rt.height_scale, tape=str(tape), start=start)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('tape', type=Path)
    ap.add_argument('--start', type=float, default=0.0)
    ap.add_argument('--dur', type=float, default=None)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--pin', action='store_true', help='move the root so planted feet stay still (video captures)')
    ap.add_argument('--hold', default='', help="joints to hold at zero: 'head', 'wrists', or joint names, comma-separated "
                                               "(a video can't show a robot's visor direction or wrist twist)")
    a = ap.parse_args()
    groups = {'head': ['neck_yaw', 'neck_pitch'],
              'wrists': [f'{s}_wrist_{j}' for s in 'LR' for j in ('yaw', 'pitch', 'roll')]}
    hold = [n for h in a.hold.split(',') if h for n in groups.get(h, [h])]
    take = retarget(a.tape, a.start, a.dur, hold=hold, pin=a.pin)
    out = a.out or ROOT / 'reports' / 'motion' / f'{a.tape.stem}_echo.npz'
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **take)
    print(f"{out}: {take['t'][-1]:.1f} s, {len(take['t'])} frames at {FPS} Hz")


if __name__ == '__main__':
    main()
