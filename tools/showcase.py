"""echo's showcase film for a phone: 1080 x 1920 in the living room. Every motion is a CPU MuJoCo replay of the v6 policy
(train/replay_forces.py npz); the exploded view is posed, not simulated. Rendered with tools/render_quality.py; sections
render separately (in parallel) and ffmpeg joins them.

  python tools/showcase.py --section tour_a --out a.mp4      (sections: SECTIONS)
  python tools/showcase.py --check                          (clearance from the furniture of every placed clip)

Room frame as tools/scene_living_room.py. The room is background only: echo never collides with it, so every clip is
placed on open floor and checked against the furniture footprints below.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_quality as RQ  # noqa: E402
from render_motion import smooth_track  # noqa: E402

Q = 'C:/Users/PC/AppData/Local/Temp/claude/714944801f6bb85e/092d1abd-f8dd-4f81-a7d0-79bb9f2082fb/scratchpad/qual'
FPS = 25
W, H, PIC, STRIP = 1080, 1920, 1560, 360
FOVY = 45.0
CREDIT = 'Motion Data by Bones Studio (bones.studio)'
TEAL = np.array(RQ.GLOWS['teal'])
GLOW = 0.14

# furniture footprints (x0, x1, y0, y1) and circles (x, y, r), from scene_living_room.py, each 0.1 m larger
BOXES = [(5.06, 7.77, -0.34, 0.37), (4.30, 9.0, 1.27, 2.6), (3.5, 4.8, 1.27, 2.6), (7.0, 11.6, -3.5, -1.95)]
CIRCLES = [(4.05, -1.6, 0.85), (9.0, -1.55, 0.7), (10.84, -1.91, 0.45)]
ROOM = (-1.35, 11.2, -2.25, 2.35)

# action clips: npz, title, start, duration, placement (x, y, heading, by path or by body yaw)
ACTION = {
    'walk': ('c1', 'Relaxed walk', 2.0, 3.4, (4.0, 0.3, 180.0, 'path')),
    'crouchwalk': ('c2', 'Crouch walk', 1.0, 4.0, (1.6, 0.0, 200.0, 'yaw')),
    'lowcrouch': ('c3', 'Low crouch', 1.5, 4.0, (1.8, 0.1, 200.0, 'yaw')),
    'dance': ('c4', 'Hip-hop dance', 3.2, 4.8, (1.6, 0.0, 200.0, 'yaw')),
    'shovecrouch': ('c6', 'Shoved mid crouch', 1.2, 5.6, (1.7, 1.25, 200.0, 'yaw')),
    'jogshove': ('c7', 'Shoved at a 2 m/s jog', 5.0, 3.6, (7.75, -1.5, 170.0, 'path')),
}
TOUR_CLIP, TOUR_T0 = '../gest/left_wave_002_p80', 0.3   # a wave: the hand above the head at tour time 3.5-6.1 s
KNEEL_CLIP, KNEEL_T0, T_BLEND = 'c3', 1.5, 8.0   # the low crouch (kneel) takes over at T_BLEND, as the camera nears the knees
# the tour stands where the walk starts, so the film runs on from the exploded view into the walk
T_DOWN, T_UP, T_DOLLY, T_EX, T_HOLD, T_IN, T_COV, T_END = 12.0, 5.0, 2.5, 1.6, 1.0, 1.6, 3.0, 0.4
K_UP = T_DOWN + T_UP
K_DOLLY = K_UP + T_DOLLY
K_EX = K_DOLLY + T_EX
K_HOLD = K_EX + T_HOLD
K_IN = K_HOLD + T_IN
K_COV = K_IN + T_COV
TOUR_LEN = K_COV + T_END
SIM_LEN = K_UP + 0.8


def load(name):
    return np.load(f'{Q}/{name}.npz', allow_pickle=True)


def placed(d, t0, dur, x, y, heading, by='yaw'):
    """root_pos and root_quat moved so that at t0 the robot is at (x, y) facing heading (degrees): by its body's yaw, or by
    the direction its path takes over the cut. Also the rotation and shift, for contact points."""
    t = d['t']
    src = round(1 / (t[1] - t[0]))
    root, qw = d['root_pos'].copy(), d['root_quat_wxyz'].copy()
    k0 = int(t0 * src)
    k1 = min(len(t) - 1, k0 + int(dur * src))
    if by == 'path':
        yaw0 = math.atan2(root[k1, 1] - root[k0, 1], root[k1, 0] - root[k0, 0])
    else:
        yaw0 = yaw_of(qw[k0])
    dy = math.radians(heading) - yaw0
    c, s = math.cos(dy), math.sin(dy)
    xy = root[:, :2] - root[k0, :2]
    root[:, 0], root[:, 1] = x + c * xy[:, 0] - s * xy[:, 1], y + s * xy[:, 0] + c * xy[:, 1]
    qz = np.array([math.cos(dy / 2), 0, 0, math.sin(dy / 2)])
    w0, v0 = qz[0], qz[1:]
    w1, v1 = qw[:, 0], qw[:, 1:]
    qw = np.column_stack((w0 * w1 - v1 @ v0, w0 * v1 + w1[:, None] * v0 + np.cross(v0, v1)))
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    shift = np.array([x, y, 0]) - rot @ np.r_[d['root_pos'][k0, :2], 0]
    return root, qw, rot, shift, src


def yaw_of(q):
    return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


def clearance(name):
    if name == 'tour':
        npz, t0, dur = TOUR_CLIP, TOUR_T0, SIM_LEN
        x, y, hd, by = (*ACTION['walk'][4][:2], walk_yaw_deg(), 'yaw')
    else:
        npz, _, t0, dur, (x, y, hd, by) = ACTION[name]
    d = load(npz)
    root, qw, _, _, src = placed(d, t0, dur, x, y, hd, by)
    m = mujoco.MjModel.from_xml_path(str(RQ.XML))
    dd = mujoco.MjData(m)
    adr = [m.jnt_qposadr[m.joint(str(n)).id] for n in d['joint_names']]
    worst, where = 9.0, None
    for k in range(int(t0 * src), min(len(root), int((t0 + dur) * src))):
        dd.qpos[:3], dd.qpos[3:7], dd.qpos[adr] = root[k], qw[k], d['q'][k]
        mujoco.mj_kinematics(m, dd)
        for p in dd.xpos[1:]:
            px, py = p[0], p[1]
            cands = []
            for x0, x1, y0, y1 in BOXES:
                dx, dy = max(x0 - px, 0, px - x1), max(y0 - py, 0, py - y1)
                cands.append(math.hypot(dx, dy) if dx or dy else -min(px - x0, x1 - px, py - y0, y1 - py))
            cands += [math.hypot(px - cx, py - cy) - r for cx, cy, r in CIRCLES]
            cands += [px - ROOM[0], ROOM[1] - px, py - ROOM[2], ROOM[3] - py]
            if min(cands) < worst:
                worst, where = min(cands), (round(px, 2), round(py, 2), k / src)
    path = root[int(t0 * src):int((t0 + dur) * src), :2]
    print(f'{name:12s} clearance {worst:5.2f} m at {where}   path {path[0].round(2)} -> {path[-1].round(2)}')


def walk_yaw_deg():
    npz, _, t0, dur, (x, y, hd, by) = ACTION['walk']
    d = load(npz)
    _, qw, _, _, src = placed(d, t0, dur, x, y, hd, by)
    return math.degrees(yaw_of(qw[int(t0 * src)]))


# ---------------------------------------------------------------- type

F_BIG, F_MID, F_SMALL = ImageFont.load_default(size=58), ImageFont.load_default(size=36), ImageFont.load_default(size=30)


def shadow_text(img, items):
    """White type with a soft shadow: items are (xy, text, font, fill)."""
    lay = Image.new('L', img.size, 0)
    dr = ImageDraw.Draw(lay)
    for xy, text, font, _ in items:
        dr.text((xy[0] + 1, xy[1] + 2), text, fill=255, font=font)
    lay = lay.filter(ImageFilter.GaussianBlur(3))
    img.paste(Image.new('RGB', img.size, (0, 0, 0)), (0, 0), lay.point(lambda v: int(v * 0.62)))
    dr = ImageDraw.Draw(img)
    for xy, text, font, fill in items:
        dr.text(xy, text, fill=fill, font=font)
    return img


def shade(img, top=320, top_k=0.34, bot=260, bot_k=0.30):
    """Soft darkening at the top and foot of the picture, so white captions read over a white ceiling."""
    a = np.asarray(img, float)
    n = a.shape[0]
    r = np.arange(n)
    g = np.clip(1 - r / top, 0, 1) ** 1.6 * top_k + np.clip(1 - (n - 1 - r) / bot, 0, 1) ** 1.6 * bot_k
    return Image.fromarray(np.clip(a * (1 - g[:, None, None]), 0, 255).astype(np.uint8))


def panel(lines):
    """The pale panel under the picture, as the force chart's, for the parts of the film without a chart."""
    im = Image.new('RGB', (W, STRIP), (247, 247, 246))
    dr = ImageDraw.Draw(im)
    y = 40
    for text, font, fill in lines:
        dr.text((40, y), text, fill=fill, font=font)
        y += font.size + 22
    return im


def compose(pic, strip):
    frame = Image.new('RGB', (W, H))
    frame.paste(pic, (0, 0))
    frame.paste(strip, (0, PIC))
    return frame


def set_pose(st, root, qw, q, adr, k):
    st.data.qpos[:3], st.data.qpos[3:7] = root[k], qw[k]
    st.data.qpos[adr] = q[k]
    mujoco.mj_kinematics(st.model, st.data)


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


# ---------------------------------------------------------------- action framing (render_forces.py's, in the room)

def action_setup(name, st):
    npz, title, t0, dur, (x, y, hd, by) = ACTION[name]
    d = load(npz)
    root, qw, rot, shift, src = placed(d, t0, dur, x, y, hd, by)
    q, t = d['q'], d['t']
    m = st.model
    adr = [m.jnt_qposadr[m.joint(str(n)).id] for n in d['joint_names']]
    yaw = np.unwrap(np.array([yaw_of(v) for v in qw]))
    heading = smooth_track(yaw[:, None], 1.0 * src)[:, 0]
    tan_v = math.tan(math.radians(FOVY / 2))
    centre, need = np.zeros((len(t), 3)), np.zeros(len(t))
    CAP, FOOT = 0.17, 0.12   # the captions at the top, the force readout and credit at the foot
    for k in range(len(t)):
        set_pose(st, root, qw, q, adr, k)
        xb = st.data.xpos[1:]
        ang = heading[k] + math.radians(135)
        side = xb[:, 0] * -math.sin(ang) + xb[:, 1] * math.cos(ang)
        centre[k] = (xb.min(0) + xb.max(0)) / 2
        top = xb[:, 2].max() + 0.3   # body origins: the crown is about 30 cm above the head's
        low = max(0.0, min(xb[:, 2].min(), root[k, 2] - 0.1) - 0.1)
        centre[k, 2] = (low + top) / 2 + 0.05
        need[k] = max((top - low + 0.3) / (2 * tan_v) / (1 - CAP - FOOT), (side.max() - side.min() + 0.45) / (2 * tan_v * W / H))
    win = int(0.5 * src)
    need = np.array([need[max(0, k - win):k + win + 1].max() for k in range(len(t))])
    need = smooth_track(np.maximum(need, 2.75)[:, None], 0.4 * src)[:, 0]
    centre = smooth_track(centre, 0.25 * src)
    centre[:, 2] += (CAP - FOOT) * need * tan_v

    # The full distance is kept: where the front-right spot falls outside the room (or behind the sofa), the camera swings
    # round the robot to the nearest azimuth that fits, smoothed, rather than closing in.
    e = math.radians(-6.0)
    offs = np.array([0, 10, -10, 20, -20, 30, -30, 40, -40, 50, -50, 60, -60, 75, -75, 90, -90, 110, -110, 130, -130])

    def eye_for(k, off):
        ang = heading[k] + math.radians(135 + off)
        fwd = np.array([math.cos(e) * math.cos(ang), math.cos(e) * math.sin(ang), math.sin(e)])
        return centre[k] - need[k] * fwd

    def fits(p):
        return ROOM[0] + 0.15 < p[0] < ROOM[1] - 0.25 and ROOM[2] + 0.2 < p[1] < (2.05 if p[0] < 3.3 else 1.2)
    off = np.zeros(len(t))
    prev = None
    for k in range(len(t)):     # of the azimuths that fit, the one nearest the last: no jumping from side to side
        ok = [o for o in offs if fits(eye_for(k, o))]
        if not ok:
            off[k] = prev if prev is not None else 0.0
            continue
        prev = min(ok, key=lambda o: (abs(o) if prev is None else abs(o - prev) + 0.05 * abs(o)))
        off[k] = prev
    off = smooth_track(off[:, None], 0.5 * src)[:, 0]

    def camera(i):
        eye = eye_for(i, off[i])
        eye[0] = np.clip(eye[0], ROOM[0] + 0.1, ROOM[1] - 0.2)    # a last guard: the camera stays inside the room
        eye[1] = np.clip(eye[1], ROOM[2] + 0.15, 2.1 if eye[0] < 3.3 else 1.3)
        return eye, centre[i].copy()
    k0, k1 = int(t0 * src), min(len(t), int((t0 + dur) * src))
    return dict(d=d, root=root, qw=qw, q=q, t=t, rot=rot, shift=shift, src=src, adr=adr, title=title,
                frames=np.arange(k0, k1, round(src / FPS)), camera=camera, k0=k0)


# ---------------------------------------------------------------- the tour

# descent keyframes: time, azimuth from echo's front (deg, + to its left, the room's open side here), distance, eye height, target height (-1: the head), weight on the head
DOWN = CubicSpline([0.0, 2.6, 5.0, 7.0, 9.7, 12.0],
                   [[12, 0.62, 1.62, -1, 1.0], [24, 0.80, 1.62, -1, 1.0], [38, 2.3, 1.50, 1.25, 0.0], [50, 3.1, 1.25, 0.92, 0.0],
                    [66, 1.85, 0.78, 0.42, 0.0], [82, 1.40, 0.36, 0.16, 0.0]], bc_type='clamped')


def path_camera(s, head, xy, h0):
    az, dist, ez, tz, hw = DOWN(s)
    tgt = np.array([xy[0], xy[1], max(tz, 0.0)])
    hp = head + [0, 0, 0.07]
    hw = float(np.clip(hw, 0, 1))
    if tz < 0:
        tgt[2] = hp[2]
    tgt = tgt * (1 - hw) + hp * hw
    a = math.radians(h0 + az)
    eye = tgt + np.array([math.cos(a) * dist, math.sin(a) * dist, 0.0])
    eye[2] = ez + (hp[2] - 1.62) * hw     # close to the face, the camera stays at the head's height when echo kneels
    eye[0] = np.clip(eye[0], ROOM[0] + 0.15, ROOM[1] - 0.2)   # inside the room: the sofa wall is on echo's right here
    eye[1] = np.clip(eye[1], ROOM[2] + 0.15, 2.0)
    return eye, tgt


class Pieces:
    """Every rigid part's flight for the exploded view: out along its own radial line from the robot's middle, a little turn."""

    def __init__(self, st, centre, radius=0.9, seed=3, r_min=0.15):
        rng = np.random.default_rng(seed)
        self.st = st
        mats = st.body_mats()
        self.items = []
        for ln, o, n, c, cov, size, role in st.pieces:
            M = mats[ln]
            cw = M[:3, :3] @ c + M[:3, 3]
            v = cw - centre
            v[2] *= 0.6
            r = np.linalg.norm(v)
            u = v / r if r > 1e-6 else rng.normal(size=3)
            u = u + rng.normal(scale=0.25, size=3)
            u /= np.linalg.norm(u)
            ax = rng.normal(size=3)
            ax /= np.linalg.norm(ax)
            # spread evenly through a ball: r = R u^(1/3), from r_min out
            rr = r_min + (radius - r_min) * rng.random() ** (1 / 3)
            self.items.append(dict(ln=ln, o=o, n=n, c=c, cov=cov, cw=cw, role=role, hero=0, goal=centre + u * rr * np.array([1, 1, 1.25]), axis=ax,
                                   ang=rng.uniform(0.4, 1.2) * rng.choice((-1, 1)), r=r, size=size))
        # big parts must not land in each other: push apart the ones that do
        big = [i for i in self.items if i['size'] > 0.05]
        for _ in range(40):
            moved = False
            for a_ in range(len(big)):
                for b_ in range(a_ + 1, len(big)):
                    A, B = big[a_], big[b_]
                    dv = B['goal'] - A['goal']
                    dd = np.linalg.norm(dv) + 1e-9
                    need_ = 0.8 * (A['size'] + B['size'])
                    if dd < need_:
                        push = dv / dd * (need_ - dd) / 2
                        A['goal'] -= push
                        B['goal'] += push
                        moved = True
            if not moved:
                break
        for i in self.items:
            i['dir'] = i['goal'] - i['cw']
        # two big motors (a hip and a knee) come towards the lens, in the frame's outer third, one each side
        motors = sorted([i for i in self.items if i['role'] == 'actuator' and ('hip' in i['ln'] or 'knee' in i['ln'])], key=lambda i: -i['size'])
        picked = []
        for i in motors:
            if all(i['ln'][0] != j['ln'][0] or ('knee' in i['ln']) != ('knee' in j['ln']) for j in picked):
                picked.append(i)
            if len(picked) == 2:
                break
        for k, i in enumerate(picked):
            i['hero'] = 1 if k == 0 else -1
        rmax = max(i['r'] for i in self.items)
        for i in self.items:
            i['rn'] = i['r'] / rmax
        covers = [i for i in self.items if i['cov']]
        order = rng.permutation(len(covers))     # covers land in a staggered order, low and inner pieces first
        keyv = np.array([0.6 * (1 - i['rn']) + 0.4 * (1 - (st.body_mats()[i['ln']][:3, :3] @ i['c'])[2] / 1.8) for i in covers]) + 0.15 * rng.random(len(covers))
        for rank, j in enumerate(np.argsort(-keyv)):
            covers[j]['rank'] = rank / max(1, len(covers) - 1)
        del order

    def draws(self, amount, covers_only=False, frame_only=False, eye=None, tgt=None):
        """amount(item) -> 0 assembled .. 1 fully out. With the camera given, the two hero motors fly to just off the
        frame's side, about 0.75 m from the lens, turning slowly."""
        mats = self.st.body_mats()
        if eye is not None:
            fwd = (tgt - eye) / np.linalg.norm(tgt - eye)
            right = np.cross(fwd, [0, 0, 1.0])
            right /= np.linalg.norm(right)
            up = np.cross(right, fwd)
            tv = math.tan(math.radians(FOVY / 2))
        out = []
        for i in self.items:
            if covers_only and not i['cov'] or frame_only and i['cov']:
                continue
            a = amount(i)
            M = mats[i['ln']]
            if a <= 0:
                out.append((M, i['o'], i['n']))
                continue
            cw = M[:3, :3] @ i['c'] + M[:3, 3]
            th = i['ang'] * a
            if i['hero'] and eye is not None:
                dz = 0.78
                half_w = dz * tv * W / H
                goal = eye + fwd * dz + right * i['hero'] * 0.70 * half_w + up * (0.30 if i['hero'] > 0 else -0.42) * dz * tv
                disp = (goal - cw) * a
                th = 1.6 * a * i['hero']
                k = up
                K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
                Rr = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K
                T = np.eye(4)
                T[:3, :3] = Rr
                T[:3, 3] = cw + disp - Rr @ cw
                out.append((T @ M, i['o'], i['n']))
                continue
            k = i['axis']
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            Rr = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K
            T = np.eye(4)
            T[:3, :3] = Rr
            T[:3, 3] = cw + i['dir'] * a - Rr @ cw
            out.append((T @ M, i['o'], i['n']))
        return out


def tour_motion(wx, wy, h0):
    """The tour's poses at 50 Hz from tour time 0: the idle, then a 0.8 s blend into the kneel, which holds its last frame
    once its recording ends. Both are physics replays; only the blend between them is not."""
    d = load(TOUR_CLIP)
    root, qw, _, _, src = placed(d, TOUR_T0, SIM_LEN, wx, wy, h0, 'yaw')
    kb = int((TOUR_T0 + T_BLEND) * src)
    c = load(KNEEL_CLIP)
    croot, cqw, _, _, _ = placed(c, KNEEL_T0, 3.0, root[kb, 0], root[kb, 1], math.degrees(yaw_of(qw[kb])), 'yaw')
    n = int((SIM_LEN + 0.5) * src)
    R_, Q_, J_ = np.zeros((n, 3)), np.zeros((n, 4)), np.zeros((n, d['q'].shape[1]))
    for i in range(n):
        tau = i / src
        ki = min(int((TOUR_T0 + tau) * src), len(root) - 1)
        kc = min(int((KNEEL_T0 + tau - T_BLEND) * src), len(croot) - 1)
        kc = max(kc, int(KNEEL_T0 * src))
        u = ease((tau - (T_BLEND - 0.4)) / 0.8)
        R_[i] = root[ki] * (1 - u) + croot[kc] * u
        qa, qb = qw[ki], cqw[kc]
        if qa @ qb < 0:
            qb = -qb
        Q_[i] = (qa * (1 - u) + qb * u) / np.linalg.norm(qa * (1 - u) + qb * u)
        J_[i] = d['q'][ki] * (1 - u) + c['q'][kc] * u
    return R_, Q_, J_, src, d['joint_names']


def tour_lines(sim):
    """The small lines at the foot of the full-frame tour."""
    if sim:
        rows = [('Physics sim: every step simulated', (240, 242, 245)), ('Motion: BONES-SEED', (240, 242, 245)),
                ('Policy: SuperMimic, trained by us in MuJoCo Warp', (240, 242, 245)), (CREDIT, (255, 225, 150))]
    else:
        rows = [('Display sequence: posed, not simulated', (240, 242, 245)), (CREDIT, (255, 225, 150))]
    y0 = H - 60 - 38 * len(rows)
    return [((46, y0 + 38 * k), text, F_SMALL, fill) for k, (text, fill) in enumerate(rows)]


def render_tour(a, t0, t1):
    wx, wy = ACTION['walk'][4][:2]
    h0 = walk_yaw_deg()
    root, qw, q, src, jn = tour_motion(wx, wy, h0)
    need_cov = t0 < K_UP + 0.3 or t1 > K_IN
    cov = RQ.Studio(W, H, SS=2, room='living', envpos=(wx, wy, 1.4), glow='teal', glow_gain=0.0) if need_cov and t0 < K_UP + 0.3 else None
    ske = RQ.Studio(W, H, SS=2, skeleton=True, room='living', envpos=(wx, wy, 1.4)) if t1 > T_DOWN - 0.5 and t0 < K_IN else None
    both = RQ.Studio(W, H, SS=2, skeleton='both', room='living', envpos=(wx, wy, 1.4), glow='teal', glow_gain=GLOW) if t1 > K_IN else None
    any_st = cov or ske or both
    adr = [any_st.model.jnt_qposadr[any_st.model.joint(str(n)).id] for n in jn]
    head_id = any_st.model.body('head').id
    k_all = np.arange(0, len(q))
    heads, mids = [], []
    for k in k_all:
        set_pose(any_st, root, qw, q, adr, k)
        heads.append(any_st.data.xpos[head_id].copy())
        xb = any_st.data.xpos[1:]
        mids.append((xb.min(0) + xb.max(0)) / 2)        # the body's middle: kneeling, the knees go well ahead of the pelvis
    heads = smooth_track(np.array(heads), 0.35 * src)
    mids = smooth_track(np.array(mids), 0.5 * src)
    # the relaxed stand at the walk's start, and the walk's first camera: where the display sequence ends
    probe = RQ.Studio(W, PIC, SS=1, marks=False)
    walk = action_setup('walk', probe)
    eye_w, tgt_w = walk['camera'](walk['k0'])
    probe.close()
    stand_q = None
    w = writer(a.out) if not a.times else None
    pieces_ske = pieces_both = None
    todo = [int(round(x * FPS)) for x in a.times] if a.times else range(int(round(t0 * FPS)), int(round(t1 * FPS)))
    for f in todo:
        t = f / FPS
        k = min(int(t * src), k_all[-1])
        head = heads[k - k_all[0]]
        sts = [s for s in (cov, ske, both) if s is not None]
        # pose: the sim until the ascent ends, then a short blend into the relaxed stand at the walk's start
        for s in sts:
            set_pose(s, root, qw, q, adr, k)
        if t >= K_UP:
            for s in sts:
                if stand_q is None or stand_q[0] is not s:
                    pass
            bl = ease((t - K_UP) / 1.4)
            for s in sts:
                qs = s.data.qpos.copy()
                s.pose_zero(RQ.BEND)
                s.place(wx, wy, math.radians(h0))
                qz = s.data.qpos.copy()
                s.data.qpos[:] = qs + (qz - qs) * bl
                s.data.qpos[3:7] /= np.linalg.norm(s.data.qpos[3:7])
                mujoco.mj_kinematics(s.model, s.data)
        # camera
        if t <= T_DOWN:
            eye, tgt = path_camera(t, head, mids[k, :2], h0)
        elif t <= K_UP:
            eye, tgt = path_camera(T_DOWN * (1 - ease((t - T_DOWN) / T_UP)), head, mids[k, :2], h0)
        else:
            e0, g0 = path_camera(0.0, heads[min(len(heads) - 1, int(K_UP * src))], mids[min(len(heads) - 1, int(K_UP * src)), :2], h0)
            u = ease((t - K_UP) / T_DOLLY)
            eye, tgt = e0 + (eye_w - e0) * u, g0 + (tgt_w - g0) * u
            wide = ease((t - K_DOLLY + 0.3) / (T_EX + 0.3)) - ease((t - K_HOLD) / (T_IN + 0.6))
            eye = tgt + (eye - tgt) * (1 + 0.15 * wide)
        fd = float(np.linalg.norm(tgt - eye))
        gain = GLOW * ease((t - 1.0) / 0.9)
        for s in (cov, ske):
            if s is not None:
                s.glow = TEAL * gain
        if t < T_DOWN - 0.45 or K_UP + 0.25 <= t < K_UP and False:
            img = cov.render(eye, tgt, FOVY, t=t, focus=fd)
        elif t < T_DOWN + 0.45:
            ci = np.asarray(cov.render(eye, tgt, FOVY, t=t, focus=fd), float)
            si = np.asarray(ske.render(eye, tgt, FOVY, t=t, focus=fd), float)
            wipe = (t - (T_DOWN - 0.45)) / 0.9                     # the covers vanish from the feet up
            yy = np.arange(H)[:, None, None] / H
            m = np.clip((yy - (1.15 - 1.3 * wipe)) / 0.12, 0, 1)
            img = Image.fromarray(np.clip(ci * (1 - m) + si * m, 0, 255).astype(np.uint8))
        elif t < K_DOLLY:
            img = ske.render(eye, tgt, FOVY, t=t, focus=fd)
        elif t < K_IN:
            if pieces_ske is None:
                pieces_ske = Pieces(ske, ske.data.xpos[ske.model.body('torso').id] * [1, 1, 0] + [0, 0, 0.95])
            if t < K_EX:
                g = (t - K_DOLLY) / T_EX

                def amount(i, g=g):
                    return ease((g - 0.25 * i['rn']) / 0.75)
            elif t < K_HOLD:
                def amount(i):
                    return 1.0
            else:
                g = (t - K_HOLD) / T_IN

                def amount(i, g=g):
                    return 1 - ease((g - 0.25 * (1 - i['rn'])) / 0.75)
            img = ske.render(eye, tgt, FOVY, t=t, focus=fd, draws=pieces_ske.draws(amount, eye=eye, tgt=tgt))
        else:
            if pieces_both is None:
                pieces_both = Pieces(both, both.data.xpos[both.model.body('torso').id] * [1, 1, 0] + [0, 0, 0.95], seed=11, radius=1.05, r_min=0.55)
            g = (t - K_IN) / T_COV

            def amount(i, g=g):
                if not i['cov']:
                    return 0.0
                return 1 - ease((g - 0.6 * i['rank']) / 0.4)     # each panel flies in over 40 % of the time, staggered
            img = both.render(eye, tgt, FOVY, t=t, focus=fd, draws=pieces_both.draws(amount))
        img = shade(img)
        if t < 3.0:
            title, sub = 'echo', 'A humanoid, 1.73 m'
        elif t < T_DOWN:
            title, sub = 'Garment on', 'Knit cover, glass visor'
        elif t < K_UP:
            title, sub = 'Covers off', 'Motors and frame'
        elif t < K_IN:
            title, sub = 'Exploded view', 'Every part, posed, not simulated'
        else:
            title, sub = 'Covers on', 'Posed, not simulated'
        img = shadow_text(img, [((48, 52), title, F_BIG, (255, 255, 255)), ((50, 124), sub, F_MID, (238, 240, 244))])
        fr = shadow_text(img, tour_lines(t < K_UP))
        if w is None:
            fr.save(f'{a.out}_{t:05.2f}.png')
        else:
            w.append_data(np.asarray(fr))
    if w is not None:
        w.close()
    for s in (cov, ske, both):
        if s is not None:
            s.close()


# ---------------------------------------------------------------- action

FOOT_N_PER_M, FOOT_ARROW_MAX = 1100.0, 0.75


def foot_arrows(st, d, i, rot, shift):
    """One thick arrow a foot: the floor's total force on it, drawn from its centre of pressure but set out beside the foot
    (away from the other one) so it never runs through the leg; small discs where the contacts are."""
    over = []
    for p in d['contact_pos'][i]:
        if np.isfinite(p).all():
            over += st.disc_draw(rot @ p + shift + [0, 0, 0.002], 0.013, (1.0, 0.45, 0.03))
    for s_, foot in ((0, 'L_foot'), (1, 'R_foot')):
        f = rot @ d['force'][i, s_]
        fn = np.linalg.norm(f)
        c = d['cop'][i, s_]
        if fn < 15 or not np.isfinite(c).all():
            continue
        c = rot @ c + shift
        fx = st.data.xmat[st.model.body(foot).id].reshape(3, 3)
        side = fx[:, 1] * (1 if s_ == 0 else -1)
        side[2] = 0
        side /= max(np.linalg.norm(side), 1e-6)
        base = c + side * 0.13 + [0, 0, 0.01]
        over += st.arrow_draws(base, base + f / fn * min(fn / FOOT_N_PER_M, FOOT_ARROW_MAX), 0.012, (1.0, 0.04, 0.03))
        over += st.disc_draw(c + [0, 0, 0.003], 0.02, (1.0, 0.45, 0.03))
    return over


def render_action(a, name):
    # one room image per process: a second MuJoCo context for it fails ('Could not allocate display lists')
    envpos = (*ACTION['walk'][4][:2], 1.4) if name == 'walk' else (3.0, 0.0, 1.3)
    st = RQ.Studio(W, H, SS=2, room='living', envpos=envpos, glow='teal', glow_gain=GLOW)
    S = action_setup(name, st)
    d, root, qw, q, t, rot, shift, src, adr = (S[k] for k in ('d', 'root', 'qw', 'q', 't', 'rot', 'shift', 'src', 'adr'))
    F, weight, other = d['force'], float(d['weight']), d['other_n']
    w = writer(a.out)
    still = None
    if name == 'walk':   # the exploded view's last picture, to dissolve from: the relaxed stand, covered, the same camera
        wx, wy = ACTION['walk'][4][:2]
        full = RQ.Studio(W, H, SS=2, skeleton='both', room='living', envpos=envpos, glow='teal', glow_gain=GLOW)
        full.pose_zero(RQ.BEND)
        full.place(wx, wy, math.radians(walk_yaw_deg()))
        eye, tgt = S['camera'](S['k0'])
        img = shade(full.render(eye, tgt, FOVY, t=TOUR_LEN, focus=float(np.linalg.norm(tgt - eye))))
        img = shadow_text(img, [((48, 52), 'Covers on', F_BIG, (255, 255, 255)), ((50, 124), 'Posed, not simulated', F_MID, (238, 240, 244))] + tour_lines(False))
        still = np.asarray(img, float)
        full.close()
    for n, i in enumerate(S['frames']):
        set_pose(st, root, qw, q, adr, i)
        over = []
        over += foot_arrows(st, d, i, rot, shift)
        neck_n = 0.0
        back = d['shove_force'][max(0, i - int(0.6 * src)):i + 1]
        hit = np.flatnonzero(np.abs(back).sum(1) > 0)
        if len(hit):
            f = rot @ back[hit[-1]]
            neck_n = float(np.linalg.norm(f))
            tip = rot @ d['shove_point'][i] + shift
            L = min(0.14 + neck_n / 3000.0, 0.36)   # thick, and short enough to stay in the picture
            over += st.arrow_draws(tip - f / neck_n * (0.05 + L), tip - f / neck_n * 0.05, 0.02, (0.10, 0.45, 1.0))
        eye, tgt = S['camera'](i)
        img = st.render(eye, tgt, FOVY, overlays=over, t=float(t[i]), focus=float(np.linalg.norm(tgt - eye)))
        img = shade(img, 360, 0.38, 300, 0.36)
        items = [((44, 40), S['title'], ImageFont.load_default(size=50), (255, 255, 255)),
                 ((46, 104), 'Physics sim: every step simulated', F_SMALL, (240, 242, 245)),
                 ((46, 142), 'Motion: BONES-SEED', F_SMALL, (240, 242, 245)),
                 ((46, 180), 'Policy: SuperMimic, trained by us in MuJoCo Warp', F_SMALL, (240, 242, 245)),
                 ((46, 218), 'Red arrows: floor force on each foot, from the solver', F_SMALL, (255, 196, 186)),
                 ((46, H - 148), f'Left foot {F[i, 0, 2]:4.0f} N    Right foot {F[i, 1, 2]:4.0f} N', F_MID, (255, 255, 255)),
                 ((46, H - 62), CREDIT, F_SMALL, (255, 225, 150))]
        if other[i] > 20:
            items.append(((46, H - 100), f'Body on floor {other[i]:4.0f} N', F_MID, (255, 150, 140)))
        if neck_n:
            items.append(((44, 266), f'SHOVE {neck_n:.0f} N at the neck', ImageFont.load_default(size=46), (150, 205, 255)))
        img = shadow_text(img, items)
        fr = np.asarray(img, float)
        if still is not None and n < 14:      # dissolve from the exploded view; the arrows and readout fade in
            u = ease(n / 14)
            fr = still * (1 - u) + fr * u
        elif still is None and n < 6:          # a quick dip from white into each clip
            fr = fr + (255 - fr) * (1 - n / 6) ** 1.5
        out = Image.fromarray(np.clip(fr, 0, 255).astype(np.uint8))
        if name in a.reset and n < 16:
            out = shadow_text(out, [((W // 2 - 150, H // 2 - 60), 'RESET', ImageFont.load_default(size=100), (255, 255, 255))])
        w.append_data(np.asarray(out))
    w.close()
    st.close()


def render_card(a):
    w = writer(a.out)
    big, small = ImageFont.load_default(size=150), ImageFont.load_default(size=36)
    for n in range(int(2.0 * FPS)):
        im = Image.new('RGB', (W, H), (14, 15, 17))
        dr = ImageDraw.Draw(im)
        x0, _, x1, _ = dr.textbbox((0, 0), 'echo', font=big)
        dr.text(((W - (x1 - x0)) // 2, 760), 'echo', fill=(245, 246, 248), font=big)
        line = 'Physics sim \u00b7 ' + CREDIT
        x0, _, x1, _ = dr.textbbox((0, 0), line, font=small)
        dr.text(((W - (x1 - x0)) // 2, 990), line, fill=(225, 228, 232), font=small)
        w.append_data((np.asarray(im, float) * min(1.0, (n + 1) / 8)).astype(np.uint8))
    w.close()


def writer(path):
    return imageio.get_writer(path, fps=FPS, codec='libx264', quality=None, pixelformat='yuv420p', macro_block_size=8,
                              ffmpeg_params=['-crf', '14', '-preset', 'veryfast'])


TOUR_SPLITS = {'tour_a': (0.0, 6.0), 'tour_b': (6.0, T_DOWN - 0.5), 'tour_c': (T_DOWN - 0.5, K_UP), 'tour_d': (K_UP, K_HOLD),
               'tour_e': (K_HOLD, TOUR_LEN)}
SECTIONS = list(TOUR_SPLITS) + ['walk', 'crouchwalk', 'lowcrouch', 'dance', 'shovecrouch', 'jogshove', 'card']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--section')
    ap.add_argument('--out')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--reset', default='jogshove', help='sections that open with RESET')
    ap.add_argument('--from', dest='t_from', type=float, default=None, help='tour sections: render only from this tour time (tests)')
    ap.add_argument('--to', dest='t_to', type=float, default=None)
    ap.add_argument('--times', type=lambda v: [float(x) for x in v.split(',')], default=None, help='tour: stills at these times, to <out>_<t>.png')
    a = ap.parse_args()
    a.reset = a.reset.split(',')
    if a.check:
        for n in ['tour'] + list(ACTION):
            clearance(n)
        return
    if a.section in TOUR_SPLITS:
        t0, t1 = TOUR_SPLITS[a.section]
        render_tour(a, a.t_from if a.t_from is not None else t0, a.t_to if a.t_to is not None else t1) if not a.times else render_tour(a, 0.0, TOUR_LEN)
    elif a.section == 'card':
        render_card(a)
    else:
        render_action(a, a.section)
    print(a.out)


if __name__ == '__main__':
    main()
