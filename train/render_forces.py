"""A train/replay_forces.py recording as a movie: the floor's force on each foot as MuJoCo's viewer shows contacts, echo in its real colours, and a strip chart
of the vertical force on each foot against the robot's weight.

  python train/render_forces.py <f.npz> --out x.mp4 [--start S] [--dur D] [--title T]
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'tools'))
from render_motion import TRACKS_HEADING, VIEWS, build_view_model, caption, smooth_track  # noqa: E402

COL = ((0.15, 0.8, 0.3, 1.0), (1.0, 0.45, 0.05, 1.0))
RGB = ((30, 170, 70), (235, 105, 10))
N_PER_M = 250.0  # arrow length, per contact point: a foot's load is shared among several
W, H, STRIP = 1024, 576, 128


def arrow(scene, a, b, rgba, width=0.022):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9), np.array(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, np.array(a, dtype=float), np.array(b, dtype=float))
    scene.ngeom += 1


def classic_room(ms):
    """MuJoCo's classic look around echo's real colours: blue gradient sky, dark blue checker floor."""
    ms.add_texture(name='sky', type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                   rgb1=[0.3, 0.5, 0.7], rgb2=[0, 0, 0], width=512, height=3072)
    ms.add_texture(name='grid', type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                   mark=mujoco.mjtMark.mjMARK_EDGE, rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.2, 0.3], markrgb=[0.8, 0.8, 0.8], width=300, height=300)
    floor = ms.add_material(name='floor', reflectance=0.2, texrepeat=[10, 10], texuniform=True)
    floor.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'grid'
    ms.worldbody.add_geom(name='view floor', type=mujoco.mjtGeom.mjGEOM_PLANE, size=[30, 30, 0.05], material='floor', contype=0, conaffinity=0)


def chart(t, fz, weight, i, window=6.0):
    im = Image.new('RGB', (W, STRIP), (250, 250, 252))
    dr = ImageDraw.Draw(im)
    x0, x1, y0, y1 = 44, W - 170, STRIP - 18, 8
    top = 1.8 * weight
    t1 = t[i]
    t0 = t1 - window * 0.8
    X = lambda v: x0 + (v - t0) / window * (x1 - x0)  # noqa: E731
    Y = lambda v: y0 + np.clip(v / top, 0, 1) * (y1 - y0)  # noqa: E731
    yw = Y(weight)
    dr.line([(x0, y0), (x1, y0)], fill=(120, 120, 120))
    dr.line([(x0, yw), (x1, yw)], fill=(90, 90, 90))
    dr.text((x0 - 40, yw - 6), 'weight', fill=(60, 60, 60))
    dr.text((x0 - 14, y0 - 6), '0 N', fill=(60, 60, 60))
    lo = max(0, int(np.searchsorted(t, t0)))
    hi = int(np.searchsorted(t, min(t1, t0 + window)))
    for s in (0, 1):
        pts = [(float(X(t[k])), float(Y(fz[k, s]))) for k in range(lo, max(lo + 2, hi))]
        dr.line(pts, fill=RGB[s], width=2)
    dr.line([(X(t1), y0), (X(t1), y1)], fill=(30, 30, 30))
    tx = x1 + 12
    dr.text((tx, 10), f'left foot   {fz[i, 0]:5.0f} N', fill=RGB[0])
    dr.text((tx, 28), f'right foot  {fz[i, 1]:5.0f} N', fill=RGB[1])
    dr.text((tx, 46), f'sum / weight  {fz[i].sum() / weight:4.2f}', fill=(30, 30, 30))
    state = 'both feet in the air' if fz[i].sum() < 1 else 'left only' if fz[i, 1] < 1 else 'right only' if fz[i, 0] < 1 else 'both on floor'
    dr.text((tx, 64), state, fill=(30, 30, 30))
    dr.text((tx, 92), 'measured by the mj_step', fill=(90, 90, 90))
    dr.text((tx, 106), 'contact solver', fill=(90, 90, 90))
    return im


def chart_portrait(t, fz, weight, i, w, h, window=6.0, other=None):
    """The same chart for a phone screen: large type, the readings in a row above the curves."""
    im = Image.new('RGB', (w, h), (250, 250, 252))
    dr = ImageDraw.Draw(im)
    big, small = ImageFont.load_default(size=38), ImageFont.load_default(size=28)
    dr.text((30, 18), f'Left foot {fz[i, 0]:4.0f} N', fill=RGB[0], font=big)
    dr.text((w // 2 + 10, 18), f'Right foot {fz[i, 1]:4.0f} N', fill=RGB[1], font=big)
    x0, x1, y0, y1 = 130, w - 30, h - 34, 86
    top = 1.8 * weight
    t1 = t[i]
    t0 = t1 - window * 0.8
    X = lambda v: x0 + (v - t0) / window * (x1 - x0)  # noqa: E731
    Y = lambda v: y0 + np.clip(v / top, 0, 1) * (y1 - y0)  # noqa: E731
    yw = Y(weight)
    dr.line([(x0, y0), (x1, y0)], fill=(120, 120, 120), width=2)
    dr.line([(x0, yw), (x1, yw)], fill=(90, 90, 90), width=2)
    dr.text((14, yw - 16), 'Weight', fill=(60, 60, 60), font=small)
    dr.text((66, y0 - 16), '0 N', fill=(60, 60, 60), font=small)
    lo = max(0, int(np.searchsorted(t, t0)))
    hi = int(np.searchsorted(t, min(t1, t0 + window)))
    for s in (0, 1):
        pts = [(float(X(t[k])), float(Y(fz[k, s]))) for k in range(lo, max(lo + 2, hi))]
        dr.line(pts, fill=RGB[s], width=5)
    dr.line([(X(t1), y0), (X(t1), y1)], fill=(30, 30, 30), width=2)
    if other is not None and other[i] > 20:
        dr.text((x0 + 10, 76), f'Body on floor {other[i]:4.0f} N', fill=(190, 30, 30), font=big)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rec')
    ap.add_argument('--out', required=True)
    ap.add_argument('--start', type=float, default=0.0)
    ap.add_argument('--dur', type=float, default=None)
    ap.add_argument('--title', default='')
    ap.add_argument('--view', default='hero')
    ap.add_argument('--dist', type=float, default=None, help='camera distance')
    ap.add_argument('--portrait', action='store_true', help='1080 x 1920, for a phone: a closer camera, large type, the chart below')
    a = ap.parse_args()
    d = np.load(a.rec, allow_pickle=True)
    t, q = d['t'], d['q']
    src = round(1 / (t[1] - t[0]))
    F, P, weight = d['force'], d['cop'], float(d['weight'])
    W_, H_, STRIP_ = (1080, 1560, 360) if a.portrait else (W, H, STRIP)
    m = build_view_model(room=classic_room)
    m.vis.global_.offwidth, m.vis.global_.offheight = max(W_, 1920), max(H_, 1080)
    data = mujoco.MjData(m)
    adr = [m.jnt_qposadr[m.joint(str(n)).id] for n in d['joint_names']]
    i0 = int(a.start * src)
    i1 = len(t) if a.dur is None else min(len(t), int((a.start + a.dur) * src))
    fps = 25
    frames = np.arange(i0, i1, max(1, round(src / fps)))
    track = smooth_track(d['root_pos'][:, :2], 0.6 * src)
    az, el, dist = VIEWS[a.view]
    qw = d['root_quat_wxyz']
    yaw = np.unwrap(np.arctan2(2 * (qw[:, 0] * qw[:, 3] + qw[:, 1] * qw[:, 2]), 1 - 2 * (qw[:, 2] ** 2 + qw[:, 3] ** 2)))
    heading = smooth_track(yaw[:, None], 1.0 * src)[:, 0]
    assert a.view in TRACKS_HEADING
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.elevation, cam.distance = el, a.dist or (2.75 if a.portrait else 3.3)
    # Aim at the middle of the body whatever its height: standing, crouched or on the floor.
    aim = np.clip(0.58 + 0.71 * (smooth_track(d['root_pos'][:, 2:], 0.4 * src)[:, 0] - 0.52), 0.3, 0.85)
    r = mujoco.Renderer(m, H_, W_)
    w = imageio.get_writer(a.out, fps=fps, codec='libx264', quality=7, pixelformat='yuv420p', macro_block_size=8, ffmpeg_params=['-movflags', '+faststart'])
    for i in frames:
        data.qpos[:3] = d['root_pos'][i]
        data.qpos[3:7] = qw[i]
        data.qpos[adr] = q[i]
        mujoco.mj_kinematics(m, data)
        mujoco.mj_camlight(m, data)
        cam.lookat[:] = [track[i, 0], track[i, 1], aim[i]]
        cam.azimuth = math.degrees(heading[i]) + az
        r.update_scene(data, cam)
        for p, f in zip(d['contact_pos'][i], d['contact_force'][i]):  # as MuJoCo's viewer shows them: each contact point and its force
            if np.isfinite(p).all():
                arrow(r.scene, p, p + [0, 0, 0.006], (1.0, 0.55, 0.1, 1.0), 0.03)
                arrow(r.scene, p, p + f / N_PER_M, (0.9, 0.1, 0.1, 1.0), 0.014)
        shove = None
        if 'push' in d.files:
            recent = np.flatnonzero(np.abs(d['push'][max(0, i - int(0.5 * src)):i + 1]).sum(1) > 0)
            if len(recent):
                kick = d['push'][max(0, i - int(0.5 * src)):i + 1][recent[-1]]
                shove = float(np.linalg.norm(kick))
                c = data.qpos[:3] + [0, 0, 0.1]
                arrow(r.scene, c - np.r_[kick, 0] / np.linalg.norm(kick) * 0.75, c - np.r_[kick, 0] / np.linalg.norm(kick) * 0.2, (0.1, 0.45, 1.0, 1.0), 0.04)
        neck_n = 0.0
        if 'shove_force' in d.files:
            back = d['shove_force'][max(0, i - int(0.6 * src)):i + 1]
            hit = np.flatnonzero(np.abs(back).sum(1) > 0)
            if len(hit):
                f = back[hit[-1]]
                neck_n = float(np.linalg.norm(f))
                tip = d['shove_point'][i]
                arrow(r.scene, tip - f / neck_n * (0.25 + neck_n / 700.0), tip - f / neck_n * 0.05, (0.1, 0.45, 1.0, 1.0), 0.05)
        img = Image.fromarray(r.render())
        if a.portrait:
            dr = ImageDraw.Draw(img)
            big, small = ImageFont.load_default(size=46), ImageFont.load_default(size=30)
            dr.text((40, 40), a.title, fill=(255, 255, 255), font=big)
            dr.text((40, 100), 'Physics sim: every step simulated', fill=(215, 225, 235), font=small)
            dr.text((40, 140), 'Red arrows: contact forces from the solver', fill=(215, 225, 235), font=small)
            if neck_n:
                dr.text((40, 190), f'SHOVE {neck_n:.0f} N at the neck', fill=(120, 190, 255), font=big)
            strip = chart_portrait(t, F[..., 2], weight, i, W_, STRIP_, other=d['other_n'])
        else:
            caption(img, (12, 10), a.title + (f'    SHOVE {shove:.1f} m/s' if shove else ''))
            caption(img, (12, H - 24), f't {t[i]:5.1f} s   1x   CPU MuJoCo, every step simulated; red arrows: contact forces from the solver')
            strip = chart(t, F[..., 2], weight, i)
        frame = Image.new('RGB', (W_, H_ + STRIP_))
        frame.paste(img, (0, 0))
        frame.paste(strip, (0, H_))
        w.append_data(np.asarray(frame))
    w.close()
    r.close()
    print(a.out, len(frames), 'frames')


if __name__ == '__main__':
    main()
