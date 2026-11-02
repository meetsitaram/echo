"""Kinematic playback of a retargeted take (tools/retarget_pico.py) as a movie: the replica in its own colours (the
spec's shapes, cuts applied, on the MJCF's bodies), a floor, a camera tracking the pelvis, and the captured human
skeleton (robot scale) walking alongside. Joint angles are set, nothing is simulated.

  python tools/render_motion.py reports/motion/<take>.npz [--view 34|side|front] [--start S] [--dur D] [--out x.mp4]
  python tools/render_motion.py reports/motion/<take>.npz --viewer     (MuJoCo's interactive viewer, looping)

Plain python (MuJoCo in the user site-packages).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checks as C  # noqa: E402
import geom as G  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / 'urdf' / 'echo.xml'
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
GLOSSY = ('#08090B', '#0A1420')
VIEWS = {'34': (150.0, -12.0, 3.2), 'side': (90.0, -6.0, 2.8), 'front': (180.0, -6.0, 3.0),
         'follow': (90.0, -4.0, 3.1)}   # azimuth (follow: from the heading), elevation, distance


def rgba(hex_color):
    h = hex_color.lstrip('#')
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [1.0]


def build_view_model(spec=None, room=None):
    """The MJCF's bodies and joints with every spec part as a coloured mesh geom (the MJCF's single grey mesh per
    link hidden in group 3), a checker floor and lights; or, with room(ms), that room instead of the studio."""
    return view_spec(spec, room).compile()


def view_spec(spec=None, room=None):
    spec = spec or G.load()
    S = C.Solids(spec)
    m0 = mujoco.MjModel.from_xml_path(str(XML))
    d0 = mujoco.MjData(m0)
    mujoco.mj_forward(m0, d0)
    ms = mujoco.MjSpec.from_file(str(XML))
    for g in ms.geoms:
        g.group = 3
    ms.visual.global_.offwidth, ms.visual.global_.offheight = 1920, 1080
    ms.visual.quality.shadowsize = 4096
    ms.visual.headlight.ambient = [0.42, 0.42, 0.42]
    ms.visual.headlight.diffuse = [0.4, 0.4, 0.4]
    ms.visual.headlight.specular = [0.05, 0.05, 0.05]
    ms.add_material(name='matte', specular=0.08, shininess=0.15)
    ms.add_material(name='gloss', specular=0.7, shininess=0.9)
    if room is None:
        # a light studio: grey gradient sky, pale checker floor (1 m squares)
        ms.add_texture(name='sky', type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                       rgb1=[0.97, 0.97, 0.98], rgb2=[0.80, 0.82, 0.85], width=512, height=3072)
        ms.add_texture(name='grid', type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                       rgb1=[0.93, 0.93, 0.94], rgb2=[0.84, 0.85, 0.87], width=512, height=512)
        floor = ms.add_material(name='floor', reflectance=0.05)
        floor.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'grid'
        floor.texrepeat = [1, 1]
        floor.texuniform = True
        ms.worldbody.add_geom(name='view floor', type=mujoco.mjtGeom.mjGEOM_PLANE, size=[30, 30, 0.05], material='floor',
                              contype=0, conaffinity=0)
    # the MJCF's own spotlight is fixed over the origin: the floor went dark once the robot walked away from it
    for lt in ms.lights:
        lt.active = False
    sun = ms.worldbody.add_light(name='sun', dir=[-0.4, 0.3, -1.0], diffuse=[0.55, 0.55, 0.55],
                                 specular=[0.15, 0.15, 0.15], castshadow=False, pos=[0, 0, 4])
    if hasattr(sun, 'type'):                  # newer MuJoCo: a light type; older: a 'directional' flag
        sun.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    else:
        sun.directional = True
    k = 0
    for (ln, i), m in S.geom.items():
        if m is None:
            continue
        g = spec['links'][ln]['geoms'][i]
        mesh = m.to_mesh()
        v = np.asarray(mesh.vert_properties[:, :3], float) - d0.xpos[m0.body(ln).id]   # body frames are world-aligned
        f = np.asarray(mesh.tri_verts, int)
        if len(f) < 4:
            continue
        name = f'part{k}'
        ms.add_mesh(name=name, uservert=v.flatten().tolist(), userface=f.flatten().tolist())
        ms.body(ln).add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, rgba=rgba(g['color']),
                             material='gloss' if g['color'] in GLOSSY else 'matte', contype=0, conaffinity=0,
                             group=1, density=0)
        k += 1
    if room is not None:
        room(ms)
    return ms


def skeleton_geoms(scn, J, alpha=1.0):
    """The captured human: joints as spheres, bones as capsules (as the X2 repo's viewer draws it)."""
    for j in range(len(J)):
        if scn.ngeom >= scn.maxgeom:
            return
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.022, 0, 0]), J[j],
                            np.eye(3).flatten(), np.array([0.15, 0.55, 0.95, alpha]))
        scn.ngeom += 1
    for j, p in enumerate(PARENTS):
        if p < 0 or scn.ngeom >= scn.maxgeom:
            continue
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                            np.eye(3).flatten(), np.array([0.25, 0.27, 0.32, alpha]))
        mujoco.mjv_connector(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, 0.012, J[p], J[j])
        scn.ngeom += 1


def smooth_track(p, sigma_frames):
    k = np.arange(-int(3 * sigma_frames), int(3 * sigma_frames) + 1)
    w = np.exp(-0.5 * (k / sigma_frames) ** 2)
    w /= w.sum()
    pad = np.pad(p, ((len(k) // 2, len(k) // 2), (0, 0)), mode='edge')
    return np.stack([np.convolve(pad[:, j], w, mode='valid') for j in range(p.shape[1])], axis=1)


def caption(img, xy, text, fill=(20, 20, 20)):
    """Text on a pale box, readable on floor and sky alike."""
    dr = ImageDraw.Draw(img, 'RGBA')
    x0, y0, x1, y1 = dr.textbbox(xy, text)
    dr.rectangle((x0 - 5, y0 - 3, x1 + 5, y1 + 3), fill=(255, 255, 255, 190))
    dr.text(xy, text, fill=fill)


def play_viewer(take_path, skeleton=True):
    """The take in MuJoCo's interactive viewer, in real time, looping (needs the PC's screen)."""
    import time
    import mujoco.viewer
    d = np.load(take_path)
    m = build_view_model()
    data = mujoco.MjData(m)
    adr = [m.jnt_qposadr[m.joint(n).id] for n in d['joint_names']]
    dt = float(d['t'][1] - d['t'][0])
    with mujoco.viewer.launch_passive(m, data) as v:
        while v.is_running():
            for i in range(len(d['t'])):
                t0 = time.time()
                data.qpos[:3], data.qpos[3:7], data.qpos[adr] = d['root_pos'][i], d['root_quat_wxyz'][i], d['q'][i]
                mujoco.mj_kinematics(m, data)
                mujoco.mj_camlight(m, data)               # light directions too (kinematics alone left the sun dark)
                if skeleton and 'skel' in d.files:
                    J = d['skel'][i].copy()
                    J[:, 1] += 0.85
                    v.user_scn.ngeom = 0
                    skeleton_geoms(v.user_scn, J)
                v.sync()
                if not v.is_running():
                    break
                time.sleep(max(0.0, dt - (time.time() - t0)))


def render(take_path, out, view='34', start=0.0, dur=None, fps=25, w=1024, h=576, skeleton=True, title=''):
    d = np.load(take_path)
    t, q = d['t'], d['q']
    src_fps = round(1 / (t[1] - t[0]))
    m = build_view_model()
    data = mujoco.MjData(m)
    names = list(d['joint_names'])
    adr = [m.jnt_qposadr[m.joint(n).id] for n in names]
    i0 = int(start * src_fps)
    i1 = len(t) if dur is None else min(len(t), int((start + dur) * src_fps))
    frames = np.arange(i0, i1, max(1, round(src_fps / fps)))
    track = smooth_track(d['root_pos'][:, :2], 0.6 * src_fps)    # the camera follows the pelvis, smoothly
    az, el, dist = VIEWS[view]
    if view == 'follow':                                          # from the robot's side as it turns (heading, 1 s)
        qw = d['root_quat_wxyz']
        yaw = np.unwrap(np.arctan2(2 * (qw[:, 0] * qw[:, 3] + qw[:, 1] * qw[:, 2]), 1 - 2 * (qw[:, 2] ** 2 + qw[:, 3] ** 2)))
        heading = smooth_track(yaw[:, None], 1.0 * src_fps)[:, 0]
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = az, el, dist
    right = np.array([math.sin(math.radians(az)), -math.cos(math.radians(az)), 0.0])   # screen right, on the floor
    r = mujoco.Renderer(m, h, w)
    opt = mujoco.MjvOption()
    writer = imageio.get_writer(out, fps=fps, codec='libx264', quality=7, pixelformat='yuv420p', macro_block_size=8)
    for n, i in enumerate(frames):
        data.qpos[:3] = d['root_pos'][i]
        data.qpos[3:7] = d['root_quat_wxyz'][i]
        data.qpos[adr] = q[i]
        mujoco.mj_kinematics(m, data)
        mujoco.mj_camlight(m, data)               # light directions too (kinematics alone left the sun dark)
        cam.lookat[:] = [track[i, 0], track[i, 1], 0.85]
        if view == 'follow':
            cam.azimuth = math.degrees(heading[i]) + az
            right = np.array([math.sin(math.radians(cam.azimuth)), -math.cos(math.radians(cam.azimuth)), 0.0])
        r.update_scene(data, cam, opt)
        if skeleton and 'skel' in d.files:
            J = d['skel'][i].copy()
            J[:, :2] += d['root_pos'][i, :2] - J[0, :2] + right[:2] * 0.85    # beside the robot, on screen right
            skeleton_geoms(r.scene, J)
        img = Image.fromarray(r.render())
        caption(img, (12, 10), title)
        caption(img, (12, h - 24), f't {t[i]:5.1f} s   1x   kinematic playback: joint angles set, no physics')
        if skeleton:
            caption(img, (w - 262, 10), 'blue: the PICO-captured operator, robot scale', (20, 50, 140))
        writer.append_data(np.asarray(img))
        if n % 250 == 0:
            print(f'  frame {n}/{len(frames)}', flush=True)
    writer.close()
    r.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('take', type=Path)
    ap.add_argument('--view', default='34', choices=list(VIEWS))
    ap.add_argument('--start', type=float, default=0.0)
    ap.add_argument('--dur', type=float, default=None)
    ap.add_argument('--no-skeleton', action='store_true')
    ap.add_argument('--viewer', action='store_true', help="play in MuJoCo's interactive viewer instead")
    ap.add_argument('--title', default='echo walking: PICO full-body capture retargeted (session 2026-08-22 seg30)')
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    if a.viewer:
        return play_viewer(a.take, not a.no_skeleton)
    out = a.out or a.take.with_name(f'{a.take.stem}_{a.view}.mp4')
    render(a.take, str(out), a.view, a.start, a.dur, skeleton=not a.no_skeleton, title=a.title)
    print(out)


if __name__ == '__main__':
    main()
