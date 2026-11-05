"""The robot with its cover off: only what holds it together and moves it, from the spec, at the angles a build photo would use.

  python tools/render_skeleton.py --out reports/parts/skeleton.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import checks as C  # noqa: E402
import geom as G  # noqa: E402
from render_motion import XML, rgba  # noqa: E402

ALU, MOTOR, STEEL = '#C4C7CB', '#303236', '#8E9298'
HARDWARE, BEARING = '#D9A520', '#2AA79B'    # for reading: real screws are black, but these are what the picture is for
ELECTRONICS = {'battery', 'battery lower', 'compute'}
BG = [0.93, 0.92, 0.90]
BEND = {'L_hip_pitch': -0.35, 'L_knee': 0.75, 'L_ankle_pitch': -0.35, 'R_hip_pitch': 0.1, 'R_knee': 0.1,
        'L_shoulder_pitch': 0.25, 'R_shoulder_pitch': -0.25, 'L_elbow': -0.5, 'R_elbow': -0.5}
VIEWS = [('Front right', 45, -6, 2.3, 0.84), ('Side', 90, -2, 2.3, 0.84), ('Back', 215, -6, 2.3, 0.84), ('Legs', 40, -4, 1.5, 0.5)]
ROLES = ('struct', 'actuator', 'linkage', 'hardware', 'bearing')


def colour(g):
    if g['name'] in ELECTRONICS:
        return g['color']
    return {'actuator': MOTOR, 'linkage': STEEL, 'hardware': HARDWARE, 'bearing': BEARING}.get(g['role'], ALU)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    spec = G.load()
    S = C.Solids(spec)
    m0 = mujoco.MjModel.from_xml_path(str(XML))
    d0 = mujoco.MjData(m0)
    mujoco.mj_forward(m0, d0)
    ms = mujoco.MjSpec.from_file(str(XML))
    for g in ms.geoms:
        g.group = 3
    for lt in ms.lights:
        lt.active = False
    ms.visual.global_.offwidth, ms.visual.global_.offheight = 1920, 1920
    ms.visual.headlight.ambient = [0.5, 0.5, 0.5]
    ms.visual.headlight.diffuse = [0.45, 0.45, 0.45]
    ms.visual.headlight.specular = [0.1, 0.1, 0.1]
    ms.add_material(name='metal', specular=0.35, shininess=0.5)
    ms.add_texture(name='sky', type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, rgb1=BG, rgb2=BG, width=64, height=384)
    n = {r: 0 for r in ROLES}
    for (ln, i), m in S.geom.items():
        g = spec['links'][ln]['geoms'][i]
        if m is None or g.get('role') not in n:
            continue
        mesh = m.to_mesh()
        v = np.asarray(mesh.vert_properties[:, :3], float) - d0.xpos[m0.body(ln).id]  # body frames are world-aligned
        f = np.asarray(mesh.tri_verts, int)
        if len(f) < 4:
            continue
        name = f'part{sum(n.values())}'
        ms.add_mesh(name=name, uservert=v.flatten().tolist(), userface=f.flatten().tolist())
        ms.body(ln).add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, rgba=rgba(colour(g)), material='metal', contype=0, conaffinity=0, group=1, density=0)
        n[g['role']] += 1
    m = ms.compile()
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    for j, q in BEND.items():
        d.qpos[m.jnt_qposadr[m.joint(j).id]] = q
    mujoco.mj_kinematics(m, d)
    mujoco.mj_camlight(m, d)
    W, H = 900, 1500
    r = mujoco.Renderer(m, H, W)
    sheet = Image.new('RGB', (W * 4, H + 90), tuple(int(c * 255) for c in BG))
    font, small = ImageFont.load_default(size=40), ImageFont.load_default(size=34)
    root = d.qpos[:2].copy()
    for k, (title, az, el, dist, z) in enumerate(VIEWS):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth, cam.elevation, cam.distance = az, el, dist
        cam.lookat[:] = [root[0], root[1], z]
        r.update_scene(d, cam, mujoco.MjvOption())
        im = Image.fromarray(r.render())
        ImageDraw.Draw(im).text((30, 24), title, fill=(50, 50, 55), font=font)
        sheet.paste(im, (k * W, 0))
    r.close()
    ImageDraw.Draw(sheet).text((30, H + 24), f'Cover off, from the spec: {n["actuator"]} motors at their STEP outline, {n["struct"]} structure shapes, {n["linkage"]} rods and cranks, {n["hardware"]} screws (gold), {n["bearing"]} bearings (teal), knees only so far. Wiring not modelled; most structure is still placeholder. Colours for reading: not finishes.', fill=(150, 40, 30), font=small)
    sheet.save(a.out)
    print(a.out, sheet.size, n)


if __name__ == '__main__':
    main()
