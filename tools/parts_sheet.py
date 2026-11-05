"""Every part in the spec laid flat on one sheet, all at the same scale, in its real colour: what exists, and what doesn't yet.

  python tools/parts_sheet.py --out reports/parts_sheet.png
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import checks as C  # noqa: E402
import geom as G  # noqa: E402

BG = (0.89, 0.89, 0.90, 1.0)
GROUPS = [('knit', 'Garment'), ('shell', 'Covers'), ('actuator', 'Actuators (bought)'), ('struct', 'Structure'), ('linkage', 'Linkages')]
SPEC_GLOSS = {'#08090B'}


def rgba(h):
    h = h.lstrip('#')
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [1.0]


def lay_flat(v, f):
    """Thinnest direction up, longest direction up the sheet, a cover's outside toward the camera."""
    mesh = trimesh.Trimesh(v, f, process=False)
    T, ext = trimesh.bounds.oriented_bounds(mesh)
    p = trimesh.transform_points(v, T)
    order = np.argsort(ext)  # thin, middle, long
    z, y = np.eye(3)[order[0]], np.eye(3)[order[2]]
    x = np.cross(y, z)
    p = p @ np.array([x, y, z]).T
    p -= (p.min(0) + p.max(0)) / 2
    c = trimesh.Trimesh(p, f, process=False).triangles_center
    a = trimesh.Trimesh(p, f, process=False).area_faces
    if (c[:, 2] * a).sum() < 0:  # most of its surface below the middle: it would lie outside down
        p = p * [-1, 1, -1]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--width', type=float, default=1.55, help='sheet width, metres of robot')
    ap.add_argument('--px', type=int, default=2160)
    a = ap.parse_args()
    spec = G.load()
    S = C.Solids(spec)
    parts = []
    skipped = Counter()
    for (ln, i), m in S.geom.items():
        g = spec['links'][ln]['geoms'][i]
        if m is None or g.get('role') not in dict(GROUPS):
            skipped[g.get('role')] += 1
            continue
        mesh = m.to_mesh()
        v, f = np.asarray(mesh.vert_properties[:, :3], float), np.asarray(mesh.tri_verts, int)
        if len(f) < 4:
            continue
        p = lay_flat(v, f)
        parts.append(dict(name=g['name'], role=g['role'], color=g['color'], v=p, f=f, ext=p.max(0) - p.min(0)))
    # Shelves, biggest first, left and right of a part next to each other; a gap and a heading between kinds.
    pad, head = 0.022, 0.05
    y = 0.0
    rows = []
    ms = mujoco.MjSpec()
    ms.visual.global_.orthographic = 1
    ms.visual.headlight.ambient = [0.45, 0.45, 0.45]
    ms.visual.headlight.diffuse = [0.55, 0.55, 0.55]
    ms.visual.headlight.specular = [0.1, 0.1, 0.1]
    ms.add_material(name='matte', specular=0.15, shininess=0.2)
    ms.add_material(name='gloss', specular=0.7, shininess=0.9)
    k = 0
    totals = {}
    for role, title in GROUPS:
        ps = sorted((q for q in parts if q['role'] == role), key=lambda q: (-round(q['ext'][1] * q['ext'][0], 4), re.sub(r'^[LR][ _]', '', q['name'])))
        totals[title] = len(ps)
        y -= head
        rows.append((title, y + head * 0.72, len(ps)))
        x, shelf = 0.0, 0.0
        for q in ps:
            w, h = q['ext'][0], q['ext'][1]
            if x + w > a.width and x > 0:
                y -= shelf + pad
                x, shelf = 0.0, 0.0
            name = f'p{k}'
            ms.add_mesh(name=name, uservert=q['v'].flatten().tolist(), userface=q['f'].flatten().tolist())
            ms.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, pos=[x + w / 2, y - h / 2, q['ext'][2] / 2 + 0.002], rgba=rgba(q['color']),
                                  material='gloss' if q['color'] in SPEC_GLOSS else 'matte', contype=0, conaffinity=0)
            k += 1
            x += w + pad
            shelf = max(shelf, h)
        y -= shelf + pad
    foot = 0.075
    H = -y + foot
    ms.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[9, 9, 0.1], rgba=BG, contype=0, conaffinity=0)
    ms.visual.global_.fovy = H * 1.02
    px_h = int(a.px * H / a.width)
    ms.visual.global_.offwidth, ms.visual.global_.offheight = a.px, px_h
    m = ms.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, px_h, a.px)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [a.width / 2, -H / 2, 0]
    cam.distance, cam.elevation, cam.azimuth = 3.0, -90, 90
    r.update_scene(d, cam)
    img = Image.fromarray(r.render())
    r.close()
    dr = ImageDraw.Draw(img)
    # the same mapping the camera uses: the sheet fills the picture's height, 2 % spare
    scale = px_h / (H * 1.02)
    u = lambda wx: a.px / 2 + (wx - a.width / 2) * scale  # noqa: E731
    w_ = lambda wy: px_h / 2 - (wy + H / 2) * scale  # noqa: E731
    big, small = ImageFont.load_default(size=int(a.px * 0.019)), ImageFont.load_default(size=int(a.px * 0.0135))
    for title, wy, n in rows:
        dr.text((u(0), w_(wy)), f'{title}: {n}', fill=(40, 40, 44), font=big)
    t = y - 0.012
    dr.text((u(0), w_(t)), 'Fasteners: 0     Bearings: 0     Wiring: 0     (none of these are in the spec yet)', fill=(180, 40, 30), font=big)
    dr.text((u(0), w_(t - 0.04)), f'{sum(totals.values())} shapes from the spec at one scale. Most structure is still a placeholder block, not a designed, makeable part.', fill=(70, 70, 75), font=small)
    img.save(a.out)
    print(a.out, img.size, totals, 'skipped', dict(skipped))


if __name__ == '__main__':
    main()
