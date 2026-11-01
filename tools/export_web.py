"""The replica for a web viewer: a coloured glTF with embedded buffers, saved as .json (one node per link at its rest
pose, one mesh per part, named '<role>|<part>|<link>|<index>') and a take's per-frame link poses and operator skeleton as JSON.

  python tools/export_web.py [reports/motion/<take>.npz] [--out reports/web] [--fps 25]

Plain python (MuJoCo in the user site-packages).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import mujoco
import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checks as C  # noqa: E402
import geom as G  # noqa: E402

XML = os.path.join(G.ROOT, 'urdf', 'echo.xml')
GLOSSY = ('#08090B', '#0A1420')


def finish(g):
    """(metallic, roughness) by the part's look."""
    if g['color'] in GLOSSY:
        return 0.0, 0.12
    if g.get('material') == 'al7075' and g['color'] == '#9EA3A8':
        return 0.7, 0.35
    if g['role'] == 'actuator':
        return 0.3, 0.45
    return 0.0, 0.8 if g.get('material') == 'knit_pa12' else 0.55


def export_glb(spec, out):
    S = C.Solids(spec)
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    scene = trimesh.Scene()
    tris = 0
    for ln in spec['links']:
        rest = d.xpos[m.body(ln).id].copy()
        T = np.eye(4)
        T[:3, 3] = rest
        scene.graph.update(frame_from=scene.graph.base_frame, frame_to=ln, matrix=T)
        for i, g in enumerate(spec['links'][ln]['geoms']):
            man = S.geom.get((ln, i))
            if man is None:
                continue
            mesh = man.to_mesh()
            tm = trimesh.Trimesh(np.asarray(mesh.vert_properties[:, :3], float) - rest, np.asarray(mesh.tri_verts, int),
                                 process=True)
            if len(tm.faces) < 4:
                continue
            tm = trimesh.graph.smooth_shade(tm, angle=np.radians(35))   # sharp edges keep their own normals
            h = g['color'].lstrip('#')
            rgb = [int(h[k:k + 2], 16) / 255 for k in (0, 2, 4)]
            met, rough = finish(g)
            tm.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
                baseColorFactor=rgb + [1.0], metallicFactor=met, roughnessFactor=rough, doubleSided=False))
            role = 'cover' if g['role'] in ('shell', 'decal') else 'inside'
            name = f"{role}|{g['name']}|{ln}|{i}"
            scene.add_geometry(tm, node_name=name, geom_name=name, parent_node_name=ln)
            tris += len(tm.faces)
    # glTF with its buffer embedded (base64), named .json: artifact pages serve .json but not .glb
    files = trimesh.exchange.gltf.export_gltf(scene, embed_buffers=True)
    data = next(v for k, v in files.items() if k.endswith('.gltf'))
    with open(out, 'wb') as f:
        f.write(data)
    return tris, len(data)


def export_motion(take_path, out, fps=25):
    tk = np.load(take_path)
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    names = [m.body(b).name for b in range(1, m.nbody)]
    adr = [m.jnt_qposadr[m.joint(n).id] for n in tk['joint_names']]
    src = round(1 / (tk['t'][1] - tk['t'][0]))
    idx = np.arange(0, len(tk['t']), max(1, round(src / fps)))
    poses, skel = [], []
    for i in idx:
        d.qpos[:3], d.qpos[3:7], d.qpos[adr] = tk['root_pos'][i], tk['root_quat_wxyz'][i], tk['q'][i]
        mujoco.mj_kinematics(m, d)
        row = []
        for n in names:
            b = m.body(n).id
            w, x, y, z = d.xquat[b]
            row += [round(float(v), 4) for v in d.xpos[b]] + [round(float(v), 4) for v in (x, y, z, w)]
        poses.append(row)
        skel.append([round(float(v), 3) for v in tk['skel'][i].ravel()])
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(dict(fps=fps, links=names, frames=poses, skeleton=skel,
                       parents=[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
                       take=os.path.basename(str(take_path))), f, separators=(',', ':'))
    return len(idx), os.path.getsize(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('take', nargs='?', default=os.path.join(G.ROOT, 'reports', 'motion', 'pico_seg30_walk_echo.npz'))
    ap.add_argument('--out', default=os.path.join(G.ROOT, 'reports', 'web'))
    ap.add_argument('--fps', type=int, default=25)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tris, size = export_glb(G.load(), os.path.join(a.out, 'echo.json'))
    print(f'echo.json (glTF): {tris} triangles, {size / 1e6:.1f} MB')
    n, size = export_motion(a.take, os.path.join(a.out, 'walk.json'), a.fps)
    print(f'walk.json: {n} frames, {size / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
