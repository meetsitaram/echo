"""Machined-metal close-ups of the skeleton, from the spec's real solids (holes, seats, fillets), with a key light and physically based materials (Open3D's Filament renderer). Offline primitives and distance trees, not CAD: see LESSONS.

  python -I tools/render_metal.py --set knee --out reports/parts/knee_metal         a sheet per bend, cover off, plus sections
  python -I tools/render_metal.py --set leg --out reports/parts/leg_metal           the whole leg
  python -I tools/render_metal.py --spec old.json --set knee --out x                any robot.json, for a before picture

Finishes (a look, not a claim about the part): bare 7075 bead-blasted and turned, black-oxide screws, bearing steel, anodised motor. Section views cut the solids by a plane and colour the cut faces.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checks as C  # noqa: E402
import geom as G  # noqa: E402

KNEE_LINKS = ['L_hip_yaw_link', 'L_knee_link']
LEG_LINKS = ['L_hip_yaw_link', 'L_knee_link', 'L_ankle_pitch_link', 'L_foot']
SKEL = ('struct', 'actuator', 'linkage', 'hardware', 'bearing')


def part_mesh(spec, ln, i, g, cuts, hi):
    """The part's real solid: finer than the checks' for a machined part, its cuts applied, returned as trimesh."""
    if g['kind'] == 'sdf' and hi:
        base = G.sdf_mesh(g, hi=True)
    else:
        base = G.mesh_of(g)
    m = C.to_manifold(base)
    if g['name'] in cuts:
        m = m - cuts[g['name']]
    o = m.to_mesh()
    return trimesh.Trimesh(np.asarray(o.vert_properties[:, :3], float), np.asarray(o.tri_verts, int), process=False)


def tree_normals(g, verts, normals):
    """A machined part's true surface normals, from its distance tree: no mesh facets in the reflections. Only where the vertex is on the tree's surface and the mesh agrees (a drilled hole's wall isn't)."""
    import sdf as SDF
    if g.get('kind') != 'sdf':
        return normals
    P = np.array(verts, float)
    if g.get('flip_y'):
        P[:, 1] *= -1
    d0 = SDF.dist(g['tree'], P)
    e = 0.00012
    N = np.empty_like(P)
    for k in range(3):
        o = np.zeros(3)
        o[k] = e
        N[:, k] = SDF.dist(g['tree'], P + o) - SDF.dist(g['tree'], P - o)
    if g.get('flip_y'):
        N[:, 1] *= -1
    l = np.linalg.norm(N, axis=1, keepdims=True)
    N = N / np.maximum(l, 1e-12)
    use = (np.abs(d0) < 0.0003) & ((N * normals).sum(1) > 0.8)
    out = np.array(normals, float)
    out[use] = N[use]
    return out


def finish(g, shell_ok=False):
    """(base colour rgb, metallic, roughness). What the part is, not its colour in the look model."""
    role, nm = g['role'], g['name']
    if role == 'hardware':
        if g.get('hw', {}).get('type') == 'retainer':
            return (0.62, 0.63, 0.65), 1.0, 0.34
        return (0.05, 0.05, 0.055), 0.85, 0.34
    if role == 'bearing':
        return (0.80, 0.81, 0.83), 1.0, 0.16
    if role == 'actuator':
        return (0.075, 0.08, 0.09), 0.8, 0.42
    if role == 'linkage':
        return (0.70, 0.71, 0.72), 1.0, 0.26
    if role in ('shell', 'decal', 'knit'):
        return tuple(int(g['color'][k:k + 2], 16) / 255 for k in (1, 3, 5)), 0.0, 0.3
    # 7075: bare, fine machined
    return (0.86, 0.865, 0.875), 1.0, 0.30


def collect(spec, links, roles=SKEL, skip=(), hi=True):
    """[(name, mesh at the zero pose, link, finish)]"""
    out = []
    for ln in links:
        l = spec['links'][ln]
        cuts = {}
        for g in l['geoms']:
            if g['role'] == 'cut':
                m = C.to_manifold(G.mesh_of(g))
                for t in g['targets']:
                    cuts[t] = m if t not in cuts else cuts[t] + m
        for i, g in enumerate(l['geoms']):
            if g['role'] == 'cut' or g['role'] not in roles or g['name'] in skip:
                continue
            out.append((g['name'], part_mesh(spec, ln, i, g, cuts, hi), ln, finish(g), g))
    return out


def clean_normals(n):
    """A degenerate triangle gives a NaN normal, and one NaN turns Filament's whole frame black."""
    n = np.nan_to_num(np.array(n, float))
    l = np.linalg.norm(n, axis=1, keepdims=True)
    return np.where(l > 1e-9, n / np.maximum(l, 1e-9), np.array([[0.0, 0.0, 1.0]]))


def smooth(mesh, angle=32.0):
    """Split vertices at edges sharper than angle, so turned and milled faces shade smooth and their edges stay crisp."""
    return trimesh.graph.smooth_shade(mesh, angle=np.radians(angle))


class Renderer:
    def __init__(self, W=1500, H=1100, bg=(0.90, 0.90, 0.905), ss=2):
        import open3d as o3d
        self.o3d = o3d
        self.ss = ss
        self.W, self.H = W, H
        self.r = o3d.visualization.rendering.OffscreenRenderer(W * ss, H * ss)
        self.bg = bg
        self._cache = {}
        self.ao = False

    def _geom(self, mesh, key, g=None):
        o3d = self.o3d
        if key in self._cache:
            return self._cache[key]
        s = smooth(mesh)
        t = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.array(s.vertices, float)), o3d.utility.Vector3iVector(np.array(s.faces, np.int32)))
        n = clean_normals(s.vertex_normals)
        if g is not None:
            n = clean_normals(tree_normals(g, s.vertices, n))
        t.vertex_normals = o3d.utility.Vector3dVector(n)
        self._cache[key] = t
        return t

    def shot(self, parts, cam, target, up=(0, 0, 1), fov=22.0, poses=None, key=(-0.55, 0.65, -0.55), key_i=95000, ibl=32000, section=None, cut_colour=(0.78, 0.27, 0.16)):
        """parts: from collect(). cam, target: world points. poses: {link: 4x4}. section: (point, normal): keep what's in front of the plane, cut faces coloured."""
        import open3d.visualization.rendering as R
        rr = self.r
        rr.scene.clear_geometry()
        rr.scene.set_background(list(self.bg) + [1.0])
        sc = rr.scene.scene
        sc.set_sun_light(list(key), [1.0, 0.98, 0.95], key_i)
        sc.enable_sun_light(True)
        sc.enable_indirect_light(True)
        sc.set_indirect_light_intensity(ibl)
        if self.ao:
            rr.scene.view.set_ambient_occlusion(True)
        for k, (name, mesh, ln, (rgb, metal, rough), gm) in enumerate(parts):
            M = np.eye(4) if poses is None else poses[ln]
            mat = R.MaterialRecord()
            mat.shader = 'defaultLit'
            mat.base_color = list(rgb) + [1.0]
            mat.base_metallic, mat.base_roughness, mat.base_reflectance = metal, rough, 0.5
            if section is None:
                g = self._geom(mesh, name, gm)
                rr.scene.add_geometry(f'p{k}', g, mat)
                rr.scene.set_geometry_transform(f'p{k}', M)
            else:
                mm = C.to_manifold(mesh)
                pt, nr = section
                Mi = np.linalg.inv(M)
                n0 = Mi[:3, :3] @ np.asarray(nr, float)
                p0 = Mi[:3, :3] @ np.asarray(pt, float) + Mi[:3, 3]
                n0 = n0 / np.linalg.norm(n0)
                cutm = mm.trim_by_plane(n0.tolist(), float(n0 @ p0))
                if cutm.is_empty():
                    continue
                o = cutm.to_mesh()
                t = trimesh.Trimesh(np.asarray(o.vert_properties[:, :3], float), np.asarray(o.tri_verts, int), process=False)
                fn = t.face_normals
                cen = t.triangles_center
                on = (np.abs(fn @ n0 + 1) < 1e-3) & (np.abs((cen - p0) @ n0) < 2e-5)
                keep = trimesh.Trimesh(t.vertices, t.faces[~on], process=False) if (~on).any() else None
                if keep is not None and len(keep.faces):
                    g = self._geom(keep, None) if False else None
                    s = smooth(keep)
                    self_n = clean_normals(tree_normals(gm, s.vertices, clean_normals(s.vertex_normals)))
                    tt = self.o3d.geometry.TriangleMesh(self.o3d.utility.Vector3dVector(np.array(s.vertices, float)), self.o3d.utility.Vector3iVector(np.array(s.faces, np.int32)))
                    tt.vertex_normals = self.o3d.utility.Vector3dVector(self_n)
                    rr.scene.add_geometry(f'p{k}', tt, mat)
                    rr.scene.set_geometry_transform(f'p{k}', M)
                if on.any():
                    capm = R.MaterialRecord()
                    capm.shader = 'defaultLit'
                    capm.base_color = list(cut_colour) + [1.0]
                    capm.base_metallic, capm.base_roughness = 0.0, 0.55
                    c = trimesh.Trimesh(t.vertices, t.faces[on], process=False)
                    tt = self.o3d.geometry.TriangleMesh(self.o3d.utility.Vector3dVector(np.array(c.vertices, float)), self.o3d.utility.Vector3iVector(np.array(c.faces, np.int32)))
                    tt.compute_vertex_normals()
                    rr.scene.add_geometry(f'c{k}', tt, capm)
                    rr.scene.set_geometry_transform(f'c{k}', M)
        rr.setup_camera(fov, list(map(float, target)), list(map(float, cam)), list(map(float, up)))
        img = np.asarray(rr.render_to_image())
        for _ in range(40):               # a big upload can hand back an empty or part-drawn frame
            if img[0, 0].sum() > 30 and img[0, -1].sum() > 30:
                break
            import time
            time.sleep(0.25)
            img = np.asarray(rr.render_to_image())
        if img[0, 0].sum() <= 30:
            raise RuntimeError('a black frame')
        if self.ss > 1:
            from PIL import Image
            img = np.asarray(Image.fromarray(img).resize((self.W, self.H), Image.LANCZOS))
        return img


def ang_cam(target, az, el, dist):
    """A camera at azimuth az (deg from +x toward +y) and elevation el, looking at target."""
    a, e = np.radians(az), np.radians(el)
    return np.asarray(target, float) + dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def knee_views():
    """(label, azimuth, elevation). The left leg: +y is its outside."""
    return [('front', 0, 3), ('side, outside', 90, 3), ('3/4 outside', 42, 14), ('rear', 180, 5), ('3/4 inside', -48, 12)]


def sheet(imgs, labels, cols, out, font_size=26, title=None):
    from PIL import Image, ImageDraw, ImageFont
    h, w = imgs[0].shape[:2]
    rows = (len(imgs) + cols - 1) // cols
    top = 56 if title else 0
    S = Image.new('RGB', (w * cols, h * rows + top), (246, 246, 246))
    dr = ImageDraw.Draw(S)
    font = ImageFont.load_default(size=font_size)
    if title:
        dr.text((14, 12), title, fill=(25, 25, 28), font=font)
    for i, im in enumerate(imgs):
        x, y = (i % cols) * w, (i // cols) * h + top
        S.paste(Image.fromarray(im), (x, y))
        dr.text((x + 12, y + 8), labels[i], fill=(25, 25, 28), font=font)
    S.save(out)
    return S.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec')
    ap.add_argument('--set', default='knee', choices=('knee', 'leg', 'section'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--bends', default='0,0.6,2.15')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    spec = G.load(a.spec) if a.spec else G.load()
    Rb = G.Robot(spec)
    origin = np.array(next(j for j in spec['joints'] if j['name'] == 'L_knee')['origin'], float)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    rn = Renderer(W=1000, H=1500) if a.set == 'leg' else Renderer()
    if a.set == 'knee':
        hub_screws = {g['name'] for l in KNEE_LINKS for g in spec['links'][l]['geoms'] if g.get('hw', {}).get('type') == 'screw' and ' foot' in g['hw'].get('pattern', '')}
        parts = collect(spec, KNEE_LINKS, skip=hub_screws)
        for q in [float(x) for x in a.bends.split(',')]:
            M = Rb.fk({'L_knee': q})
            # the world stays with the thigh at the zero pose; the shank bends
            P = {l: np.linalg.inv(M['L_hip_yaw_link']) @ M[l] for l in KNEE_LINKS}
            tgt = origin + np.array([0.0, 0.0, 0.0])
            imgs, labs = [], []
            for lab, az, el in knee_views():
                imgs.append(rn.shot(parts, ang_cam(tgt, az, el, 0.85), tgt, poses=P))
                labs.append(f'{lab}, knee {q:.2f} rad')
            print(sheet(imgs, labs + [], 3, f'{a.out}_{q:.2f}.png', title=f'Knee, cover off, {q:.2f} rad. {a.tag}'), flush=True)
    if a.set == 'leg':
        parts = collect(spec, LEG_LINKS)
        tgt = origin + np.array([0.0, 0.0, -0.08])
        imgs, labs = [], []
        for q, qa in ((0.0, 0.0), (0.6, -0.3)):
            M = Rb.fk({'L_knee': q, 'L_ankle_pitch': qa})
            P = {l: np.linalg.inv(M['L_hip_yaw_link']) @ M[l] for l in LEG_LINKS}
            for lab, az, el in (('front', 0, 3), ('side', 90, 3), ('3/4', 42, 10), ('rear', 180, 3)):
                imgs.append(rn.shot(parts, ang_cam(tgt, az, el, 2.05), tgt, poses=P, fov=26.0))
                labs.append(f'{lab}, knee {q:.1f}')
        print(sheet(imgs, labs, 4, f'{a.out}.png', title=f'Thigh and shank, cover off. {a.tag}'))
    if a.set == 'section':
        parts = collect(spec, KNEE_LINKS)
        imgs, labs = [], []
        tgt = origin
        for lab, cam_az, nr, pt in (('through the axis, front half removed, looking back', 0, (-1, 0, 0), origin),
                                    ('through a stator screw', 0, (-1, 0, 0), origin + np.array([-0.0265, 0, 0]))):
            imgs.append(rn.shot(parts, ang_cam(tgt, cam_az, 2, 0.8), tgt, section=(pt, nr)))
            labs.append(lab)
        imgs.append(rn.shot(parts, ang_cam(tgt, 90, 2, 0.8), tgt, section=(origin + np.array([0, 0.0, 0]), (0, -1, 0))))
        labs.append('through the motor\'s middle, looking inboard')
        imgs.append(rn.shot(parts, ang_cam(tgt, 90, 2, 0.8), tgt, section=(origin + np.array([0, -0.0305, 0]), (0, -1, 0))))
        labs.append('through the inner bearing, looking inboard')
        print(sheet(imgs, labs, 2, f'{a.out}.png', title=f'Knee in section; cut faces red. {a.tag}'))


if __name__ == '__main__':
    main()
