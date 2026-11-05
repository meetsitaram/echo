"""Offline renders of the replica in any pose, from the spec's own shapes (cuts applied, each part in its colour),
posed with the same kinematics the checks use, drawn with MuJoCo's renderer. For when Fusion cannot save images
(2026-10-31: with the PC's screen asleep its viewport saved nothing) and for poses Fusion is slow to set.

  python tools/render_pose.py           -> reports/poses.png (the joint-range poses below)
  python tools/render_pose.py --views   -> reports/compare/replica_<view>.png, framed as build.render's (for
                                           tools/compare.py when Fusion cannot render; same robot box within 2 px)

Plain python (MuJoCo lives in the user site-packages, which -I skips); it reads only our own files.
"""
import math
import os
import sys
import tempfile

import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checks as C  # noqa: E402
import geom as G  # noqa: E402

GLOSSY = ('#08090B', '#0A1420')
VIEWS = {'front': (0, 0), 'side': (90, 0), 'back': (180, 0), '34': (60, 8), 'hero': (40, 8)}


def rgba(hex_color):
    h = hex_color.lstrip('#')
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [1.0]


def stl(path, v, f):
    """Binary STL (MuJoCo reads it; normals are recomputed)."""
    tri = v[f].astype(np.float32)
    rec = np.zeros(len(f), dtype=[('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
    rec['v'] = tri
    with open(path, 'wb') as fh:
        fh.write(b'\0' * 80)
        fh.write(np.uint32(len(f)).tobytes())
        fh.write(rec.tobytes())


class Scene:
    def __init__(self, spec=None):
        self.spec = spec or G.load()
        self.S = C.Solids(self.spec)
        self.R = G.Robot(self.spec)
        self.parts = []                                   # (link, verts, faces, colour, role)
        for (ln, i), m in self.S.geom.items():
            if m is None:
                continue
            g = self.spec['links'][ln]['geoms'][i]
            mesh = m.to_mesh()
            self.parts.append((ln, np.asarray(mesh.vert_properties[:, :3], float),
                               np.asarray(mesh.tri_verts, int), g['color'], g['role']))

    def model(self, q, tmp, hide=()):
        M = self.R.fk(q)
        assets, geoms = [], []
        for k, (ln, v, f, color, role) in enumerate(self.parts):
            if len(f) == 0 or role in hide:
                continue
            stl(os.path.join(tmp, f'p{k}.stl'), G.apply(M[ln], v), f)
            assets.append(f'<mesh name="p{k}" file="p{k}.stl"/>')
            mat = 'gloss' if color in GLOSSY else 'matte'
            geoms.append(f'<geom type="mesh" mesh="p{k}" material="{mat}" rgba="{" ".join(f"{c:.3f}" for c in rgba(color))}"/>')
        xml = f"""<mujoco><compiler meshdir="{tmp}"/>
<visual><global offwidth="1600" offheight="1600"/><headlight ambient=".42 .42 .42" diffuse=".45 .45 .45" specular=".1 .1 .1"/>
<quality shadowsize="0"/></visual>
<asset><material name="matte" specular=".05" shininess=".1"/><material name="gloss" specular=".7" shininess=".9" reflectance="0"/>
{''.join(assets)}</asset>
<worldbody><light directional="true" diffuse=".35 .35 .35" specular=".2 .2 .2" dir="-0.6 -0.3 -1" castshadow="false"/>
{''.join(geoms)}</worldbody></mujoco>"""
        path = os.path.join(tmp, 'scene.xml')
        with open(path, 'w') as fh:
            fh.write(xml)
        return mujoco.MjModel.from_xml_path(path)

    def render(self, q, view='front', w=800, h=600, height_m=1.9, target_z=0.86, target_y=0.0, hide=()):
        with tempfile.TemporaryDirectory() as tmp:
            m = self.model(q, tmp, hide)
            d = mujoco.MjData(m)
            mujoco.mj_forward(m, d)
            az, el = (math.radians(a) for a in VIEWS[view])
            dv = np.array([math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)])
            x = np.cross([0, 0, 1], dv)
            x /= np.linalg.norm(x)
            y = np.cross(dv, x)
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [0, target_y, target_z]
            cam.distance = 4.0
            cam.azimuth = math.degrees(math.atan2(-dv[1], -dv[0]))
            cam.elevation = -math.degrees(el)
            m.vis.global_.orthographic = 1
            m.vis.global_.fovy = height_m
            r = mujoco.Renderer(m, h, w)
            r.update_scene(d, cam)
            img = r.render().copy()
            r.enable_segmentation_rendering()
            r.update_scene(d, cam)
            seg = r.render()[:, :, 0]
            r.close()
        img[seg < 0] = 255
        return Image.fromarray(img)


COVERS = ('shell', 'decal')
POSES = [
    # (title, angles, view, picture height m, target z m, target y m, roles hidden)
    ('shoulder roll 90 / 149 deg (max), neck nod down 26 deg (max)',
     {'L_shoulder_roll': 1.57, 'R_shoulder_roll': -2.6, 'neck_pitch': 0.45}, 'front', 1.1, 1.3, 0.0, ()),
    ('same, 40 deg view', {'L_shoulder_roll': 1.57, 'R_shoulder_roll': -2.6, 'neck_pitch': 0.45}, 'hero', 1.1,
     1.3, 0.0, ()),
    ('neck nod down 0.45 rad (max)', {'neck_pitch': 0.45}, 'side', 0.42, 1.52, 0.0, ()),
    ('neck nod up -0.40 rad (min)', {'neck_pitch': -0.40}, 'side', 0.42, 1.52, 0.0, ()),
    ('at rest: shoulder socket and neck collar', {}, 'hero', 0.42, 1.45, 0.08, ()),
    ('at rest, back: shoulder roll lugs', {}, 'back', 0.42, 1.42, 0.0, ()),
    ('inside, front: neck yaw low in the torso, post, pitch at the neck base', {}, 'front', 0.3, 1.47, 0.0, COVERS),
    ('inside, back: roll output through the cap\'s back, lug, step, foot', {}, 'back', 0.42, 1.38, 0.0, COVERS),
]


def main():
    sc = Scene()
    if '--views' in sys.argv:
        for v in ('side', '34', 'front', 'back'):
            out = os.path.join(G.ROOT, 'reports', 'compare', f'replica_{v}.png')
            sc.render({}, v, 1600, 1200, 1.9, 0.86).save(out)   # twice the frames' size: the close-up crops stay sharp
            print(out, flush=True)
        with open(os.path.join(G.ROOT, 'reports', 'compare', 'replica_source.txt'), 'w', encoding='utf-8') as fh:
            fh.write('offline render of the spec shapes')    # compare.py's panel label; build.render removes it
        return
    tiles = []
    for title, q, view, hm, tz, ty, hide in POSES:
        im = sc.render(q, view, 640, 480, hm, tz, ty, hide)
        ImageDraw.Draw(im).text((8, 6), title, fill=(0, 0, 0))
        tiles.append(im)
        print(title, flush=True)
    cols = 2
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new('RGB', (cols * 640, rows * 480), 'white')
    for k, im in enumerate(tiles):
        sheet.paste(im, ((k % cols) * 640, (k // cols) * 480))
    out = os.path.join(G.ROOT, 'reports', 'poses.png')
    sheet.save(out)
    print(out)


if __name__ == '__main__':
    main()
