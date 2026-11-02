"""A living room for echo to work in, laid out from the living-room clip of Figure's home-work video: a vaulted
ceiling on light beams, a grey plaster fireplace between two tall windows with blinds, a long sofa on the left wall
with a side table at its near end, a long coffee table on a rug, a dark armchair and a tall plant by the right window,
a TV on a white console along the angled right wall, and a teal armchair in the near right corner.

Where things are: floor points marked on the clip's empty-room frame (the median of 60 frames; the corners and feet
in LAYOUT's comments, pixels in the 1280 x 720 frame) traced out to the floor through the camera that
tools/nlf_to_tape.py calibrated (1.30 m up, horizon at row 330), scaled by the take's height ratio (0.942), so the
room and the captured motion share one frame. Heights on the back wall the same way. Nothing from the footage is in
the scene: the textures are Poly Haven's (CC0, scenes/living_room/cc0/, see SOURCES.md there) and a generated rug.

  python tools/scene_living_room.py                    # stills from the clip's camera and a 3/4 view
  python tools/scene_living_room.py --pose stand       # one pose only (stand | carry | reach)

Room frame = the take's: the camera at (0, 0, 1.23) looking +x (1.6 deg down) at the fireplace wall (x = 11.55); the
sofa's wall is +y. Metres, floor z = 0.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_motion as RM  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / 'scenes' / 'living_room'
CC0 = TEX / 'cc0'
BUILD = TEX / 'build'
OUT = ROOT / 'reports' / 'scene'

# measured (see the docstring); y of a back-wall pixel column u: 1.07 - (u - 548) * 0.01048
BACK_X, LEFT_Y, FRONT_X = 11.55, 2.48, -1.5
RIGHT_BACK_Y, RIGHT_TURN_X = -2.40, 9.9        # the right wall's back stretch; from here it angles out toward the camera
RIGHT_ANGLE = math.radians(28)                 # the TV wall's angle to the room's x (the console's front edge)
EAVE_Z, RIDGE_Z, RIDGE_Y = 2.78, 3.35, 0.05    # eaves (u 415 and 880, v 180) and ridge (u 640, v 125) on the back wall
CAMERA = dict(pos=(0.0, 0.0, 1.226), pitch=math.radians(1.6), fovy=36.0)   # the clip's camera (60 deg across)


# ---------------------------------------------------------------- generated textures (rug, blinds)

def _noise(shape, sigma, rng):
    n = gaussian_filter(rng.standard_normal(shape), sigma)
    return (n - n.mean()) / (n.std() + 1e-9)


def _save(name, rgb):
    """Write a generated texture (MuJoCo reads PNG) to the build folder (not in the repo; made on every run)."""
    BUILD.mkdir(parents=True, exist_ok=True)
    p = BUILD / f'{name}.png'
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(p)
    return p


def tex_rug(name, size=1024, seed=3):
    """A faded Persian-style rug, as in the clip: a rose-beige field of small repeating motifs round a central
    medallion, framed by a dark main border between two guard stripes; worn and low in contrast."""
    rng = np.random.default_rng(seed)
    yy, xx = (np.mgrid[0:size, 0:size] + 0.5) / size
    u, v = (xx - 0.5) * 2, (yy - 0.5) * 2                        # -1..1 (u along the rug's length)
    rose, sand, slate, rust, ink = (np.array(c, float) for c in ((168, 132, 120), (196, 172, 146), (112, 118, 128),
                                                                    (140, 86, 70), (70, 60, 62)))
    tile = np.sin(xx * math.pi * 36) ** 2 * np.sin(yy * math.pi * 24) ** 2
    rosette = np.cos(np.hypot(((xx * 18) % 1) - 0.5, ((yy * 12) % 1) - 0.5) * 14) > 0.6
    img = rose * (1 - 0.25 * tile[..., None]) + (sand - rose) * 0.35 * rosette[..., None]
    med = np.abs(u / 0.42) + np.abs(v / 0.55)                     # the central medallion, a lozenge
    img = np.where((med < 1)[..., None], sand * 0.9 + (rust - sand) * 0.5 * (np.cos(med * 9) > 0)[..., None], img)
    img = np.where(((med > 1) & (med < 1.08))[..., None], ink, img)
    edge = np.minimum(1 - np.abs(u), 1 - np.abs(v))
    border = (edge > 0.035) & (edge < 0.16)
    vine = np.sin((u + v) * 40) * np.sin((u - v) * 40) > 0.2
    img = np.where(border[..., None], slate * 0.75 + (sand - slate * 0.75) * 0.5 * vine[..., None], img)
    img = np.where((((edge > 0.025) & (edge < 0.035)) | ((edge > 0.16) & (edge < 0.172)))[..., None], rust, img)
    img = np.where((edge < 0.025)[..., None], sand * 0.85, img)
    img += 9 * _noise((size, size), 0.7, rng)[..., None]                                    # pile
    wear = np.clip(_noise((size, size), 60, rng), 0, None)[..., None]
    img = img * (1 - 0.15 * wear) + 12 * wear                                              # worn, washed out
    img = 0.72 * img + 0.28 * img.mean(axis=2, keepdims=True)                              # faded colour
    return _save(name, img)


def recolour(src, rgb, contrast=1.5):
    """A photo texture's detail (its luminance about the mean, times contrast) on the colour rgb (0-255): Poly Haven's
    velvet is red and its linen blue; the room's are teal, charcoal and oatmeal."""
    a = np.asarray(Image.open(src).convert('RGB'), float)
    lum = a @ [0.299, 0.587, 0.114]
    detail = 1 + contrast * (lum / lum.mean() - 1)
    return _save(f'{Path(src).stem}_{"".join(f"{c:02x}" for c in rgb)}', detail[..., None] * np.array(rgb))


def tex_slats(name, size=512, n=24):
    """Window blinds: white slats with a soft shadow line under each."""
    yy = np.mgrid[0:size, 0:size][0]
    phase = (yy % (size / n)) / (size / n)
    v = 248 - 30 * np.exp(-((phase - 0.92) / 0.05) ** 2) - 6 * phase
    return _save(name, np.repeat(v[..., None], 3, axis=2))


# ---------------------------------------------------------------- geometry helpers

def quat_xyz(rx=0.0, ry=0.0, rz=0.0):
    """Quaternion (w, x, y, z) of rotations about x, then y, then z (world axes)."""
    def q(ax, a):
        c, s = math.cos(a / 2), math.sin(a / 2)
        return np.array([c] + [s if i == ax else 0.0 for i in range(3)])

    def mul(a, b):
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])
    return mul(q(2, rz), mul(q(1, ry), q(0, rx)))


def _box_mesh(half, tile, once=False):
    """A box's 24 vertices (4 per face, outward winding), 12 triangles and texture coordinates in tiles: each face
    is mapped by its own size, so a texture keeps its real scale on every face."""
    hx, hy, hz = half
    faces = [((0, 1, 2), +1), ((0, 1, 2), -1), ((1, 2, 0), +1), ((1, 2, 0), -1), ((2, 0, 1), +1), ((2, 0, 1), -1)]
    verts, tris, uv = [], [], []
    h = np.array(half)
    for (n, a, b), s in faces:
        base = len(verts)
        for ca, cb in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            p = np.zeros(3)
            p[n], p[a], p[b] = s * h[n], ca * h[a], cb * h[b]
            verts.append(p)
            uv.append(((ca + 1) / 2, (cb + 1) / 2) if once else (ca * h[a] / tile, cb * h[b] / tile))
        if s > 0:
            tris += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
        else:
            tris += [(base, base + 2, base + 1), (base, base + 3, base + 2)]
    return np.array(verts), np.array(tris), np.array(uv)


def _soft_mesh(half, tile, e=0.22, n=24):
    """A rounded box (a superellipsoid, exponent e: 0 is a box, 1 an ellipsoid) as vertices, triangles and texture
    coordinates in tiles (wrapped round it, so a fabric's weave keeps its scale)."""
    a = np.asarray(half, float)

    def sp(w, p):
        return math.copysign(abs(w) ** p, w)
    us = np.linspace(-math.pi, math.pi, 2 * n + 1)
    vs = np.linspace(-math.pi / 2, math.pi / 2, n + 1)
    verts, uv = [], []
    perim = 4 * (a[0] + a[1])
    for v in vs:
        for u in us:
            verts.append([a[0] * sp(math.cos(v), e) * sp(math.cos(u), e),
                          a[1] * sp(math.cos(v), e) * sp(math.sin(u), e), a[2] * sp(math.sin(v), e)])
            uv.append(((u + math.pi) / (2 * math.pi) * perim / tile,
                       (v + math.pi / 2) / math.pi * (2 * a[2] + a[0]) / tile))
    W = len(us)
    tris = []
    for i in range(n):
        for j in range(2 * n):
            k = i * W + j
            tris += [(k, k + 1, k + W + 1), (k, k + W + 1, k + W)]
    return np.array(verts), np.array(tris), np.array(uv)


class Room:
    def __init__(self, ms):
        self.ms, self.wb, self.n = ms, ms.worldbody, 0
        self.tile = {}

    def material(self, name, tex=None, tile=1.0, rgba=(1, 1, 1, 1), plane=False, **kw):
        """A material; with tex (an image), its texture repeats every `tile` metres."""
        if tex is not None:
            self.ms.add_texture(name=name, file=str(tex), type=mujoco.mjtTexture.mjTEXTURE_2D)
        m = self.ms.add_material(name=name, rgba=list(rgba), **kw)
        if tex is not None:
            m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name
            if plane:
                m.texuniform, m.texrepeat = True, [1 / tile, 1 / tile]
        self.tile[name] = tile
        return name

    def _geom(self, c, q, mat, rgba):
        self.n += 1
        g = self.wb.add_geom(name=f'room{self.n}', pos=list(c), contype=0, conaffinity=0, group=0)
        if q is not None:
            g.quat = list(q)
        if mat:
            g.material = mat
        if rgba is not None:
            g.rgba = list(rgba)
        return g

    def box(self, c, half, mat=None, rgba=None, q=None, once=False):
        """A box; textured materials get a mesh with real-scale texture coordinates (once: the texture stretched
        once over each face, as a rug's pattern)."""
        if mat in self.tile and self.ms.material(mat).textures[mujoco.mjtTextureRole.mjTEXROLE_RGB]:
            v, f, uv = _box_mesh(half, self.tile[mat], once)
            name = f'roommesh{self.n + 1}'
            self.ms.add_mesh(name=name, uservert=v.flatten().tolist(), userface=f.flatten().tolist(),
                             usertexcoord=uv.flatten().tolist())
            g = self._geom(c, q, mat, rgba)
            g.type, g.meshname = mujoco.mjtGeom.mjGEOM_MESH, name
            return g
        g = self._geom(c, q, mat, rgba)
        g.type, g.size = mujoco.mjtGeom.mjGEOM_BOX, list(half)
        return g

    def soft(self, c, half, mat, rgba=None, q=None, e=0.22):
        """A rounded box: upholstery and cushions."""
        v, f, uv = _soft_mesh(half, self.tile.get(mat, 0.5), e)
        name = f'roommesh{self.n + 1}'
        self.ms.add_mesh(name=name, uservert=v.flatten().tolist(), userface=f.flatten().tolist(),
                         usertexcoord=uv.flatten().tolist())
        g = self._geom(c, q, mat, rgba)
        g.type, g.meshname = mujoco.mjtGeom.mjGEOM_MESH, name
        return g

    def leaf(self, c, length, width, yaw, droop, rgba):
        """A leaf: a thin ellipsoid pointing out along yaw, drooping by droop (rad)."""
        g = self._geom(c, quat_xyz(ry=droop, rz=yaw), None, rgba)
        g.type, g.size = mujoco.mjtGeom.mjGEOM_ELLIPSOID, [length / 2, width / 2, 0.004]
        return g

    def cyl(self, c, r, hh, mat=None, rgba=None, kind=mujoco.mjtGeom.mjGEOM_CYLINDER, q=None):
        g = self._geom(c, q, mat, rgba)
        g.type, g.size = kind, [r, hh, 0]
        return g

    def wall(self, p0, p1, z0, z1, mat, t=0.1):
        """A vertical slab from p0 to p1 (x, y on the floor; its room side on the left going p0 -> p1)."""
        p0, p1 = np.array(p0, float), np.array(p1, float)
        d = p1 - p0
        L = np.linalg.norm(d)
        a = math.atan2(d[1], d[0])
        left = np.array([-d[1], d[0]]) / L
        c = (p0 + p1) / 2 - left * t / 2
        return self.box((c[0], c[1], (z0 + z1) / 2), (L / 2, t / 2, (z1 - z0) / 2), mat, q=quat_xyz(rz=a))


# ---------------------------------------------------------------- the room

def add_living_room(ms):
    R = Room(ms)
    ms.visual.global_.fovy = CAMERA['fovy']
    ms.visual.headlight.ambient = [0.40, 0.40, 0.40]
    ms.visual.headlight.diffuse = [0.32, 0.32, 0.32]
    ms.visual.headlight.specular = [0.02, 0.02, 0.02]
    ms.visual.quality.shadowsize = 4096
    ms.add_texture(name='sky', type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                   rgb1=[0.95, 0.96, 0.97], rgb2=[0.86, 0.88, 0.90], width=256, height=1536)

    floor = R.material('floor', recolour(CC0 / 'wood_floor.jpg', (118, 88, 62), 1.2), tile=2.0, plane=True,
                       reflectance=0.08, specular=0.3, shininess=0.5)
    wall = R.material('wall', recolour(CC0 / 'plastered_wall_04.jpg', (236, 234, 229), 1.0), tile=2.5,
                      specular=0.02, emission=0.3)
    plaster = R.material('plaster', recolour(CC0 / 'plaster_grey_04.jpg', (170, 165, 154), 2.0), tile=1.6,
                         specular=0.05, emission=0.15)
    ceil = R.material('ceil', recolour(CC0 / 'white_planks_clean.jpg', (242, 242, 240), 1.2), tile=1.2,
                      specular=0.02, emission=0.45)
    beam = R.material('beam', recolour(CC0 / 'white_oak_veneer.jpg', (196, 160, 118), 1.6), tile=1.0, specular=0.1,
                      emission=0.2)
    walnut = R.material('walnut', recolour(CC0 / 'american_walnut_veneer.jpg', (128, 86, 56), 2.5), tile=0.8,
                        specular=0.25, shininess=0.4)
    dark = R.material('darkwood', recolour(CC0 / 'black_walnut_veneer_01.jpg', (62, 47, 40), 1.5), tile=0.8,
                      specular=0.2, shininess=0.4)
    linen = R.material('linen', recolour(CC0 / 'rough_linen.jpg', (204, 198, 188), 1.3), tile=0.5, specular=0.02,
                       emission=0.05)
    teal = R.material('teal', recolour(CC0 / 'velour_velvet.jpg', (72, 108, 116), 1.6), tile=0.4, specular=0.08)
    charcoal = R.material('charcoal', recolour(CC0 / 'velour_velvet.jpg', (58, 62, 70), 1.6), tile=0.4,
                          specular=0.06)
    rug = R.material('rug', tex_rug('rug'), tile=4.9, specular=0.02)
    slats = R.material('slats', tex_slats('slats'), tile=1.2, emission=0.55, specular=0.0)
    white = R.material('white', rgba=(0.93, 0.93, 0.91, 1), specular=0.2, emission=0.05)
    black = R.material('black', rgba=(0.04, 0.04, 0.045, 1), specular=0.6, shininess=0.9)

    # shell: floor, walls, a vaulted ceiling in two slopes meeting at the ridge
    ms.worldbody.add_geom(name='room floor', type=mujoco.mjtGeom.mjGEOM_PLANE, pos=[5, -1, 0], size=[8, 8, 0.05],
                          material=floor, contype=0, conaffinity=0)
    turn = (RIGHT_TURN_X, RIGHT_BACK_Y)
    near = (FRONT_X, RIGHT_BACK_Y - (RIGHT_TURN_X - FRONT_X) * math.tan(RIGHT_ANGLE))
    R.wall((BACK_X, RIGHT_BACK_Y - 0.1), (BACK_X, LEFT_Y + 0.1), 0, RIDGE_Z + 0.2, wall)      # fireplace wall
    R.wall((BACK_X + 0.1, LEFT_Y), (FRONT_X, LEFT_Y), 0, EAVE_Z + 0.05, wall)                # sofa wall
    R.wall(turn, (BACK_X + 0.1, RIGHT_BACK_Y), 0, EAVE_Z + 0.05, wall)                       # right, back stretch
    R.wall(near, turn, 0, EAVE_Z + 0.05, wall)                                              # the angled TV wall
    R.wall((FRONT_X, LEFT_Y), (FRONT_X, near[1]), 0, RIDGE_Z, wall)                          # behind the camera
    span_x, cx = BACK_X - FRONT_X, 0.5 * (BACK_X + FRONT_X)
    for y_wall in (LEFT_Y, RIGHT_BACK_Y):
        run, rise = abs(y_wall - RIDGE_Y), RIDGE_Z - EAVE_Z
        ang = math.atan2(rise, run) * (-1 if y_wall > RIDGE_Y else 1)
        mid_y, mid_z = 0.5 * (y_wall + RIDGE_Y), 0.5 * (EAVE_Z + RIDGE_Z)
        q = quat_xyz(rx=ang)
        R.box((cx, mid_y, mid_z + 0.03), (span_x / 2, math.hypot(run, rise) / 2 + 0.05, 0.03), ceil, q=q)
        for bx in (2.2, 5.3, 8.4):                                                           # rafters down each slope
            R.box((bx, mid_y, mid_z - 0.10), (0.09, math.hypot(run, rise) / 2, 0.12), beam, q=q)
    R.box((cx, RIDGE_Y, RIDGE_Z - 0.14), (span_x / 2, 0.1, 0.15), beam)                      # ridge beam
    alcove_y = 0.5 * (RIGHT_BACK_Y + near[1])                                                # flat ceiling over the
    R.box((0.5 * (FRONT_X + RIGHT_TURN_X), alcove_y, EAVE_Z + 0.03),                        # TV wall's widening
          ((RIGHT_TURN_X - FRONT_X) / 2, abs(near[1] - RIGHT_BACK_Y) / 2, 0.03), ceil)

    # fireplace (u 548-736): a grey plaster column to the ridge, the opening (u 594-688, 0.72 m) in a surround
    R.box((BACK_X - 0.15, 0.085, RIDGE_Z / 2), (0.15, 0.985, RIDGE_Z / 2), plaster)
    R.box((BACK_X - 0.31, 0.095, 0.42), (0.02, 0.61, 0.42), plaster)
    R.box((BACK_X - 0.32, 0.095, 0.36), (0.015, 0.495, 0.36), rgba=(0.07, 0.065, 0.06, 1))
    R.box((BACK_X - 0.45, 0.095, 0.025), (0.15, 0.75, 0.025), plaster)                      # hearth
    # tall windows with blinds (u 418-543 and 737-865, from 0.15 m to 2.08 m), white frames
    for y0, y1 in ((2.43, 1.12), (-0.91, -2.25)):
        yc, hw = 0.5 * (y0 + y1), 0.5 * abs(y0 - y1)
        R.box((BACK_X - 0.02, yc, 1.115), (0.02, hw + 0.04, 1.0), white)
        R.box((BACK_X - 0.045, yc, 1.115), (0.008, hw - 0.02, 0.95), slats)

    # sofa on the left wall (feet at u 322 and 475: x 4.56-8.71, front y 1.37); seat 0.45, back 0.85
    sx0, sx1, sy0 = 4.5, 8.75, 1.32
    sxc, sl, sd = 0.5 * (sx0 + sx1), 0.5 * (sx1 - sx0), 0.5 * (LEFT_Y - 0.02 - sy0)
    syc = sy0 + sd
    R.soft((sxc, syc, 0.25), (sl, sd, 0.13), linen, e=0.12)
    for k in range(3):
        c = sx0 + 0.15 + (k + 0.5) * (2 * sl - 0.3) / 3
        R.soft((c, syc - 0.05, 0.42), ((2 * sl - 0.3) / 6 - 0.005, sd - 0.12, 0.075), linen, e=0.3)
        R.soft((c, LEFT_Y - 0.18, 0.66), ((2 * sl - 0.3) / 6 - 0.005, 0.13, 0.21), linen, q=quat_xyz(rx=-0.12),
               e=0.35)
    for e in (sx0 + 0.075, sx1 - 0.075):
        R.soft((e, syc, 0.43), (0.075, sd, 0.19), linen, e=0.3)
    for x, col in ((sx0 + 0.45, (0.55, 0.57, 0.52, 1)), (sx0 + 0.9, (0.78, 0.76, 0.70, 1)),
                   (sx1 - 0.55, (0.60, 0.58, 0.50, 1))):
        R.soft((x, LEFT_Y - 0.32, 0.67), (0.23, 0.08, 0.21), linen, rgba=col, q=quat_xyz(rx=-0.3, rz=0.15), e=0.5)
    for ex in (sx0 + 0.06, sx1 - 0.06):
        for ey in (sy0 + 0.06, LEFT_Y - 0.08):
            R.cyl((ex, ey, 0.05), 0.02, 0.05, black)

    # side table at the sofa's near end (front leg at u 275: x 4.31, y 1.47), with a lamp base, books and a plant
    tx, ty = 4.0, 1.97
    R.box((tx, ty, 0.6), (0.3, 0.48, 0.02), dark)
    R.box((tx, ty, 0.2), (0.28, 0.46, 0.015), dark)
    for dx in (-0.28, 0.28):
        for dy in (-0.46, 0.46):
            R.box((tx + dx, ty + dy, 0.3), (0.015, 0.015, 0.3), rgba=(0.1, 0.09, 0.08, 1))
    R.cyl((tx + 0.05, ty - 0.2, 0.7), 0.08, 0.08, white)
    R.box((tx - 0.05, ty + 0.15, 0.64), (0.13, 0.1, 0.02), rgba=(0.85, 0.83, 0.78, 1), q=quat_xyz(rz=0.3))
    R.box((tx, ty, 0.24), (0.14, 0.11, 0.025), rgba=(0.82, 0.62, 0.22, 1), q=quat_xyz(rz=-0.2))
    R.cyl((tx - 0.1, ty + 0.3, 0.68), 0.07, 0.06, white)
    rng = np.random.default_rng(5)
    for k in range(22):
        a = k * 2.4
        L = rng.uniform(0.1, 0.18)
        R.leaf((tx - 0.1 + 0.5 * L * math.cos(a), ty + 0.3 + 0.5 * L * math.sin(a), 0.76 + rng.uniform(0, 0.08)), L,
               0.035, a, rng.uniform(0.1, 0.7), (0.30, 0.44, 0.27, 1))

    # frames on the sofa wall (u 188-243 and 278-318: x 5.95-6.74 and 7.39-8.27, z 1.39-1.93)
    for x0, x1 in ((5.95, 6.74), (7.39, 8.27)):
        xc, hx = 0.5 * (x0 + x1), 0.5 * (x1 - x0) * 0.62
        R.box((xc, LEFT_Y - 0.015, 1.66), (hx, 0.015, 0.29), rgba=(0.1, 0.09, 0.08, 1))
        R.box((xc, LEFT_Y - 0.032, 1.66), (hx - 0.04, 0.003, 0.25), rgba=(0.95, 0.94, 0.91, 1))
        R.box((xc, LEFT_Y - 0.036, 1.66), (hx * 0.35, 0.002, 0.1), rgba=(0.62, 0.55, 0.48, 1))

    # rug (corners at u 255 / 910 near, 470 / 775 far: x 4.0-8.95, y -1.0-1.45) and the coffee table on it
    # (front legs u 598 / 694: x 5.19, y -0.21-0.24; the top's far edge at v 445: x 7.64; 0.43 high)
    R.box((6.48, 0.22, 0.0015), (2.47, 1.22, 0.0015), rug, once=True)
    cx0, cx1 = 5.19, 7.64
    R.box((0.5 * (cx0 + cx1), 0.015, 0.41), (0.5 * (cx1 - cx0), 0.24, 0.025), walnut)
    for e in (cx0 + 0.03, cx1 - 0.03):
        R.box((e, 0.015, 0.2), (0.03, 0.22, 0.2), walnut)
    R.box((0.5 * (cx0 + cx1), 0.015, 0.1), (0.5 * (cx1 - cx0) - 0.06, 0.2, 0.015), walnut)

    # dark armchair and tall plant by the right window (legs at u 750 / 818: around x 8.0-9.3, y -0.8 - -1.2)
    ax_, ay = 9.0, -1.55
    q = quat_xyz(rz=math.radians(-35))
    R.soft((ax_, ay, 0.3), (0.38, 0.4, 0.08), charcoal, q=q, e=0.2)
    R.soft((ax_, ay, 0.43), (0.34, 0.36, 0.07), charcoal, q=q, e=0.35)
    back = np.array([ax_, ay]) + np.array([math.cos(math.radians(-35)), math.sin(math.radians(-35))]) * 0.33
    R.soft((back[0], back[1], 0.64), (0.07, 0.4, 0.26), charcoal, q=q, e=0.35)
    R.soft((ax_ + 0.1, ay - 0.06, 0.6), (0.08, 0.2, 0.14), linen, rgba=(0.75, 0.73, 0.66, 1), q=q, e=0.5)
    for dx, dy in ((-0.3, -0.33), (-0.3, 0.33), (0.3, -0.33), (0.3, 0.33)):
        rot = np.array([[math.cos(-0.61), -math.sin(-0.61)], [math.sin(-0.61), math.cos(-0.61)]])
        p = np.array([ax_, ay]) + rot @ [dx, dy]
        R.cyl((p[0], p[1], 0.11), 0.012, 0.11, black)
    px, py = 10.84, -1.91                                         # a fiddle-leaf fig: big leaves up a bare stem
    R.cyl((px, py, 0.18), 0.17, 0.18, rgba=(0.88, 0.87, 0.84, 1))
    R.cyl((px, py, 1.0), 0.012, 0.82, rgba=(0.32, 0.25, 0.18, 1))
    rng = np.random.default_rng(7)
    for k in range(22):
        h = 0.8 + 1.0 * k / 21
        a = k * 2.4 + rng.uniform(-0.3, 0.3)
        L = rng.uniform(0.2, 0.3)
        R.leaf((px + 0.5 * L * math.cos(a), py + 0.5 * L * math.sin(a), h), L, 0.15, a, rng.uniform(-0.2, 0.5),
               (0.1, 0.18 + 0.05 * rng.uniform(), 0.11, 1))

    # TV on a white console along the angled wall (the console's front at u 880 / 1140: (9.68, -2.01) to (7.34, -3.24))
    p0, p1 = np.array([9.68, -2.01]), np.array([7.34, -3.24])
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    inward = np.array([-d[1], d[0]])
    if inward @ (np.array([6.0, 0.0]) - p0) < 0:
        inward = -inward
    yaw = math.atan2(d[1], d[0])
    c = 0.5 * (p0 + p1) - inward * 0.22
    L = np.linalg.norm(p1 - p0) / 2
    R.box((c[0], c[1], 0.3), (L, 0.22, 0.26), white, q=quat_xyz(rz=yaw))
    R.box((c[0], c[1], 0.035), (L - 0.05, 0.2, 0.035), rgba=(0.2, 0.2, 0.2, 1), q=quat_xyz(rz=yaw))
    tv = c - inward * 0.08
    R.box((tv[0], tv[1], 1.2), (0.95, 0.025, 0.55), rgba=(0.03, 0.03, 0.035, 1), q=quat_xyz(rz=yaw))   # 85 in
    R.box((tv[0], tv[1], 0.6), (0.3, 0.1, 0.02), black, q=quat_xyz(rz=yaw))

    # teal armchair in the near right corner (a front leg at u 885, v 705: x 3.6, y -0.77), facing the sofa
    tq = quat_xyz(rz=math.radians(105))
    tc = np.array([4.05, -1.6])
    R.soft((tc[0], tc[1], 0.27), (0.46, 0.48, 0.15), teal, q=tq, e=0.2)
    R.soft((tc[0], tc[1], 0.47), (0.4, 0.38, 0.08), teal, q=tq, e=0.35)
    bk = tc + np.array([math.cos(math.radians(105)), math.sin(math.radians(105))]) * -0.4
    R.soft((bk[0], bk[1], 0.66), (0.12, 0.48, 0.3), teal, q=tq, e=0.4)
    for e in (-1, 1):
        arm = tc + np.array([-math.sin(math.radians(105)), math.cos(math.radians(105))]) * 0.4 * e
        R.soft((arm[0], arm[1], 0.52), (0.46, 0.09, 0.2), teal, q=tq, e=0.4)
    R.soft((tc[0] - 0.12, tc[1] + 0.03, 0.66), (0.1, 0.24, 0.15), linen, rgba=(0.82, 0.84, 0.86, 1), q=tq, e=0.5)
    for dx, dy in ((-0.38, -0.4), (-0.38, 0.4), (0.38, -0.4), (0.38, 0.4)):
        rot = np.array([[math.cos(1.83), -math.sin(1.83)], [math.sin(1.83), math.cos(1.83)]])
        p = tc + rot @ [dx, dy]
        R.cyl((p[0], p[1], 0.06), 0.015, 0.06, black)

    # daylight through the back windows, and a soft fill from the ceiling
    sun = ms.worldbody.add_light(name='window sun', dir=[-0.75, -0.1, -0.5], diffuse=[0.5, 0.49, 0.47],
                                 specular=[0.08, 0.08, 0.08], castshadow=True, pos=[14, 0, 5])
    sun.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    for lx in (3.0, 6.5, 9.5):
        fill = ms.worldbody.add_light(name=f'fill{lx}', pos=[lx, 0.0, 3.1], dir=[0, 0, -1],
                                      diffuse=[0.16, 0.16, 0.16], specular=[0, 0, 0], castshadow=False,
                                      attenuation=[1, 0.05, 0.0], cutoff=85)
        fill.type = mujoco.mjtLightType.mjLIGHT_SPOT
    return R


# ---------------------------------------------------------------- echo's poses in the room

def pose_q(kind):
    """Joint angles by name. Leg angles from absolute segment pitches (thigh forward T, shin back S, pelvis lean P),
    so the feet stay flat: hip = -T - P, knee = S + T, ankle = -S."""
    q = {}
    T, S, P = {'stand': (0.0, 0.0, 0.0), 'carry': (0.08, 0.1, 0.02), 'reach': (0.45, 0.35, 0.9)}[kind]
    for s in 'LR':
        q[f'{s}_hip_pitch'], q[f'{s}_knee'], q[f'{s}_ankle_pitch'] = -T - P, S + T, -S
    if kind == 'carry':                   # a basket held in front at the waist
        for s, sg in (('L', 1), ('R', -1)):
            q[f'{s}_shoulder_pitch'], q[f'{s}_shoulder_roll'], q[f'{s}_elbow'] = -0.35, sg * 0.32, -1.3
            q[f'{s}_wrist_yaw'] = -sg * 0.4
    if kind == 'reach':                   # bending to pick laundry off the sofa seat
        for s, sg in (('L', 1), ('R', -1)):
            q[f'{s}_shoulder_pitch'], q[f'{s}_shoulder_roll'], q[f'{s}_elbow'] = -0.55, sg * 0.15, -0.35
        q['neck_pitch'] = 0.3
    return q, P


def place(m, d, q, lean, xy, yaw):
    """Set the joints, lean the pelvis, then put the lowest foot point on the floor under xy."""
    for n, v in q.items():
        d.qpos[m.jnt_qposadr[m.joint(n).id]] = v
    d.qpos[3:7] = quat_xyz(ry=lean, rz=yaw)             # lean forward about the body's y, then turn
    d.qpos[:3] = [0, 0, 1.0]
    mujoco.mj_kinematics(m, d)
    lo, feet = 1e9, []
    for b in ('L_foot', 'R_foot'):
        bid = m.body(b).id
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != bid or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_group[g] != 1:
                continue
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            w = d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T
            lo = min(lo, w[:, 2].min())
            feet.append(w[:, :2].mean(0))
    mid_feet = np.mean(feet, axis=0)
    d.qpos[0] += xy[0] - mid_feet[0]
    d.qpos[1] += xy[1] - mid_feet[1]
    d.qpos[2] -= lo
    mujoco.mj_kinematics(m, d)
    mujoco.mj_camlight(m, d)


def camera(view):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    if view == 'clip':
        p = np.array(CAMERA['pos'])
        l = p + 4.0 * np.array([math.cos(CAMERA['pitch']), 0.0, -math.sin(CAMERA['pitch'])])
    else:                                  # a 3/4 view from the teal chair's corner, toward the sofa and fireplace
        p, l = np.array([2.2, -1.9, 1.7]), np.array([7.0, 1.2, 0.8])
    dv = l - p
    cam.lookat[:] = l
    cam.distance = float(np.linalg.norm(dv))
    cam.azimuth = math.degrees(math.atan2(dv[1], dv[0]))
    cam.elevation = math.degrees(math.asin(dv[2] / np.linalg.norm(dv)))
    return cam


def render_still(m, kind, view, out, w=1280, h=720):
    d = mujoco.MjData(m)
    q, lean = pose_q(kind)
    spot = {'stand': ((6.2, 0.75), math.pi / 2), 'carry': ((6.2, 0.75), math.pi / 2),
            'reach': ((6.4, 0.7), math.pi / 2)}[kind]
    place(m, d, q, lean, *spot)
    r = mujoco.Renderer(m, h, w)
    r.update_scene(d, camera(view))
    img = Image.fromarray(r.render())
    r.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pose', choices=['stand', 'carry', 'reach'], default=None)
    ap.add_argument('--view', choices=['clip', '34'], default=None)
    a = ap.parse_args()
    m = RM.build_view_model(room=add_living_room)
    for kind in [a.pose] if a.pose else ['stand', 'reach']:
        for view in [a.view] if a.view else ['clip', '34']:
            print(render_still(m, kind, view, OUT / f'living_room_{kind}_{view}.png'))


if __name__ == '__main__':
    main()
