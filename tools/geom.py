"""Shapes from spec/robot.json: watertight meshes, exact point-inside tests, and the robot's forward kinematics.

Everything is in the zero-pose world frame (metres). A link's pose is a 4x4 matrix M that maps its zero-pose
world coordinates to posed world coordinates.
"""
import json
import os

import numpy as np
import trimesh
from scipy.interpolate import PchipInterpolator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AXIS = {'x': np.array([1.0, 0, 0]), 'y': np.array([0, 1.0, 0]), 'z': np.array([0, 0, 1.0])}
RIGID = ('struct', 'actuator')


def rbox_fillet(g):
    """An rbox's fillet, never more than half its thinnest side. Past that the corner spheres cross over: 3 x 46 x 45 with a
    2 mm fillet meshed 5 mm thick while inside() read about 3, so mass and contact used one thickness and the stress notes
    another (audit item 18, 2026-11-05). A box can't have a fillet bigger than half of itself, so this is the shape meant."""
    return min(g['fillet'], min(g['size']) / 2)


def load(path=None):
    return json.load(open(path or os.path.join(ROOT, 'spec', 'robot.json'), encoding='utf-8'))


# ---------------------------------------------------------------- bought motors at their real outline

_OUTLINES = {}
_MOTOR_LOCAL = {}


def outline_data(model):
    """spec/actuator_outlines.json (tools/motor_outlines.py, measured from the maker's STEP): bands of radius per angle along the axis."""
    if not _OUTLINES:
        _OUTLINES.update(json.load(open(os.path.join(ROOT, 'spec', 'actuator_outlines.json'), encoding='utf-8')))
    return _OUTLINES[model]


def _star(rad):
    n = len(rad)
    th = (np.arange(n) + 0.5) * 2 * np.pi / n
    r = np.maximum(np.asarray(rad, float), 0.5)
    return np.column_stack([r * np.cos(th), r * np.sin(th)])


def _motor_local(model):
    """The motor as one manifold in its own frame, metres: STEP x, STEP y, t toward the output."""
    import manifold3d as mf
    if model in _MOTOR_LOCAL:
        return _MOTOR_LOCAL[model]
    d = outline_data(model)
    ms = []
    for k, (t0, t1, rad) in enumerate(d['bands']):
        # a hair of overlap along the axis, so the stack unions into one body (hidden inside the bigger band)
        lo, hi = t0 - (0.01 if k else 0.0), t1
        ms.append(mf.CrossSection([_star(rad)]).extrude(hi - lo).translate((0.0, 0.0, lo)))
    p = d.get('pins')
    if p:
        for a in p['angles']:
            x, y = p['bc'] / 2 * np.cos(np.radians(a)), p['bc'] / 2 * np.sin(np.radians(a))
            ms.append(mf.Manifold.cylinder(p['height'] + 0.01, p['d'] / 2, p['d'] / 2, 32).translate((float(x), float(y), d['t_hub_face'] - 0.01)))
    u = mf.Manifold.batch_boolean(ms, mf.OpType.Add).scale((0.001, 0.001, 0.001))
    _MOTOR_LOCAL[model] = u
    return u


def motor_axes(g):
    """Unit vectors (STEP x, STEP y, output) in the world, from a motor geom's 'outline' frame."""
    o = g['outline']
    return np.array(o['x'], float), np.array(o['y'], float), np.array(o['out'], float)


def motor_point(g, r_mm, ang_deg, t_mm):
    """World point on the motor: radius and angle in the STEP's own polar frame, and t along the output from the middle of h."""
    ex, ey, eo = motor_axes(g)
    a = np.radians(ang_deg)
    return np.asarray(g['c'], float) + (r_mm * np.cos(a) * ex + r_mm * np.sin(a) * ey + t_mm * eo) / 1000.0


def motor_mesh(g):
    ex, ey, eo = motor_axes(g)
    k = g['r'] / g['outline']['r0']                   # a motor enlarged in a --bad case grows across its axis, as the plain cylinder did
    T = np.column_stack([k * ex, k * ey, eo, np.asarray(g['c'], float)])
    m = _motor_local(g['outline']['model']).transform(T.astype(np.float64)).to_mesh()
    return trimesh.Trimesh(np.asarray(m.vert_properties[:, :3], float), np.asarray(m.tri_verts, int))


def motor_inside(g, p):
    d = outline_data(g['outline']['model'])
    ex, ey, eo = motor_axes(g)
    k = g['r'] / g['outline']['r0']
    q = np.atleast_2d(p) - np.asarray(g['c'], float)
    x, y, t = q @ ex * 1000 / k, q @ ey * 1000 / k, q @ eo * 1000
    r = np.hypot(x, y)
    ang = np.arctan2(y, x) % (2 * np.pi)
    ok = np.zeros(len(q), bool)
    for t0, t1, rad in d['bands']:
        n = len(rad)
        sel = (t >= t0) & (t <= t1)
        if sel.any():
            f = ang[sel] / (2 * np.pi) * n - 0.5
            i0 = np.floor(f).astype(int) % n
            w = f - np.floor(f)
            rad = np.asarray(rad, float)
            ok[sel] = r[sel] <= rad[i0] * (1 - w) + rad[(i0 + 1) % n] * w
    pin = d.get('pins')
    if pin:
        for a in pin['angles']:
            px, py = pin['bc'] / 2 * np.cos(np.radians(a)), pin['bc'] / 2 * np.sin(np.radians(a))
            ok |= (np.hypot(x - px, y - py) <= pin['d'] / 2) & (t >= d['t_hub_face']) & (t <= d['t_hub_face'] + pin['height'])
    return ok


# ---------------------------------------------------------------- lofts

def _loft_params(g):
    s = g['sections']
    zs = np.array([q['z'] for q in s])
    keys = ('cx', 'cy', 'a', 'b', 'n')
    vals = {k: np.array([q[k] for q in s]) for k in keys}
    if len(s) >= 3:
        f = {k: PchipInterpolator(zs, vals[k]) for k in keys}
    else:
        f = {k: (lambda z, k=k: np.interp(z, zs, vals[k])) for k in keys}
    return zs[0], zs[-1], f


def superellipse(cx, cy, a, b, n, m=64):
    t = np.linspace(0, 2 * np.pi, m, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    return np.stack([cx + a * np.sign(c) * np.abs(c) ** (2 / n), cy + b * np.sign(s) * np.abs(s) ** (2 / n)], 1)


def loft_mesh(g, levels=40, m=64):
    z0, z1, f = _loft_params(g)
    zs = np.linspace(z0, z1, levels)
    rings = []
    for z in zs:
        xy = superellipse(*(float(f[k](z)) for k in ('cx', 'cy', 'a', 'b', 'n')), m=m)
        rings.append(np.column_stack([xy, np.full(m, z)]))
    ends = [[float(f['cx'](z0)), float(f['cy'](z0)), z0], [float(f['cx'](z1)), float(f['cy'](z1)), z1]]
    v = np.vstack(rings + [np.array(ends)])
    faces = []
    for i in range(levels - 1):
        for k in range(m):
            a, b = i * m + k, i * m + (k + 1) % m
            c, d = a + m, b + m
            faces += [[a, b, d], [a, d, c]]
    bot, top = levels * m, levels * m + 1
    for k in range(m):
        faces.append([bot, (k + 1) % m, k])
        faces.append([top, (levels - 1) * m + k, (levels - 1) * m + (k + 1) % m])
    mesh = trimesh.Trimesh(v, np.array(faces), process=True)
    mesh.fix_normals()
    return mesh


def loft_inside(g, p):
    z0, z1, f = _loft_params(g)
    p = np.atleast_2d(p)
    z = np.clip(p[:, 2], z0, z1)
    cx, cy, a, b, n = (np.asarray(f[k](z), float) for k in ('cx', 'cy', 'a', 'b', 'n'))
    r = np.abs((p[:, 0] - cx) / a) ** n + np.abs((p[:, 1] - cy) / b) ** n
    return (r <= 1.0) & (p[:, 2] >= z0) & (p[:, 2] <= z1)


# ---------------------------------------------------------------- other primitives

def axis_vec(a):
    """A cylinder's axis: 'x' / 'y' / 'z', or any direction [x, y, z] (decals that follow a sloping surface)."""
    if isinstance(a, str):
        return AXIS[a]
    a = np.asarray(a, float)
    return a / np.linalg.norm(a)


def _frame_to(axis_vec):
    """Rotation taking +Z to axis_vec."""
    z = np.array([0, 0, 1.0])
    v = np.cross(z, axis_vec)
    s, c = np.linalg.norm(v), float(np.dot(z, axis_vec))
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1, -1, -1.0])
    k = v / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def clips(g):
    """g['clip'] as a list of {'p': point, 'n': normal}: one plane (a visor cut from a head) or several (a shoe's
    collar, a band between two planes); the part kept is in front of every one."""
    c = g.get('clip')
    return [] if not c else ([c] if isinstance(c, dict) else c)


def clipped(g, mesh):
    """Keep the part of mesh in front of all of g's clip planes, and inside its 'keep' sphere ({'c', 'r'}: a domed
    top, round about a vertical line through c, so a joint turning about that line never sweeps it)."""
    if not clips(g) and not g.get('keep'):
        return mesh
    import manifold3d as mf
    m = mf.Manifold(mf.Mesh(vert_properties=np.asarray(mesh.vertices, np.float32),
                            tri_verts=np.asarray(mesh.faces, np.uint32)))
    for c in clips(g):
        n = np.asarray(c['n'], float)
        n /= np.linalg.norm(n)
        m = m.trim_by_plane(n.tolist(), float(n @ np.asarray(c['p'], float)))
    if g.get('keep'):
        m = m ^ mf.Manifold.sphere(g['keep']['r'], 512).translate(list(map(float, g['keep']['c'])))
    out = m.to_mesh()
    return trimesh.Trimesh(out.vert_properties[:, :3], out.tri_verts)


def _revolve_r(g, t):
    """Radius of a solid of revolution at axial position t (negative outside its length)."""
    ts = [q[0] for q in g['profile']]
    rs = [q[1] for q in g['profile']]
    return np.where((t >= min(ts)) & (t <= max(ts)), np.interp(t, ts, rs), -1.0)


def sdf_mesh(g, hi=False):
    """A machined part from its distance tree (tools/sdf.py): the checks' mesh, or with hi the finer one the renderers use. A right side's part is the left's mesh reflected."""
    import sdf as SDF
    if hi:
        vox = g.get('voxel_hi', 0.0005)
        v, f = SDF.cached(g['tree'], vox, vox * 0.004)
    else:
        vox = g.get('voxel', 0.001)
        v, f = SDF.cached(g['tree'], vox, vox * 0.03)
    if g.get('flip_y'):
        v = v * np.array([1.0, -1.0, 1.0])
        f = f[:, ::-1]
    return trimesh.Trimesh(np.asarray(v, float), np.asarray(f, int), process=False)


FINISH = {'silver': ((0.86, 0.865, 0.875), 1.0, 0.30), 'black': ((0.035, 0.036, 0.04), 0.9, 0.36), 'bronze': ((0.70, 0.52, 0.31), 1.0, 0.34)}


def finish_of(g, pts=None):
    """A machined part's finish: its name, or for points, an array of names by its zones (masked anodising). A right side's zone is the left's reflected."""
    base = g.get('finish') or ('black' if g.get('role') in ('actuator',) else 'silver')
    if pts is None:
        return base
    out = np.full(len(pts), base, dtype=object)
    q = np.asarray(pts, float) * (np.array([1.0, -1.0, 1.0]) if g.get('flip_y') else 1.0)
    for z in g.get('zones', []):
        i = 'xyz'.index(z['axis'])
        d = q - np.asarray(z['c'], float)
        t = d[:, i]
        rr = np.sqrt(np.maximum((d * d).sum(1) - t * t, 0.0))
        out[(rr <= z['r']) & (t >= z.get('t0', -9.0)) & (t <= z.get('t1', 9.0))] = z['finish']
    return out


def mesh_of(g):
    k = g['kind']
    if g.get('outline'):
        return motor_mesh(g)
    if k == 'sdf':
        return sdf_mesh(g)
    if k == 'loft':
        return clipped(g, loft_mesh(g))
    if k == 'revolve':
        # profile (t, r): t along the axis (absolute coordinate), r its radius; first and last r are 0
        pts = np.array([[r, t] for t, r in g['profile']], float)
        m = trimesh.creation.revolve(pts, sections=64)
        T = np.eye(4)
        T[:3, :3] = _frame_to(axis_vec(g['axis']))
        T[:3, 3] = g['c']
        m.apply_transform(T)
        m.fix_normals()
        return m
    if k == 'prism':
        # a convex polygon in the plane x = x0, points (y, z), extruded t along +x
        P = np.array(g['pts'], float)
        n = len(P)
        v = np.vstack([np.column_stack([np.full(n, g['x0']), P]),
                       np.column_stack([np.full(n, g['x0'] + g['t']), P])])
        f = [[0, i, i + 1] for i in range(1, n - 1)] + [[n, n + i + 1, n + i] for i in range(1, n - 1)]
        for i in range(n):
            j = (i + 1) % n
            f += [[i, n + i, n + j], [i, n + j, j]]
        m = trimesh.Trimesh(v, np.array(f), process=True)
        m.fix_normals()
        return m
    if k == 'ellipsoid':
        m = trimesh.creation.icosphere(subdivisions=4)
        m.apply_scale(g['r'])
        m.apply_translation(g['c'])
        return m
    if k == 'cyl':
        m = trimesh.creation.cylinder(radius=g['r'], height=g['h'], sections=g.get('segs', 64))
        T = np.eye(4)
        T[:3, :3] = _frame_to(axis_vec(g['axis']))
        T[:3, 3] = g['c']
        m.apply_transform(T)
        return m
    if k == 'rbox':
        f = rbox_fillet(g)
        half = np.array(g['size']) / 2 - f
        s = trimesh.creation.icosphere(subdivisions=3, radius=f).vertices
        pts = np.vstack([s + np.array(g['c']) + half * np.array(sg) for sg in
                         [(i, j, l) for i in (-1, 1) for j in (-1, 1) for l in (-1, 1)]])
        return trimesh.convex.convex_hull(pts)
    if k == 'capsule':
        p0, p1 = np.array(g['p0']), np.array(g['p1'])
        L = np.linalg.norm(p1 - p0)
        m = trimesh.creation.capsule(height=L, radius=g['r'], count=[32, 32])
        T = np.eye(4)
        T[:3, :3] = _frame_to((p1 - p0) / L)
        T[:3, 3] = (p0 + p1) / 2
        m.apply_transform(T)
        return m
    raise ValueError(k)


def inner_of(g, t):
    """The cavity of a skin of wall thickness t: g shrunk by t. A loft's open ends (g['open'], default both:
    lofts are sleeves) run 2 cm past the end so the wall has no cap there."""
    g = json.loads(json.dumps(g))
    k = g['kind']
    if k == 'loft':
        op = g.get('open', ['top', 'bottom'])
        for s in g['sections']:
            s['a'] -= t
            s['b'] -= t
        lo, hi = g['sections'][0], g['sections'][-1]
        ext_lo, ext_hi = dict(lo), dict(hi)
        ext_lo['z'] = lo['z'] - 0.02 if 'bottom' in op else lo['z'] + t
        ext_hi['z'] = hi['z'] + 0.02 if 'top' in op else hi['z'] - t
        # a closed end moved in by t can pass the next section (a small pad's sections are closer than the wall)
        mid = [s for s in g['sections'][1:-1] if ext_lo['z'] < s['z'] < ext_hi['z']]
        g['sections'] = [ext_lo] + mid + [ext_hi]
    elif k == 'ellipsoid':
        g['r'] = [r - t for r in g['r']]
    elif k == 'revolve':
        prof = g['profile']
        g['profile'] = [[prof[0][0] + t, 0.0]] + [[q[0], max(q[1] - t, 0.0005)] for q in prof[1:-1]] + \
                       [[prof[-1][0] - t, 0.0]]
    elif k == 'rbox':
        g['size'] = [s - 2 * t for s in g['size']]
        g['fillet'] = max(g['fillet'] - t, 0.001)
    elif k == 'cyl':
        g['r'] -= t
        g['h'] -= 2 * t
    elif k == 'capsule':
        g['r'] -= t
    return g


def inner_closed(g, t):
    """inner_of with each clip plane moved t into the part, so the skin has a wall on its flat cut faces too: Fusion
    draws those faces as solid surfaces (a hip lobe's flat inner face sliced through the hip motors, 2026-10-31)."""
    g = inner_of(g, t)
    for c in clips(g):
        n = np.asarray(c['n'], float)
        c['p'] = list(np.asarray(c['p'], float) + t * n / np.linalg.norm(n))
    if g.get('keep'):
        g['keep']['r'] -= t
    return g


def inside(g, p):
    """Exact point-in-solid test for the primitive g, for an (N, 3) array of points."""
    p = np.atleast_2d(np.asarray(p, float))
    k = g['kind']
    if g.get('outline'):
        return motor_inside(g, p)
    if k == 'sdf':
        import sdf as SDF
        q = p * np.array([1.0, -1.0, 1.0]) if g.get('flip_y') else p
        return SDF.dist(g['tree'], q) <= 0.0
    if k == 'loft':
        ok = loft_inside(g, p)
        for c in clips(g):
            ok &= (p - np.asarray(c['p'])) @ np.asarray(c['n'], float) >= 0
        if g.get('keep'):
            ok &= np.linalg.norm(p - np.asarray(g['keep']['c'], float), axis=1) <= g['keep']['r']
        return ok
    if k == 'ellipsoid':
        return (((p - g['c']) / g['r']) ** 2).sum(1) <= 1.0
    if k == 'revolve':
        a = axis_vec(g['axis'])
        d = p - g['c']
        t = d @ a
        return np.linalg.norm(d - np.outer(t, a), axis=1) <= _revolve_r(g, t)
    if k == 'prism':
        P = np.array(g['pts'], float)
        ok = (p[:, 0] >= g['x0']) & (p[:, 0] <= g['x0'] + g['t'])
        q = p[:, 1:3]
        s = None
        for i in range(len(P)):
            e = P[(i + 1) % len(P)] - P[i]
            c = e[0] * (q[:, 1] - P[i][1]) - e[1] * (q[:, 0] - P[i][0])
            s = np.sign(c) if s is None else s
            ok &= np.sign(c) == s
        return ok
    if k == 'cyl':
        a = axis_vec(g['axis'])
        d = p - g['c']
        t = d @ a
        rr = np.linalg.norm(d - np.outer(t, a), axis=1)
        return (np.abs(t) <= g['h'] / 2) & (rr <= g['r'])
    if k == 'rbox':
        f = rbox_fillet(g)
        half = np.array(g['size']) / 2 - f
        q = np.maximum(np.abs(p - g['c']) - half, 0)
        return np.linalg.norm(q, axis=1) <= f
    if k == 'capsule':
        p0, p1 = np.array(g['p0']), np.array(g['p1'])
        d = p1 - p0
        t = np.clip(((p - p0) @ d) / (d @ d), 0, 1)
        return np.linalg.norm(p - (p0 + np.outer(t, d)), axis=1) <= g['r']
    raise ValueError(k)


def volume_samples(g, n=4000, rng=None):
    """Points uniformly inside g (rejection sampling in its bounding box)."""
    rng = rng or np.random.default_rng(0)
    m = mesh_of(g)
    lo, hi = m.bounds
    out = []
    while sum(len(o) for o in out) < n:
        q = rng.uniform(lo, hi, size=(n, 3))
        out.append(q[inside(g, q)])
    return np.vstack(out)[:n]


# ---------------------------------------------------------------- kinematics

def rot(axis, q):
    a = AXIS[axis]
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K


class Robot:
    def __init__(self, spec):
        self.spec = spec
        self.links = spec['links']
        self.joints = {j['name']: j for j in spec['joints']}
        self.parent_joint = {j['child']: j for j in spec['joints']}
        self.children = {}
        for j in spec['joints']:
            self.children.setdefault(j['parent'], []).append(j)
        self.order = []
        stack = [spec['root']]
        while stack:
            l = stack.pop(0)
            self.order.append(l)
            stack += [j['child'] for j in self.children.get(l, [])]

    def subtree(self, link):
        out, stack = [], [link]
        while stack:
            l = stack.pop()
            out.append(l)
            stack += [j['child'] for j in self.children.get(l, [])]
        return out

    def chain(self, link):
        """Joints from the root down to link."""
        out = []
        while link in self.parent_joint:
            j = self.parent_joint[link]
            out.append(j)
            link = j['parent']
        return out[::-1]

    def distance(self, a, b):
        """Number of joints between links a and b in the tree."""
        ca = [j['child'] for j in self.chain(a)]
        cb = [j['child'] for j in self.chain(b)]
        k = 0
        while k < min(len(ca), len(cb)) and ca[k] == cb[k]:
            k += 1
        return len(ca) + len(cb) - 2 * k

    def adjacent(self, a, b):
        pa, pb = self.parent_joint.get(a), self.parent_joint.get(b)
        return (pa and pa['parent'] == b) or (pb and pb['parent'] == a)

    def fk(self, q=None, base=None):
        """{link: 4x4} for joint angles q {name: rad}; missing joints are 0. Right-side joints default to the
        mirror of the left when only the left one is given and mirror=True was used to build q."""
        q = q or {}
        M = {self.spec['root']: np.eye(4) if base is None else base}
        for l in self.order[1:]:
            j = self.parent_joint[l]
            o = np.array(j['origin'])
            T = np.eye(4)
            R = rot(j['axis'], q.get(j['name'], 0.0))
            T[:3, :3] = R
            T[:3, 3] = o - R @ o
            M[l] = M[j['parent']] @ T
        return M


def mirror_pose(q):
    """Copy L_ angles to R_ (pitch keeps its sign, roll and yaw flip) unless R_ is given."""
    out = dict(q)
    for k, v in q.items():
        if k.startswith('L_'):
            r = 'R_' + k[2:]
            if r not in q:
                ax = k.split('_')[-1]
                out[r] = -v if ax in ('roll', 'yaw') else v
    return out


def link_geoms(spec, link, roles=None):
    return [g for g in spec['links'][link]['geoms'] if roles is None or g['role'] in roles]


def apply(M, p):
    return p @ M[:3, :3].T + M[:3, 3]
