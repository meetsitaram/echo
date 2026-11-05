"""Machined shapes as signed distance trees: the one primitive here that gives real fillets where a tube runs into a fork or a neck into a barrel.

A geom of kind 'sdf' holds a tree of plain JSON (built by the helpers in spec/hardware.py). geom.py asks this file for two things: the distance at points (exact inside tests), and a watertight mesh, cached on disk by
the tree's hash (.cache/sdf, git-ignored; a first build takes seconds to a minute a part, a cached one milliseconds).

Leaves (metres, world frame at the zero pose):
  cyl   c, axis, r, h, e            a cylinder; e rounds its two end edges
  rev   c, axis, poly [[r, t]...]   a closed polygon in (radius, position along the axis), revolved. t is measured from c along +axis
  ext   axis, t0, t1, poly, e       a closed polygon in the plane square to axis, extruded from t0 to t1. Plane axes: axis y: (x, z); axis x: (y, z); axis z: (x, y)
  loft  secs [[z, cx, cy, a, b, n]...], zlo, zhi, e    superellipse sections along z, PCHIP between them as geom.py's lofts
  box   c, size, e                  a box; e rounds all its edges
Operations: {'t': 'u', 'k': k, 'of': [...]} union, {'t': 'i', 'k': k, 'of': [...]} intersection, {'t': 'd', 'k': k, 'a': node, 'b': [...]} subtraction. k is the fillet's size (m): 0 is a sharp join.

The distances are exact on the surface and at worst underestimates away from it, which is all meshing and inside tests need.
"""
import hashlib
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, '.cache', 'sdf')
AX = {'x': 0, 'y': 1, 'z': 2}
PLANE = {'y': (0, 2), 'x': (1, 2), 'z': (0, 1)}


def _smin(a, b, k):
    if k <= 0:
        return np.minimum(a, b)
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h * h * k / 4


def _smax(a, b, k):
    return -_smin(-a, -b, k)


try:
    import numba

    @numba.njit(parallel=True, cache=False, fastmath=True)
    def _poly_sd(px, py, vx, vy):
        n, m = len(px), len(vx)
        out = np.empty(n)
        for i in numba.prange(n):
            x, y = px[i], py[i]
            d = (x - vx[0]) ** 2 + (y - vy[0]) ** 2
            s = 1.0
            j = m - 1
            for k in range(m):
                ex, ey = vx[j] - vx[k], vy[j] - vy[k]
                wx, wy = x - vx[k], y - vy[k]
                t = (wx * ex + wy * ey) / (ex * ex + ey * ey + 1e-30)
                t = 0.0 if t < 0 else (1.0 if t > 1 else t)
                bx, by = wx - ex * t, wy - ey * t
                q = bx * bx + by * by
                if q < d:
                    d = q
                c1, c2, c3 = y >= vy[k], y < vy[j], ex * wy > ey * wx
                if (c1 and c2 and c3) or ((not c1) and (not c2) and (not c3)):
                    s = -s
                j = k
            out[i] = s * np.sqrt(d)
        return out
except ImportError:                # slower, the same answer
    def _poly_sd(px, py, vx, vy):
        out = np.empty(len(px))
        v = np.column_stack([vx, vy])
        for start in range(0, len(px), 200000):
            P = np.column_stack([px[start:start + 200000], py[start:start + 200000]])
            d = np.full(len(P), 1e30)
            s = np.ones(len(P))
            j = len(v) - 1
            for k in range(len(v)):
                e, w = v[j] - v[k], P - v[k]
                t = np.clip((w @ e) / (e @ e + 1e-30), 0, 1)
                b = w - np.outer(t, e)
                d = np.minimum(d, (b * b).sum(1))
                c1, c2, c3 = P[:, 1] >= v[k, 1], P[:, 1] < v[j, 1], e[0] * w[:, 1] > e[1] * w[:, 0]
                s = np.where((c1 & c2 & c3) | (~c1 & ~c2 & ~c3), -s, s)
                j = k
            out[start:start + 200000] = s * np.sqrt(d)
        return out


def poly_sd(q, poly):
    v = np.asarray(poly, float)
    return _poly_sd(np.asarray(q[:, 0], float), np.asarray(q[:, 1], float), v[:, 0], v[:, 1])


def _capped(d2, dt, e):
    """A 2D shape extruded: distance from its 2D distance d2 and its distance along the axis dt, end edges rounded by e."""
    d2, dt = d2 + e, dt + e
    return np.minimum(np.maximum(d2, dt), 0.0) + np.hypot(np.maximum(d2, 0.0), np.maximum(dt, 0.0)) - e


def _loft_f(node):
    from scipy.interpolate import PchipInterpolator
    s = np.asarray(node['secs'], float)
    zs = s[:, 0]
    if len(s) >= 3:
        return [PchipInterpolator(zs, s[:, k]) for k in range(1, 6)], zs
    return [(lambda z, k=k: np.interp(z, zs, s[:, k])) for k in range(1, 6)], zs


def dist(node, p):
    """Distance from the shape at points p (N, 3)."""
    t = node['t']
    if t == 'u':
        d = dist(node['of'][0], p)
        for x in node['of'][1:]:
            d = _smin(d, dist(x, p), node.get('k', 0.0))
        return d
    if t == 'i':
        d = dist(node['of'][0], p)
        for x in node['of'][1:]:
            d = _smax(d, dist(x, p), node.get('k', 0.0))
        return d
    if t == 'd':
        d = dist(node['a'], p)
        for x in node['b']:
            d = _smax(d, -dist(x, p), node.get('k', 0.0))
        return d
    if t == 'cyl':
        i = AX[node['axis']]
        q = p - np.asarray(node['c'], float)
        rr = np.sqrt(np.maximum((q * q).sum(1) - q[:, i] ** 2, 0.0))
        return _capped(rr - node['r'], np.abs(q[:, i]) - node['h'] / 2, node.get('e', 0.0))
    if t == 'rev':
        i = AX[node['axis']]
        q = p - np.asarray(node['c'], float)
        rr = np.sqrt(np.maximum((q * q).sum(1) - q[:, i] ** 2, 0.0))
        return poly_sd(np.column_stack([rr, q[:, i]]), node['poly'])
    if t == 'ext':
        i = AX[node['axis']]
        u, v = PLANE[node['axis']]
        d2 = poly_sd(np.column_stack([p[:, u], p[:, v]]), node['poly'])
        mid, half = (node['t0'] + node['t1']) / 2, abs(node['t1'] - node['t0']) / 2
        return _capped(d2, np.abs(p[:, i] - mid) - half, node.get('e', 0.0))
    if t == 'loft':
        f, zs = _loft_f(node)
        z = np.clip(p[:, 2], zs[0], zs[-1])
        cx, cy, a, b, n = (np.asarray(g(z), float) for g in f)
        L = (np.abs((p[:, 0] - cx) / a) ** n + np.abs((p[:, 1] - cy) / b) ** n) ** (1.0 / n)
        d2 = np.minimum(a, b) * (L - 1.0)
        zlo, zhi = node.get('zlo', zs[0]), node.get('zhi', zs[-1])
        return _capped(d2, np.maximum(zlo - p[:, 2], p[:, 2] - zhi), node.get('e', 0.0))
    if t == 'box':
        e = node.get('e', 0.0)
        q = np.abs(p - np.asarray(node['c'], float)) - np.asarray(node['size'], float) / 2 + e
        return np.minimum(q.max(1), 0.0) + np.linalg.norm(np.maximum(q, 0.0), axis=1) - e
    raise ValueError(t)


def bounds(node):
    """A box that holds the shape."""
    t = node['t']
    if t == 'u':
        bs = [bounds(x) for x in node['of']]
        return np.min([b[0] for b in bs], axis=0), np.max([b[1] for b in bs], axis=0)
    if t == 'i':
        bs = [bounds(x) for x in node['of']]
        return np.max([b[0] for b in bs], axis=0), np.min([b[1] for b in bs], axis=0)
    if t == 'd':
        return bounds(node['a'])
    if t in ('cyl', 'rev'):
        i = AX[node['axis']]
        c = np.asarray(node['c'], float)
        if t == 'cyl':
            r, lo_t, hi_t = node['r'], -node['h'] / 2, node['h'] / 2
        else:
            P = np.asarray(node['poly'], float)
            r, lo_t, hi_t = P[:, 0].max(), P[:, 1].min(), P[:, 1].max()
        lo, hi = c - r, c + r
        lo[i], hi[i] = c[i] + lo_t, c[i] + hi_t
        return lo, hi
    if t == 'ext':
        i = AX[node['axis']]
        u, v = PLANE[node['axis']]
        P = np.asarray(node['poly'], float)
        lo, hi = np.zeros(3), np.zeros(3)
        lo[u], hi[u], lo[v], hi[v] = P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max()
        lo[i], hi[i] = min(node['t0'], node['t1']), max(node['t0'], node['t1'])
        return lo, hi
    if t == 'loft':
        f, zs = _loft_f(node)
        z = np.linspace(zs[0], zs[-1], 200)
        cx, cy, a, b, _ = (np.asarray(g(z), float) for g in f)
        return np.array([(cx - a).min(), (cy - b).min(), node.get('zlo', zs[0])]), np.array([(cx + a).max(), (cy + b).max(), node.get('zhi', zs[-1])])
    if t == 'box':
        c, s = np.asarray(node['c'], float), np.asarray(node['size'], float)
        return c - s / 2, c + s / 2
    raise ValueError(t)


def _upsample(a, n, axis):
    """Linear interpolation of a along axis, to n samples."""
    m = a.shape[axis]
    x = np.linspace(0, m - 1, n)
    i = np.minimum(x.astype(int), m - 2)
    w = (x - i).reshape([-1 if k == axis else 1 for k in range(a.ndim)]).astype(np.float32)
    return np.take(a, i, axis) * (1 - w) + np.take(a, i + 1, axis) * w


def field(node, voxel):
    """The distance on a grid: coarse first, then exact only near the surface. Returns (F float32, origin)."""
    lo, hi = bounds(node)
    lo, hi = lo - 3 * voxel, hi + 3 * voxel
    n = np.ceil((hi - lo) / voxel).astype(int) + 1
    k = 4
    nc = (n - 1) // k + 2
    xs = [lo[i] + np.arange(nc[i]) * voxel * k for i in range(3)]
    Fc = np.empty(nc, np.float32)
    for a in range(nc[0]):
        Y, Z = np.meshgrid(xs[1], xs[2], indexing='ij')
        P = np.column_stack([np.full(Y.size, xs[0][a]), Y.ravel(), Z.ravel()])
        Fc[a] = dist(node, P).reshape(Y.shape)
    nf = (nc - 1) * k + 1
    F = Fc
    for ax in range(3):
        F = _upsample(F, nf[ax], ax)
    band = 2.2 * voxel * k
    idx = np.nonzero(np.abs(F) < band)
    for s in range(0, len(idx[0]), 2000000):
        I = [c[s:s + 2000000] for c in idx]
        P = np.column_stack([lo[i] + I[i] * voxel for i in range(3)])
        F[I[0], I[1], I[2]] = dist(node, P)
    return F, lo


def mesh(node, voxel, simplify=0.0):
    """A watertight (vertices, faces), from the tree. simplify: the distance a vertex may move to merge triangles, m."""
    import manifold3d as mf
    from skimage.measure import marching_cubes
    F, lo = field(node, voxel)
    # a level no grid value can equal: a flat face on a grid plane gives exact zeros, and those make non-manifold cells
    m = None
    for level in (voxel * 1.37e-3, voxel * 7.3e-3, voxel * 0.031):
        v, f, _, _ = marching_cubes(F, level, spacing=(voxel, voxel, voxel), method='lewiner')
        v = v.astype(np.float64) + lo
        m = mf.Manifold(mf.Mesh(vert_properties=v.astype(np.float32), tri_verts=f.astype(np.uint32)))
        if m.status() == mf.Error.NoError:
            break
    del F
    if m.status() != mf.Error.NoError:
        raise ValueError(f'sdf mesh is not manifold: {m.status()}')
    if m.volume() < 0:
        o = m.to_mesh()
        m = mf.Manifold(mf.Mesh(vert_properties=o.vert_properties, tri_verts=o.tri_verts[:, ::-1]))
    if simplify > 0:
        m = m.simplify(simplify)
    o = m.to_mesh()
    return np.asarray(o.vert_properties[:, :3], np.float64), np.asarray(o.tri_verts, np.int64)


def cached(node, voxel, simplify=0.0):
    key = hashlib.sha1(json.dumps([node, voxel, simplify], sort_keys=True).encode()).hexdigest()[:24]
    path = os.path.join(CACHE, key + '.npz')
    if os.path.exists(path):
        z = np.load(path)
        return z['v'], z['f']
    v, f = mesh(node, voxel, simplify)
    os.makedirs(CACHE, exist_ok=True)
    tmp = path + f'.{os.getpid()}.tmp.npz'
    np.savez_compressed(tmp, v=v.astype(np.float32), f=f.astype(np.int32))
    os.replace(tmp, path)
    return v, f
