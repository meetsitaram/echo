"""Quick look at spec/robot.json without Fusion: front, side and back views of the skins (and the internals).

  python -I tools/preview.py [--inside] [--pose name] [--out reports/preview.png]
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom as G  # noqa: E402


def shade(color, normals, light):
    c = np.array([int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    k = 0.45 + 0.55 * np.clip(normals @ light, 0, 1)
    return np.clip(c[None] * k[:, None] + 0.04, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inside', action='store_true', help='draw rigid internals instead of skins')
    ap.add_argument('--pose', default=None)
    ap.add_argument('--out', default=os.path.join(G.ROOT, 'reports', 'preview.png'))
    a = ap.parse_args()
    spec = G.load()
    R = G.Robot(spec)
    q = {}
    if a.pose:
        import checks
        q = G.mirror_pose(checks.POSES[a.pose][0])
    M = R.fk(q)
    import checks
    shown = checks.exposed_names(spec)
    parts = []
    for ln, l in spec['links'].items():
        for g in l['geoms']:
            visible = g['role'] in ('shell', 'decal') or g['name'] in shown
            if (a.inside and g['role'] not in G.RIGID) or (not a.inside and not visible):
                continue
            m = G.mesh_of(g)
            m.apply_transform(M[ln])
            parts.append((m, g['color']))
    views = [('front', (0, 0)), ('side', (0, -90)), ('back', (0, 180)), ('3/4', (12, -35))]
    fig = plt.figure(figsize=(16, 9), dpi=120)
    for k, (title, (elev, azim)) in enumerate(views):
        ax = fig.add_subplot(1, 4, k + 1, projection='3d', computed_zorder=False)
        az, el = np.radians(azim), np.radians(elev)
        view = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        light = view + np.array([0, 0, 0.6])
        light /= np.linalg.norm(light)
        tris, cols = [], []
        for m, color in parts:
            t = m.triangles
            n = m.face_normals
            vis = n @ view > -0.05
            tris.append(t[vis])
            cols.append(shade(color, n[vis], light))
        tris, cols = np.vstack(tris), np.vstack(cols)
        depth = tris.mean(1) @ view
        order = np.argsort(depth)
        ax.add_collection3d(Poly3DCollection(tris[order], facecolors=cols[order], edgecolors='none'))
        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.3, 0.3)
        ax.set_zlim(0, 1.75)
        ax.set_box_aspect((0.6, 0.6, 1.75), zoom=2.1)
        ax.view_init(elev=elev, azim=azim)
        ax.set_proj_type('ortho')
        ax.set_axis_off()
        ax.set_title(title)
    plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0, wspace=0)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    plt.savefig(a.out, facecolor='white')
    print(a.out)


if __name__ == '__main__':
    main()
