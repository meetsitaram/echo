"""Bottom-up mass model: every part's mass, centre of mass and inertia from its material and shape, summed per link.

  python -I tools/mass_model.py          -> reports/mass.md (per link and per material), prints the total

Rules (spec/robot.py MATERIALS; a geom's 'material' defaults by role and colour in build()):
  shell     a cover: its 3 mm wall (the same wall the collision checks use) x the material's density
  decal     a layer: 'visor'-like decals 2 mm thick, small marks as solids; a decal 'material' with layer 0 is solid
  struct    7075 frames x a fill factor by the part's thinnest dimension (our blocks stand for machined, pocketed or
            hollow frames: <= 8 mm plates 0.8, <= 20 mm 0.5, thicker 0.25), unless the geom gives 'fill'
  actuator  the catalogue mass (spec/actuators.json), a solid cylinder
  linkage   7075 rods and cranks
  'mass'    on any geom: that mass, spread over its solid (battery, compute, the hands' internals)
  'add_mass' on any geom: extra mass carried by that part (boards on a frame)
  allowance WIRING of each link's mass (cables, fasteners, bearings, thermal pads), on its frames
"""
from __future__ import annotations

import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checks as C  # noqa: E402
import geom as G  # noqa: E402

WIRING = 0.06
LAYER = 0.002


def fill_of(g, m):
    if 'fill' in g:
        return g['fill']
    b = m.bounding_box()
    t = min(b[3] - b[0], b[4] - b[1], b[5] - b[2])
    return 0.8 if t <= 0.0085 else 0.5 if t <= 0.0205 else 0.25


def _props(m):
    """(volume, com, inertia about the com per unit density) of a manifold."""
    mesh = m.to_mesh()
    tm = trimesh.Trimesh(np.asarray(mesh.vert_properties[:, :3], float), np.asarray(mesh.tri_verts, int))
    if tm.volume <= 0:
        tm.invert()
    return tm.volume, tm.center_mass, tm.moment_inertia / max(tm.volume, 1e-12)


def parts(spec, S, ln):
    """[(name, material, mass, com, inertia about its com)] for one link, before the wiring allowance."""
    mats = spec['materials']
    A = spec['actuators']
    out = []
    for i, g in enumerate(spec['links'][ln]['geoms']):
        if g['role'] == 'cut' or S.geom.get((ln, i)) is None:
            continue
        if g['role'] == 'actuator':
            m = A[g['name'].split()[-1]]['mass']
            r, h = g['r'], g['h']
            ia, ir = 0.5 * m * r * r, m * (3 * r * r + h * h) / 12
            I = np.diag([ir, ir, ir])
            k = 'xyz'.index(g['axis'])
            I[k, k] = ia
            out.append((g['name'], 'actuator', m, np.array(g['c'], float), I))
            continue
        mat = g['material']
        solid = S.geom[(ln, i)]
        body = solid
        if 'mass' not in g:
            if g['role'] == 'shell':
                body = S.wall[(ln, i)]
            elif g['role'] == 'knit':
                body = S.soft[(ln, i)]
            elif g['role'] == 'decal' and mats[mat].get('layer'):
                body = solid - C.to_manifold(G.mesh_of(G.inner_of(g, mats[mat]['layer'])))
        vol, com, Iu = _props(body)
        if 'mass' in g:
            m = g['mass']
        else:
            m = vol * mats[mat]['density'] * (fill_of(g, solid) if g['role'] in ('struct', 'linkage') else g.get('wall_fill', 1.0))
        out.append((g['name'], mat, m, com, Iu * m))
        if g.get('add_mass'):
            out.append((g['name'] + ' (carried)', g.get('add_what', 'carried'), g['add_mass'], com, Iu * g['add_mass']))
    return out


def combine(ps):
    mt = sum(p[2] for p in ps)
    com = sum(p[2] * p[3] for p in ps) / mt
    I = np.zeros((3, 3))
    for _, _, m, c, Ic in ps:
        d = c - com
        I += Ic + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return mt, com, I


def link_inertials(spec, S=None):
    """{link: (mass, com, inertia about the com, parts)} in world coordinates at the zero pose, with the wiring
    allowance spread over the link's frames (or over the link when it has none)."""
    S = S or C.Solids(spec)
    out = {}
    for ln in spec['links']:
        ps = parts(spec, S, ln)
        frames = [p for p in ps if p[1] == 'al7075']
        base = sum(p[2] for p in ps if p[1] != 'cells')
        extra = spec['links'][ln].get('wiring', WIRING) * base
        carrier = frames or ps
        fm = sum(p[2] for p in carrier)
        ps = [(n, mat, m * (1 + extra / fm), c, I * (1 + extra / fm)) if any(p[0] == n for p in carrier) else
              (n, mat, m, c, I) for n, mat, m, c, I in ps]
        out[ln] = (*combine(ps), ps)
    return out


def report(spec, inert, path):
    mats = sorted({p[1] for v in inert.values() for p in v[3]})
    total = sum(v[0] for v in inert.values())
    by_mat = {m: sum(p[2] for v in inert.values() for p in v[3] if p[1] == m) for m in mats}
    lines = ['# Mass model', '',
             'Bottom-up: every part from its material and shape (`tools/mass_model.py`, materials in `spec/robot.py`), '
             f'a {WIRING * 100:.0f} % wiring / fastener / bearing allowance per link on its frames.', '',
             f"**Total {total:.2f} kg** (Figure 03: {spec.get('target_mass', 61.0)} kg).", '',
             '| material | kg | share |', '|---|---|---|']
    lines += [f'| {m} | {by_mat[m]:.2f} | {by_mat[m] / total * 100:.0f} % |' for m in sorted(by_mat, key=lambda k: -by_mat[k])]
    lines += ['', '| link | mass kg | budget kg (hand estimate) | CoM (m, world, zero pose) | principal inertia (kg m2) |',
              '|---|---|---|---|---|']
    for ln, (m, com, I, _) in inert.items():
        ev = np.linalg.eigvalsh(I)
        lines.append(f"| {ln} | {m:.2f} | {spec['links'][ln].get('budget', float('nan')):.2f} | "
                     f"{com[0]:+.3f} {com[1]:+.3f} {com[2]:+.3f} | {ev[0]:.4f} {ev[1]:.4f} {ev[2]:.4f} |")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return total, by_mat


def main():
    spec = G.load()
    inert = link_inertials(spec)
    total, by_mat = report(spec, inert, os.path.join(G.ROOT, 'reports', 'mass.md'))
    for m, v in sorted(by_mat.items(), key=lambda kv: -kv[1]):
        print(f'{m:12s} {v:6.2f} kg')
    print(f'total {total:.2f} kg -> reports/mass.md')


if __name__ == '__main__':
    main()
