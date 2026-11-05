"""The maker's real motor outlines: read each RobStride STEP file and write spec/actuator_outlines.json.

  python -I tools/motor_outlines.py --step RS00=<path> RS02=<path> RS03=<path> RS04=<path> RS06=<path>

The STEP files are not in the repo (RobStride's, github.com/RobStride/Product_Information); this file and its JSON are. Run it again when a new
STEP comes, with python -I since it reads downloaded files. Only the regex-level parser below touches them: trimesh can't read STEP, and cadquery
and OCP aren't installed. It was written for the data pass (reports/actuator_interfaces.md), whose hole patterns matched a survey of the same files.

Method: every face that isn't parallel to the motor's axis is projected along that axis, and rays along the axis (a polar grid, 2 degrees by 0.5 mm)
collect the heights where they cross faces. Crossings pair up into solid intervals. The envelope at each height is the largest occupied radius per angle.
Heights whose envelope barely changes merge into one band, taking the larger radius at each angle, so the outline can only err outward, by
under MERGE_MM. What that gives: the real revolved profile with its ears, bumps and cable tab as lumps on it. Recesses inside the outline (hub recesses,
screw holes, the gap round a hub) are filled: this is an outside shape for fit checks and renders, not a section. Holes are in the interface file.

Output, per model: h and the frame of spec/actuators.json; 'bands' [[t0, t1, [r per angle]]] with t along the output direction, measured from the middle of
h, so t = +h/2 is the output's extreme, as the old cylinder's end was; 'pins' for pins that stand off the hub (RS04); 'keepouts' for connectors, cable
exits and bumps (from the interface file's own measured text), in the STEP's own polar frame.
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENT = re.compile(r"#(\d+)\s*=\s*([A-Z_0-9]*)\s*\((.*?)\)\s*;\s*(?=#\d+\s*=|ENDSEC)", re.S)
NA, DR, DZ = 180, 0.5, 0.1
MERGE_MM = 0.75                    # a band may overstate the real radius by this much, never understate it
MIN_BAND = 0.3
CLOSE = 3                          # samples each way: a dip narrower than 12 deg (a connector recess, an unpaired ray) is filled, so the outline errs outward


def refs(s):
    return [int(x) for x in re.findall(r"#(\d+)", s)]


class Model:
    def __init__(self, path):
        t = open(path, encoding='latin1').read()
        self.E = {int(m.group(1)): (m.group(2), m.group(3)) for m in ENT.finditer(t, t.index('DATA;'))}
        self.pt, self.dr = {}, {}
        for k, (n, a) in self.E.items():
            if n in ('CARTESIAN_POINT', 'DIRECTION'):
                v = np.array([float(x) for x in re.search(r"\(([^()]*)\)", a).group(1).split(',')])
                (self.pt if n[0] == 'C' else self.dr)[k] = v

    def ax(self, k):
        r = refs(self.E[k][1])
        o = self.pt[r[0]]
        z = self.dr[r[1]] if len(r) > 1 else np.array([0, 0, 1.0])
        x = self.dr[r[2]] if len(r) > 2 else np.array([1.0, 0, 0])
        x = x - z * (x @ z)
        x = x / np.linalg.norm(x)
        return o, z, x, np.cross(z, x)

    def vert(self, k):
        return self.pt[refs(self.E[k][1])[0]]

    def edge_pts(self, ec, n=24):
        """Ordered points along an edge. Lines end to end; circles sampled. Anything else: its end points (fine for a shrink-wrap's few splines at this grid)."""
        a = self.E[ec]
        r = refs(a[1])
        v1, v2, c = self.vert(r[0]), self.vert(r[1]), r[2]
        if self.E[c][0] != 'CIRCLE':
            return [v1, v2]
        o, z, x, y = self.ax(refs(self.E[c][1])[0])
        R = float(self.E[c][1].split(',')[-1])
        ang = lambda p: math.atan2((p - o) @ y, (p - o) @ x)
        same = '.T.' in a[1].split(',')[-1]
        a1, a2 = (ang(v1), ang(v2)) if same else (ang(v2), ang(v1))
        d = (a2 - a1) % (2 * math.pi)
        d = 2 * math.pi if d < 1e-9 else d
        pts = [o + R * (math.cos(a1 + t) * x + math.sin(a1 + t) * y) for t in np.linspace(0, d, n)]
        return pts if same else pts[::-1]

    def faces(self):
        """[(surface type, plane or None, loops)] of the faces that a ray along Z can cross."""
        out = []
        for k, (n, a) in self.E.items():
            if n != 'ADVANCED_FACE':
                continue
            m = re.match(r"\s*'[^']*'\s*,\s*\((.*?)\)\s*,\s*#(\d+)\s*,\s*\.(T|F)\.", a, re.S)
            bounds, surf = refs(m.group(1)), int(m.group(2))
            stype = self.E[surf][0]
            if stype in ('CYLINDRICAL_SURFACE', 'PLANE'):
                o, z, _, _ = self.ax(refs(self.E[surf][1])[0])
                if (stype == 'CYLINDRICAL_SURFACE' and abs(abs(z[2]) - 1) < 1e-3) or (stype == 'PLANE' and abs(z[2]) < 1e-4):
                    continue                        # parallel to the rays
            loops = []
            for b in bounds:
                lp = refs(self.E[b][1])[0]
                poly = []
                for oe in refs(self.E[lp][1]):
                    t = self.E[oe][1]
                    seq = self.edge_pts(refs(t)[-1])
                    poly += list(seq if '.T.' in t.split(',')[-1] else seq[::-1])
                loops.append(np.array(poly))
            plane = None
            if stype == 'PLANE':
                o, z, _, _ = self.ax(refs(self.E[surf][1])[0])
                plane = (o, z)
            out.append((stype, plane, loops))
        return out


def inside_poly(px, py, poly):
    x, y = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    n = np.zeros(len(px), int)
    for a, b, a2, b2 in zip(x, y, x2, y2):
        if b == b2:
            continue
        c = (b > py) != (b2 > py)
        n[c] += px[c] < (a + (py[c] - b) * (a2 - a) / (b2 - b))
    return n % 2 == 1


def envelope(path, axis_xy, rmax):
    M = Model(path)
    th = (np.arange(NA) + 0.5) * 2 * math.pi / NA                # half a step off, so a ray rarely runs along a seam
    rs = np.arange(0.25, rmax + 3.0, DR)
    Th, Rr = np.meshgrid(th, rs, indexing='ij')
    px, py = (Rr * np.cos(Th)).ravel() + axis_xy[0], (Rr * np.sin(Th)).ravel() + axis_xy[1]
    hits = [[] for _ in range(len(px))]
    faces = M.faces()
    for stype, plane, loops in faces:
        al = np.vstack(loops)
        idx = np.nonzero((px >= al[:, 0].min()) & (px <= al[:, 0].max()) & (py >= al[:, 1].min()) & (py <= al[:, 1].max()))[0]
        if not len(idx):
            continue
        par = np.zeros(len(idx), int)
        for l in loops:
            par += inside_poly(px[idx], py[idx], l)
        for i in idx[par % 2 == 1]:
            if plane is not None:
                o, n = plane
                z = o[2] - ((px[i] - o[0]) * n[0] + (py[i] - o[1]) * n[1]) / n[2]
            else:
                d = np.hypot(al[:, 0] - px[i], al[:, 1] - py[i])
                z = al[np.argsort(d)[:4], 2].mean()
            hits[i].append(float(z))
    return th, rs, hits, len(faces)


def bands_of(th, rs, hits, z_hub_extreme, z_rear, zdir, h, drop_beyond=None):
    """Bands in t (output +). z_hub_extreme: the output's extreme in STEP z, where t = +h/2. drop_beyond: t above this holds pins, added separately."""
    t_of = lambda z: h / 2 - zdir * (z_hub_extreme - z)
    tall = np.arange(-h / 2 - 2.0, h / 2 + DZ, DZ)
    occ = np.zeros((NA, len(rs), len(tall)), bool)
    odd = 0
    for i, zs in enumerate(hits):
        if not zs:
            continue
        zs = sorted(round(z, 4) for z in zs)
        uniq = []                                                # a coincident pair of faces (a seam on a face boundary) cancels out
        for z in zs:
            if uniq and abs(uniq[-1] - z) < 1e-3:
                uniq.pop()
            else:
                uniq.append(z)
        if len(uniq) % 2:
            odd += 1
            uniq = [uniq[0], uniq[-1]]
        a, r = np.unravel_index(i, (NA, len(rs)))
        for z0, z1 in zip(uniq[::2], uniq[1::2]):
            t0, t1 = sorted((t_of(z0), t_of(z1)))
            occ[a, r, (tall >= t0) & (tall <= t1)] = True
    rad = np.zeros((NA, len(tall)))
    for k in range(len(tall)):
        for a in range(NA):
            w = np.nonzero(occ[a, :, k])[0]
            rad[a, k] = rs[w[-1]] + DR / 2 if len(w) else 0.0
    dil = np.max([np.roll(rad, i, axis=0) for i in range(-CLOSE, CLOSE + 1)], axis=0)      # closing: dilate, then erode, round the circle
    rad = np.min([np.roll(dil, i, axis=0) for i in range(-CLOSE, CLOSE + 1)], axis=0)
    keep = [k for k in range(len(tall)) if rad[:, k].max() > 0 and (drop_beyond is None or tall[k] <= drop_beyond)]
    out, start, cur = [], keep[0], rad[:, keep[0]].copy()
    for a, b in zip(keep, keep[1:] + [None]):
        nxt = None if b is None else rad[:, b]
        if nxt is not None and b == a + 1 and (np.abs(np.maximum(cur, nxt) - nxt).max() <= MERGE_MM and np.abs(np.maximum(cur, nxt) - cur).max() <= MERGE_MM or tall[b] - tall[start] < MIN_BAND):
            cur = np.maximum(cur, nxt)
            continue
        out.append([round(float(tall[start]) - DZ / 2, 3), round(float(tall[a]) + DZ / 2, 3), [round(float(x), 2) for x in cur]])
        if nxt is not None:
            start, cur = b, nxt.copy()
    return out, odd


def v(x):
    return x['value'] if isinstance(x, dict) and 'value' in x else x


# What must stay clear of a screw, from the interface file's measured 'cable_exit' and 'envelope' text, in the STEP's own frame (angle deg ccw about +Z from +X).
# (name, kind, angle lo, angle hi, r lo, r hi, z lo, z hi). Half widths turned to angles at the feature's radius.
KEEPOUTS = {
    'RS00': [('power connector A', 'connector', -16.5, 16.5, 22.0, 28.5, 36.64, 42.94), ('power connector B', 'connector', 163.5, 196.5, 22.0, 28.5, 36.64, 42.94)],
    'RS02': [('power connector A', 'connector', 3.0, 25.6, 30.0, 37.0, 7.3, 12.5), ('power connector B', 'connector', 183.0, 205.6, 30.0, 37.0, 7.3, 12.5)],
    'RS03': [('power connector A', 'connector', -15.0, 18.0, 33.0, 40.0, 50.45, 55.45), ('power connector B', 'connector', 162.0, 195.0, 33.0, 40.0, 50.45, 55.45),
             ('bump of unstated function', 'bump', 83.0, 97.0, 45.0, 51.0, 8.0, 38.8)],
    'RS04': [('cable outlet tab', 'cable', 70.0, 80.0, 45.0, 58.0, -23.9, -16.7), ('connector detail', 'connector', 240.0, 250.0, 35.0, 42.0, -26.0, -22.0),
             ('wire clamp cover slot', 'cable', 145.0, 155.0, 55.0, 60.0, -13.5, 20.0)],
    'RS06': [('power connector A', 'connector', 78.0, 102.0, 30.0, 35.0, 36.33, 41.53), ('power connector B', 'connector', 258.0, 282.0, 30.0, 35.0, 36.33, 41.53),
             ('bump of unstated function', 'bump', 172.0, 188.0, 36.0, 43.5, 0.0, 33.0)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', nargs='+', required=True, help='MODEL=path to its STEP file')
    ap.add_argument('--out', default=os.path.join(ROOT, 'spec', 'actuator_outlines.json'))
    a = ap.parse_args()
    I = json.load(open(os.path.join(ROOT, 'spec', 'actuator_interfaces.json'), encoding='utf-8'))
    A = json.load(open(os.path.join(ROOT, 'spec', 'actuators.json'), encoding='utf-8'))
    out = {'_about': __doc__.split('\n\n')[2].replace('\n', ' '), '_angles': f'{NA} radii per band, at (k + 0.5) x {360 / NA:.0f} deg about STEP +Z from STEP +X, in mm. t in mm.', '_tool': 'tools/motor_outlines.py'}
    for item in a.step:
        name, path = item.split('=', 1)
        e, h = I[name], A[name]['h'] * 1000
        faces = v(e['frame']['faces_z_mm'])
        zdir = 1 if '+Z' in e['frame']['output_faces'] else -1
        ext = faces.get('pin_tips', faces['output_hub'])
        rear = faces['rear_cover']
        xy = (54.084, 40.164) if name == 'RS02' else (0.0, 0.0)  # the model is off-origin: interface file, frame
        rmax = v(e['envelope']['round_dia_mm']) / 2
        th, rs, hits, nf = envelope(path, xy, rmax)
        hub_t = h / 2 - abs(ext - faces['output_hub'])
        bands, odd = bands_of(th, rs, hits, ext, rear, zdir, h, drop_beyond=hub_t if 'pin_tips' in faces else None)
        t_rear = h / 2 - abs(ext - rear)
        bands[0][0], bands[-1][1] = round(t_rear, 3), round(hub_t, 3)       # the end faces exactly, from the interface file's measured faces
        rec = dict(h=h, step_output_sign=zdir, z_output_extreme=ext, t_hub_face=round(hub_t, 3), t_stator_front=round(h / 2 - abs(ext - faces['stator_front']), 3),
                   t_rear=round(h / 2 - abs(ext - rear), 3), bands=bands, keepouts=[dict(zip(('name', 'kind', 'a0', 'a1', 'r0', 'r1', 'z0', 'z1'), k)) for k in KEEPOUTS[name]], faces_read=nf, odd_rays=odd)
        f = e['output_flange']
        if 'protruding' in str(f['locating'].get('type', '')):
            rec['pins'] = dict(d=v(f['locating']['dia_mm']), bc=v(f['locating']['bolt_circle_mm']), height=v(f['locating']['height_mm']), angles=v(f['locating']['angles_step_deg']))
        out[name] = rec
        rr = np.array([b[2] for b in bands])
        print(f"{name}: {nf} faces, {len(bands)} bands, odd rays {odd}, t {bands[0][0]:.2f}..{bands[-1][1]:.2f} (h/2 {h / 2:.2f}), rmax {rr.max():.2f}, x {max(b[2][k] * math.cos(th[k]) for b in bands for k in range(NA)):.1f}, y {max(b[2][k] * math.sin(th[k]) for b in bands for k in range(NA)):.1f}")
        for b in bands:
            x = np.array(b[2])
            print(f'   t {b[0]:7.2f} {b[1]:7.2f}  r min {x.min():6.2f} median {np.median(x):6.2f} max {x.max():6.2f} at {np.degrees(th[int(x.argmax())]):5.0f} deg')
    with open(a.out, 'w', encoding='utf-8') as fp:
        json.dump(out, fp)
    print('->', a.out, os.path.getsize(a.out), 'bytes')


if __name__ == '__main__':
    sys.exit(main())
