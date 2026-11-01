"""Collisions along a recorded motion: the combined joint motions the single-joint sweeps of tools/checks.py do not
test (a hip pitching, rolling and yawing at once). Every Nth frame is posed and each leg link is tested against
every link not jointed to it (across one joint the motion is a single joint's, which the sweep covers).

  python -I tools/motion_collisions.py [reports/motion/<take>.npz] [--step 5] [--bad collide]

--bad: break the design on purpose (tools/checks.py's cases) and show that the test sees it. Prints each colliding
pair with its number of frames and worst overlap; exits 1 if any.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checks as C  # noqa: E402
import geom as G  # noqa: E402

LEG = ('hip', 'knee', 'ankle', 'foot')


def hits(spec, take, step):
    S, R = C.Solids(spec), G.Robot(spec)
    names, Q = list(take['joint_names']), take['q']
    legs = [l for l in spec['links'] if any(k in l for k in LEG)]
    pairs = sorted({tuple(sorted((a, b))) for a in legs for b in spec['links'] if a != b and not R.adjacent(a, b)})
    out = {}
    for i in range(0, len(Q), step):
        M = R.fk(dict(zip(names, Q[i])))
        P = {l: C.posed(S.hard[l], M[l]) for l in spec['links'] if S.hard[l] is not None}
        for a, b in pairs:
            v = C.overlap(P.get(a), P.get(b))
            if v > C.TOUCH:
                h = out.setdefault((a, b), [0, 0.0, i])
                h[0] += 1
                if v > h[1]:
                    h[1], h[2] = v, i
    return out, len(range(0, len(Q), step)), len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('take', nargs='?', default=os.path.join(G.ROOT, 'reports', 'motion', 'pico_seg30_walk_echo.npz'))
    ap.add_argument('--step', type=int, default=5)
    ap.add_argument('--bad')
    a = ap.parse_args()
    spec = G.load()
    if a.bad:
        spec = C.break_spec(spec, a.bad)
    take = np.load(a.take, allow_pickle=True)
    found, frames, npairs = hits(spec, take, a.step)
    broken = f' (broken: {a.bad})' if a.bad else ''
    print(f'{os.path.basename(a.take)}: {frames} frames, {npairs} link pairs{broken}')
    for (x, y), (n, v, i) in sorted(found.items(), key=lambda kv: -kv[1][0]):
        print(f'  {x} / {y}: {n} frames, worst {v * 1e6:.1f} cm3 at {take["t"][i]:.1f} s')
    print('no collisions' if not found else f'{len(found)} colliding pairs')
    sys.exit(1 if found else 0)


if __name__ == '__main__':
    main()
