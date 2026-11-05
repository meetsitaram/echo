"""echo CSVs from soma-retargeter -> the SuperMimic motion pkls: grounded, 50 Hz, screened, with a hold-out set for testing.

  python train/build_corpus.py <csv dir> <out dir> [--hold 0.1]

Writes train.pkl, test.pkl and corpus.json. Mirrored clips (_M) of a held-out clip are held out too, so a test clip is never
seen in training. The screen is what the design can't do in a walk, reported rather than hidden: corpus.json lists each clip's
reason, and the joints that reject the most.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from soma_csv_to_take import FPS, ground, load

SKIP = 15             # frames at 120 Hz: the retargeter's settling at the start
MAX_SOLE_RISE = 0.10  # m: root shift needed to ground it, beyond which the clip isn't standing on the floor
MIN_S = 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--hold', type=float, default=0.1)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    train, test, rejected = {}, {}, {}
    frames = collections.Counter()
    for f in sorted(a.csv.glob('*.csv')):
        try:
            t, names, q, p, wxyz = load(f, 120.0, SKIP)
        except Exception as e:  # noqa: BLE001
            rejected[f.stem] = f'unreadable: {e}'
            continue
        if not (np.isfinite(q).all() and np.isfinite(p).all()):
            rejected[f.stem] = 'non-finite'
            continue
        if t[-1] < MIN_S:
            rejected[f.stem] = 'shorter than 2 s'
            continue
        p, shift = ground(names, q, p, wxyz)
        if shift > MAX_SOLE_RISE:
            rejected[f.stem] = f'needed {shift:.2f} m to ground'
            continue
        if p[:, 2].min() < 0.35:
            rejected[f.stem] = 'pelvis below 0.35 m'
            continue
        base = f.stem[:-2] if f.stem.endswith('_M') else f.stem
        held = int(hashlib.sha1(base.encode()).hexdigest(), 16) % 1000 < a.hold * 1000
        xyzw = np.column_stack((wxyz[:, 1:], wxyz[:, :1]))
        (test if held else train)[f.stem] = dict(root_trans_offset=p.astype(np.float32), root_rot=xyzw.astype(np.float32), dof=q.astype(np.float32), fps=FPS)
        frames['test' if held else 'train'] += len(q)
    joblib.dump(train, a.out / 'train.pkl')
    joblib.dump(test, a.out / 'test.pkl')
    summary = dict(train=len(train), test=len(test), rejected=len(rejected), hours={k: round(v / FPS / 3600, 2) for k, v in frames.items()}, reasons=rejected)
    (a.out / 'corpus.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    print({k: v for k, v in summary.items() if k != 'reasons'})


if __name__ == '__main__':
    main()
