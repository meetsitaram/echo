"""The machined knee beside what it replaced and beside the reference frame, at one scale. Local only: it reads a reference picture, so its output goes in the git-ignored reports/compare/ (detail_*).

  python -I tools/compare_knee.py --before old_robot.json --figure <knee crop> --out reports/compare/detail_knee_metal

The Figure 01 frame is there for principles (round, closed, motors on the joint). Cover-off is judged as well-made machinery, not by resemblance.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom as G  # noqa: E402
import render_metal as RM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--before', required=True)
    ap.add_argument('--figure', required=True)
    ap.add_argument('--rotate', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--only', help='one panel, "bend,view": a process per panel, as the renderer can run dry over many scenes')
    a = ap.parse_args()
    font = ImageFont.load_default(size=26)
    rn = RM.Renderer(W=900, H=1100, ss=1)
    now, old = G.load(), G.load(a.before)
    views = [('front', 0, 3), ('side', 90, 3), ('3/4', 42, 14), ('rear', 180, 5)]
    fig = Image.open(a.figure).convert('RGB')
    if a.rotate:
        fig = fig.rotate(a.rotate, expand=True)
    fig = fig.resize((int(fig.width * 1100 / fig.height), 1100), Image.LANCZOS)
    for q in (0.0, 0.6, 2.15):
        for lab, az, el in views:
            if a.only and a.only != f'{q:.2f},{lab}':
                continue
            ims = []
            for spec, who in ((old, 'as of the last round'), (now, 'this round')):
                R = G.Robot(spec)
                M = R.fk({'L_knee': q})
                P = {l: np.linalg.inv(M['L_hip_yaw_link']) @ M[l] for l in RM.KNEE_LINKS}
                o = np.array(next(j for j in spec['joints'] if j['name'] == 'L_knee')['origin'], float)
                parts = RM.collect(spec, RM.KNEE_LINKS)
                rn._cache = {}
                im = Image.fromarray(rn.shot(parts, RM.ang_cam(o, az, el, 0.85), o, poses=P))
                ImageDraw.Draw(im).text((12, 8), f'{who}: {lab}, {q:.2f} rad', fill=(20, 20, 24), font=font)
                ims.append(im)
            S = Image.new('RGB', (1800 + fig.width, 1100), (246, 246, 246))
            S.paste(ims[0], (0, 0))
            S.paste(ims[1], (900, 0))
            S.paste(fig, (1800, 0))
            ImageDraw.Draw(S).text((1812, 8), 'Figure 01 frame (principles only)', fill=(255, 255, 255), font=font)
            S.save(f"{a.out}_{q:.2f}_{lab.replace('/', '-')}.png")
            print(f"{a.out}_{q:.2f}_{lab.replace('/', '-')}.png", flush=True)


if __name__ == '__main__':
    main()
