"""Side-by-side look comparison: Figure 03 reference frames vs Fusion renders of the replica.

  python -I tools/compare.py      -> reports/compare/compare_<view>.png and reports/compare/compare.md

Each pair is cut to the robot and scaled to the same height (the published 1.727 m). The target landmark heights
(spec/targets.json, measured on Figure's frames) are drawn as labelled lines across BOTH pictures, so a replica
feature off its line is off Figure's proportions. The table compares the targets with the replica's own geometry
(tools/checks.py measure), not with a detector run on the render: the first version detected black parts on both
pictures and merged the elbow, hand and pelvis into one blob (2026-10-31).

Reference frames (reports/compare/figure_*.jpg): figure.ai reveal video, side at 0:21.5 (mid-stride, so legs read
a little short), 3/4 at 0:24.5 standing (about 60 deg from the front: fit of silhouette widths), front at 1:50
(standing in a doorway; head top and feet are out of frame, so it is scaled by chin and knee instead).
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checks as C  # noqa: E402
import geom as G  # noqa: E402

DIR = os.path.join(G.ROOT, 'reports', 'compare')
PAIRS = [('side', 'figure_side.jpg', 'replica_side.png'), ('34', 'figure_34.jpg', 'replica_34.png'),
         ('front', 'figure_front.jpg', 'replica_front.png')]
# frames the robot does not fit in: (top, bottom, left, right) of the full robot in pixels, from chin (row 56,
# 1.522 m) and knee discs (row 503, 0.54 m) for the height and the arms' outer edges for the width
FRAME_BOX = {'figure_front.jpg': (56 - (1.727 - 1.522) * 447 / 0.982, 503 + 0.54 * 447 / 0.982, 318, 548)}
LINES = ['chin_z', 'shoulder_top_z', 'elbow_z', 'pelvis_band_top_z', 'wrist_z', 'hips_black_bottom_z', 'fingertip_z',
         'knee_z']
COLORS = [(214, 39, 40), (31, 119, 180), (44, 160, 44), (148, 103, 189), (255, 127, 14), (140, 86, 75),
          (227, 119, 194), (23, 190, 207)]


def robot_box(path):
    if os.path.basename(path) in FRAME_BOX:
        return tuple(int(round(v)) for v in FRAME_BOX[os.path.basename(path)])
    g = np.asarray(Image.open(path).convert('L')).astype(float)
    if path.endswith('.jpg'):
        g = g[:578]                                     # caption bar under the video
        bg = np.maximum(ndimage.median_filter(g, size=(1, 41)), np.percentile(g, 75))
        robot = g < bg - 18
    else:
        robot = g < 246
    robot = ndimage.binary_opening(robot, iterations=1)
    lab, n = ndimage.label(robot)
    sizes = ndimage.sum(robot, lab, range(1, n + 1))
    robot = np.isin(lab, np.where(sizes > 0.02 * sizes.max())[0] + 1)
    rows, cols = np.where(robot.any(1))[0], np.where(robot.any(0))[0]
    return rows.min(), rows.max(), cols.min(), cols.max()


def crop_pad(im, box):
    """Crop, padding with white where the box runs past the picture (a frame that cuts the robot off), so every
    panel puts the robot's head and feet at the same rows."""
    out = Image.new('RGB', (box[2] - box[0], box[3] - box[1]), 'white')
    out.paste(im, (-box[0], -box[1]))
    return out


def panel(path, h_px, targets, title):
    im = Image.open(path).convert('RGB')
    top, bot, left, right = robot_box(path)
    pad = int(0.10 * (bot - top))
    box = (left - pad, top - pad // 3, right + pad, bot + pad // 3)
    im = crop_pad(im, box)
    s = h_px / (bot - top)
    im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    y0, y1 = (top - box[1]) * s, (bot - box[1]) * s
    d = ImageDraw.Draw(im)
    for (k, z), c in zip(targets, COLORS):
        y = y1 - z / C.json.load(open(os.path.join(G.ROOT, 'spec', 'targets.json')))['published']['height_m'] \
            * (y1 - y0)
        d.line([(0, y), (im.width, y)], fill=c, width=1)
    d.text((6, 4), title, fill=(0, 0, 0))
    return im


BANDS = {'head': (1.36, 1.75), 'shoulders': (1.05, 1.55), 'hips': (0.70, 1.20),
         'knees': (0.30, 0.80),                            # close-ups for review (metres above the floor)
         'elbows': (1.03, 1.29, 1.0)}                      # (wider: both elbows in the front view)


def detail(view, fig, rep, band, h_px=520):
    """The same height band cut from both pictures (scaled by robot height) and shown side by side, larger.
    band: (low, high) in metres, optionally a third value: the crop's half width per unit of its height."""
    out = []
    for path, title in ((fig, 'Figure 03'), (rep, 'replica')):
        im = Image.open(os.path.join(DIR, path)).convert('RGB')
        top, bot, left, right = robot_box(os.path.join(DIR, path))
        H = bot - top
        y0 = int(bot - band[1] / 1.727 * H)
        y1 = int(bot - band[0] / 1.727 * H)
        cx = (left + right) // 2
        half = int((y1 - y0) * (band[2] if len(band) > 2 else 0.45))
        crop = crop_pad(im, (cx - half, y0, cx + half, y1))
        s = h_px / crop.height
        crop = crop.resize((int(crop.width * s), h_px), Image.LANCZOS)
        ImageDraw.Draw(crop).text((6, 4), title, fill=(0, 0, 0))
        out.append(crop)
    img = Image.new('RGB', (out[0].width + out[1].width + 10, h_px), 'white')
    img.paste(out[0], (0, 0))
    img.paste(out[1], (out[0].width + 10, 0))
    return img


def main():
    tg = json.load(open(os.path.join(G.ROOT, 'spec', 'targets.json'), encoding='utf-8'))
    for view, fig, rep in PAIRS:
        for name, band in BANDS.items():
            detail(view, fig, rep, band).save(os.path.join(DIR, f'detail_{view}_{name}.png'))
    spec = G.load()
    S = C.Solids(spec)
    m = C.measure(spec, S)
    lines = [(k, tg['photo'][k][0]) for k in LINES]
    for view, fig, rep in PAIRS:
        p1 = panel(os.path.join(DIR, fig), 600, lines, 'Figure 03 (figure.ai reveal video)')
        src = os.path.join(DIR, 'replica_source.txt')       # tools/render_pose.py --views writes it; Fusion not
        how = open(src, encoding='utf-8').read().strip() if os.path.exists(src) else 'Fusion render'
        p2 = panel(os.path.join(DIR, rep), 600, lines, f'replica ({how})')
        legend_h = 18 * len(lines) + 8
        out = Image.new('RGB', (p1.width + p2.width + 10, max(p1.height, p2.height) + legend_h), 'white')
        out.paste(p1, (0, 0))
        out.paste(p2, (p1.width + 10, 0))
        d = ImageDraw.Draw(out)
        y = max(p1.height, p2.height) + 4
        for (k, z), c in zip(lines, COLORS):
            d.line([(8, y + 7), (40, y + 7)], fill=c, width=3)
            d.text((48, y), f'{k.replace("_z", "")}: Figure {z:.3f} m, replica {m[k]:.3f} m ({m[k] - z:+.3f})',
                   fill=(0, 0, 0))
            y += 18
        out.save(os.path.join(DIR, f'compare_{view}.png'))
    rows = ['# Figure 03 vs replica: measurements', '',
            'Figure 03: published specs (figure.ai) and landmarks measured on reveal-video frames '
            '(spec/targets.json). Replica: its own geometry (spec/robot.json).', '',
            '| measure | Figure 03 | tolerance | replica | difference | |', '|---|---|---|---|---|---|']
    rows.append(f"| height (m) | {tg['published']['height_m']:.3f} | 0.010 | {m['height_m']:.3f} | "
                f"{m['height_m'] - tg['published']['height_m']:+.3f} | "
                f"{'ok' if abs(m['height_m'] - tg['published']['height_m']) <= 0.01 else 'OFF'} |")
    for k, (v, tol) in tg['photo'].items():
        rows.append(f'| {k} | {v:.3f} | {tol:.3f} | {m[k]:.3f} | {m[k] - v:+.3f} | '
                    f"{'ok' if abs(m[k] - v) <= tol else 'OFF'} |")
    mass = sum(v[0] for v in C.MM.link_inertials(spec).values())   # bottom-up, from the materials
    lo, hi = tg['published']['mass_kg'] * (1 - C.MASS_BELOW), tg['published']['mass_kg'] + 1.5
    rows.append(f"| mass (kg, bottom-up) | {tg['published']['mass_kg']:.1f} | -10 % / +1.5 | {mass:.1f} | "
                f"{mass - tg['published']['mass_kg']:+.1f} | {'ok' if lo <= mass <= hi else 'OFF'} |")
    with open(os.path.join(DIR, 'compare.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(rows))
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
