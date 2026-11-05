"""Close-ups of one joint's real connections, from the spec's solids: a section through the axis, part-ID colours, and an exploded look along the axis. What the cover-off view can't show.

  python tools/render_knee.py --out reports/parts/knee_section.png                 the left knee, sectioned in the plane through the axis, by part
  python tools/render_knee.py --out reports/parts/knee_faces.png --view faces      thigh-side and motor-back views, with the garment and hubs off

Section: each part's real solid (holes, seats, cuts) is cut by a plane and filled flat, so it reads like a drawing. Not a render: no shading to hide a gap. Dimensions in mm from the leg's centre.
"""
from __future__ import annotations

import argparse
import colorsys
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import checks as C  # noqa: E402
import geom as G  # noqa: E402

ROLE_COLOR = {'hardware': (217, 165, 32), 'bearing': (42, 167, 155), 'actuator': (70, 74, 80), 'shell': (40, 40, 44)}
KEEP = ('struct', 'actuator', 'hardware', 'bearing', 'shell', 'linkage')


def part_color(g, i):
    if g['role'] in ROLE_COLOR:
        return ROLE_COLOR[g['role']]
    h = (i * 0.618) % 1.0
    r, gg, b = colorsys.hsv_to_rgb(h, 0.45, 0.85)
    return int(r * 255), int(gg * 255), int(b * 255)


def section(S, spec, links, n, at, u, v, box, scale, W, H, font, label=True, skip=()):
    im = Image.new('RGB', (W, H), (248, 247, 244))
    dr = ImageDraw.Draw(im)
    Rm = np.vstack([u, v, n]) * 1000.0
    T = np.column_stack([Rm, np.zeros(3)])
    x0, y0 = box
    px = lambda p: (W / 2 - (170 if label else 0) + (p[0] - x0) * scale, H / 2 - (p[1] - y0) * scale)
    legend = []
    i = 0
    order = {'shell': 0, 'struct': 1, 'actuator': 2, 'bearing': 3, 'hardware': 4, 'linkage': 1}
    items = []
    for ln in links:
        for k, g in enumerate(spec['links'][ln]['geoms']):
            if g['role'] not in KEEP or S.geom.get((ln, k)) is None or g['name'] in skip:
                continue
            items.append((order[g['role']], ln, k, g))
    for _, ln, k, g in sorted(items, key=lambda x: x[0]):
        body = S.wall.get((ln, k), S.geom[(ln, k)])
        cs = body.transform(T.astype(np.float64)).slice(at)
        if cs.area() < 0.01:
            continue
        col = part_color(g, i)
        i += 1
        for part in cs.decompose():
            polys = [np.asarray(q, float) for q in part.to_polygons()]
            area = lambda q: 0.5 * (np.dot(q[:, 0], np.roll(q[:, 1], -1)) - np.dot(q[:, 1], np.roll(q[:, 0], -1)))
            polys.sort(key=lambda q: -abs(area(q)))
            sign = np.sign(area(polys[0]))
            layer = Image.new('L', (W, H), 0)
            d2 = ImageDraw.Draw(layer)
            for q in polys:
                d2.polygon([px(p) for p in q], fill=255 if np.sign(area(q)) == sign else 0)
            im.paste(Image.new('RGB', (W, H), col), mask=layer)
            for q in polys:
                dr.line([px(p) for p in q] + [px(q[0])], fill=(30, 30, 34), width=1)
        legend.append((g['name'], col))
    if label:
        y = 10
        for name, col in legend[:60]:
            dr.rectangle((W - 330, y + 2, W - 316, y + 14), fill=col, outline=(30, 30, 30))
            dr.text((W - 310, y), name[:42], fill=(30, 30, 34), font=font)
            y += 17
    return im, px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--view', default='section', choices=('section', 'faces'))
    a = ap.parse_args()
    spec = G.load()
    S = C.Solids(spec)
    font = ImageFont.load_default(size=13)
    big = ImageFont.load_default(size=20)
    links = ['L_knee_link', 'L_hip_yaw_link']
    HY, KZ = next(j for j in spec['joints'] if j['name'] == 'L_knee')['origin'][1:]
    if a.view == 'section':
        # plane x = 0: the axis in it. u is y (across the leg), v is z. Then the plane through a hub screw, where the M5s and M4s read in section, at x = screw's x.
        W, H = 1500, 1020
        scale = 7.4
        hub = next(g['hw'] for l in links for g in spec['links'][l]['geoms'] if g.get('hw', {}).get('pattern') == 'L knee hub' and g['hw']['type'] == 'screw' and abs(g['hw']['pos'][0]) > 0.012)
        st = next(g['hw'] for l in links for g in spec['links'][l]['geoms'] if g.get('hw', {}).get('pattern') == 'L knee stator' and g['hw']['type'] == 'screw')
        sheets = []
        for title, x in ((f'Through the axis (x = 0)', 0.0), (f"Through a hub screw (x = {hub['pos'][0] * 1000:.1f})", hub['pos'][0]), (f"Through a stator screw (x = {st['pos'][0] * 1000:.1f})", st['pos'][0])):
            u, v, n = np.array([0, 1.0, 0]), np.array([0, 0, 1.0]), np.array([1.0, 0, 0])
            im, px = section(S, spec, links, n, x * 1000.0, u, v, (HY * 1000, KZ * 1000), scale, W, H, font)
            dr = ImageDraw.Draw(im)
            dr.text((16, 12), f'Left knee, part ID section. {title}. Scale: 7.4 px per mm; inboard to the left. Gold screws, teal bearings.', fill=(30, 30, 34), font=big)
            for y in (-61.5, -43, -37, -34.5, -30.65, -26.5, -24.3, 27.85, 31.65, 33.4, 41.4, 43, 61.5):
                p = px(((HY * 1000 + y), KZ * 1000 - 78))
                dr.line([(p[0], p[1]), (p[0], p[1] + 14)], fill=(120, 30, 30), width=1)
                dr.text((p[0] - 12, p[1] + 16 + (10 if int(abs(y) * 2) % 3 == 0 else 0)), f'{y:g}', fill=(120, 30, 30), font=font)
            sheets.append(im)
        out = Image.new('RGB', (W, H * len(sheets)), (248, 247, 244))
        for i, s in enumerate(sheets):
            out.paste(s, (0, i * H))
        out.save(a.out)
        print(a.out, out.size)
    else:
        W, H = 1000, 1000
        scale = 6.6
        sheets = []
        for title, y in (('Cup floor and barrel plate: hub screws, pin holes, stator screws', -27.5), ('Rear cap\'s skirt on the motor\'s back, cable notch', 16.0), ('Outer bearing, in its cheek, on the post', 37.4), ('Inner bearing, in the barrel, on the cup', -30.5)):
            u, v, n = np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), np.array([0, -1.0, 0])
            im, px = section(S, spec, links, n, -(HY * 1000 + y), u, v, (0.0, KZ * 1000), scale, W, H, font, label=False)
            ImageDraw.Draw(im).text((16, 12), f'{title}. y = {y:g} from the leg\'s centre. Looking inboard, x forward to the right', fill=(30, 30, 34), font=big)
            sheets.append(im)
        out = Image.new('RGB', (W * 2, H * 2), (248, 247, 244))
        for i, s in enumerate(sheets):
            out.paste(s, ((i % 2) * W, (i // 2) * H))
        out.save(a.out)
        print(a.out, out.size)


if __name__ == '__main__':
    main()
