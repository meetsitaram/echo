"""Checks on real standard parts: motors at their true outline, screws, holes, bolted joints, bearings, patterns, tool access. tools/checks.py runs them as groups.

  motorfit  a motor at its STEP outline inside any part, of its own link too (the old cylinder fits nothing by name)
  fastener  thread engaged against the rule; the screw no longer than its blind hole; nothing it doesn't belong in; through the joint's sweep
  holes     each hole's edge distance and ligament, read from a slice of the part with all its holes cut
  bolted    slip, separation, shear, tension and strip per pattern against strength.py's load cases (formulas: tools/strength.py bolted_rows)
  bearing   catalogue ratings against strength.py's load cases; seat wall and seat land; the ring stopped both ways; clear of what it isn't seated in
  pattern   each motor's stator and hub pattern fully used; no drawn hole over a connector, cable exit or bump; whether the right side's motor can be the same part
  access    a straight line along each screw's axis from its head: can a key reach. A pattern in the build order is asked at its moment: only what's on by then is in the way
  assembly  the build order (spec/robot.py, HW.step): each part goes in along its path without cutting what's on

Failures here are the design's open items: most of the robot has no screws yet, and each row says so.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom as G  # noqa: E402

# ---------------------------------------------------------------- thresholds (each with its reason)

# Under this a screw and a part share nothing that isn't a meshing error: 1 mm3 is a 1 mm cube, far past what polygon-sided cylinders disagree by (chord sag 0.03 mm on d51).
TOUCH_MM3 = 1.0
SEAT_DEPTH = 0.0005
# Edge distance: from a hole's centre to the part's nearest edge, at least 1.2 x the hole. EN 1993-1-8 Table 3.3's minimum for a bolted plate (from memory, an estimate); 7075's lower ductility argues for no less. Below it, the plate tears out behind the screw.
EDGE_MIN = 1.2
# Ligament: material left between two holes, at least half the hole and at least 1.5 mm. A joint held by friction isn't a shear joint: EN's 2.2 d pitch doesn't apply, and the maker's own hub would fail it (9.3 mm pitch on M5).
# Half a diameter keeps a drill's +-0.1 wander from running into its neighbour and keeps the band from yielding under the screw heads; 1.5 mm is the wall the process holds, as the wall check on made parts does.
LIG_FRACTION, LIG_MIN_MM = 0.5, 1.5
# A counterbore's wall to an edge or a neighbour. Same process limit.
WALL_MIN_MM = 1.5
# The seat's wall outside a race (or the shaft's, for a hollow one), at least the race's own section. IKO: 'when the rigidity of the mounting parts is not sufficient ... performance will be deteriorated significantly', with no number:
# a housing no stiffer than the ring it holds is where that starts. An estimate, not a rule of theirs.
SEAT_WALL_FRACTION = 0.5     # of (D - d) / 2: the outer race is about half the section
# Seat land: at least 80 % of the race's width in contact. A race half-supported tilts under its own load. An estimate; IKO gives shoulders, not a land.
SEAT_LAND_MIN = 0.8
# Key: its corners' reach is the socket's across-flats / sqrt 3, plus a hair. Reach: 60 mm is the short arm of a hex key at these sizes plus a hand behind it (estimate).
KEY_REACH = 0.060
KEY_CLEAR = 0.0002


def carrying(role):
    return role in ('struct', 'actuator', 'linkage')


def named(spec):
    return {g['name']: (ln, i, g) for ln, l in spec['links'].items() for i, g in enumerate(l['geoms']) if g['role'] != 'cut'}


def left(spec):
    return [ln for ln in spec['links'] if not ln.startswith('R_')]


def all_patterns(spec):
    """Screws by pattern, left side and centre: {pattern: [(link, index, geom)]}."""
    out = {}
    for ln in left(spec):
        for i, g in enumerate(spec['links'][ln]['geoms']):
            if g.get('hw', {}).get('type') == 'screw':
                out.setdefault(g['hw']['pattern'], []).append((ln, i, g))
    return out


def to_m(mesh_like):
    import checks as C
    return C.to_manifold(mesh_like)


def motor_of(spec, h):
    if not str(h['into']).startswith('motor:'):
        return None
    return h['into'][6:]


# ---------------------------------------------------------------- motorfit

def check_motorfit(spec, S, R, rep):
    """A motor against everything rigid or solid near it, at the STEP outline. check_static leaves a link's own parts alone, as brackets overlap to read as joined, and that hid motors built into frames."""
    import checks as C
    for ln in left(spec):
        l = spec['links'][ln]
        for i, g in enumerate(l['geoms']):
            if g['role'] != 'actuator' or g['name'].startswith('R_'):
                continue
            jn = g['name'].split(' actuator')[0]
            own_flange = f'{jn} flange'
            axis = G.AXIS[g['axis']]
            m = S.geom[(ln, i)]
            hits = []
            for l2 in spec['links']:
                for k, h in enumerate(spec['links'][l2]['geoms']):
                    if (l2, k) == (ln, i) or h['role'] not in ('struct', 'actuator', 'shell') or S.geom.get((l2, k)) is None or h['name'] == own_flange:
                        continue
                    body = S.wall.get((l2, k), S.geom[(l2, k)])
                    v = C.overlap(m, body)
                    if v > C.TOUCH:
                        # a part drawn 0.5 mm or less into an end face is seated on it, as parts are drawn here so they read as touching. That isn't a collision
                        lo, hi = C.bbox(m ^ body)
                        if float((hi - lo) @ axis) > SEAT_DEPTH:
                            hits.append((v, h['name']))
            hits.sort(reverse=True)
            rep.add('motorfit', g['name'], not hits, ('inside: ' + ', '.join(f'{n} {v * 1e6:.1f} cm3' for v, n in hits[:4])) if hits else 'clear', value=-sum(v for v, _ in hits) * 1e6)


# ---------------------------------------------------------------- fastener

def thread_zone(h):
    """The cylinder of the screw's engaged length, thread diameter + a hair: what it's allowed to share with the part it threads into."""
    pos, d = np.array(h['pos'], float), np.array(h['dir'], float)
    face = np.array(h['face'], float)
    tip = pos + d * h['length'] / 1000
    L = (tip - face) @ d
    return dict(kind='cyl', c=(face + tip) / 2, axis=list(d), r=h['d'] / 2000 + 0.0005, h=max(float(L), 1e-4) + 0.0004)


def check_fastener(spec, S, R, rep):
    import checks as C
    import strength as ST
    rule = ST.assumed(spec, 'min_engagement')
    idx = named(spec)
    rigid_all = [(ln, i, g) for ln, l in spec['links'].items() for i, g in enumerate(l['geoms']) if g['role'] in ('struct', 'actuator', 'linkage', 'bearing', 'hardware')]
    for pat, screws in all_patterns(spec).items():
        hs = [g['hw'] for _, _, g in screws]
        # the rule
        rows = [(ST.engagement(h)[0], ST.engagement(h)[1], h) for h in hs]
        le, depth, h0 = min(rows, key=lambda x: x[0])
        into = 'the motor\'s drawn thread' if str(h0['into']).startswith('motor:') else h0['into']
        ratio = le / h0['d']
        Fp, strip, proof, As, limit = ST.screw_load(spec, h0, le)
        rep.add('fastener', f'{pat}: engagement', ratio >= rule, f"{h0['thread']} x {h0['length']:.0f}, {le:.1f} mm in {into} ({ratio:.2f} d, the rule is {rule:.1f} d). Strip force {strip / 1000:.1f} kN at this length, so preload {Fp / 1000:.1f} kN, by {limit}", value=ratio)
        # no longer than its hole
        over = max(r[0] - r[1] for r in rows)
        rep.add('fastener', f'{pat}: tip inside its hole', over <= 0, f'{-over:.1f} mm short of the {depth:.1f} mm hole' if over <= 0 else f'{over:.1f} mm past the {depth:.1f} mm hole: it bottoms out before it clamps', value=-over)
        # into nothing else, along its whole length
        bad = []
        for ln, i, g in screws:
            h = g['hw']
            m = S.geom[(ln, i)]
            allowed = h['into'][6:] if str(h['into']).startswith('motor:') else h['into']
            zone = to_m(G.mesh_of(thread_zone(h)))
            for l2, k, o in rigid_all:
                if o['name'] == g['name'] or S.geom.get((l2, k)) is None:
                    continue
                x = m - zone if o['name'] == allowed else m
                v = C.overlap(x, S.geom[(l2, k)]) * 1e9
                if v > TOUCH_MM3:
                    bad.append((v, g['name'], o['name']))
        bad.sort(reverse=True)
        worst = '; '.join(f'{a} in {b} {v:.0f} mm3' for v, a, b in bad[:4])
        rep.add('fastener', f'{pat}: clear of what it does not belong in', not bad, worst or 'head and shank clear; clearance holes and counterbores cut', value=-sum(v for v, _, _ in bad))
        # through the joint's range, against the links next to its own
        link = screws[0][0]
        j = R.parent_joint.get(link)
        pairs = [(jx, jx['child'] if jx['parent'] == link else jx['parent']) for jx in spec['joints'] if link in (jx['parent'], jx['child'])]
        hit = []
        # the thread's zone goes with the screw: a screw in a hub, where the mesh has the hub and stator as one motor, turns with the hub
        heads = C.union([S.geom[(ln, i)] - to_m(G.mesh_of(thread_zone(g['hw']))) for ln, i, g in screws])
        for jx, other in pairs:
            # a screw into a motor's hub turns with the hub; the motor's mesh is hub, pins and stator in one, so it isn't asked
            own_motors = {g['hw']['into'][6:] for _, _, g in screws if str(g['hw']['into']).startswith('motor:')}
            hard = C.union([S.wall.get((other, k), S.geom[(other, k)]) for k, o in enumerate(spec['links'][other]['geoms']) if o['role'] in ('struct', 'actuator', 'shell') and o['name'] not in own_motors and S.geom.get((other, k)) is not None])
            if hard is None:
                continue
            for q in np.linspace(jx['lower'], jx['upper'], 13):
                Mx = R.fk({jx['name']: float(q)})
                Mo, Ml = Mx[other], Mx[link]
                v = C.overlap(C.posed(heads, Ml), C.posed(hard, Mo)) * 1e9
                if v > TOUCH_MM3:
                    hit.append((v, jx['name'], float(q), other))
        hit.sort(reverse=True)
        rep.add('fastener', f'{pat}: clear through the joints beside it', not hit, f'{hit[0][1]} at {hit[0][2]:+.2f} rad: {hit[0][0]:.0f} mm3 into {hit[0][3]}' if hit else f"{len(pairs)} joints swept, 13 steps", value=-(hit[0][0] if hit else 0.0))
    # the bought parts stay inside the skin
    check_hidden(spec, rep)


def check_hidden(spec, rep):
    """Screws and bearings stay under the skin, as check_skin asks of rigid parts: a head through the knit would change the look we were told not to."""
    import checks as C
    rng = np.random.default_rng(2)
    shells = [g for l in spec['links'].values() for g in l['geoms'] if g['role'] == 'shell']
    shells += [dict(G.inner_of(g, C.WALL), name=g['name'], role='knit') for l in spec['links'].values() for g in l['geoms'] if g['role'] == 'knit']
    # a part that shows on purpose hides what's behind it (a cheek's disc in the band, closing the hub's gap)
    shells += [g for l in spec['links'].values() for g in l['geoms'] if g['role'] in G.RIGID + ('hardware',) and g.get('exposed')]
    cuts = {}
    for l in spec['links'].values():
        for c in l['geoms']:
            for t in c.get('targets', []) if c['role'] == 'cut' else []:
                cuts.setdefault(t, []).append(c)
    for pat, screws in all_patterns(spec).items():
        worst, who = 0.0, ''
        for _, _, g in screws:
            p = G.volume_samples(g, 300, rng)
            inn = np.zeros(len(p), bool)
            for s in shells:
                hit = G.inside(s, p)
                if s['role'] in ('shell', 'knit'):               # a cut through a cover uncovers; a hole in a part that shows is where the screw is
                    for c in cuts.get(s['name'], []):
                        if c.get('internal') or c.get('hw', {}).get('pattern') == pat or c['name'].startswith(g['name']):
                            continue
                        hit &= ~G.inside(c, p)
                inn |= hit
            f = 1 - inn.mean()
            if f > worst:
                worst, who = f, g['name']
        rep.add('fastener', f'{pat}: under the skin', worst <= 0.005, f'{worst * 100:.1f} % of {who} outside' if who else 'all inside', value=-worst)


# ---------------------------------------------------------------- holes

def part_slice(m, axis, at_pos):
    """The part cut square to a hole at a point: polygons in mm, in the plane's own axes, and the hole's centre in them."""
    import strength as ST
    n = G.axis_vec(axis)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(a, n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    Rm = np.vstack([u, v, n]) * 1000.0
    T = np.column_stack([Rm, np.zeros(3)])
    p = np.asarray(at_pos, float)
    cs = m.transform(T.astype(np.float64)).slice(float(p @ n) * 1000.0)
    return [np.asarray(q, float) for q in cs.to_polygons()], np.array([p @ u, p @ v]) * 1000.0


def seg_dist(p, poly):
    a, b = poly, np.roll(poly, -1, axis=0)
    ab = b - a
    t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    return float(np.min(np.linalg.norm(a + ab * t[:, None] - p, axis=1)))


def spans_of(g, idx, S):
    """Planes to read a hole at: the middle of where it runs through each of its parts. (Its own span from the cut's c and h, clipped by each part's extent along the axis.)"""
    a = G.axis_vec(g['axis'])
    c = np.array(g['c'], float)
    lo, hi = c @ a - g['h'] / 2, c @ a + g['h'] / 2
    out = []
    for part in g['hw']['parts']:
        if part not in idx:
            continue
        m = S.geom[idx[part][:2]]
        v = np.asarray(m.to_mesh().vert_properties[:, :3], float) @ a
        p0, p1 = max(lo, v.min()), min(hi, v.max())
        if p1 > p0:
            out.append((p0 + p1) / 2)
    return out


def hole_readings(S, solids, h):
    """For one hole in its part: (edge distance mm, ligament mm to the nearest other small hole). None where the part doesn't reach the plane."""
    polys, c = part_slice(solids, h['axis'], h['pos'])
    if not polys:
        return None
    d = h['d']
    own = None
    rest = []
    for q in polys:
        ctr = q.mean(0)
        area = 0.5 * abs(np.dot(q[:, 0], np.roll(q[:, 1], -1)) - np.dot(q[:, 1], np.roll(q[:, 0], -1)))
        small = area < math.pi * (20.0 / 2) ** 2       # under d20: a screw hole, not a pocket or a bore
        if own is None and np.linalg.norm(ctr - c) < d / 4 and small and area < 2 * math.pi * d * d / 4:
            own = q
        else:
            rest.append((q, small))
    if own is None:
        return (0.0, 0.0)                               # it broke through the edge: its polygon merged with the outline
    edge = min((seg_dist(c, q) for q, s in rest if not s), default=999.0)
    lig = min((seg_dist(c, q) - d / 2 for q, s in rest if s), default=999.0)
    return edge, lig


def check_holes(spec, S, R, rep):
    idx = named(spec)
    import checks as C
    pats = {}
    for ln in left(spec):
        for g in spec['links'][ln]['geoms']:
            if g['role'] == 'cut' and g.get('hw', {}).get('type') == 'hole':
                pats.setdefault((g['hw']['pattern'], g['hw']['kind']), []).append(g)
    for (pat, kind), cuts in pats.items():
        worst_edge, worst_lig = (999.0, ''), (999.0, '')
        need_edge = need_lig = 0.0
        d0 = cuts[0]['hw']['d']
        for g in cuts:
            h = g['hw']
            parts = [p for p in h['parts'] if p in idx]
            # the parts it runs through, as one body: plates drawn overlapping are one piece of metal, and a hole near the edge of one isn't near an edge if the next carries on past it
            body = C.union([S.geom[idx[p][:2]] for p in parts])
            a = G.axis_vec(g['axis'])
            c = np.array(g['c'], float)
            for at in spans_of(g, idx, S):
                got = hole_readings(S, body, dict(h, pos=(c + a * (at - c @ a)).tolist()))
                if got is None:
                    continue
                e, l = got
                who = ' + '.join(parts)
                if e < worst_edge[0]:
                    worst_edge = (e, who)
                if l < worst_lig[0]:
                    worst_lig = (l, who)
        if kind in ('clearance', 'tapped', 'pin', 'insert'):
            need_edge = EDGE_MIN * d0
            need_lig = max(LIG_FRACTION * d0, LIG_MIN_MM)
        else:                                            # a counterbore or register: only a wall
            need_edge = d0 / 2 + WALL_MIN_MM
            need_lig = WALL_MIN_MM
        rep.add('holes', f'{pat}, {kind} d{d0:.1f}: edge distance', worst_edge[0] >= need_edge, f'{worst_edge[0]:.1f} mm to the edge of {worst_edge[1]} (needs {need_edge:.1f})' if worst_edge[0] < 999 else 'no edge in the plane', value=worst_edge[0] - need_edge)
        rep.add('holes', f'{pat}, {kind} d{d0:.1f}: ligament', worst_lig[0] >= need_lig, f'{worst_lig[0]:.1f} mm to the next hole in {worst_lig[1]} (needs {need_lig:.1f})' if worst_lig[0] < 999 else 'no other hole in the plane', value=worst_lig[0] - need_lig)


# ---------------------------------------------------------------- bolted

def check_bolted(spec, S, R, rep):
    import strength as ST
    M = ST.Model(spec, S)
    for r in ST.bolted_rows(M):
        if r.get('why'):
            rep.add('bolted', f"{r['name']}", False, r['why'], value=0.0)
            continue
        s = r['sf_each']
        f = lambda x: 'inf' if x == float('inf') else f'{x:.1f}'
        rep.add('bolted', f"{r['name']}, {r['label']}", r['ok'], ('Not the design\'s load: a bracket on what the motor\'s bearing could be asked. ' if r['whatif'] and not r['ok'] else '') + f"{r['n']} {r['thread']}, Fp {r['Fp'] / 1000:.1f} kN by {r['limit']}, T {r['T']:.0f} N m, V {r['V']:.0f} N, Fa {r['Fa']:.0f} N, Mt {r['Mt']:.0f} N m: slip {f(s['slip'])}, separation {f(s['separation'])}, shear {f(s['shear'])}, tension {f(s['tension'])} (needs {ST.SF_TENSION}), strip {f(s['strip'])}; a hand estimate", value=r['sf'], warn=r['whatif'])


# ---------------------------------------------------------------- bearing

def ring(c, axis, r_out, r_in, h):
    a = dict(kind='cyl', c=list(c), axis=axis, r=r_out, h=h)
    m = to_m(G.mesh_of(a))
    if r_in > 0:
        m = m - to_m(G.mesh_of(dict(a, r=r_in, h=h + 0.002)))
    return m


def seat_reading(S, idx, parts, h, outer):
    """What's round a race: (wall thickness mm, land fraction of its width). Slices the seating parts at planes across the race's width; wall from the smallest radial thickness found."""
    import checks as C
    ms = [S.geom[(idx[p][0], idx[p][1])] for p in parts if p in idx]
    if not ms:
        return 0.0, 0.0
    u = C.union(ms)
    a = G.axis_vec(h['axis'])
    c = np.array(h['pos'], float)
    B = h['B'] / 1000
    R_in, R_out = h['d'] / 2000, h['D'] / 2000
    # material present just outside the outer race / just inside the inner one, slice by slice along the width
    hits, walls = 0, []
    planes = np.linspace(-0.45, 0.45, 19)
    for s in planes:
        pos = c + a * s * B
        probe = (ring(pos, h['axis'], R_out + 0.0004, R_out + 0.0001, B / 20) if outer else ring(pos, h['axis'], R_in - 0.0001, max(R_in - 0.0004, 0), B / 20))
        whole = probe.volume()
        got = (probe ^ u).volume()
        if got > 0.8 * whole:
            hits += 1
            polys, ctr = part_slice(u, h['axis'], pos)
            if polys:
                if outer:
                    radii = [np.linalg.norm(q - ctr, axis=1) for q in polys]
                    # the wall out to the nearest edge, past the seat's own bore
                    out_edge = [seg_dist(ctr, q) for q in polys if seg_dist(ctr, q) > R_out * 1000 + 0.3]
                    outermost = min(out_edge) if out_edge else max(r.max() for r in radii)
                    walls.append(outermost - R_out * 1000)
                else:
                    in_edge = [seg_dist(ctr, q) for q in polys if seg_dist(ctr, q) < R_in * 1000 - 0.3]
                    walls.append(R_in * 1000 - (max(in_edge) if in_edge else 0.0))
    return (min(walls) if walls else 0.0), hits / len(planes)


def stop_reading(S, spec, h, ring_name, sign):
    """Is there a part of the race's own link right beside its shoulder band, on this side? Probe: a ring at the catalogue's shoulder band, machining-clearance-thin, just past the race's face."""
    import checks as C
    a = G.axis_vec(h['axis'])
    c = np.array(h['pos'], float) + a * sign * (h['B'] / 2000 + 0.00015)
    R_in, R_out = h['d'] / 2000, h['D'] / 2000
    da, Da = (h.get('da') or h['d'] * 1.05) / 2000, (h.get('Da') or h['D'] * 0.95) / 2000
    probe = ring(c, h['axis'], da, R_in, 0.0002) if ring_name == 'inner' else ring(c, h['axis'], R_out, Da, 0.0002)
    link = h['inner_link'] if ring_name == 'inner' else h['outer_link']
    parts = [S.geom[(link, i)] for i, g in enumerate(spec['links'][link]['geoms']) if g['role'] in ('struct', 'hardware') and S.geom.get((link, i)) is not None]
    u = C.union(parts)
    return (probe ^ u).volume() / max(probe.volume(), 1e-15)


def check_bearing(spec, S, R, rep):
    import checks as C
    import strength as ST
    M = ST.Model(spec, S)
    idx = named(spec)
    clear = ST.assumed(spec, 'machining_clearance')
    rows = ST.bearing_rows(M)
    if not rows:
        return
    for r in rows:
        nm = r['name']
        h = idx[nm][2]['hw']
        rep.add('bearing', f'{nm}: static at 3,400 N', r['fs_peak'] >= r['fs_min'], f"{r['designation']}: C0 {r['C0']:.0f} N, Fr {r['Fr_peak']:.0f} N (lever rule on {r['span']:.0f} mm), fs {r['fs_peak']:.2f}, needs {r['fs_min']:.1f}", value=r['fs_peak'])
        rep.add('bearing', f'{nm}: static with the side force', r['fs_side'] >= r['fs_min'], f"adding {r['Fa_side']:.0f} N axial, an upper bound: fs {r['fs_side']:.2f}, needs {r['fs_min']:.1f}", value=r['fs_side'])
        ok = r['life_valid'] and r['cycles'] >= r['need']
        rep.add('bearing', f'{nm}: life on the walk', ok, f"P {r['P']:.0f} N (C {r['C']:.0f} N{'' if r['life_valid'] else ': over C / 2, the formula does not apply'}), {r['cycles'] / 1e6:.0f} M cycles of 60 deg, needs {r['need'] / 1e6:.1f} M (a hand figure)", value=r['cycles'] if r['life_valid'] else 0.0)
        section = (h['D'] - h['d']) / 2
        for which, parts, outer in (('outer', h['outer_parts'], True), ('inner', h['inner_parts'], False)):
            wall, land = seat_reading(S, idx, parts, h, outer)
            need = SEAT_WALL_FRACTION * section
            rep.add('bearing', f'{nm}: {which} race seat wall', wall >= need, f"{wall:.1f} mm of {' + '.join(parts)}, needs {need:.1f} (the race's section)" if wall else f"no seat: {' + '.join(parts)} doesn't reach the race", value=wall)
            rep.add('bearing', f'{nm}: {which} race seat land', land >= SEAT_LAND_MIN, f"{land * 100:.0f} % of its {h['B']:.0f} mm width seated, needs {SEAT_LAND_MIN * 100:.0f} %", value=land)
            for sign, side in ((+1, '+y' if h['axis'] == 'y' else '+'), (-1, '-y' if h['axis'] == 'y' else '-')):
                f = stop_reading(S, spec, h, which, sign)
                rep.add('bearing', f'{nm}: {which} race stopped toward {side}', f > 0.25, f'{f * 100:.0f} % of its shoulder band has a part of its own link beside it' if f > 0.25 else f'nothing of its own link beside the shoulder band ({f * 100:.0f} %): it could walk that way. A standard circlip needs about 1.75 mm and a groove (estimate); the slot leaves {0.5:.1f} mm', value=f)
        # it fits: the ring in nothing, and running clear of the other link's parts
        ln, i, g = idx[nm]
        m = S.geom[(ln, i)]
        overl, near = [], []
        for l2, l in spec['links'].items():
            for k, o in enumerate(l['geoms']):
                if o['name'] == nm or o['role'] not in ('struct', 'actuator', 'shell', 'hardware', 'bearing') or S.geom.get((l2, k)) is None:
                    continue
                body = S.wall.get((l2, k), S.geom[(l2, k)])
                v = C.overlap(m, body) * 1e9
                if v > TOUCH_MM3:
                    overl.append((v, o['name']))
                elif o['name'] not in h['outer_parts'] + h['inner_parts'] and o['role'] in ('struct', 'shell') and l2 in (h['inner_link'], h['outer_link']):
                    # each race's shoulder may touch its own link; the other link must run clear of it
                    da, Da = (h.get('da') or h['d'] * 1.05) / 2000, (h.get('Da') or h['D'] * 0.95) / 2000
                    band = ring(h['pos'], h['axis'], h['D'] / 2000, Da, h['B'] / 1000) if l2 == h['inner_link'] else ring(h['pos'], h['axis'], da, h['d'] / 2000, h['B'] / 1000)
                    (ax, bx) = C.bbox(body)
                    (ay, by) = C.bbox(band)
                    if np.any(ax > by + 0.002) or np.any(ay > bx + 0.002):
                        continue
                    gp = band.min_gap(body, 0.002) * 1000
                    if gp < clear:
                        near.append((gp, o['name']))
        ok = not overl and not near
        txt = '; '.join([f'in {n} {v:.0f} mm3' for v, n in sorted(overl, reverse=True)[:4]] + [f'{gp:.2f} mm from {n}, a running gap needs {clear}' for gp, n in near[:3]])
        rep.add('bearing', f'{nm}: goes in', ok, txt or f"{h['B']:.0f} mm wide; nothing in its way, {clear} mm or more from what turns against it", value=-sum(v for v, _ in overl) - sum(clear - gp for gp, _ in near))


# ---------------------------------------------------------------- pattern

def face_ranges(model, which, entry, length_out=25.0):
    """z-range in the STEP's own frame a screw at this pattern occupies: its thread below the face and the screw and head above it."""
    import re
    if which == 'hub':
        z = float(re.search(r'z=(-?[\d.]+)', entry['face']).group(1))
    else:
        z = float(re.search(r'z=(-?[\d.]+)', entry['face']).group(1))
    return z


def _v(x):
    return x['value'] if isinstance(x, dict) and 'value' in x else x


def keepout_hits(model, entry, head_r, angles_override=None, outward=None):
    """Drawn holes that land on a connector, cable exit or bump, from the head's reach through the part it clamps to the thread's bottom."""
    d = G.outline_data(model)
    r = _v(entry['bolt_circle_mm']) / 2
    z = face_ranges(model, None, entry)
    depth = _v(entry.get('thread_depth_mm')) or 0
    sign = d['step_output_sign']
    # toward the output along STEP z is +sign. A stator's front or hub screw comes from outside the output, and a rear one from behind: which way is out of the face?
    rear = 'rear' in entry.get('face', '')
    outward_sign = -sign if rear else sign
    zs = sorted((z - outward_sign * depth, z + outward_sign * (outward if outward is not None else 25.0)))
    out = []
    for ang in (angles_override if angles_override is not None else _v(entry['angles_step_deg'])):
        for k in d['keepouts']:
            lo, hi = sorted((k['z0'], k['z1']))
            if hi < zs[0] or lo > zs[1]:
                continue
            a0, a1 = k['a0'], k['a1']
            mid = (a0 + a1) / 2
            half = (a1 - a0) / 2 + math.degrees(head_r / max(r, 1e-6))
            da = (ang - mid + 180) % 360 - 180
            if abs(da) <= half and r + head_r >= k['r0'] and r - head_r <= k['r1']:
                out.append((ang, k['name'], k['kind']))
    return out


def mirror_ok(model):
    """Can the right side's motor be this same part, turned? Only if all its patterns, and its pins and tab, read the same reflected across one line through the axis."""
    import json
    I = json.load(open(os.path.join(G.ROOT, 'spec', 'actuator_interfaces.json'), encoding='utf-8'))[model]
    sets = [_v(s['angles_step_deg']) for s in I['stator_mounting'] if _v(s.get('thread_depth_mm')) is not None] + [_v(I['output_flange']['angles_step_deg'])]
    if isinstance(I['output_flange'].get('locating'), dict) and _v(I['output_flange']['locating'].get('angles_step_deg')):
        sets.append(_v(I['output_flange']['locating']['angles_step_deg']))
    for alpha in np.arange(0, 180, 0.5):
        good = True
        for s in sets:
            a = np.array(s, float)
            b = (2 * alpha - a) % 360
            if not all(np.min(np.abs((b - x + 180) % 360 - 180)) < 1.0 for x in a):
                good = False
                break
        if good:
            return float(alpha)
    return None


def check_pattern(spec, S, R, rep):
    import json
    I = json.load(open(os.path.join(G.ROOT, 'spec', 'actuator_interfaces.json'), encoding='utf-8'))
    pats = {}
    for ln in left(spec):
        for g in spec['links'][ln]['geoms']:
            if g.get('hw', {}).get('type') == 'screw' and g['hw'].get('which'):
                pats.setdefault((g['hw']['joint'], g['hw']['which']), []).append(g['hw'])
    seen = set()
    for ln in left(spec):
        for g in spec['links'][ln]['geoms']:
            if g['role'] != 'actuator' or g['name'].startswith('R_'):
                continue
            jn = g['name'].split(' actuator')[0].replace(' ', '_')
            model = g['outline']['model']
            e = I[model]
            hub = pats.get((jn, 'hub'), [])
            hub_n = _v(e['output_flange']['count'])
            rep.add('pattern', f'{g["name"]}: hub', len(hub) == hub_n, f'{len(hub)} of {hub_n} {_v(e["output_flange"]["thread"])} placed' + ('' if hub else ': no screws in the spec yet'), value=float(len(hub)))
            st = {w: pats.get((jn, w), []) for w in ('stator_front', 'stator_rear')}
            n_front = _v(e['stator_mounting'][0]['count'])
            best = max(st.values(), key=len)
            rep.add('pattern', f'{g["name"]}: stator', len(best) >= n_front, f'{len(best)} of {n_front} {_v(e["stator_mounting"][0]["thread"])} placed' + ('' if best else ': no screws in the spec yet'), value=float(len(best)))
            # placed screws, by the angle they were really placed at
            for w, hs in {'hub': hub, **st}.items():
                if not hs:
                    continue
                entry = e['output_flange'] if w == 'hub' else next(s for s in e['stator_mounting'] if ('rear' in s.get('face', '')) == (w == 'stator_rear'))
                hits = keepout_hits(model, entry, hs[0]['head_d'] / 2, [h['angle'] for h in hs], outward=hs[0]['length'])
                rep.add('pattern', f'{g["name"]}: {w} screws clear of connector, cable and bumps', not hits, ('; '.join(f'at {a:.0f} deg: {n}' for a, n, _ in hits[:4]) if hits else f'{len(hs)} checked'), value=-float(len(hits)))
            # is the same part usable on the right?
            if g['name'].startswith('L') and model not in seen:
                seen.add(model)
                a = mirror_ok(model)
                rep.add('pattern', f'{model}: the same part fits the right side', a is not None, f'its patterns read the same reflected across {a:.1f} deg' if a is not None else "its patterns don't mirror onto themselves: the right side's motor is the left's turned, so its holes land elsewhere. Place each side from the real part, not the mirror, when that joint's screws go in", warn=True)
    # every model's drawn patterns against its own connectors
    for model in ('RS00', 'RS02', 'RS03', 'RS04', 'RS06'):
        e = I[model]
        for label, entry in [('hub', e['output_flange'])] + [(f"stator {s.get('face', '')}", s) for s in e['stator_mounting'] if _v(s.get('thread_depth_mm')) is not None]:
            head = 3.5 if '4' in str(_v(entry['thread'])) else 2.75 if '3' in str(_v(entry['thread'])) else 4.25
            hits = keepout_hits(model, entry, head)
            rep.add('pattern', f'{model} drawn {label}: no hole on a connector, cable or bump', not hits, '; '.join(f'at {a:.0f} deg: {n}' for a, n, _ in hits[:4]) if hits else 'clear', value=-float(len(hits)))


# ---------------------------------------------------------------- access

def build_state(spec):
    """The build order as sets. Returns (steps, later) where later[k] holds names not yet on before step k, and each pattern's step."""
    steps = spec.get('hardware', {}).get('build', [])
    pattern_step = {}
    for k, st in enumerate(steps):
        for p in st['screws']:
            pattern_step[p] = k
    return steps, pattern_step


def absent_at(spec, steps, pattern_step, k, driving=None):
    """Names not yet on the robot while step k is under way: parts of later steps, and screws of patterns driven later (or of this one, except the screw in hand)."""
    out = set()
    for m, st in enumerate(steps):
        if m > k:
            out |= set(st['parts'])
    for ln in spec['links']:
        for g in spec['links'][ln]['geoms']:
            h = g.get('hw', {})
            if h.get('type') == 'screw' and h.get('pattern') in pattern_step and pattern_step[h['pattern']] >= k:
                out.add(g['name'])
    return out


def check_assembly(spec, S, R, rep):
    """Each step's parts, started off along their path, slide home without passing through what's already on. Their seats are drawn into each other by a hair, so the last millimetre isn't asked."""
    import checks as C
    steps, pattern_step = build_state(spec)
    if not steps:
        return
    idx = named(spec)
    links = {idx[n][0] for st in steps for n in st['parts'] if n in idx}
    for k, st in enumerate(steps):
        gone = absent_at(spec, steps, pattern_step, k)
        moving = [n for n in st['parts'] if n in idx]
        missing = [n for n in st['parts'] if n not in idx]
        on = []
        for ln in links:
            for i, g in enumerate(spec['links'][ln]['geoms']):
                if g['role'] in ('struct', 'actuator', 'linkage', 'hardware', 'bearing', 'shell') and g['name'] not in gone and g['name'] not in moving and S.geom.get((ln, i)) is not None:
                    on.append((g['name'], S.wall.get((ln, i), S.geom[(ln, i)])))
        d = np.array(st['path'], float) if st['path'] else None
        worst = (0.0, '', '')
        if d is not None:
            for n in moving:
                m = S.geom[idx[n][:2]]
                for s in (1.0, 0.75, 0.5, 0.25, max(0.0015 / st['distance'], 0.02)):
                    x = m.translate((d * st['distance'] * s).tolist())
                    for nm, b in on:
                        v = C.overlap(x, b) * 1e9
                        if v > worst[0]:
                            worst = (v, n, nm)
        ok = worst[0] <= TOUCH_MM3 and not missing
        txt = st['why'] or ', '.join(st['parts'])
        rep.add('assembly', f"step {k + 1}: {', '.join(st['parts'])[:80]}", ok, (txt + ': clear along its path') if ok else (f"{worst[1]} passes through {worst[2]}, {worst[0]:.0f} mm3, on its way in" if worst[0] > TOUCH_MM3 else f'not in the spec: {missing}'), value=-worst[0])
    # every moving part named once, and every pattern of this joint driven once
    all_named = [n for st in steps for n in st['parts']]
    rep.add('assembly', 'each part goes on once', len(all_named) == len(set(all_named)), '', value=-float(len(all_named) - len(set(all_named))))


def blocked_by(S, spec, h, name, M=None, tool_r=None, covers=True, absent=()):
    """Which parts a straight key along this screw's axis meets, from just above its head up KEY_REACH. Rigid parts, covers' walls, other hardware, at the poses M (default zero). The knit comes off, so it isn't counted."""
    import checks as C
    pos = np.array(h['pos'], float)
    d = np.array(h['dir'], float)
    start = pos - d * (h['head_k'] / 1000 + KEY_CLEAR)
    r = (tool_r if tool_r else h['socket'] / math.sqrt(3) / 1000 + KEY_CLEAR)
    tool = to_m(G.mesh_of(dict(kind='cyl', c=list(start - d * KEY_REACH / 2), axis=list(-d), r=r, h=KEY_REACH)))
    hits = {}
    for ln, l in spec['links'].items():
        for i, g in enumerate(l['geoms']):
            if g['role'] == 'cut' or g['name'] == name or g['role'] not in ('struct', 'actuator', 'shell', 'hardware', 'bearing', 'linkage') or S.geom.get((ln, i)) is None:
                continue
            if g['role'] == 'shell' and not covers:
                continue
            if g['name'] in absent:
                continue
            body = S.wall.get((ln, i), S.geom[(ln, i)])
            if M is not None:
                body = C.posed(body, M[ln])
            if C.overlap(tool, body) > 1e-9:
                hits[g['name']] = ln
    return hits


def check_access(spec, S, R, rep):
    import checks as C
    steps, pattern_step = build_state(spec)
    for pat, screws in all_patterns(spec).items():
        n = len(screws)
        if pat in pattern_step:                # asked at its moment in the build order
            k = pattern_step[pat]
            gone = absent_at(spec, steps, pattern_step, k)
            free, blockers = 0, {}
            for ln, i, g in screws:
                hits = blocked_by(S, spec, g['hw'], g['name'], absent=gone)
                if hits:
                    for x in hits:
                        blockers[x] = blockers.get(x, 0) + 1
                else:
                    free += 1
            who = ', '.join(f'{a} x{b}' for a, b in sorted(blockers.items(), key=lambda kv: -kv[1])[:5])
            rep.add('access', pat, free == n, f'{free} of {n} reachable at step {k + 1} of the build order, {KEY_REACH * 1000:.0f} mm of key' + (f'; blocked by {who}' if who else ''), value=float(free))
            continue
        free, blockers, poses, bare = 0, {}, set(), 0
        for ln, i, g in screws:
            h = g['hw']
            hits = blocked_by(S, spec, h, g['name'])
            bare += 0 if blocked_by(S, spec, h, g['name'], covers=False) else 1
            if hits:
                # does bending the joint next door clear it? Only parts of one other link in the way, and that link joined to this one
                others = set(hits.values())
                cleared = False
                if ln not in others and len(others) == 1:
                    other = next(iter(others))
                    jx = next((j for j in spec['joints'] if {j['parent'], j['child']} == {ln, other}), None)
                    if jx is not None:
                        for q in np.linspace(jx['lower'], jx['upper'], 9):
                            Mx = R.fk({jx['name']: float(q)})
                            # the key comes with the screw's own link
                            Minv = np.linalg.inv(Mx[ln])
                            Mrel = {k: Minv @ Mx[k] for k in Mx}
                            if not blocked_by(S, spec, h, g['name'], Mrel):
                                cleared = True
                                poses.add(f"{jx['name']} {float(q):+.2f}")
                                break
                if cleared:
                    free += 1
                    continue
                for k in hits:
                    blockers[k] = blockers.get(k, 0) + 1
            else:
                free += 1
        who = ', '.join(f'{k} x{v}' for k, v in sorted(blockers.items(), key=lambda kv: -kv[1])[:5])
        rep.add('access', pat, free == n, f'{free} of {n} reachable along their axis, {KEY_REACH * 1000:.0f} mm of key' + (f', with {", ".join(sorted(poses))}' if poses else '') + (f'; the rest blocked by {who} (build order not considered); {bare} of {n} with the covers and hubs off' if who else ''), value=float(free) + 0.01 * bare)
