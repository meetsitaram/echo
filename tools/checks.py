"""Physical-validity checks for the replica, run offline on spec/robot.json before anything is built in Fusion.

  python -I tools/checks.py                 all checks, report in reports/checks.md, exit 1 on any FAIL
  python -I tools/checks.py --only sweep    one group (targets mass axis static connect skin sweep poses
                                            loadpath contact pivot linkage stator)
  python -I tools/checks.py --bad torque    break the design on purpose and show the matching check FAILs
                                            (cases: torque collide offaxis floating mass poke; and for the load checks hipshrink
                                            spineshrink norods token tokenedge pa12 rolebad newpart nostator thinpin; for hardware: motorinto shortthread
                                            longscrew protrude edge ligament fewscrews smallbearing thinseat noshoulder missingscrew ontab blockkey)
  python -I tools/checks.py --good pin      fix one thing on purpose and show the matching row goes to PASS: a check that can never
                                            pass proves as little as one that can never fail (cases: pin link stator frame; hardware: deepthread bigbearing keyclear)

What each group guards against (the CDX-3R lesson it carries over in brackets):
  targets  size and proportions match Figure 03 (published specs + photo estimates)
  mass     the bottom-up mass (tools/mass_model.py: every part from its material) at most 10 % under Figure 03's
           61 kg and never more than 1.5 kg over
  axis     every directly driven actuator is coaxial with its joint
  static   at the zero pose no rigid part is inside another part, and no skin is inside a non-adjacent link
  covers   no cover (closed on its flat cut faces, as Fusion draws it) passes through a motor of its own link or a
           neighbouring one
           (collision_check: every bought part modelled, in every pose)
  connect  each link is one connected solid and each joint axis passes through both links it joins
           (design_rules 'floating': nothing hangs in thin air). It proves only that shapes touch: a 2.7 x 3.5 mm hip bracket,
           a 2 x 2 mm spine and deleted ankle rods all passed. Kept for the covers and the axes; the groups below say whether it holds
  loadpath each named section (tools/strength.py SECTIONS: cut from the real solids) at safety factor 2 to yield under 3,400 N, walking,
           motor peak and payload; only struct or linkage parts of a material with a strength count; every rigid part is on a path or excused
  contact  each rigid interface on a path has the area its force needs; each link's frame is one piece at 100 mm2 a contact, with motors
           and passengers not counting as frame (a part held by a motor's case, or a 0.02 mm touch, is not joined)
  pivot    every passive joint and every drawn bearing seat has a modelled pin or bearing touching both links, sized for its load
  linkage  each remote drive really reaches: motor, crank, rod, output; rod ends have pivots; moving parts clear of fixed plates
  stator   each motor is held through a real mounting face (its end), not by a part overlapping its side wall
  skin     internal parts stay inside the skin, except parts meant to show (exposed actuators)
  sweep    each joint swept through its limits: limits are reachable without collisions (motion sweep)
  poses    task poses: static joint torques vs actuator ratings, centre of mass over the feet, no collisions
  motorfit fastener holes bolted bearing pattern access: real standard parts (tools/hw_checks.py, thresholds and reasons there). What rests on assumptions is in spec/assumptions.json, printed at the end of the report
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import manifold3d as mf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom as G  # noqa: E402
import hw_checks as HC  # noqa: E402
import mass_model as MM  # noqa: E402

ROOT = G.ROOT
G_ACC = 9.81
TOUCH = 0.3e-6            # m^3 (0.3 cm^3): intersections below this count as touching, not colliding
ARM_KEYS, LEG_KEYS = ('shoulder', 'elbow', 'wrist', 'hand'), ('hip', 'knee', 'ankle', 'foot')


# ---------------------------------------------------------------- report

class Report:
    def __init__(self):
        self.rows = []
        self.assumptions = []

    def add(self, group, item, ok, detail='', warn=False, value=None):
        """value: a number where higher is better, so --bad can tell a row that got worse from one that was already failing."""
        status = 'PASS' if ok else ('WARN' if warn else 'FAIL')
        self.rows.append(dict(group=group, item=item, status=status, detail=detail, value=value))
        return ok

    def failing(self):
        return [r for r in self.rows if r['status'] == 'FAIL']

    def write(self, path, header=''):
        lines = [f'# Physical checks\n', header, '',
                 f"{len(self.rows)} checks: {sum(r['status'] == 'PASS' for r in self.rows)} pass, "
                 f"{sum(r['status'] == 'WARN' for r in self.rows)} warn, **{len(self.failing())} fail**", '']
        for grp in dict.fromkeys(r['group'] for r in self.rows):
            rows = [r for r in self.rows if r['group'] == grp]
            lines += [f'## {grp}', '', '| status | item | detail |', '|---|---|---|']
            lines += [f"| {r['status']} | {r['item']} | {r['detail']} |" for r in rows]
            lines.append('')
        if self.assumptions:
            lines += ['## What the hardware checks assume', '', 'From `spec/assumptions.json`: change one there and run the checks. All status `assumed`.', '', '| id | value | basis | source | what it risks |', '|---|---|---|---|---|']
            lines += [f"| {r['id']} | {r['value']} {r['unit']} | {r['basis']} | {r['source']} | {r['risk']} |" for r in self.assumptions]
            lines.append('')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        with open(path.replace('.md', '.json'), 'w', encoding='utf-8') as f:
            json.dump(self.rows, f, indent=1)


# ---------------------------------------------------------------- solids

def to_manifold(mesh):
    m = mf.Manifold(mf.Mesh(vert_properties=np.asarray(mesh.vertices, np.float32),
                            tri_verts=np.asarray(mesh.faces, np.uint32)))
    if m.status() != mf.Error.NoError:
        raise ValueError(f'not a manifold mesh: {m.status()}')
    return m


def union(ms):
    ms = [m for m in ms if m is not None]
    return mf.Manifold.batch_boolean(ms, mf.OpType.Add) if len(ms) > 1 else (ms[0] if ms else None)


def posed(m, M):
    return m.transform(M[:3, :].astype(np.float64))


def overlap(a, b):
    if a is None or b is None:
        return 0.0
    (alo, ahi), (blo, bhi) = bbox(a), bbox(b)
    if np.any(alo > bhi) or np.any(blo > ahi):
        return 0.0
    return (a ^ b).volume()


def bbox(m):
    b = m.bounding_box()
    return np.array(b[:3]), np.array(b[3:])


WALL = 0.003              # skin wall thickness for collisions (PA12 cover / fabric over a frame)


class Solids:
    """Manifolds per geom and per link at the zero pose.

    all:   every solid of the link (skins as solids): for connectivity and the look
    hard:  what can actually collide: rigid parts plus the skins' WALLS (a skin is hollow; parts of other links
           may sit in its cavity as long as they never touch the wall)
    """

    def __init__(self, spec):
        self.spec = spec
        self.geom, self.wall, self.soft = {}, {}, {}
        for ln, l in spec['links'].items():
            # 'cut' geoms are not parts: they are removed from their targets (a window through a shoe)
            cuts = {}
            for g in l['geoms']:
                if g['role'] == 'cut':
                    m = to_manifold(G.mesh_of(g))
                    for t in g['targets']:
                        cuts[t] = m if t not in cuts else cuts[t] + m
            for i, g in enumerate(l['geoms']):
                if g['role'] == 'cut':
                    self.geom[(ln, i)] = None
                    continue
                self.geom[(ln, i)] = to_manifold(G.mesh_of(g))
                if g['name'] in cuts:
                    self.geom[(ln, i)] = self.geom[(ln, i)] - cuts[g['name']]
                if g['role'] == 'shell':
                    cav = G.inner_of(g, WALL)
                    self.wall[(ln, i)] = self.geom[(ln, i)] - to_manifold(G.mesh_of(cav))
                if g['role'] == 'knit':       # the garment's bulk (2026-11-05): between far links it still takes up room
                    self.soft[(ln, i)] = self.geom[(ln, i)] - to_manifold(G.mesh_of(G.inner_of(g, WALL)))
        self.all, self.shell, self.rigid, self.hard, self.bulk = {}, {}, {}, {}, {}
        for ln, l in spec['links'].items():
            idx = range(len(l['geoms']))
            role = lambda i: l['geoms'][i]['role']
            # bought screws and bearings aren't part of the link's own connected body: a screw through a gap mustn't read as a bracket (they have their own checks, tools/hw_checks.py)
            self.all[ln] = union([self.geom[(ln, i)] for i in idx if role(i) not in ('hardware', 'bearing')])
            self.shell[ln] = union([self.geom[(ln, i)] for i in idx if role(i) in ('shell', 'knit')])
            self.rigid[ln] = union([self.geom[(ln, i)] for i in idx if role(i) in G.RIGID])
            self.hard[ln] = union([self.geom[(ln, i)] for i in idx if role(i) in G.RIGID] +
                                  [self.wall[(ln, i)] for i in idx if role(i) == 'shell'])
            self.bulk[ln] = union([self.geom[(ln, i)] for i in idx if role(i) in G.RIGID] +
                                  [self.wall[(ln, i)] for i in idx if role(i) == 'shell'] +
                                  [self.soft[(ln, i)] for i in idx if role(i) == 'knit'])


def exposed_names(spec):
    """Geoms meant to show: exposed actuators, their flanges, and geoms flagged exposed."""
    out = {g['name'] for l in spec['links'].values() for g in l['geoms'] if g.get('exposed')}
    for j in spec['joints']:
        if j['act'] and j['exposed'] == 'flange':        # only the flange shows (in a seam); the actuator is covered
            out.add(f"{j['name']} flange")
        elif j['act'] and j['exposed']:
            out |= interface(j)
    return out


def interface(j):
    """Names of the geoms that are meant to touch at joint j: its actuator and its flange."""
    return {f"{j['name']} actuator (buy) {j['act']}", f"{j['name']} flange"}


def group(link):
    if any(k in link for k in ARM_KEYS):
        return 'arm'
    if any(k in link for k in LEG_KEYS):
        return 'leg'
    return 'body'


# ---------------------------------------------------------------- checks

def measure(spec, S):
    T = {ln: S.shell[ln] or S.all[ln] for ln in spec['links']}     # what is seen: the skins
    lo = {ln: bbox(m)[0] for ln, m in T.items()}
    hi = {ln: bbox(m)[1] for ln, m in T.items()}
    J = {j['name']: j for j in spec['joints']}
    lobe = next(S.geom[(ln, i)] for ln, l in spec['links'].items() for i, g in enumerate(l['geoms'])
                if g['name'] == 'L hip lobe')           # with its cuts
    verts = np.vstack([m.to_mesh().vert_properties[:, :3] for m in T.values()])
    band = verts[(verts[:, 2] > 1.2) & (verts[:, 2] < 1.45)]
    # the head shape alone: the head link also carries the neck (it tilts with the head, iteration 18)
    hd = G.mesh_of([g for g in spec['links']['head']['geoms'] if g['name'] == 'head'][0]).bounds
    return {
        'height_m': max(h[2] for h in hi.values()),
        'chin_z': hd[0][2],
        'shoulder_top_z': float(max(v[2] for i, g in enumerate(spec['links']['torso']['geoms']) if g['name'] == 'L shoulder yoke'
                                    for v in S.geom[('torso', i)].to_mesh().vert_properties[:, :3] if abs(v[1]) > 0.17)),
        'elbow_z': J['L_elbow']['origin'][2],
        'wrist_z': J['L_wrist_pitch']['origin'][2],
        'fingertip_z': lo['L_hand'][2],
        'knee_z': J['L_knee']['origin'][2],
        # the black hips' lower edge in the front and side views: the lobe's underside in its front half
        'hips_black_bottom_z': float(min(v[2] for v in lobe.to_mesh().vert_properties[:, :3] if v[0] >= -0.02)),
        # where the grey torso ends and the black waist begins
        'pelvis_band_top_z': G.mesh_of([g for g in spec['links']['torso']['geoms'] if g['name'] == 'torso'][0])
        .bounds[0][2],
        'head_height_m': hd[1][2] - hd[0][2],
        'head_depth_m': hd[1][0] - hd[0][0],
        'head_width_m': hd[1][1] - hd[0][1],
        'chest_depth_m': hi['torso'][0] - lo['torso'][0],
        'bideltoid_width_m': band[:, 1].max() - band[:, 1].min(),
        'foot_length_m': hi['L_foot'][0] - lo['L_foot'][0],
    }


def check_targets(spec, S, rep):
    tg = json.load(open(os.path.join(ROOT, 'spec', 'targets.json'), encoding='utf-8'))
    m = measure(spec, S)
    h = m['height_m']
    rep.add('targets', 'height', abs(h - tg['published']['height_m']) <= 0.01,
            f"{h:.3f} m (Figure 03: {tg['published']['height_m']} m, +/- 0.010)")
    for k, (v, tol) in tg['photo'].items():
        rep.add('targets', k, abs(m[k] - v) <= tol, f'{m[k]:.3f} (photo {v} +/- {tol})')
    return m


MASS_BELOW = 0.10          # the bottom-up mass may be this much under Figure's, never over it by more than 1.5 kg


def check_mass(spec, S, rep):
    """The bottom-up mass (every part from its material, tools/mass_model.py) against Figure 03's. Until
    2026-10-31 the link masses were hand budgets summing to 61 kg exactly, with the battery as the balance."""
    tg = json.load(open(os.path.join(ROOT, 'spec', 'targets.json'), encoding='utf-8'))['published']['mass_kg']
    inert = MM.link_inertials(spec, S)
    total = sum(v[0] for v in inert.values())
    rep.add('mass', 'total', tg * (1 - MASS_BELOW) <= total <= tg + 1.5,
            f'{total:.2f} kg bottom-up (Figure 03: {tg} kg; allowed {tg * (1 - MASS_BELOW):.1f} .. {tg + 1.5:.1f})')


def check_axis(spec, rep):
    for j in spec['joints']:
        if not j['act'] or j['remote']:
            continue
        a, o, c = G.AXIS[j['axis']], np.array(j['origin']), np.array(j['act_c'])
        d = c - o
        off = np.linalg.norm(d - (d @ a) * a)
        rep.add('axis', j['name'], off < 1e-4, f"{j['act']} {off * 1000:.1f} mm off the axis")


def check_static(spec, S, R, rep):
    links = list(spec['links'])
    bad = 0
    # rigid vs rigid, every pair of links (adjacent too), except each joint's actuator/flange interface
    allowed = set()
    for j in spec['joints']:
        if j['act']:
            allowed.add(frozenset(interface(j)))
    rigid = [(ln, i, g) for ln in links for i, g in enumerate(spec['links'][ln]['geoms']) if g['role'] in G.RIGID]
    for x in range(len(rigid)):
        for y in range(x + 1, len(rigid)):
            (la, ia, ga), (lb, ib, gb) = rigid[x], rigid[y]
            if la == lb or frozenset((ga['name'], gb['name'])) in allowed:
                continue
            v = overlap(S.geom[(la, ia)], S.geom[(lb, ib)])
            if v > TOUCH:
                bad += 1
                rep.add('static', f"{ga['name']} / {gb['name']}", False, f'rigid parts overlap {v * 1e6:.1f} cm3')
    # skin vs anything of a non-adjacent link
    for x in range(len(links)):
        for y in range(x + 1, len(links)):
            a, b = links[x], links[y]
            if R.adjacent(a, b):
                continue
            # the garment's bulk counts between far links only: near a joint the skin folds
            sa, sb = (S.bulk, S.bulk) if R.distance(a, b) > NEAR else (S.hard, S.hard)
            v = overlap(sa[a], sb[b])
            if v > TOUCH:
                bad += 1
                rep.add('static', f'{a} / {b}', False, f'non-adjacent links touch (skin wall or rigid) {v * 1e6:.1f} cm3')
    rep.add('static', 'zero pose', bad == 0, f'{bad} overlaps')


def check_covers(spec, S, R, rep):
    """A cover must not pass through a motor of its own link or of a link next to it. The skin tested here is
    closed on the cover's flat cut faces (Fusion draws them as solid faces); the collision walls leave those faces
    open, so a hip lobe's flat inner face sliced through both hip motors unseen (user, 2026-10-31)."""
    acts = [(ln, i, g) for ln, l in spec['links'].items() for i, g in enumerate(l['geoms']) if g['role'] == 'actuator']
    bad = 0
    for ln, l in spec['links'].items():
        for i, g in enumerate(l['geoms']):
            if g['role'] != 'shell' or S.geom.get((ln, i)) is None:
                continue
            skin = S.geom[(ln, i)] - to_manifold(G.mesh_of(G.inner_closed(g, WALL)))
            for la, ia, ga in acts:
                if la != ln and not R.adjacent(la, ln):
                    continue
                v = overlap(skin, S.geom[(la, ia)])
                if v > TOUCH:
                    bad += 1
                    rep.add('covers', f"{g['name']} / {ga['name']}", False,
                            f'the cover passes through the motor: {v * 1e6:.1f} cm3')
    rep.add('covers', 'motors clear of the covers', bad == 0, f'{bad} cover / motor crossings')


def check_connect(spec, S, rep):
    """Rigid parts of a link must be one structure on their own (covers carry no load: a missing bracket hidden
    under a cover is still missing, 2026-10-31 calibration), and every cover must be mounted on it."""
    # pieces under 0.01 cm3 are slivers the boolean union leaves where cut pieces meet, not parts (2026-10-31: a
    # zero-volume sliver where the right shoe's toe and heel pieces overlap)
    # 2026-11-05: decompose hands back a sealed cavity as a piece of negative volume (the real RS00's steps under the wrist cuff closed two). A hole inside a part isn't a loose part
    pieces = lambda m: len([p for p in m.decompose() if p.volume() > 1e-8])
    for ln in spec['links']:
        if S.rigid[ln] is not None:
            n = pieces(S.rigid[ln])
            rep.add('connect', f'{ln} structure', n == 1,
                    f'rigid parts form {n} separate pieces (a bracket or frame is missing)' if n != 1 else '')
        n = pieces(S.all[ln])
        rep.add('connect', f'{ln} covers mounted', n == 1, f'{n} separate pieces' if n != 1 else '')
    t = np.linspace(-0.18, 0.18, 361)
    for j in spec['joints']:
        a, o = G.AXIS[j['axis']], np.array(j['origin'])
        line = o + np.outer(t, a)
        hit = {}
        for side in ('parent', 'child'):
            gs = spec['links'][j[side]]['geoms']
            hit[side] = any(G.inside(g, line).any() for g in gs if g['role'] in G.RIGID) or \
                any(G.inside(g, line).any() for g in gs)
        rigid_hit = {side: any(G.inside(g, line).any() for g in spec['links'][j[side]]['geoms'] if g['role'] in G.RIGID)
                     for side in ('parent', 'child')}
        ok = rigid_hit['parent'] and rigid_hit['child']
        miss = [s for s in ('parent', 'child') if not rigid_hit[s]]
        rep.add('connect', f"{j['name']} axis", ok,
                '' if ok else f"axis passes through no rigid part of the {' and '.join(miss)} "
                              f"({', '.join(j[s] for s in miss)})")


# ---------------------------------------------------------------- does it hold (thresholds and their reasons: tools/strength.py)

def left_links(spec):
    """Left side and centre: the right is built as their mirror in spec/robot.py, as the sweep assumes."""
    return [ln for ln in spec['links'] if not ln.startswith('R_')]


def carrying_parts(spec, ln):
    import strength as ST
    return [(i, g) for i, g in enumerate(spec['links'][ln]['geoms']) if g['role'] != 'cut' and ST.why_not_carrying(g) is None]


def frame_components(spec, S, ln):
    """A link's frame: its carrying parts joined only by contacts a bolt could use. Motors are not frame: a part held only by a
    motor's case is not joined, as nothing says the case is rated for it. Returns (components as sets of indices, every contact found)."""
    import strength as ST
    parts = carrying_parts(spec, ln)
    par = {i: i for i, _ in parts}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    seen = []
    for a in range(len(parts)):
        for b in range(a + 1, len(parts)):
            i, k = parts[a][0], parts[b][0]
            ma, mb = S.geom[(ln, i)], S.geom[(ln, k)]
            (la, ha), (lb, hb) = bbox(ma), bbox(mb)
            if np.any(la > hb + 1e-3) or np.any(lb > ha + 1e-3):
                continue
            c = ST.contact(ma, mb)
            if c['kind'] == 'gap':
                continue
            seen.append((i, k, c))
            if c['kind'] in ('overlap', 'flush') and c['area'] >= ST.CONTACT_MIN_MM2:
                par[find(i)] = find(k)
    comps = {}
    for i, _ in parts:
        comps.setdefault(find(i), set()).add(i)
    return list(comps.values()), seen


def check_loadpath(spec, S, rep):
    import strength as ST
    M = ST.Model(spec, S)
    for r in ST.evaluate_all(M):
        if r['why'] or not r['case']:
            rep.add('loadpath', r['name'], False, r['why'] or 'nothing to read', value=0.0)
            continue
        g = r['results'][r['case']]
        ok = r['sf'] >= ST.SF_MIN
        rep.add('loadpath', r['name'], ok, f"SF {r['sf']:.2f} to yield (needs {ST.SF_MIN:.0f}), {ST.CASE_LABEL[r['case']]}: {g['vm']:.0f} MPa on {r['size']} mm {r['material']}; a hand estimate", value=r['sf'])
        if ok and r['sf_hollow'] is not None and r['sf_hollow'] < ST.SF_MIN:
            rep.add('loadpath', f"{r['name']}, at the mass model's fill", False, f"SF {r['sf_hollow']:.2f} if as hollow as its mass assumes (fill {r['fill']:.2f}); solid {r['sf']:.2f}", warn=True)
    # coverage: a rigid part on no path must say why, so a new part can't slip past this check
    named = {n for x in ST.SECTIONS if x['parts'] != '*' for n in x['parts']} | {n for _, a, b, _ in ST.INTERFACES for n in (a, b)} | set(ST.NOT_ON_PATH)
    loose = [g['name'] for ln in left_links(spec) for g in spec['links'][ln]['geoms'] if g['role'] in ('struct', 'linkage') and g['name'] not in named]
    rep.add('loadpath', 'coverage: every rigid part on a path or excused', not loose, 'not in strength.py SECTIONS, INTERFACES or NOT_ON_PATH: ' + ', '.join(loose) if loose else '', value=-float(len(loose)))
    gone = sorted(n for n in named if n not in M.part)
    rep.add('loadpath', 'coverage: every named part exists', not gone, 'in strength.py but not in the spec: ' + ', '.join(gone) if gone else '', value=-float(len(gone)))
    # role against material: metal drawn as a cover counts for nothing here, and that should be a choice
    for ln in left_links(spec):
        for g in spec['links'][ln]['geoms']:
            if g['role'] in ('shell', 'knit') and g.get('material') == 'al7075':
                rep.add('loadpath', f"role: {g['name']}", False, f"7075 with role {g['role']}: not counted as load-bearing. If it carries load, make it struct", warn=True)


def check_contact(spec, S, rep):
    import strength as ST
    M = ST.Model(spec, S)
    for r in ST.evaluate_interfaces(M):
        extra = f", {r['gap']:.2f} mm of air" if r['kind'] == 'gap' and r['gap'] is not None else ''
        rep.add('contact', f"{r['a']} / {r['b']}", r['ok'], f"{r['area']:.0f} mm2 ({r['kind']}{extra}, {r['depth']:.2f} deep), needs {r['need']:.0f} to pass {r['v']:.0f} N", value=r['area'] if r['kind'] in ('overlap', 'flush') else 0.0)
    for ln in left_links(spec):
        gs = spec['links'][ln]['geoms']
        comps, seen = frame_components(spec, S, ln)
        if comps:
            n = len(comps)
            names = '; '.join(' + '.join(gs[i]['name'] for i in sorted(c)) for c in comps) if n > 1 else ''
            weak = [f"{gs[i]['name']} / {gs[k]['name']} {c['area']:.0f} mm2 {c['kind']}" for i, k, c in seen if any(i in x and k not in x for x in comps)]
            rep.add('contact', f'{ln}: frame in one piece', n == 1, f'{n} pieces: {names}' + (f". Only by: {'; '.join(weak)}" if weak else '. Nothing joins them') if n > 1 else '', value=-float(n))
        # passengers (a pack, a computer) need a seat on the frame: a seat on another passenger is not one
        for i, g in enumerate(gs):
            if g['role'] != 'struct' or ST.why_not_carrying(g) is None:
                continue
            best = 0.0
            for k, _ in carrying_parts(spec, ln):
                c = ST.contact(S.geom[(ln, i)], S.geom[(ln, k)])
                if c['kind'] in ('overlap', 'flush'):
                    best = max(best, c['area'])
            rep.add('contact', f"{g['name']}: seated on the frame", best >= ST.CONTACT_MIN_MM2, f'{best:.0f} mm2 on carrying parts (needs {ST.CONTACT_MIN_MM2:.0f})', value=best)


PIVOT_GAP = 1e-4          # a modelled bearing or pin touches both links: within this, as contact is read elsewhere


def pivot_row(spec, S, rep, M, j, c, label):
    """Is there a pin or bearing at this pivot? A cylinder on the axis, in either link, with the two links' frames touching there."""
    import strength as ST
    a = G.AXIS[j['axis']]
    c = np.asarray(c, float)
    # a catalogue bearing here, its races touching the parts of both links: the row is that fact; its rating is read in group 'bearing'
    for ln in (j['parent'], j['child']):
        for i, g in enumerate(spec['links'][ln]['geoms']):
            h = g.get('hw', {})
            if h.get('type') == 'bearing' and h['axis'] == j['axis']:
                d = np.array(h['pos'], float) - c
                if np.linalg.norm(d - (d @ a) * a) < 5e-4 and abs(d @ a) < 0.012:
                    idx = {x['name']: (l2, k) for l2 in (j['parent'], j['child']) for k, x in enumerate(spec['links'][l2]['geoms'])}
                    m = S.geom[(ln, i)]
                    gaps = [m.min_gap(S.geom[idx[p]], 0.05) for p in h['inner_parts'] + h['outer_parts'] if p in idx]
                    inner = min(m.min_gap(S.geom[idx[p]], 0.05) for p in h['inner_parts'] if p in idx) if any(p in idx for p in h['inner_parts']) else 1.0
                    outer = min(m.min_gap(S.geom[idx[p]], 0.05) for p in h['outer_parts'] if p in idx) if any(p in idx for p in h['outer_parts']) else 1.0
                    ok = max(inner, outer) <= PIVOT_GAP
                    rep.add('pivot', label, ok, f"{h['designation']}: inner race on {h['inner_parts'][0]}, outer in {h['outer_parts'][0]}" + ('; rating in group bearing' if ok else f"; {max(inner, outer) * 1000:.2f} mm of air at a race"), value=-max(inner, outer))
                    return
    zone = to_manifold(G.mesh_of(dict(kind='cyl', c=list(c), axis=j['axis'], r=0.03, h=0.02 if 'seat' in label else 0.07)))
    pins, near = [], {}
    for side in ('parent', 'child'):
        ln = j[side]
        ms = []
        bored = {t for g in spec['links'][ln]['geoms'] if g['role'] == 'cut' for t in g['targets']}
        for i, g in carrying_parts(spec, ln):
            # not the motor's own output (the question here is the other support), nor the rods, which pass by on their own business
            if g['name'].endswith(' flange') or g['role'] == 'linkage':
                continue
            ms.append(S.geom[(ln, i)])
            # a disc with a bore in it is the seat, not the pin
            if g['kind'] == 'cyl' and g['axis'] == j['axis'] and g['name'] not in bored:
                d = np.array(g['c'], float) - c
                if np.linalg.norm(d - (d @ a) * a) < 5e-4 and abs(d @ a) < 0.05:
                    pins.append(g)
        u = union(ms)
        near[side] = None if u is None else u ^ zone
    gap = None
    if near['parent'] is not None and near['child'] is not None and near['parent'].volume() > 0 and near['child'].volume() > 0:
        gap = near['parent'].min_gap(near['child'], 0.05)
    touching = gap is not None and gap <= PIVOT_GAP
    if not pins:
        rep.add('pivot', label, False, f"no pin or bearing on the axis; {gap * 1000:.2f} mm of air between the links" if gap is not None else 'no pin or bearing on the axis', value=-(gap or 1.0))
        return
    if not touching:
        rep.add('pivot', label, False, f"no bearing modelled: {pins[0]['name']} is on the axis, with {gap * 1000:.2f} mm of air to the other link" if gap is not None else 'no bearing modelled', value=-(gap or 1.0))
        return
    # sized for its load, in double shear: the smallest pin that touches
    force = ST.F_PEAK * (1.0 if 'ankle' in j['name'] else 0.5)
    if 'knee' not in j['name'] and 'ankle' not in j['name']:
        force = M.cap(j['name']) / 0.0665
    r = min(g['r'] for g in pins)
    sy = min(ST.yield_of(g) for g in pins)
    tau = force / (2 * math.pi * (r * 1000) ** 2)
    sf = 0.577 * sy / tau
    rep.add('pivot', label, sf >= ST.SF_MIN, f"pin d{2000 * r:.0f}, {force:.0f} N in double shear: SF {sf:.1f} to yield (a hand estimate)", value=sf)


def check_pivot(spec, S, rep):
    """Every passive joint needs a modelled pin or bearing, and so does every bearing seat drawn in a part: until one is there, the load
    has no way across. Rod ends are in 'linkage'."""
    import strength as ST
    M = ST.Model(spec, S)
    for j in spec['joints']:
        if j['name'].startswith('R_') or j['act']:
            continue
        pivot_row(spec, S, rep, M, j, j['origin'], f"{j['name']}: passive joint")
    for ln in left_links(spec):
        for g in spec['links'][ln]['geoms']:
            if g['role'] != 'cut' or ' bore' not in g['name'] or g['kind'] != 'cyl':
                continue
            c = np.array(g['c'], float)
            for j in spec['joints']:
                if ln not in (j['parent'], j['child']) or j['axis'] != g['axis']:
                    continue
                d = c - np.array(j['origin'], float)
                a = G.AXIS[j['axis']]
                if np.linalg.norm(d - (d @ a) * a) < 5e-4:
                    pivot_row(spec, S, rep, M, j, c, f"{g['name']}: bearing seat")
                    break


def check_linkage(spec, S, rep):
    """A remote joint's drive must reach: motor to crank to rod to the output. Rods and cranks are role linkage, out of the collision
    checks, so nothing else looks at them: deleted outright, every other check still passed."""
    import strength as ST
    for j in spec['joints']:
        if j['name'].startswith('R_') or not j.get('remote') or j['act']:
            continue
        prefix = j['name'].replace('_', ' ')
        # the drive sits wherever its motor is: the ankle roll's parent is the cross, its motor is up in the calf
        ln = next((l for l in spec['links'] for g in spec['links'][l]['geoms'] if g['role'] == 'actuator' and g['name'].startswith(prefix)), j['parent'])
        geoms = spec['links'][ln]['geoms']
        idx = {g['name']: i for i, g in enumerate(geoms)}
        motor = next((g for g in geoms if g['role'] == 'actuator' and g['name'].startswith(prefix)), None)
        crank = next((g for g in geoms if g['role'] == 'linkage' and g['name'] == f'{prefix} crank'), None)
        rod = next((g for g in geoms if g['role'] == 'linkage' and g['name'] == f'{prefix} rod'), None)
        name = j['name']
        rep.add('linkage', f'{name}: has a motor, crank and rod', bool(motor and crank and rod), f"missing: {', '.join(n for n, x in (('motor', motor), ('crank', crank), ('rod', rod)) if not x)}" if not (motor and crank and rod) else '', value=float(bool(motor)) + bool(crank) + bool(rod))
        if not (motor and crank and rod):
            continue
        mm, mc, mr = (S.geom[(ln, idx[g['name']])] for g in (motor, crank, rod))
        gap = mm.min_gap(mc, 0.05)
        rep.add('linkage', f'{name}: crank reaches the motor', gap <= ST.CONTACT_MIN_DEPTH, f'{gap * 1000:.2f} mm of air and no shaft between the crank and the motor' if gap > ST.CONTACT_MIN_DEPTH else '', value=-gap)
        gap = mc.min_gap(mr, 0.05)
        rep.add('linkage', f'{name}: rod reaches the crank', gap <= ST.CONTACT_MIN_DEPTH, f'{gap * 1000:.2f} mm' if gap > ST.CONTACT_MIN_DEPTH else '', value=-gap)
        gaps = [(mr.min_gap(S.geom[(l, i)], 0.05), g['name']) for l in G.Robot(spec).subtree(j['child']) for i, g in carrying_parts(spec, l)]
        best = min(gaps) if gaps else (1.0, 'nothing')
        rep.add('linkage', f'{name}: rod reaches the output', best[0] <= ST.CONTACT_MIN_DEPTH, f'{best[0] * 1000:.2f} mm to {best[1]}' if best[0] > ST.CONTACT_MIN_DEPTH else f'at {best[1]}', value=-best[0])
        # a rod's two ends are passive joints: a modelled pin or bearing (a cylinder, by name) at each, as for any other pivot
        for end, p in (('crank end', rod['p0']), ('output end', rod['p1'])):
            pts = [g for l in spec['links'].values() for g in l['geoms'] if g['role'] != 'cut' and g['kind'] == 'cyl' and any(w in g['name'] for w in (' pin', ' bearing', ' rod end'))
                   and np.linalg.norm(np.array(g['c'], float) - np.array(p, float)) < 0.01]
            rep.add('linkage', f'{name}: rod {end} has a pivot', bool(pts), 'capsules overlapping, no pin or bearing part' if not pts else pts[0]['name'], value=float(bool(pts)))
        # moving parts through fixed plates: nothing else looks, as linkage is out of the collision checks
        for g, m in ((crank, mc), (rod, mr)):
            for i, h in carrying_parts(spec, ln):
                if h['role'] == 'linkage':
                    continue
                v = overlap(m, S.geom[(ln, i)])
                if v > 1e-8:          # 0.01 cm3, the size the structure check calls a sliver
                    rep.add('linkage', f"{g['name']} clear of {h['name']}", False, f'moving part inside a fixed one, {v * 1e6:.2f} cm3', value=-v)


STATOR_SEAT_DEPTH = 0.002  # a part sunk further than this into a motor is overlapping its body, not seated on its face...
STATOR_POCKET = 0.30       # ...unless it covers this much of the face: a pocket round the motor's end, which does hold it


def check_stator(spec, S, rep):
    """A motor is held at its end: a bolt circle on its back, or the housing ring round the output. Its side wall has no fixings, whatever a
    plate overlapping it looks like. Reads each motor's end faces against the frame of the link that carries it; a part on the back that
    hangs off the motor alone (a post) does not hold the motor to the link. Retailer drawings show the bolt circles: confirm on the maker's."""
    import strength as ST
    for ln in left_links(spec):
        geoms = spec['links'][ln]['geoms']
        comps, _ = frame_components(spec, S, ln)
        vol = lambda c: sum(S.geom[(ln, i)].volume() for i in c)
        main = max(comps, key=vol) if comps else set()
        for i, g in enumerate(geoms):
            if g['role'] != 'actuator':
                continue
            axis = g['axis']
            a = G.AXIS[axis]
            c = np.array(g['c'], float)
            jn = g['name'].split(' actuator')[0]
            j = next((x for x in spec['joints'] if x['name'] == jn), None)
            out = j['out'] if j else 0
            motor = S.geom[(ln, i)]
            best, who = 0.0, ''
            for k in main:
                part = S.geom[(ln, k)]
                inside = motor ^ part
                sunk = 0.0
                if inside.volume() > 1e-13:
                    lo, hi = bbox(inside)
                    sunk = float((hi - lo) @ np.abs(a))
                for sign in (1, -1):
                    is_output = out != 0 and sign == out
                    plane = c + sign * a * g['h'] / 2
                    zone = to_manifold(G.mesh_of(dict(kind='cyl', c=list(plane), axis=axis, r=g['r'], h=0.001)))
                    if is_output:         # the output turns: only the housing ring round it can hold anything
                        zone = zone - to_manifold(G.mesh_of(dict(kind='cyl', c=list(plane), axis=axis, r=0.6 * g['r'], h=0.002)))
                    x = zone ^ part
                    v = x.volume()
                    if v < 1e-13:
                        continue
                    lo, hi = bbox(x)
                    area = v / max(float((hi - lo) @ np.abs(a)), 1e-6) * 1e6
                    face = math.pi * (g['r'] * 1000) ** 2
                    if sunk > STATOR_SEAT_DEPTH and (is_output or area < STATOR_POCKET * face):
                        continue
                    if area > best:
                        best, who = area, geoms[k]['name']
            ok = best >= ST.CONTACT_MIN_MM2
            rep.add('stator', g['name'], ok, f'held by {who}, {best:.0f} mm2 on its end face' if ok else f'no mounting face: {best:.0f} mm2 on its ends (needs {ST.CONTACT_MIN_MM2:.0f}); what touches it overlaps its side wall', value=best)


def check_skin(spec, R, rep):
    """Internal parts must be hidden: inside the skins (any link, zero pose), unless meant to show."""
    rng = np.random.default_rng(1)
    # the garment's allowance: a part under the knit must fit WALL inside it
    shells = [g for l in spec['links'].values() for g in l['geoms'] if g['role'] == 'shell']
    shells += [dict(G.inner_of(g, WALL), name=g['name']) for l in spec['links'].values() for g in l['geoms'] if g['role'] == 'knit']
    cuts = {}                                             # shell name -> the cuts through it (they uncover)
    for l in spec['links'].values():
        for c in l['geoms']:
            for t in c.get('targets', []) if c['role'] == 'cut' else []:
                cuts.setdefault(t, []).append(c)
    exposed = exposed_names(spec)
    for ln, l in spec['links'].items():
        for g in l['geoms']:
            if g['role'] not in G.RIGID or g['name'] in exposed:
                continue
            # 0.5 %: a 5 mm corner of an actuator through the calf (0.3 %) showed as dark marks in Fusion
            # (2026-10-31); the first limit, 5 %, let it through
            p = G.volume_samples(g, 4000, rng)
            inn = np.zeros(len(p), bool)
            for s in shells:
                hit = G.inside(s, p)
                for c in cuts.get(s['name'], []):
                    hit &= ~G.inside(c, p)
                inn |= hit
            frac = 1 - inn.mean()
            rep.add('skin', g['name'], frac <= 0.005, f'{frac * 100:.2f} % outside the skin')


NEAR = 3                  # links at most this many joints apart: a collision between them is structural


def culprit(spec, S, a, Ma, b, Mb):
    """The pair of geoms (one from link a, one from link b) that overlap most, by name."""
    best = (0, '')
    hard = lambda ln, i: S.wall[(ln, i)] if (ln, i) in S.wall else S.geom[(ln, i)]
    for i, ga in enumerate(spec['links'][a]['geoms']):
        if hard(a, i) is None:                            # a cut, not a part
            continue
        ma = posed(hard(a, i), Ma)
        for k, gb in enumerate(spec['links'][b]['geoms']):
            if hard(b, k) is None:
                continue
            v = overlap(ma, posed(hard(b, k), Mb))
            if v > best[0]:
                best = (v, f"{ga['name']} / {gb['name']}")
    return best[1]


def check_sweep(spec, S, R, rep, steps=25):
    """Sweep each joint alone from the zero pose. Collisions between links near each other in the tree are
    structural (FAIL: the limit cannot be reached in any pose). Collisions with far links (a hanging arm meeting
    the hip as the waist turns) depend on the pose: WARN, and left to self-collision avoidance."""
    for j in spec['joints']:
        if j['name'].startswith('R_'):
            continue                                  # mirror of the left side
        moving = set(R.subtree(j['child']))
        static = [l for l in spec['links'] if l not in moving]
        pairs = [(a, b) for a in moving for b in static if not R.adjacent(a, b)]
        # across the joint itself (parent vs child): rigid parts AND the covers' walls, minus only the actuator /
        # flange interface (user, 2026-10-31: "move every joint to min and max, the body must not come in the
        # way"; until then the covers either side of a joint were allowed to overlap, as fabric)
        P, C = spec['links'][j['parent']]['geoms'], spec['links'][j['child']]['geoms']
        ii = interface(j) if j['act'] else set()
        hard = lambda ln, i: S.wall.get((ln, i)) if (ln, i) in S.wall else \
            (S.geom[(ln, i)] if spec['links'][ln]['geoms'][i]['role'] in G.RIGID else None)
        rig = [(hard(j['parent'], x), hard(j['child'], y), P[x]['name'], C[y]['name'])
               for x in range(len(P)) for y in range(len(C))
               if hard(j['parent'], x) is not None and hard(j['child'], y) is not None
               and {P[x]['name'], C[y]['name']} != ii]
        def hits(qv):
            """[(volume, description, structural)] at angle qv, worst first."""
            M = R.fk({j['name']: qv})
            out = []
            for a, b in pairs:
                H = S.bulk if R.distance(a, b) > NEAR else S.hard
                v = overlap(posed(H[a], M[a]), H[b])
                if v > TOUCH:
                    out.append((v, f'{a} / {b}: {culprit(spec, S, a, M[a], b, M[b])}',
                                R.distance(a, b) <= NEAR))
            for ma, mb, na, nb in rig:
                v = overlap(ma, posed(mb, M[j['child']]))
                if v > TOUCH:
                    out.append((v, f'{na} / {nb} (across the joint)', True))
            return sorted(out, key=lambda h: (not h[2], -h[0]))
        msgs, warns, ok = [], [], True
        for side, end in (('lower', j['lower']), ('upper', j['upper'])):
            far_seen = False
            for qv in np.linspace(0, end, steps):
                h = hits(qv)
                near = [x for x in h if x[2]]
                if near:
                    ok = False
                    msgs.append(f"{side} {end:+.2f}: blocked at {qv:+.2f} rad by {near[0][1]} "
                                f"({near[0][0] * 1e6:.0f} cm3)" if qv else f'collides at 0: {near[0][1]}')
                    break
                if h and not far_seen:
                    far_seen = True
                    warns.append(f"{side}: {h[0][1]} at {qv:+.2f} rad (pose-dependent)")
        if ok:
            rep.add('sweep', j['name'], not warns, f"[{j['lower']:+.2f}, {j['upper']:+.2f}] rad clear"
                    + ('; ' + '; '.join(warns) if warns else ''), warn=True)
        else:
            rep.add('sweep', j['name'], False, '; '.join(msgs + warns))


# ---------------------------------------------------------------- statics

POSES = {
    # name: (angles for the left side, mirrored unless given; support; category; payload kg per hand)
    'stand': ({}, 'double', 'continuous', 0),
    'walk crouch': ({'L_hip_pitch': -0.25, 'L_knee': 0.5, 'L_ankle_pitch': -0.25, 'L_shoulder_roll': 0.12},
                    'double', 'continuous', 0),
    # Figure 03's 20 kg payload, taken as a short carry (peak); 10 kg held for as long as needed (rated)
    # payload (user decision 2026-10-31: smaller RS06 elbows, less holding capacity than Figure's 20 kg):
    # 3 kg held for as long as needed (rated; 4 kg needed 109 % of the RS06 rating), 10 kg as a short carry (peak),
    # forearms level
    'carry 3 kg at the chest': ({'L_hip_pitch': -0.15, 'L_knee': 0.3, 'L_ankle_pitch': -0.15,
                                 'L_shoulder_pitch': -0.45, 'L_elbow': -1.3}, 'double', 'continuous', 1.5),
    'carry 10 kg at the chest': ({'L_hip_pitch': -0.15, 'L_knee': 0.3, 'L_ankle_pitch': -0.15,
                                  'L_shoulder_pitch': -0.45, 'L_elbow': -1.3}, 'double', 'peak', 5.0),
    # torso leaning 21 deg as a person's does (iteration 12): with the hips at -1.3 the torso stayed upright and the
    # balance needed 0.86 rad of ankle, past the shoe's 0.8; now 0.77, and less knee torque (70 vs 78 N m)
    'deep squat': ({'L_hip_pitch': -1.6, 'L_knee': 2.0, 'L_ankle_pitch': -0.75},
                   'double', 'peak', 0),
    'arms straight out, 5 kg': ({'L_shoulder_pitch': -1.57}, 'double', 'peak', 2.5),
    # the shoulder roll on its own: both arms raised sideways to horizontal, empty hands (a short move)
    'arms out to the sides': ({'L_shoulder_roll': 1.57}, 'double', 'peak', 0),
    'lift 10 kg from a low table': ({'L_hip_pitch': -0.9, 'L_knee': 1.4, 'L_ankle_pitch': -0.55,
                                     'L_shoulder_pitch': -0.5, 'L_elbow': -0.6},
                                    'double', 'peak', 5.0),
    'single-leg stance (walking)': ({'L_hip_pitch': -0.15, 'L_knee': 0.3, 'L_ankle_pitch': -0.15,
                                     'L_shoulder_roll': 0.12,
                                     'R_hip_pitch': -0.5, 'R_knee': 1.0, 'R_ankle_pitch': 0.0, 'R_hip_roll': -0.08},
                                    'L', 'peak', 0),
    'step up a stair': ({'L_hip_pitch': -1.0, 'L_knee': 1.5, 'L_ankle_pitch': -0.45,
                         'R_hip_pitch': 0.15, 'R_knee': 0.4, 'R_ankle_pitch': 0.0},
                        'L', 'peak', 0),
}
PEAK_MARGIN = 1.5         # peak torque must exceed static torque x this (accelerations, impacts)


class Mass:
    """Link masses and centres of mass from the parts' materials (tools/mass_model.py)."""

    def __init__(self, spec, S):
        self.com, self.m = {}, {}
        for ln, (m, com, I, ps) in MM.link_inertials(spec, S).items():   # every part from its material
            self.com[ln] = com
            self.m[ln] = m


def statics(spec, R, MS, q, support, payload):
    """Joint torques (N m) holding pose q still, feet flat. Returns torques, CoM, support polygon, poses."""
    feet = ['L_foot', 'R_foot'] if support == 'double' else [f'{support}_foot']
    M0 = R.fk(q)
    base = np.linalg.inv(M0[feet[0]])
    M = R.fk(q, base=base)
    loads = [(MS.m[l], G.apply(M[l], MS.com[l][None])[0], l) for l in spec['links']]
    for side in 'LR':
        if payload:
            hand = f'{side}_hand'
            palm = [g for g in spec['links'][hand]['geoms'] if 'palm' in g['name']][0]
            loads.append((payload, G.apply(M[hand], np.array(palm['c'])[None])[0], hand))
    gvec = np.array([0, 0, -G_ACC])
    tau = {}
    stance_joints = {j['name'] for f in feet for j in R.chain(f)}
    legs = {f: set(R.subtree(f'{f[0]}_hip_pitch_link')) for f in feet}
    if support == 'double':
        # Two feet on the ground close the chain: each leg takes half the upper body as a vertical force at its
        # hip plus half of the upper body's moment about the hip midpoint (the sideways part of that moment is
        # carried by the two hips pushing unequally, which a symmetric pose makes zero).
        upper = [x for x in loads if x[2] not in legs['L_foot'] | legs['R_foot']]
        wu = sum(m for m, _, _ in upper)
        hips = {f: G.apply(M['pelvis'], np.array(R.joints[f'{f[0]}_hip_pitch']['origin'])[None])[0] for f in feet}
        mid = (hips['L_foot'] + hips['R_foot']) / 2
        m_upper = sum(m * np.cross(c - mid, gvec) for m, c, _ in upper)
    for j in spec['joints']:
        a = M[j['parent']][:3, :3] @ G.AXIS[j['axis']]
        o = G.apply(M[j['parent']], np.array(j['origin'])[None])[0]
        sub = set(R.subtree(j['child']))
        if j['name'] in stance_joints:
            f = [f for f in feet if j['name'] in {x['name'] for x in R.chain(f)}][0]
            leg = legs[f]
            if support == 'double':
                own = [x for x in loads if x[2] in leg and x[2] not in sub]
                share = np.cross(hips[f] - o, np.array([0, 0, -wu * G_ACC / 2])) + m_upper / 2
                t = share @ a + sum(m * np.cross(c - o, gvec) @ a for m, c, _ in own)
            else:
                t = sum(m * np.cross(c - o, gvec) @ a for m, c, l in loads if l not in sub)
        else:
            t = sum(m * np.cross(c - o, gvec) @ a for m, c, l in loads if l in sub)
        tau[j['name']] = abs(t)
    mt = sum(x[0] for x in loads)
    com = sum(x[0] * x[1] for x in loads) / mt
    poly = []
    for f in feet:
        sole = [g for g in spec['links'][f]['geoms'] if 'sole' in g['name']][0]
        sole = sole.get('contact', sole)                  # a shaped outsole carries its contact box
        c, s = np.array(sole['c']), np.array(sole['size']) / 2
        for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            poly.append(G.apply(M[f], (c + [dx * s[0], dy * s[1], -s[2]])[None])[0][:2])
    return tau, com, np.array(poly), M


def hull_margin(poly, p):
    """Signed distance from p to the edge of the convex hull of poly (positive inside)."""
    from scipy.spatial import ConvexHull
    h = ConvexHull(poly)
    return min(-(eq[:2] @ p + eq[2]) for eq in h.equations)


def balance(spec, R, MS, q, support, payload):
    """Lean at the ankles (and the stance hip roll) until the CoM sits over the middle of the support."""
    q = dict(q)
    stance = ['L', 'R'] if support == 'double' else [support]
    for _ in range(60):
        tau, com, poly, M = statics(spec, R, MS, q, support, payload)
        target = poly.mean(0)
        ex, ey = com[:2] - target
        if abs(ex) < 0.002 and (support == 'double' or abs(ey) < 0.002):
            break
        for s in stance:
            q[f'{s}_ankle_pitch'] = q.get(f'{s}_ankle_pitch', 0.0) + 1.2 * ex
        if support != 'double':
            s = stance[0]
            q[f'{s}_ankle_roll'] = q.get(f'{s}_ankle_roll', 0.0) - 1.2 * ey
            q[f'{s}_hip_roll'] = -q[f'{s}_ankle_roll']
    return q


def capacity(spec, j):
    c = j['capacity']
    if not c:
        return None
    a = spec['actuators'][c['model']]
    k = c['count'] * c['ratio']
    return a['rated_nm'] * k, a['peak_nm'] * k


def check_poses(spec, S, R, rep, collide=True):
    MS = Mass(spec, S)
    worst = {}
    for name, (q, support, cat, payload) in POSES.items():
        q = G.mirror_pose(q)
        q = balance(spec, R, MS, q, support, payload)
        tau, com, poly, M = statics(spec, R, MS, q, support, payload)
        margin = hull_margin(poly, com[:2])
        rep.add('poses', f'{name}: balance', margin >= 0.01,
                f"CoM {margin * 100:.1f} cm inside the {'feet' if support == 'double' else 'stance foot'}")
        # balancing leans the ankles: it must stay inside the limits (iteration 12: the deep squat balanced at
        # 0.86 rad of dorsiflexion, past a 0.8 limit, and only a collision gave it away)
        out = [f"{k} {v:+.2f} outside [{R.joints[k]['lower']:+.2f}, {R.joints[k]['upper']:+.2f}]"
               for k, v in q.items() if not R.joints[k]['lower'] - 1e-9 <= v <= R.joints[k]['upper'] + 1e-9]
        rep.add('poses', f'{name}: within limits', not out, '; '.join(out) or 'yes')
        for j in spec['joints']:
            cap = capacity(spec, j)
            if cap is None:
                continue
            need = tau[j['name']] * (PEAK_MARGIN if cat == 'peak' else 1.0)
            have = cap[1] if cat == 'peak' else cap[0]
            r = need / have
            w = worst.get(j['name'])
            if w is None or r > w[0]:
                worst[j['name']] = (r, name, tau[j['name']], have, cat)
        if collide:
            hits = []
            links = list(spec['links'])
            for x in range(len(links)):
                for y in range(x + 1, len(links)):
                    a, b = links[x], links[y]
                    if R.adjacent(a, b):
                        continue
                    H = S.bulk if R.distance(a, b) > NEAR else S.hard
                    v = overlap(posed(H[a], M[a]), posed(H[b], M[b]))
                    if v > TOUCH:
                        hits.append(f'{a} / {b} {v * 1e6:.0f} cm3')
            rep.add('poses', f'{name}: collisions', not hits, '; '.join(hits[:4]) or 'none')
    for jn, (r, pose, t, have, cat) in sorted(worst.items(), key=lambda kv: -kv[1][0]):
        j = next(x for x in spec['joints'] if x['name'] == jn)
        c = j['capacity']
        kind = 'peak' if cat == 'peak' else 'rated'
        fac = f' x {PEAK_MARGIN}' if cat == 'peak' else ''
        rep.add('torque', jn, r <= 1.0,
                f"{t:.1f} N m{fac} in '{pose}' vs {kind} {have:.0f} N m "
                f"({c['count']} x {c['model']}{' x %.1f' % c['ratio'] if c['ratio'] != 1 else ''}): "
                f"{r * 100:.0f} % used")


# ---------------------------------------------------------------- deliberate breakage

def break_spec(spec, case):
    s = copy.deepcopy(spec)
    J = {j['name']: j for j in s['joints']}
    if case == 'torque':
        for n in ('L_knee', 'R_knee'):
            J[n]['capacity'] = dict(model='RS02', count=1, ratio=1.0)
    elif case == 'collide':
        for ln in ('L_shoulder_roll_link', 'L_shoulder_yaw_link', 'L_elbow_link', 'L_wrist_yaw_link',
                   'L_wrist_pitch_link', 'L_hand'):
            for g in s['links'][ln]['geoms']:
                for k in ('c', 'p0', 'p1'):
                    if k in g:
                        g[k][1] -= 0.07
                if g['kind'] == 'loft':
                    for sec in g['sections']:
                        sec['cy'] -= 0.07
    elif case == 'offaxis':
        J['L_hip_yaw']['act_c'][0] += 0.02
    elif case == 'floating':
        s['links']['L_hip_pitch_link']['geoms'] = [g for g in s['links']['L_hip_pitch_link']['geoms']
                                                   if g['name'] != 'L hip bracket']
    elif case == 'mass':
        next(g for g in s['links']['torso']['geoms'] if g['name'] == 'battery')['mass'] += 20
    elif case == 'poke':
        for g in s['links']['L_shoulder_roll_link']['geoms']:
            if g['role'] == 'actuator':
                g['r'] = 0.06
    elif case == 'cover':
        # the hip lobe without its openings round the hip motors: its flat inner face cuts through both again
        for l in s['links'].values():
            l['geoms'] = [x for x in l['geoms'] if not (x['role'] == 'cut' and 'L hip lobe' in x['name'])]
    elif case == 'window':
        # a window cut through the calf over the upper ankle actuator: what a cut uncovers must count as showing
        sec = lambda z: dict(z=z, cx=-0.01, cy=0.17, a=0.06, b=0.06, n=4.0)
        s['links']['L_knee_link']['geoms'].append(dict(kind='loft', name='L calf window', role='cut',
                                                       targets=['L shank'], color='#FFFFFF', open=[],
                                                       sections=[sec(0.40), sec(0.49)]))
    elif case in ('hipshrink', 'pa12', 'token', 'tokenedge'):
        for side, k in (('L', 1), ('R', -1)):
            g = next(x for x in s['links'][f'{side}_hip_pitch_link']['geoms'] if x['name'] == f'{side} hip bracket')
            if case == 'hipshrink':       # 13 x 15 to 2.7 x 3.5, as the audit's run that check_connect passed
                g['size'][1], g['size'][2] = 0.0027, 0.0035
                g['fillet'] = 0.0005
            elif case == 'pa12':          # same shape, still role struct: printed plastic
                g['material'] = 'pa12'
            else:
                f = next(x for x in s['links'][f'{side}_hip_pitch_link']['geoms'] if x['name'].endswith('hip_pitch flange'))
                if case == 'token':       # the same face, by 0.02 mm of depth
                    g['c'][1] = k * (abs(f['c'][1]) + f['h'] / 2 - 0.00002 + g['size'][1] / 2)
                else:                     # its square edge just grazing the flange's rim
                    g['fillet'] = 0.0002
                    g['c'][2] = f['c'][2] + f['r'] - 0.00002 + g['size'][2] / 2
    elif case == 'spineshrink':
        g = next(x for x in s['links']['torso']['geoms'] if x['name'] == 'spine')
        g['size'][0], g['size'][1] = 0.002, 0.002
        g['fillet'] = 0.0005
    elif case == 'norods':
        for ln in ('L_knee_link', 'R_knee_link'):
            s['links'][ln]['geoms'] = [g for g in s['links'][ln]['geoms'] if g['role'] != 'linkage']
    elif case == 'rolebad':           # a cover can't carry: the bracket as a shell, the thigh frame as knit
        next(x for x in s['links']['L_hip_pitch_link']['geoms'] if x['name'] == 'L hip bracket')['role'] = 'shell'
        next(x for x in s['links']['L_hip_yaw_link']['geoms'] if x['name'] == 'L thigh frame')['role'] = 'knit'
    elif case == 'newpart':           # a new bracket nobody added to a load path
        o = J['L_knee']['origin']
        s['links']['L_hip_yaw_link']['geoms'].append(dict(kind='rbox', name='L new bracket', c=[0.024, o[1], o[2] + 0.15], size=[0.01, 0.01, 0.01], fillet=0.001, color='#000000', role='struct', material='al7075', exposed=True))
    elif case == 'nostator':          # the knee motor's plate and post off its back, so nothing holds it by its face
        for g in s['links']['L_knee_link']['geoms']:
            if g['name'] in ('L knee stator plate', 'L knee post'):
                g['c'][1] += 0.02
    elif case == 'motorinto':         # a plate run into the knee motor's side
        s['links']['L_knee_link']['geoms'].append(dict(kind='rbox', name='L plate into motor', c=[0.0, J['L_knee']['origin'][1], J['L_knee']['origin'][2] - 0.05], size=[0.03, 0.01, 0.03], fillet=0.001, color='#000000', role='struct', material='al7075', exposed=True))
        s['links']['L_knee_link']['geoms'][-1]['_bad'] = True
    elif case in ('shortthread', 'longscrew', 'fewscrews', 'missingscrew', 'ontab', 'protrude', 'blockkey'):
        link = next(l for l in s['links'].values() if any(g.get('hw', {}).get('pattern') == 'L knee hub' for g in l['geoms']))
        if case in ('shortthread', 'longscrew'):
            for g in link['geoms']:
                if g.get('hw', {}).get('pattern') == 'L knee hub' and g['hw']['type'] == 'screw':
                    dL = -3.5 if case == 'shortthread' else 4.0
                    g['hw']['length'] += dL
                    sg = 1 if g['profile'][-1][0] > 0 else -1
                    tip = (g['profile'][0] if sg < 0 else g['profile'][-1])[0]
                    new = tip + sg * dL / 1000
                    g['profile'] = [[new if p[0] == tip else p[0], p[1]] for p in g['profile']]
        elif case == 'fewscrews':     # 6 of the hub's 9 screws gone
            drop = [g for g in link['geoms'] if g.get('hw', {}).get('pattern') == 'L knee hub' and g['hw']['type'] == 'screw'][3:]
            link['geoms'] = [g for g in link['geoms'] if g not in drop]
        elif case == 'missingscrew':  # one gone: the pattern isn't fully used
            drop = next(g for g in link['geoms'] if g.get('hw', {}).get('pattern') == 'L knee hub' and g['hw']['type'] == 'screw')
            link['geoms'] = [g for g in link['geoms'] if g is not drop]
        elif case == 'ontab':         # a stator screw claimed at the cable tab's angle
            stator = next(l for l in s['links'].values() if any(g.get('hw', {}).get('pattern') == 'L knee stator' for g in l['geoms']))
            next(g for g in stator['geoms'] if g.get('hw', {}).get('pattern') == 'L knee stator' and g['hw']['type'] == 'screw')['hw']['angle'] = 75.0
        elif case == 'protrude':      # a block in the way of a screw's tip, beyond the hole: the M5 runs into it
            h = next(g for g in link['geoms'] if g.get('hw', {}).get('pattern') == 'L knee hub' and g['hw']['type'] == 'screw')['hw']
            s['links']['L_knee_link']['geoms'].append(dict(kind='rbox', name='L block on the screw', c=[h['pos'][0], h['pos'][1], h['pos'][2]], size=[0.012, 0.004, 0.012], fillet=0.0005, color='#000000', role='struct', material='al7075', exposed=True))
        else:                         # a plate between a stator screw's head and the key
            stator = next(l for l in s['links'].values() if any(g.get('hw', {}).get('pattern') == 'L knee stator' for g in l['geoms']))
            hs = [g['hw'] for g in stator['geoms'] if g.get('hw', {}).get('pattern') == 'L knee stator' and g['hw']['type'] == 'screw']
            free = [h for h in hs if abs(h['pos'][2] - J['L_knee']['origin'][2]) < 0.03][0]
            s['links']['pelvis']['geoms'].append(dict(kind='rbox', name='block over a screw', c=[free['pos'][0], free['pos'][1] - free['dir'][1] * 0.02, free['pos'][2]], size=[0.02, 0.004, 0.02], fillet=0.0005, color='#000000', role='struct', material='al7075', exposed=True))
    elif case == 'edge':              # the stator rim down to r 55: its holes on r 53 too near its edge
        g = next(x for x in s['links']['L_knee_link']['geoms'] if x['name'] == 'L knee stator rim')
        g['r'] = 0.0545
    elif case == 'ligament':          # one hub hole turned next to its neighbour
        l = next(l for l in s['links'].values() if any(g.get('hw', {}).get('pattern') == 'L knee hub' for g in l['geoms']))
        cuts = [g for g in l['geoms'] if g['role'] == 'cut' and g.get('hw', {}).get('pattern') == 'L knee hub']
        a, b = cuts[0], cuts[1]
        for g in (a,):
            g['c'][0] += 0.65 * (b['c'][0] - a['c'][0])
            g['c'][2] += 0.65 * (b['c'][2] - a['c'][2])
            g['hw']['pos'] = list(g['c'])
    elif case == 'smallbearing':      # the same seat, a bearing with a third of the rating
        g = next(x for x in s['links']['L_hip_yaw_link']['geoms'] if x['name'] == 'L knee outer bearing')
        g['hw']['C0'], g['hw']['C'] = 800.0, 600.0
    elif case == 'thinseat':          # the cup's wall down to 1 mm round the race, and the collar too
        for n in ('L knee axle in', 'L knee cheek disc in'):
            g = next(x for x in s['links']['L_hip_yaw_link']['geoms'] if x['name'] == n)
            g['r'] = 0.0265
    elif case == 'noshoulder':        # the post's collar gone: the outer bearing's inner race has no stop
        s['links']['L_knee_link']['geoms'] = [g for g in s['links']['L_knee_link']['geoms'] if g['name'] != 'L knee post collar']
    elif case == 'deepthread':        # --good: if the hub's hole were deep enough, and the screw long enough
        l = next(l for l in s['links'].values() if any(g.get('hw', {}).get('pattern') == 'L knee hub' for g in l['geoms']))
        for g in l['geoms']:
            if g.get('hw', {}).get('pattern') == 'L knee hub' and g['hw']['type'] == 'screw':
                g['hw']['length'] += 4.5
                g['hw']['hole_depth'] = 12.0
    elif case == 'keyclear':          # --good: with the outer hub and cheek gone, a key reaches every stator screw
        s['links']['L_knee_link']['geoms'] = [g for g in s['links']['L_knee_link']['geoms'] if g['name'] != 'L knee hub out']
        s['links']['L_hip_yaw_link']['geoms'] = [g for g in s['links']['L_hip_yaw_link']['geoms'] if g['name'] not in ('L knee cheek out', 'L knee cheek out top')]
    elif case == 'bigbearing':        # --good: a CRBS 508's rating in the seat
        for n in ('L knee outer bearing', 'L knee inner bearing'):
            g = next(x for x in s['links']['L_hip_yaw_link']['geoms'] if x['name'] == n)
            g['hw']['C0'], g['hw']['C'] = 6170.0, 4900.0
    elif case == 'thinpin':           # the pins of --good pin, one far too small for the load, though it touches both links
        s = break_spec(spec, 'pin')
        next(g for g in s['links']['L_ankle_pitch_link']['geoms'] if g['name'] == 'L ankle pitch pin')['r'] = 0.001
    # ------- --good: one thing fixed on purpose, to show the matching row can pass
    elif case == 'pin':
        o = J['L_ankle_pitch']['origin']
        s['links']['L_ankle_pitch_link']['geoms'].append(dict(kind='cyl', name='L ankle pitch pin', c=list(o), axis='y', r=0.008, h=0.06, color='#000000', role='struct', material='al7075', exposed=True))
        s['links']['L_foot']['geoms'].append(dict(kind='cyl', name='L ankle roll pin d', c=list(o), axis='x', r=0.007, h=0.05, color='#000000', role='struct', material='al7075', exposed=True))
    elif case == 'link':              # cranks run on to the motors' faces
        for g in s['links']['L_knee_link']['geoms']:
            if g['role'] == 'linkage' and g['name'].endswith('crank'):
                m = next(x for x in s['links']['L_knee_link']['geoms'] if x['role'] == 'actuator' and x['name'].startswith(g['name'].rsplit(' ', 1)[0]))
                sign = 1 if g['p0'][1] > m['c'][1] else -1
                g['p0'][1] = g['p1'][1] = m['c'][1] + sign * (m['h'] / 2 + g['r'] - 0.001)
    elif case == 'stator':            # a plate on the hip roll motor's back, joined to the bracket
        m = next(g for g in s['links']['L_hip_pitch_link']['geoms'] if g['role'] == 'actuator')
        b = next(g for g in s['links']['L_hip_pitch_link']['geoms'] if g['name'] == 'L hip bracket')
        back = m['c'][0] - m['h'] / 2
        s['links']['L_hip_pitch_link']['geoms'] += [
            dict(kind='cyl', name='L hip roll back plate', c=[back - 0.002, m['c'][1], m['c'][2]], axis='x', r=0.05, h=0.006, color='#000000', role='struct', material='al7075', exposed=True),
            dict(kind='rbox', name='L hip roll back arm', c=[(back + b['c'][0]) / 2, b['c'][1], b['c'][2]], size=[b['c'][0] - back + 0.004, b['size'][1], b['size'][2]], fillet=0.002, color='#000000', role='struct', material='al7075', exposed=True)]
    elif case == 'frame':             # two side rails from the hub to the shoulder beam
        t = {g['name']: g for g in s['links']['torso']['geoms']}
        z0, z1 = t['waist hub']['c'][2], t['shoulder beam']['c'][2]
        for y in (0.045, -0.045):
            s['links']['torso']['geoms'].append(dict(kind='rbox', name=f'rail {y}', c=[-0.005, y, (z0 + z1) / 2], size=[0.04, 0.02, z1 - z0], fillet=0.002, color='#000000', role='struct', material='al7075', exposed=True))
    else:
        raise SystemExit(f'unknown case {case}')
    return s


EXPECT = {'torque': 'torque', 'collide': 'static', 'offaxis': 'axis', 'floating': 'connect', 'mass': 'mass', 'window': 'skin',
          'cover': 'covers',
          'poke': 'skin'}
# The load checks already fail on today's design, so "a row fails" would prove nothing for them. These cases compare with the unbroken
# spec: caught means a row newly fails, or an already failing row got worse.
EXPECT_LOAD = {'hipshrink': ['loadpath', 'contact'], 'spineshrink': ['loadpath'], 'norods': ['linkage', 'loadpath'], 'token': ['contact'], 'tokenedge': ['contact'],
               'pa12': ['loadpath'], 'rolebad': ['loadpath'], 'newpart': ['loadpath'], 'nostator': ['stator'], 'thinpin': ['pivot'],
               'motorinto': ['motorfit'], 'shortthread': ['fastener'], 'longscrew': ['fastener'], 'protrude': ['fastener'], 'edge': ['holes'], 'ligament': ['holes'], 'fewscrews': ['bolted'],
               'smallbearing': ['bearing'], 'thinseat': ['bearing'], 'noshoulder': ['bearing'], 'missingscrew': ['pattern'], 'ontab': ['pattern'], 'blockkey': ['access']}
BASE_FOR = {'thinpin': 'pin'}     # today's design has no pin to make too thin: compare with the one that has
# And the reverse: the fix must turn its row to PASS.
EXPECT_GOOD = {'pin': ['pivot'], 'link': ['linkage'], 'stator': ['stator'], 'frame': ['loadpath', 'contact'], 'deepthread': ['fastener'], 'bigbearing': ['bearing'], 'keyclear': ['access']}
GROUPS = {'targets': lambda spec, S, R, rep: check_targets(spec, S, rep),
          'mass': lambda spec, S, R, rep: check_mass(spec, S, rep),
          'axis': lambda spec, S, R, rep: check_axis(spec, rep),
          'static': lambda spec, S, R, rep: check_static(spec, S, R, rep),
          'covers': lambda spec, S, R, rep: check_covers(spec, S, R, rep),
          'connect': lambda spec, S, R, rep: check_connect(spec, S, rep),
          'skin': lambda spec, S, R, rep: check_skin(spec, R, rep),
          'sweep': lambda spec, S, R, rep: check_sweep(spec, S, R, rep),
          'poses': lambda spec, S, R, rep: check_poses(spec, S, R, rep),
          'torque': lambda spec, S, R, rep: check_poses(spec, S, R, rep, collide=False),
          'loadpath': lambda spec, S, R, rep: check_loadpath(spec, S, rep),
          'contact': lambda spec, S, R, rep: check_contact(spec, S, rep),
          'pivot': lambda spec, S, R, rep: check_pivot(spec, S, rep),
          'linkage': lambda spec, S, R, rep: check_linkage(spec, S, rep),
          'stator': lambda spec, S, R, rep: check_stator(spec, S, rep),
          'motorfit': HC.check_motorfit, 'fastener': HC.check_fastener, 'holes': HC.check_holes, 'bolted': HC.check_bolted, 'bearing': HC.check_bearing, 'pattern': HC.check_pattern, 'access': HC.check_access}


def run_groups(spec, groups):
    R, S, rep = G.Robot(spec), Solids(spec), Report()
    for grp in groups:
        t = time.time()
        GROUPS[grp](spec, S, R, rep)
        print(f'{grp}: {time.time() - t:.1f} s', file=sys.stderr)
    return rep


def compare(base, spec, case, groups, good):
    """Run the broken or fixed spec against the unbroken one and print what changed. Returns exit code: 0 if it did what it should."""
    was = {(r['group'], r['item']): r for r in run_groups(base, groups).rows}
    rows = run_groups(spec, groups).rows
    if good:
        hits = [r for r in rows if r['status'] == 'PASS' and was.get((r['group'], r['item']), {}).get('status') == 'FAIL']
    else:
        def worse(r):
            b = was.get((r['group'], r['item']))
            if r['status'] != 'FAIL':
                return False
            if b is None or b['status'] != 'FAIL':
                return True
            return b['value'] is not None and r['value'] is not None and r['value'] < b['value'] - 1e-6 - 0.01 * abs(b['value'])
        hits = [r for r in rows if worse(r)]
    label = 'FIXED' if good else 'CAUGHT'
    print(f"--{'good' if good else 'bad'} {case}: {label if hits else 'NOT ' + label} ({len(hits)} rows changed)")
    for r in hits[:8]:
        b = was.get((r['group'], r['item']))
        print('  ', r['group'], '|', r['item'], '|', r['status'], '|', r['detail'][:150], '| was:', (b['status'] + ' ' + b['detail'][:80]) if b else 'no such row')
    return 0 if hits else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--bad')
    ap.add_argument('--good')
    ap.add_argument('--spec')
    ap.add_argument('--out', default=os.path.join(ROOT, 'reports', 'checks.md'))
    a = ap.parse_args()
    spec = G.load(a.spec)
    if a.good or a.bad in EXPECT_LOAD:
        case = a.good or a.bad
        groups = a.only or (EXPECT_GOOD[case] if a.good else EXPECT_LOAD[case])
        base = break_spec(spec, BASE_FOR[case]) if case in BASE_FOR else spec
        return compare(base, break_spec(spec, case), case, groups, bool(a.good))
    if a.bad:
        spec = break_spec(spec, a.bad)
        a.only = a.only or [EXPECT[a.bad]]
    groups = a.only or ['targets', 'mass', 'axis', 'static', 'covers', 'connect', 'loadpath', 'contact', 'pivot', 'linkage', 'stator', 'skin', 'sweep', 'poses', 'motorfit', 'fastener', 'holes', 'bolted', 'bearing', 'pattern', 'access']
    R = G.Robot(spec)
    t0 = time.time()
    S = Solids(spec)
    rep = Report()
    rep.assumptions = spec.get('hardware', {}).get('assumptions', [])
    for grp in groups:
        t = time.time()
        GROUPS[grp](spec, S, R, rep)
        print(f'{grp}: {time.time() - t:.1f} s', file=sys.stderr)
    fails = rep.failing()
    if a.bad:
        caught = [r for r in fails if r['group'] in (EXPECT[a.bad], 'torque')]
        print(f"--bad {a.bad}: {'CAUGHT' if caught else 'NOT CAUGHT'} ({len(caught)} failing rows)")
        for r in caught[:5]:
            print('  ', r['group'], r['item'], r['detail'])
        return 0 if caught else 1
    rep.write(a.out, f"spec: `spec/robot.json`, {time.strftime('%Y-%m-%d %H:%M')}, {time.time() - t0:.0f} s")
    for r in rep.rows:
        if r['status'] != 'PASS':
            print(f"{r['status']} {r['group']:8} {r['item']}: {r['detail']}")
    print(f"{len(rep.rows)} checks, {len(fails)} failing -> {a.out}")
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
