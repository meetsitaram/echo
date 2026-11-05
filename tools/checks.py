"""Physical-validity checks for the replica, run offline on spec/robot.json before anything is built in Fusion.

  python -I tools/checks.py                 all checks, report in reports/checks.md, exit 1 on any FAIL
  python -I tools/checks.py --only sweep    one group (targets mass axis static connect skin sweep poses)
  python -I tools/checks.py --bad torque    break the design on purpose and show the matching check FAILs
                                            (cases: torque collide offaxis floating mass poke)

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
           (design_rules 'floating': nothing hangs in thin air)
  skin     internal parts stay inside the skin, except parts meant to show (exposed actuators)
  sweep    each joint swept through its limits: limits are reachable without collisions (motion sweep)
  poses    task poses: static joint torques vs actuator ratings, centre of mass over the feet, no collisions
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import manifold3d as mf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom as G  # noqa: E402
import mass_model as MM  # noqa: E402

ROOT = G.ROOT
G_ACC = 9.81
TOUCH = 0.3e-6            # m^3 (0.3 cm^3): intersections below this count as touching, not colliding
ARM_KEYS, LEG_KEYS = ('shoulder', 'elbow', 'wrist', 'hand'), ('hip', 'knee', 'ankle', 'foot')


# ---------------------------------------------------------------- report

class Report:
    def __init__(self):
        self.rows = []

    def add(self, group, item, ok, detail='', warn=False):
        status = 'PASS' if ok else ('WARN' if warn else 'FAIL')
        self.rows.append(dict(group=group, item=item, status=status, detail=detail))
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
        self.geom, self.wall = {}, {}
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
        self.all, self.shell, self.rigid, self.hard = {}, {}, {}, {}
        for ln, l in spec['links'].items():
            idx = range(len(l['geoms']))
            role = lambda i: l['geoms'][i]['role']
            self.all[ln] = union([self.geom[(ln, i)] for i in idx])
            self.shell[ln] = union([self.geom[(ln, i)] for i in idx if role(i) == 'shell'])
            self.rigid[ln] = union([self.geom[(ln, i)] for i in idx if role(i) in G.RIGID])
            self.hard[ln] = union([self.geom[(ln, i)] for i in idx if role(i) in G.RIGID] +
                                  [self.wall[(ln, i)] for i in idx if role(i) == 'shell'])


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
        'shoulder_top_z': hi['L_shoulder_pitch_link'][2],
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
            v = overlap(S.hard[a], S.hard[b])
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
    pieces = lambda m: len([p for p in m.decompose() if abs(p.volume()) > 1e-8])
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


def check_skin(spec, R, rep):
    """Internal parts must be hidden: inside the skins (any link, zero pose), unless meant to show."""
    rng = np.random.default_rng(1)
    shells = [g for l in spec['links'].values() for g in l['geoms'] if g['role'] == 'shell']
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
                v = overlap(posed(S.hard[a], M[a]), S.hard[b])
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
                    v = overlap(posed(S.hard[a], M[a]), posed(S.hard[b], M[b]))
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
    else:
        raise SystemExit(f'unknown case {case}')
    return s


EXPECT = {'torque': 'torque', 'collide': 'static', 'offaxis': 'axis', 'floating': 'connect', 'mass': 'mass', 'window': 'skin',
          'cover': 'covers',
          'poke': 'skin'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--bad')
    ap.add_argument('--spec')
    ap.add_argument('--out', default=os.path.join(ROOT, 'reports', 'checks.md'))
    a = ap.parse_args()
    spec = G.load(a.spec)
    if a.bad:
        spec = break_spec(spec, a.bad)
        a.only = a.only or [EXPECT[a.bad]]
    groups = a.only or ['targets', 'mass', 'axis', 'static', 'covers', 'connect', 'skin', 'sweep', 'poses']
    R = G.Robot(spec)
    t0 = time.time()
    S = Solids(spec)
    rep = Report()
    for grp in groups:
        t = time.time()
        {'targets': lambda: check_targets(spec, S, rep),
         'mass': lambda: check_mass(spec, S, rep),
         'axis': lambda: check_axis(spec, rep),
         'static': lambda: check_static(spec, S, R, rep),
         'covers': lambda: check_covers(spec, S, R, rep),
         'connect': lambda: check_connect(spec, S, rep),
         'skin': lambda: check_skin(spec, R, rep),
         'sweep': lambda: check_sweep(spec, S, R, rep),
         'poses': lambda: check_poses(spec, S, R, rep),
         'torque': lambda: check_poses(spec, S, R, rep, collide=False)}[grp]()
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
