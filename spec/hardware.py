"""Standard parts as data: one call places a screw pattern, a bearing or a pin, with its holes cut on the real parts and its mass from the catalogue.

spec/robot.py calls these after the links and joints are written. Each screw, pin, insert and retainer is a geom of role 'hardware', each bearing a geom of role
'bearing', so they show as parts and count on the parts sheet. Every such geom carries 'hw': the record the checks read (tools/hw_checks.py, tools/strength.py).
Holes are 'cut' geoms carrying the same kind of record. Positions in 'hw' are metres in the world at the zero pose; sizes in 'hw' are millimetres, as the catalogue's.

  pattern(name, joint, which, link, seat_t, length, ...)   a motor's whole stator or hub pattern, from spec/actuator_interfaces.json, placed by the motor's clock
  fastener(name, link, thread, length, head, pos, axis, sign, clamps, into, ...)     one screw, clearance holes and a tapped hole in our own part
  pin_holes(name, joint, target)                           holes for the pins that stand on a motor's hub
  dowel(name, link, d, length, pos, axis, parts)           a bought pin, with a hole in each part
  register(name, link, target, c, axis, d, depth)          a spigot recess that locates one part in another
  insert(name, link, part, thread, pos, axis, sign)        a heat-set insert, with its hole
  bearing(name, designation, c, axis, inner_link, inner_parts, outer_link, outer_parts)  a catalogue bearing: which link carries which race
  axle(name, link, c, axis, d, length)                     a pin or stub: role struct, on a load path
  retainer(...)                                            a bought stop for a race, recorded on the bearing

Everything unconfirmed comes from spec/assumptions.json. python -I spec/hardware.py runs a self test of every helper.
"""
import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
_j = lambda f: json.load(open(os.path.join(HERE, f), encoding='utf-8'))
CAT, IFACE, OUT, ASSUME = _j('hardware_catalog.json'), _j('actuator_interfaces.json'), _j('actuator_outlines.json'), _j('assumptions.json')
AXES = {'x': (1.0, 0.0, 0.0), 'y': (0.0, 1.0, 0.0), 'z': (0.0, 0.0, 1.0)}
SCREW_COLOR, PIN_COLOR, BEARING_COLOR = '#2B2D31', '#6E7278', '#C9CDD2'     # black-oxide 12.9, ground pin, bearing steel
HEAD_CLEAR = 0.6                                                             # counterbore over the head's diameter, mm (about what ISO 4762 tables give)

_S = dict(links=None, joints=None, frame=None, act=None)
PATTERNS, BEARINGS, SINGLE, GROUPS, BUILD = [], [], [], [], []


def setup(links, joints, act, motor_frame):
    _S.update(links=links, joints=joints, act=act, frame=motor_frame)


def value(x):
    return x['value'] if isinstance(x, dict) and 'value' in x else x


def assumed(key):
    for r in ASSUME['rows']:
        if r['id'] == key:
            return r['value']
    raise KeyError(key)


def _add(link, *geoms):
    _S['links'][link]['geoms'].extend(geoms)


def _cyl(name, c, axis, r, h, color, role, **kw):
    return dict(kind='cyl', name=name, c=list(c), axis=axis, r=r, h=h, color=color, role=role, **kw)


def _cut(name, c, axis, r, h, targets, hw=None):
    g = dict(kind='cyl', name=name, c=list(c), axis=axis, r=r, h=h, color='#000000', role='cut', targets=list(targets))
    if hw:
        g['hw'] = hw
    return g


def add_v(p, a, s):
    return [p[0] + a[0] * s, p[1] + a[1] * s, p[2] + a[2] * s]


# ---------------------------------------------------------------- catalogue lookups

def screw_data(thread, head):
    """(pitch, head dia, head height, socket across flats, proof load N, stress area mm2, break load N, tap drill, clearance medium). Raises if the catalogue has no row:
    a part with no citable data isn't placed."""
    h = {'iso4762': CAT['screws']['iso4762_socket_head'], 'iso7380': CAT['screws']['iso7380_button_head'], 'iso10642': CAT['screws']['iso10642_countersunk']}[head]
    row = h[thread]
    if row is None:
        raise ValueError(f'{thread} {head}: not in the catalogue')
    holes = CAT['holes'][thread]
    cap = CAT['screw_capacity_N']['class_12_9_steel_iso898_1'][thread]
    if cap is None:
        raise ValueError(f'{thread}: no proof load in the catalogue')
    if head == 'iso4762':
        hd, hk, af = row[1], row[2], row[3]
    elif head == 'iso7380':
        hd, hk, af = row[0], row[1], row[2]
    else:
        hd, hk, af = row[0], row[2], row[3]
    return dict(pitch=holes[0], head_d=hd, head_k=hk, socket=af, proof=cap[1], As=cap[0], breaking=cap[2], tap=holes[1], clear=holes[3], clear_close=holes[2])


def bearing_table():
    """Every bearing in the catalogue, normalised: {designation: dict(d, D, B, C, C0 in N, mass g, kind, da, Da, ...)}. Shoulders only where the catalogue gives them."""
    out = {}
    t = CAT['thin_section_bearings']
    for k, r in t['deep_groove_618'].items():
        if isinstance(r, list) and k != 'columns':
            out[k] = dict(d=r[0], D=r[1], B=r[2], C=r[3] * 1000, C0=r[4] * 1000, mass=r[5], kind='ball', source=t['deep_groove_618']['source'], read_via=t['deep_groove_618']['read_via'])
    c = t['crossed_roller_iko']
    shoulders = c.get('shoulders_da_Da', {})
    for src in (c, c['just_outside_the_fit_rule']):
        for k, r in src.items():
            if isinstance(r, list) and len(r) == 6 and k[:3] == 'CRB':
                name = k.split(' (')[0]
                out[name] = dict(d=r[0], D=r[1], B=r[2], C=r[3], C0=r[4], mass=r[5], kind='roller', source=c['source'], read_via=c['read_via'], da=shoulders.get(name, [None, None])[0], Da=shoulders.get(name, [None, None])[1])
    return out


def strip_area(d, pitch, le):
    """Stripping area of an internal thread, mm2: pi n Le d [1 / (2n) + 0.57735 (d - D1)], with n threads per mm and D1 = d - 1.0825 P (FED-STD-H28 form, nominal sizes)."""
    n = 1.0 / pitch
    d1 = d - 1.0825 * pitch
    return math.pi * n * le * d * (1 / (2 * n) + 0.57735 * (d - d1))


# ---------------------------------------------------------------- motors

def motor_face(model, which):
    """(interface entry, t of its face in mm). which: 'hub', 'stator_front' or 'stator_rear'. A rear pattern with no drawn thread is refused: assumptions.json 'rear_threads'."""
    e, o = IFACE[model], OUT[model]
    if which == 'hub':
        return e['output_flange'], o['t_hub_face']
    want = 'rear' if which == 'stator_rear' else None
    for s in e['stator_mounting']:
        rear = 'rear' in s.get('face', '')
        if (want and rear) or (not want and not rear):
            if value(s['thread_depth_mm']) is None:
                raise ValueError(f'{model} {which}: this thread is not on a drawing: no load on it (assumptions.json rear_threads)')
            z = float(re.search(r'z=(-?[\d.]+)', s['face']).group(1))
            ext = o['z_output_extreme']
            return s, o['h'] / 2 - abs(ext - z)
    raise ValueError(f'{model}: no {which} pattern')


def thread_depth(model, which, entry):
    """The hub's depth comes from the assumptions table (drawing, not the shallower STEP); a stator's is the drawing's."""
    if which == 'hub':
        for r in ASSUME['rows']:
            if r['id'] == f'hub_thread_depth_{model}':
                return r['value']
    return value(entry['thread_depth_mm'])


def _joint(name):
    return next(j for j in _S['joints'] if j['name'] == name)


def _motor(joint):
    j = _joint(joint)
    return j, j['act'], _S['frame'](j['act'], j['axis'], j['out'], j['clock'], _S['act'][j['act']]['r'])


def _on_motor(j, f, r, ang, t):
    a = math.radians(ang)
    return [j['act_c'][k] + (r * math.cos(a) * f['x'][k] + r * math.sin(a) * f['y'][k] + t * f['out'][k]) / 1000 for k in range(3)]


# ---------------------------------------------------------------- screws

def screw_mass(d, length, data, head):
    rho = assumed('steel_density')
    vol = data['As'] * length                                                  # the threaded shank at its stress area
    vol += math.pi / 4 * data['head_d'] ** 2 * data['head_k'] * (0.55 if head == 'iso10642' else 0.85)      # head less socket
    return vol * 1e-9 * rho


def screw_geom(name, thread, length, head, pos, axis, sign, grade, clamps, into, **hw):
    """A screw as a solid of revolution. pos: under its head, on the seat. sign: +1 if head to tip runs along +axis. length: mm, under head to tip (ISO 4762's nominal)."""
    d = float(thread[1:])
    data = screw_data(thread, head)
    L, hd, hk = length / 1000, data['head_d'] / 2000, data['head_k'] / 1000
    r = d / 2000
    if head == 'iso10642':                                                     # head sits flush: its cone is inside the part, the seat is the top face
        prof = [(0.0, 0.0), (0.0, hd), (hk, r), (L, r), (L, 0.0)]
    else:
        prof = [(-hk, 0.0), (-hk, hd), (0.0, hd), (0.0, r), (L, r), (L, 0.0)]
    if sign < 0:
        prof = [(-t, x) for t, x in reversed(prof)]
    dirv = [sign * v for v in AXES[axis]]
    return dict(kind='revolve', name=name, axis=axis, c=list(pos), profile=[list(p) for p in prof], color=SCREW_COLOR, role='hardware', material='steel', mass=screw_mass(d, length, data, head),
                hw=dict(type='screw', thread=thread, d=d, length=length, head=head, grade=grade, pos=list(pos), dir=dirv, clamps=list(clamps), into=into, head_d=data['head_d'], head_k=data['head_k'], socket=data['socket'], pitch=data['pitch'], **hw))


def pattern(name, joint, which, link, seat_t, length, head='iso4762', clamps=(), counterbore_to=None, grade=None, skip=(), note=''):
    """One call: every hole of a motor's pattern gets a screw, with a clearance hole in each clamped part (and a counterbore back to t = counterbore_to).
    joint: the joint whose motor it is. which: 'hub', 'stator_front' or 'stator_rear'. link: the link the screws belong to (the one they clamp).
    seat_t: where the head sits, mm along the motor's output direction from the middle of its length. length: nominal, mm. skip: indices to leave out."""
    j, model, f = _motor(joint)
    entry, t_face = motor_face(model, which)
    thread = value(entry['thread'])
    data = screw_data(thread, head)
    depth = thread_depth(model, which, entry)
    r = value(entry['bolt_circle_mm']) / 2
    angles = value(entry['angles_step_deg'])
    grade = grade or assumed('screw_grade')
    # head to tip runs from the seat toward the motor's face
    run = 1 if t_face > seat_t else -1
    axis = j['axis']
    sign = int(round(sum(a * b for a, b in zip(f['out'], AXES[axis])))) * run
    motor_name = f"{joint} actuator (buy) {model}"
    names = []
    clamp_len = abs(t_face - seat_t)
    for k, ang in enumerate(angles):
        if k in skip:
            continue
        pos = _on_motor(j, f, r, ang, seat_t)
        nm = f'{name} screw {k + 1}'
        face = _on_motor(j, f, r, ang, t_face)
        g = screw_geom(nm, thread, length, head, pos, axis, sign, grade, clamps, 'motor:' + motor_name, pattern=name, face=face, hole_depth=depth, angle=ang, r_bc=r, which=which, joint=joint, model=model)
        _add(link, g)
        names.append(nm)
        if clamps:
            ctr = _on_motor(j, f, r, ang, (seat_t + t_face) / 2)
            _add(link, _cut(f'{name} hole {k + 1}', ctr, axis, data['clear'] / 2000, (clamp_len + 1.0) / 1000, clamps,
                            hw=dict(type='hole', kind='clearance', thread=thread, d=data['clear'], pattern=name, pos=ctr, axis=axis, parts=list(clamps))))
        if counterbore_to is not None:
            cbd = data['head_d'] + HEAD_CLEAR
            ctr = _on_motor(j, f, r, ang, (seat_t + counterbore_to) / 2)
            _add(link, _cut(f'{name} counterbore {k + 1}', ctr, axis, cbd / 2000, (abs(seat_t - counterbore_to) + 0.4) / 1000, clamps,
                            hw=dict(type='hole', kind='counterbore', thread=thread, d=cbd, pattern=name, pos=ctr, axis=axis, parts=list(clamps))))
    PATTERNS.append(dict(name=name, joint=joint, model=model, which=which, link=link, thread=thread, count=len(angles), placed=names, r_bc=r, clamps=list(clamps), motor=motor_name, note=note))
    return names


def thread_r(thread):
    return float(thread[1:]) / 2000


def fastener(name, link, thread, length, head, pos, axis, sign, clamps=(), into=None, grade=None, clear_len=None, note='', group=None):
    """One screw. pos: on the seat under its head. Cuts a clearance hole in each clamped part, and a tapped hole in 'into' (a part of ours), 1.5 pitch deeper than the tip for the tap's run-out."""
    grade = grade or assumed('screw_grade')
    data = screw_data(thread, head)
    dirv = [sign * v for v in AXES[axis]]
    run = 1.5 * data['pitch']
    span = clear_len if clear_len is not None else 0.0
    g = screw_geom(name, thread, length, head, pos, axis, sign, grade, clamps, into, pattern=group or name, face=add_v(pos, dirv, span / 1000), hole_depth=length - span + run, single=True, note=note)
    _add(link, g)
    if head == 'iso10642' and clamps:             # its cone, in the part it clamps
        hr, hk, r = data['head_d'] / 2000 + 0.0002, data['head_k'] / 1000 + 0.0001, thread_r(thread) + 0.0002
        prof = [[0.0, 0.0], [0.0, hr], [hk, r], [hk, 0.0]] if sign > 0 else [[-hk, 0.0], [-hk, r], [0.0, hr], [0.0, 0.0]]
        _add(link, dict(kind='revolve', name=f'{name} seat', axis=axis, c=list(pos), profile=prof, color='#000000', role='cut', targets=list(clamps), internal=True))
    if clamps and span:
        c = add_v(pos, dirv, span / 2000)
        _add(link, _cut(f'{name} hole', c, axis, data['clear'] / 2000, (span + 0.4) / 1000, clamps, hw=dict(type='hole', kind='clearance', thread=thread, d=data['clear'], pattern=name, pos=c, axis=axis, parts=list(clamps))))
    if into and not into.startswith('motor:'):
        c = add_v(pos, dirv, (span + length + run) / 2000)
        _add(link, _cut(f'{name} tapped hole', c, axis, data['tap'] / 2000, (length - span + run) / 1000, [into], hw=dict(type='hole', kind='tapped', thread=thread, d=data['tap'], pattern=name, pos=c, axis=axis, parts=[into])))
    SINGLE.append(dict(name=name, link=link, thread=thread, count=1, placed=[name], clamps=list(clamps), into=into, note=note))
    return name


def pin_holes(name, joint, target, fit=0.02):
    """Holes for the pins a motor carries on its hub (RS04: 3 x d6, stand 3 proud). The pins are the motor's own, drawn with it; these are the holes in the part on the hub."""
    j, model, f = _motor(joint)
    p = value(OUT[model].get('pins'))
    t0 = OUT[model]['t_hub_face']
    for k, ang in enumerate(p['angles']):
        ctr = _on_motor(j, f, p['bc'] / 2, ang, t0 + p['height'] / 2 + 0.25)
        _add(_link_of(target), _cut(f'{name} {k + 1}', ctr, j['axis'], (p['d'] + fit) / 2000, (p['height'] + 0.5) / 1000, [target],
                                    hw=dict(type='hole', kind='pin', d=p['d'] + fit, pattern=name, pos=ctr, axis=j['axis'], parts=[target])))


def _link_of(part):
    """The link a part is in, so a cut goes beside its target. A flange doesn't exist yet: it lands on the link its joint's output goes to."""
    m = re.match(r'(.+) flange$', part)
    if m:
        j = _joint(m.group(1))
        return j['parent'] if j.get('stator_on_child') else j['child']
    for ln, l in _S['links'].items():
        if any(g['name'] == part for g in l['geoms']):
            return ln
    raise KeyError(part)


# ---------------------------------------------------------------- pins, registers, inserts

def dowel(name, link, d, length, pos, axis, parts=()):
    """A bought hardened pin (ISO 8734, hardware_catalog.json for its tolerance), centred at pos, with a hole in each part."""
    rho = assumed('steel_density')
    mass = math.pi / 4 * d * d * length * 1e-9 * rho
    g = _cyl(name, pos, axis, d / 2000, length / 1000, PIN_COLOR, 'hardware', material='steel', mass=mass, hw=dict(type='dowel', d=d, length=length, pos=list(pos), axis=axis, clamps=list(parts), pattern=name, count=1))
    _add(link, g)
    if parts:
        _add(link, _cut(f'{name} hole', pos, axis, d / 2000, (length + 0.2) / 1000, parts, hw=dict(type='hole', kind='pin', d=d, pattern=name, pos=list(pos), axis=axis, parts=list(parts))))
    return name


def register(name, link, target, c, axis, d, depth):
    """A spigot: a round recess of diameter d in target, centred at c on its seat face, that locates it on a boss of the mating part. Recorded; no part of its own."""
    _add(link, _cut(name, c, axis, d / 2000, depth / 1000, [target], hw=dict(type='hole', kind='register', d=d, pattern=name, pos=list(c), axis=axis, parts=[target])))
    return name


def insert(name, link, part, thread, pos, axis, sign, length=None, family='pa12_design_values'):
    """A heat-set insert in a printed part: hole and depth from the catalogue's table; a missing row raises, as no citable size exists for it."""
    t = CAT['heat_set_inserts'][family][thread]
    if t is None:
        raise ValueError(f'{thread}: no {family} row in the catalogue')
    hole_d, hole_depth = t[0], t[1]
    L = length or CAT['heat_set_inserts']['ruthex_length'][thread]
    dirv = [sign * v for v in AXES[axis]]
    c = add_v(pos, dirv, hole_depth / 2000)
    rho = 8500.0                                                               # brass
    mass = math.pi / 4 * hole_d ** 2 * L * 1e-9 * rho * 0.6
    _add(link, _cyl(name, c, axis, hole_d / 2000, L / 1000, '#B8983C', 'hardware', material='brass', mass=mass, hw=dict(type='insert', thread=thread, d=hole_d, length=L, pos=list(pos), dir=dirv, clamps=[], into=part, pattern=name, count=1)),
         _cut(f'{name} hole', c, axis, hole_d / 2000, hole_depth / 1000, [part], hw=dict(type='hole', kind='insert', thread=thread, d=hole_d, pattern=name, pos=c, axis=axis, parts=[part])))
    return name


# ---------------------------------------------------------------- bearings, axles

def bearing(name, designation, c, axis, inner_link, inner_parts, outer_link, outer_parts, joint, partner=None, note=''):
    """A catalogue bearing centred at c: which link carries each race, and the parts that seat each. Drawn as its ring, a part of its outer link.
    Doesn't cut its seat: the part that holds it is shaped in spec/robot.py, where the look is decided, and the 'bearing' checks read those solids."""
    b = bearing_table()[designation]
    mass = b['mass']
    if mass is None:                                                           # the catalogue has no mass for this one: a solid ring at 60 % fill, flagged by the note
        mass = math.pi / 4 * (b['D'] ** 2 - b['d'] ** 2) * b['B'] * 1e-9 * 7850 * 1000 * 0.6
    hw = dict(type='bearing', designation=designation, d=b['d'], D=b['D'], B=b['B'], C=b['C'], C0=b['C0'], kind=b['kind'], da=b.get('da'), Da=b.get('Da'), source=b['source'], read_via=b['read_via'],
              pos=list(c), axis=axis, inner_link=inner_link, inner_parts=list(inner_parts), outer_link=outer_link, outer_parts=list(outer_parts), joint=joint, partner=partner, note=note)
    _add(outer_link, _cyl(name, c, axis, b['D'] / 2000, b['B'] / 1000, BEARING_COLOR, 'bearing', material='steel', mass=mass / 1000, hw=hw),
         dict(_cut(f'{name} inner', c, axis, b['d'] / 2000, (b['B'] + 1) / 1000, [name]), segs=128))
    BEARINGS.append(dict(name=name, designation=designation, joint=joint, partner=partner))
    return name


def axle(name, link, c, axis, d, length, color='#1B1C1F', exposed=True, **kw):
    """A pin or stub, role struct so a section can read it: a real diameter on a load path, with a record of what it is."""
    return dict(_cyl(name, c, axis, d / 2000, length / 1000, color, 'struct', exposed=exposed, material='al7075', fill=1.0, hw=dict(type='axle', d=d, length=length, pos=list(c), axis=axis), **kw))


def retainer(name, link, c, axis, ro, ri, h, part_of, kind='retaining ring', mass=None, **kw):
    """A bought or thin stop beside a race. Round, on the axis. kind says what it is. Its stopping is read from the solids by the 'bearing' checks, whatever it's called."""
    rho = assumed('steel_density')
    m = mass if mass is not None else math.pi * (ro ** 2 - ri ** 2) * h * rho
    _add(link, _cyl(name, c, axis, ro, h, PIN_COLOR, 'hardware', material='steel', mass=m, hw=dict(type='retainer', kind=kind, part_of=part_of, pos=list(c), axis=axis, count=1, pattern=name, clamps=[]), **kw),
         _cut(f'{name} inner', c, axis, ri, h + 0.001, [name]))
    return name


def bolt_group(name, link, thread, length, head, positions, axis, sign, clamps, into, clear_len, load, note=''):
    """A row of our own screws in our own parts: one fastener per position, read as one joint by the checks. load: dict(V N, T N m, Mt N m, Fa N, r m): what the joint must pass, for 'bolted'."""
    names = []
    for k, p in enumerate(positions):
        names.append(fastener(f'{name} screw {k + 1}', link, thread, length, head, p, axis, sign, clamps=clamps, into=into, clear_len=clear_len, note=note, group=name))
    GROUPS.append(dict(name=name, link=link, thread=thread, count=len(positions), placed=names, load=load, note=note))
    return names


def step(part_names, path=None, distance=0.05, screws=(), why=''):
    """One step of the build order, as data: these parts go in along path (a world direction) by distance (m), then these screw patterns are driven. A part named in no step is in place from the start or isn't part of the joint."""
    BUILD.append(dict(parts=list(part_names), path=list(path) if path else None, distance=distance, screws=list(screws), why=why))


def records():
    return dict(patterns=PATTERNS, bearings=BEARINGS, singles=SINGLE, groups=GROUPS, build=BUILD, assumptions=ASSUME['rows'])


# ---------------------------------------------------------------- machined shapes (tools/sdf.py reads these trees). Metres. What the knee is made of, for the next joints

def rounded_poly(pts, n=6):
    """A polygon with its corners filleted. pts: [(u, v)] or [(u, v, radius)], in either winding; arcs of n segments, tangent to both edges. Radius clamped to what the two edges allow."""
    P = [(p[0], p[1], p[2] if len(p) > 2 else 0.0) for p in pts]
    m = len(P)
    out = []
    for i in range(m):
        ax, ay, _ = P[i - 1]
        bx, by, r = P[i]
        cx, cy, _ = P[(i + 1) % m]
        u, v = (ax - bx, ay - by), (cx - bx, cy - by)
        lu, lv = math.hypot(*u), math.hypot(*v)
        if r <= 0 or lu < 1e-9 or lv < 1e-9:
            out.append([bx, by])
            continue
        u, v = (u[0] / lu, u[1] / lu), (v[0] / lv, v[1] / lv)
        cosang = max(min(u[0] * v[0] + u[1] * v[1], 1.0), -1.0)
        ang = math.acos(cosang)
        if abs(ang - math.pi) < 1e-6:
            out.append([bx, by])
            continue
        t = r / math.tan(ang / 2)
        t = min(t, lu * 0.49, lv * 0.49)
        r = t * math.tan(ang / 2)
        w = math.hypot(u[0] + v[0], u[1] + v[1])
        c = (bx + (u[0] + v[0]) / w * r / math.sin(ang / 2), by + (u[1] + v[1]) / w * r / math.sin(ang / 2))
        a0 = math.atan2(by + u[1] * t - c[1], bx + u[0] * t - c[0])
        a1 = math.atan2(by + v[1] * t - c[1], bx + v[0] * t - c[0])
        da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        for k in range(n + 1):
            a = a0 + da * k / n
            out.append([c[0] + r * math.cos(a), c[1] + r * math.sin(a)])
    return out


def lug(c0, r0, c1, r1, n=48):
    """The outline a lug has in its own plane: the hull of two circles, tangent lines between them. c: (u, v). Goes round counter-clockwise."""
    dx, dy = c1[0] - c0[0], c1[1] - c0[1]
    d = math.hypot(dx, dy)
    base = math.atan2(dy, dx)
    beta = math.acos(max(min((r0 - r1) / d, 1.0), -1.0))          # from the centre line to where the tangent leaves
    pts = []
    for k in range(n + 1):                                         # round the big end, the long way
        a = base + beta + (2 * math.pi - 2 * beta) * k / n
        pts.append([c0[0] + r0 * math.cos(a), c0[1] + r0 * math.sin(a)])
    for k in range(n // 2 + 1):                                    # round the small end
        a = base - beta + 2 * beta * k / (n // 2)
        pts.append([c1[0] + r1 * math.cos(a), c1[1] + r1 * math.sin(a)])
    return pts


def s_union(k, *of):
    return dict(t='u', k=k, of=list(of))


def s_inter(k, *of):
    return dict(t='i', k=k, of=list(of))


def s_cut(k, a, *b):
    return dict(t='d', k=k, a=a, b=list(b))


def s_cyl(c, axis, r, h, e=0.0):
    return dict(t='cyl', c=list(c), axis=axis, r=r, h=h, e=e)


def s_rev(c, axis, poly):
    return dict(t='rev', c=list(c), axis=axis, poly=[list(p) for p in poly])


def s_ext(axis, t0, t1, poly, e=0.0):
    return dict(t='ext', axis=axis, t0=t0, t1=t1, poly=[list(p) for p in poly], e=e)


def s_loft(secs, zlo=None, zhi=None, e=0.0):
    secs = sorted([list(s) for s in secs])
    return dict(t='loft', secs=secs, zlo=secs[0][0] if zlo is None else zlo, zhi=secs[-1][0] if zhi is None else zhi, e=e)


def s_box(c, size, e=0.0):
    return dict(t='box', c=list(c), size=list(size), e=e)


def machined(name, tree, color, voxel=0.001, voxel_hi=0.0005, exposed=False, **kw):
    """A machined 7075 part: a distance tree, truly the shape it is, so it weighs what its solid weighs (fill 1). Its bores and threads are 'cut' geoms, as for any part."""
    g = dict(kind='sdf', name=name, tree=tree, color=color, role='struct', material='al7075', fill=1.0, voxel=voxel, voxel_hi=voxel_hi, **kw)
    if exposed:
        g['exposed'] = True
    return g


def round_housing(c, axis, od_r, t0, t1, face_t=None, collar_r=None, od_fillet=0.005, end_fillet=0.0008):
    """A round housing round a motor, as a tree: the solid outside; its bores are cut by the caller, on the real faces. Closed at t0 by a plate, open at t1, where the motor goes in. t0, t1: its ends along +axis from c.
    collar_r: a collar standing past the plate's face (face_t) to t0, for a bearing seat. od_fillet rounds the plate into the outside. First proposal for any RobStride motor: bore 0.05 over its radius, wall 4 to 5 mm."""
    pts = [(0.0, t0)]
    if collar_r:
        pts += [(collar_r, t0, 0.001), (collar_r, face_t, 0.0015), (od_r, face_t, od_fillet)]
    else:
        pts += [(od_r, t0, od_fillet)]
    pts += [(od_r, t1, end_fillet), (0.0, t1)]
    return s_rev(c, axis, rounded_poly(pts))


def fork_yoke(axis_c, eye_r, top_c, top_r, front_secs, slot_t0, slot_t1, arch_r, fillet=0.005, pocket=None):
    """A two-sided fork that runs into a tube, as a tree. Side outline: a lug about the joint's y axis (eye_r) to the tube's end (top_r at top_c). Front outline: front_secs, a z loft whose sections go from squared at the eyes to round at the tube.
    Cut by the slot (slot_t0 to slot_t1 along y), its roof a cylinder of arch_r about the axis, so what turns in the slot never sweeps the roof. pocket: a tree that hollows it."""
    x0, y0, z0 = axis_c
    side = s_ext('y', y0 - 0.2, y0 + 0.2, lug((x0, z0), eye_r, top_c, top_r))
    outer = s_inter(fillet, side, s_loft(front_secs))
    slot = s_union(0.0, s_cyl((x0, (slot_t0 + slot_t1) / 2, z0), 'y', arch_r, slot_t1 - slot_t0),
                   s_box((x0, (slot_t0 + slot_t1) / 2, z0 - 0.15), (2 * arch_r, slot_t1 - slot_t0, 0.3)))
    return s_cut(0.0015, outer, slot, *([pocket] if pocket else []))


def closed_tube(z0, z1, cx, cy, r, wall, top_collar=None):
    """A closed round tube along z, as (outer, cavity) trees the caller blends in. top_collar: (r, t, fillet): a flange at its top end, for screws from below."""
    pts = [(0.0, z0), (r, z0)]
    if top_collar:
        pts += [(r, z1 - top_collar[1], top_collar[2]), (top_collar[0], z1 - top_collar[1], 0.0008), (top_collar[0], z1, 0.0008), (0.0, z1)]
    else:
        pts += [(r, z1), (0.0, z1)]
    outer = s_rev((cx, cy, 0.0), 'z', rounded_poly(pts))
    return outer, s_cyl((cx, cy, (z0 + z1) / 2), 'z', r - wall, z1 - z0 + 0.002)


def end_cap(c, axis, spigot_r, spigot_t, flange_r, t0, t1, post=None):
    """A round cap on a housing's open end, as a tree: a spigot that registers in the bore (spigot_t long, from t0), a flange on the housing's end face, up to t1 along +axis; optionally a post on its axis, post=(r, t_end, shoulder_r, shoulder_t)."""
    pts = [(0.0, t0), (spigot_r, t0, 0.0003), (spigot_r, t0 + spigot_t), (flange_r, t0 + spigot_t, 0.0004), (flange_r, t1, 0.0012)]
    if post:
        # a 0.3 mm undercut at the shoulder, as for any seat a race must sit square on: no fillet in the race's corner
        pts += [(post[2], t1, 0.0006), (post[2], post[3], 0.0003), (post[0] + 0.0003, post[3]), (post[0] + 0.0003, post[3] - 0.00025), (post[0] - 0.0003, post[3] - 0.00025), (post[0] - 0.0003, post[3] + 0.00025), (post[0], post[3] + 0.00025), (post[0], post[1], 0.0006), (0.0, post[1])]
    else:
        pts += [(0.0, t1)]
    return s_rev(c, axis, rounded_poly(pts))


def self_test():
    """Each helper on a toy plate. Exercises what the knee's proof doesn't use: dowel, register, insert."""
    links = {'a': dict(geoms=[dict(name='plate', role='struct')])}
    joints = [dict(name='J', act='RS04', axis='y', out=-1, clock=165.0, act_c=[0, 0, 0], parent='a', child='a', stator_on_child=False)]
    frame = lambda model, axis, out, clock, r0: dict(x=[1.0, 0, 0], y=[0, 0, 1.0], out=[0, -1.0, 0])
    setup(links, joints, dict(RS04=dict(r=0.06)), frame)
    pattern('p', 'J', 'hub', 'a', 31.0, 12, clamps=['plate'])
    fastener('f', 'a', 'M4', 10, 'iso4762', (0, 0, 0), 'y', 1, clamps=['plate'], into='plate', clear_len=2.0)
    dowel('d', 'a', 4, 10, (0, 0, 0), 'y', ['plate'])
    register('r', 'a', 'plate', (0, 0, 0), 'y', 50.0, 0.5)
    insert('i', 'a', 'plate', 'M3', (0, 0, 0), 'y', 1)
    bearing('b', 'CRBT 405 A', (0, 0, 0), 'y', 'a', ['plate'], 'a', ['plate'], 'J')
    retainer('c', 'a', (0, 0, 0), 'y', 0.02, 0.01, 0.001, 'b')
    a = axle('x', 'a', (0, 0, 0), 'y', 40.0, 10.0)
    gs = links['a']['geoms']
    kinds = {g['hw']['type'] for g in gs if g.get('hw')} | {a['hw']['type']}
    assert kinds >= {'screw', 'hole', 'dowel', 'insert', 'bearing', 'retainer', 'axle'}, kinds
    assert sum(1 for g in gs if g.get('hw', {}).get('type') == 'screw' and g['hw'].get('pattern') == 'p') == 9
    assert all(g['mass'] > 0 for g in gs if g['role'] in ('hardware', 'bearing'))
    print('hardware helpers ok:', len(gs), 'geoms,', {k: sum(1 for g in gs if g.get('hw', {}).get('type') == k) for k in sorted(kinds)})
    # the shapes: polygons close and stay closed, a lug goes round
    assert len(lug((0, 0), 0.05, (0, 0.15), 0.03)) > 50
    assert len(rounded_poly([(0, 0), (0.01, 0, 0.002), (0.01, 0.01), (0, 0.01)])) > 4
    for rec in (PATTERNS, BEARINGS, SINGLE, GROUPS, BUILD):
        rec.clear()


if __name__ == '__main__':
    self_test()
