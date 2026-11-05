"""echo, a Figure 03 replica: the one source of truth for kinematics, actuators, masses and outer shape.

Run `python -I spec/robot.py` to write spec/robot.json, which the offline checks, the URDF writer and the Fusion
build all read. Never hand-edit robot.json or the Fusion document: change this file and rebuild.

Frame: metres, X forward, Y left, Z up, floor at z = 0. Zero pose: standing straight, arms hanging, palms in.
All geometry is written in WORLD coordinates at the zero pose. Each link's frame sits at its parent joint's origin
with world-aligned axes. The left side is written once; the right side is its mirror (y -> -y).

Internals follow Asimov 1's layout (one actuator per joint, stator in the parent link, 7075 brackets, PA12 covers)
scaled to Figure's size, with RobStride modules we can buy (spec/actuators.json). The knee is driven through a 2:1
linkage from an RS04 in the thigh, and the ankle by two RS06 in the calf through rods (parallel ankle, like Asimov
and Unitree G1), so those joints are 'remote' (actuator not on the joint axis).

Geometry roles:
  shell     outer cover (fabric / PA12). May overlap the shells and parts of ADJACENT links (it is the skin).
  knit      the garment (iteration 39; user: "Figure has a fabric cover on the outside", a design rule for all grey). One soft skin over a hidden
            frame and foam, sewn at seams, open only where hubs, hands, waist and shoes come through. It does not block a joint: the rigid
            parts under it must clear each other, and fit inside it with an allowance (WALL). Across far links it still has bulk (hand on thigh).
  decal     cosmetic surface detail sunk into a skin (camera ports, logo, light bar). Must touch its link.
  struct    rigid internal part (7075 bracket, fork, battery). Must not overlap rigid parts of any other link.
  actuator  bought actuator, stator side. Rigid. Lives in the joint's PARENT link, centred on the joint axis.
"""
import copy
import math
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                  # python -I leaves the script's folder off the path
import hardware as HW  # noqa: E402
ACT = json.load(open(os.path.join(HERE, 'actuators.json'), encoding='utf-8'))
OUTLINES = json.load(open(os.path.join(HERE, 'actuator_outlines.json'), encoding='utf-8'))   # measured from the maker's STEP (tools/motor_outlines.py)

# WHITE (iteration 37): the shoe's upper and the sock; Figure's upper is a light grey, not white (side and 3/4 frames)
GREY, BLACK, GLOSS, WHITE, SOLE = '#BCC0C4', '#1B1C1F', '#08090B', '#C6C9CF', '#2B2B2E'   # iteration 42: the shoe a light grey, as Figure's frames (1.24 x the knit there): it read white
ALU, CELL, ACCENT = '#9EA3A8', '#2E3B4E', '#7C6FE6'
FLANGE_T = 0.006          # output flange thickness on the child's side of each actuator

LINKS, JOINTS = {}, []

# Materials (tools/mass_model.py): a part's mass is its material's density times its volume (a cover's 3 mm wall, a
# decal's layer, a frame's solid times a fill factor), unless the part gives an explicit 'mass' (user, 2026-10-31:
# weights by material, actuator and joint, to make the model realistic).
MATERIALS = {
    'knit_pa12': dict(density=1110, note='PA12 (MJF) cover under a knit sleeve: 1010 kg/m3 plus 0.3 kg/m2 of knit '
                                         'over a 3 mm wall'),
    'pa12': dict(density=1010, note='PA12: black covers, pads, marks'),
    'pc': dict(density=1200, layer=0.002, note='polycarbonate: the glossy black covers; the visor is a 2 mm layer'),
    'tpu': dict(density=1200, note='TPU: outsole and shoe upper'),
    'eva': dict(density=250, layer=0.0, note='EVA foam midsole (solid)'),
    'al7075': dict(density=2810, note='7075-T6 frames, brackets, flanges, cranks and rods, with a fill factor'),
    'cells': dict(density=2500, note='battery pack: explicit mass'),
    'electronics': dict(density=1500, note='compute, power board, cameras: explicit masses'),
    'garment': dict(density=1110, note='the grey skin, per m2 as the 3 mm PA12 and knit cover it replaced (3.3 kg/m2): knit about 0.5, foam about 0.3, and about 2.5 '
                                       'held for the inner frames, which are not designed. An estimate carried over so the masses do not fall on a guess'),
    'hand': dict(density=1500, note="the hand's motors, tendons and palm: explicit mass"),
    'steel': dict(density=7850, note='screws, pins and bearings: role hardware or bearing, explicit mass from spec/hardware.py (catalogue, or geometry times density)'),
    'brass': dict(density=8500, note='heat-set inserts: explicit mass from spec/hardware.py'),
}


def material_of(g):
    """Default material by role and colour (a geom may name its own)."""
    if g.get('material'):
        return g['material']
    role, col = g['role'], g['color']
    if role in ('struct', 'linkage'):
        return 'cells' if col == CELL else 'al7075'
    if role == 'knit':
        return 'garment'
    if role == 'decal':
        return 'pc' if col in (GLOSS, LENS) else 'eva' if col == SOLE else 'pa12'
    if role == 'shell':
        return {GREY: 'knit_pa12', GLOSS: 'pc', WHITE: 'tpu', SOLE: 'tpu'}.get(col, 'pa12')
    return role


# ---------------------------------------------------------------- primitives (world coords, zero pose)

def cyl(name, c, axis, r, h, color=ALU, role='struct'):
    return dict(kind='cyl', name=name, c=list(c), axis=axis, r=r, h=h, color=color, role=role)


def ell(name, c, r, color=GREY, role='shell'):
    return dict(kind='ellipsoid', name=name, c=list(c), r=list(r), color=color, role=role)


def rbox(name, c, size, fillet, color=ALU, role='struct', exposed=False, **extra):
    """extra: mass-model fields ('mass', 'add_mass', 'add_what', 'material', 'fill'; tools/mass_model.py)."""
    return dict(kind='rbox', name=name, c=list(c), size=list(size), fillet=fillet, color=color, role=role,
                exposed=exposed, **extra)


def cap(name, p0, p1, r, color=BLACK, role='shell'):
    return dict(kind='capsule', name=name, p0=list(p0), p1=list(p1), r=r, color=color, role=role)


def loft(name, sections, color=GREY, role='shell', open=('top', 'bottom'), clip=None, keep=None):
    """sections: (z, cx, cy, a, b, n): superellipse |dx/a|^n + |dy/b|^n = 1 in the plane z, bottom to top or
    top to bottom. n = 2 is an ellipse, larger n is squarer. open: ends where the skin is a sleeve (no cap).
    clip: (point, normal): keep only the part in front of this plane (a visor cut from a head shape); or a list of
    them: keep the part in front of every one (a shoe collar, a band between two planes).
    keep: (centre, radius): keep only the part inside this sphere (a domed top, round about the vertical through
    the centre, so a yaw joint on that line turns it in place)."""
    secs = [dict(z=z, cx=cx, cy=cy, a=a, b=b, n=n) for z, cx, cy, a, b, n in sections]
    g = dict(kind='loft', name=name, sections=sorted(secs, key=lambda s: s['z']), color=color, role=role,
             open=list(open))
    if clip:
        planes = [dict(p=list(c[0]), n=list(c[1]), **({'r': c[2]} if len(c) > 2 else {}))
                  for c in (clip if isinstance(clip, list) else [clip])]
        g['clip'] = planes if isinstance(clip, list) else planes[0]
    if keep:
        g['keep'] = dict(c=list(keep[0]), r=keep[1])
    return g


def rounded(plane, r):
    """A clip plane whose cut rim is rounded with radius r (in Fusion; the checks keep the sharp, larger shape)."""
    return plane[0], plane[1], r


def cut(g, *targets):
    """g removed from the named geoms of the same link (a window through a shoe): role 'cut', no body of its own."""
    return dict(g, role='cut', targets=list(targets))


def below(p, q):
    """Clip plane through the side-view points p = (x, z) and q, keeping what is below the line pq (any y)."""
    dx, dz = q[0] - p[0], q[1] - p[1]
    n = (dz, 0.0, -dx) if dx > 0 else (-dz, 0.0, dx)        # the normal pointing down
    return (p[0], 0.0, p[1]), n


def above(p, q):
    pt, n = below(p, q)
    return pt, tuple(-v for v in n)


def ahead(x):
    """Clip plane keeping x > x0 (in front)."""
    return (x, 0.0, 0.0), (1.0, 0.0, 0.0)


def behind(x):
    return (x, 0.0, 0.0), (-1.0, 0.0, 0.0)


LENS = '#0A1420'


def front_x(sections, y, z):
    """x of a loft's front surface at (y, z), interpolated between sections the way the loft is (PCHIP; a straight
    line between sections left the face mark buried 2-3 mm in Fusion, 2026-10-31)."""
    from scipy.interpolate import PchipInterpolator
    s = sorted(sections)
    f = [PchipInterpolator([q[0] for q in s], [q[k] for q in s]) for k in (1, 2, 3, 4, 5)]
    cx, cy, a, b, n = (float(g(z)) for g in f)
    return cx + a * (1 - abs((y - cy) / b) ** n) ** (1 / n)


def camera_port(name, sections, z, r):
    """A camera port on the centre line of a loft's front, square to the surface (the face slopes): dark ring
    0.6 mm proud, glossy lens 1.5 mm proud."""
    xs = front_x(sections, 0, z)
    slope = (front_x(sections, 0, z + 0.002) - front_x(sections, 0, z - 0.002)) / 0.004
    n = (1.0 / (1 + slope * slope) ** 0.5, 0.0, -slope / (1 + slope * slope) ** 0.5)     # outward normal
    at = lambda d: (xs + n[0] * d, 0.0, z + n[2] * d)
    return [cyl(f'{name} ring', at(0.0006 - 0.003), list(n), r, 0.006, '#2A2C30', 'decal'),
            cyl(f'{name} lens', at(0.0015 - 0.003), list(n), 0.5 * r, 0.006, LENS, 'decal')]


def revolve(name, axis, c, profile, color=GREY, role='shell'):
    """Solid of revolution about the axis-aligned line through c; profile (t, r) with t the coordinate along the
    axis (first and last r = 0). Round about its axis, so it never sweeps as a joint on that axis turns."""
    return dict(kind='revolve', name=name, axis=axis, c=list(c), profile=[list(p) for p in profile], color=color,
                role=role)


def prism(name, x0, t, pts, color, role='decal'):
    """A flat convex polygon in the plane x = x0, points (y, z), extruded t along +x."""
    return dict(kind='prism', name=name, x0=x0, t=t, pts=[list(p) for p in pts], color=color, role=role)


def tri_mark(name, sections, zc, w, color):
    """Our mark (no Figure logo, for trademark reasons): an inverted triangle w wide, 0.6 mm proud of a loft's
    front at its centre."""
    h = w * 0.866
    pts = [(-w / 2, zc + h / 3), (w / 2, zc + h / 3), (0.0, zc - 2 * h / 3)]
    return [prism(name, front_x(sections, 0, zc) + 0.0006 - 0.004, 0.004, pts, color)]


def motor_frame(model, axis, out, clock, r0):
    """How a bought motor sits: where its STEP's +X and +Y point in the world. out: +1 or -1, its output along +axis or -axis. clock: degrees about the
    STEP's own +Z. At clock 0 STEP +X points along the world's reference for that axis (+x for a y or z axis, +z for an x axis)."""
    import numpy as np
    a = np.array({'x': (1.0, 0, 0), 'y': (0, 1.0, 0), 'z': (0, 0, 1.0)}[axis]) * out
    z = a * OUTLINES[model]['step_output_sign']
    ref = np.array((0, 0, 1.0) if axis == 'x' else (1.0, 0, 0))
    y0 = np.cross(z, ref)
    t = np.radians(clock)
    x = np.cos(t) * ref + np.sin(t) * y0
    y = np.cross(z, x)
    return dict(model=model, clock=clock, r0=r0, x=x.round(9).tolist(), y=y.round(9).tolist(), out=a.tolist())


def motor(name, model, c, axis, out, clock, color='#3A3D42', r=None):
    """A bought motor at its real outline: a cylinder's fields (for mass and axis checks) plus the frame geom.py reads the STEP-measured shape through."""
    r = r if r is not None else ACT[model]['r']
    return dict(cyl(name, c, axis, r, ACT[model]['h'], color, 'actuator'), outline=motor_frame(model, axis, out, clock, r))


# Each motor's turn about its axis (2026-11-05): nothing fixed it while motors were plain cylinders. Scanned every 7.5 deg against every part at the zero pose:
# most fits don't care. These do: the ankle RS03 (ears up and down, as the maker's drawing allows: 0.13 % outside the calf at the worst turn) and the knee, where a
# screw must miss the cable: the tab points down and back, along the shank, and two of the stator's holes straddle the hub's tail. The others are the scan's best.
MOTOR_CLOCK = {'waist_yaw': 15.0, 'neck_yaw': 45.0, 'neck_pitch': 90.0, 'L_shoulder_pitch': 262.5, 'L_shoulder_roll': 0.0, 'L_shoulder_yaw': 75.0, 'L_elbow': 0.0, 'L_wrist_yaw': 90.0,
               'L_wrist_pitch': 30.0, 'L_wrist_roll': 165.0, 'L_hip_pitch': 15.0, 'L_hip_roll': 60.0, 'L_hip_yaw': 15.0, 'L_knee': 195.0, 'L_ankle_pitch': 90.0, 'L_ankle_roll': 60.0}


def link(name, budget, geoms, note=''):
    """budget: the hand estimate of the link's mass (kg), kept for comparison; its real mass, centre of mass and
    inertia come from its parts' materials (tools/mass_model.py)."""
    LINKS[name] = dict(name=name, budget=budget, geoms=list(geoms), note=note)


def joint(name, parent, child, origin, axis, lim, act=None, act_c=None, out=+1, exposed=False, remote=False,
          capacity=None, stator_on_child=False, clock=None, flange=None):
    """act: actuator model whose stator is added to `parent`, centred at act_c (default: the joint origin) and
    aligned with `axis`. out = +1/-1: which end of the actuator faces the child (where its flange goes).
    stator_on_child: the stator rides on the child and the output drives the parent (the knee: the hub turns with the shank).
    capacity: what drives the joint, {'model', 'count', 'ratio'} (default: one `act`, ratio 1).
    flange: a part of our own to stand for the output's flange (named '<joint> flange'), in place of the plain disc.
    clock: how far the motor is turned about its axis (motor_frame); default MOTOR_CLOCK."""
    j = dict(name=name, parent=parent, child=child, origin=list(origin), axis=axis, lower=lim[0], upper=lim[1],
             act=act, act_c=list(act_c or origin), out=out, exposed=exposed, remote=remote, stator_on_child=stator_on_child, clock=clock if clock is not None else MOTOR_CLOCK.get(name, 0.0),
             capacity=capacity or (dict(model=act, count=1, ratio=1.0) if act else None))
    if flange:
        j['flange'] = flange
    JOINTS.append(j)
    return j


# ---------------------------------------------------------------- proportions (iterate here)

# Anchor heights. Every part below is placed relative to these, so a proportion change is one number.
# Iteration 2 (2026-10-31), from tools/compare.py on Figure 03 frames (reveal video side 0:21.5, 3/4 0:24.5, front
# 1:50): head 0.205 tall x 0.14 wide x 0.18 deep with the chin at 1.52; shoulder tops at 1.445; elbow 1.155;
# wrist 0.955; fingertips 0.71; grey torso down to 1.085; black hips 0.86-1.085; knee 0.54.
H = 1.727                       # overall height (figure.ai)
HY, HZ = 0.10, 0.93             # hip centre: half spacing, height
KZ, AZ = 0.535, 0.085           # knee and ankle axes
SY, SZ = 0.205, 1.39            # shoulder centre
EZ, WZ = 1.155, 0.955           # elbow and wrist pitch axes
HEM = 1.085                     # where the grey torso ends (black waist below)
CHIN = 1.522                    # lowest point of the head
WY = HZ + 0.06                  # waist yaw actuator centre
# neck (iteration 18): the pitch axis at the neck's base, level with the torso's top, crossing the yaw axis (NX), so
# the neck tilts with the head and its base can be round about both axes
NX, NP = -0.012, 1.496
NZ = NP - 0.0755                # neck yaw actuator centre, under the pitch actuator
CAP_R = 0.0575                  # shoulder cap radius (round about the pitch axis; its outer end a hemisphere)


def span(name, lo, hi, x=0.0, y=0.0, sx=0.04, sy=0.04, f=0.004, **kw):
    """A rigid block filling the height lo..hi (for frames that join parts at two given heights)."""
    return rbox(name, (x, y, (lo + hi) / 2), (sx, sy, hi - lo), f, **kw)


# ---------------------------------------------------------------- pelvis and waist

# Hips: like Figure, the black hip mechanism is the visible housing. A closed skirt over it cannot work: the hip
# roll brackets swing through ~16 cm radius as the thigh flexes (2026-10-31 sweep).
# Figure 03's look (figure.ai back view): grey torso ending in a chamfered hem with a glossy black rim; a narrow
# glossy black waist column dropping into the pelvis; a wide glossy black pelvis made of a central block and a big
# rounded lobe over each hip, with an arch between the legs. Here the lobes ride on the hip pitch link (they move
# with the leg, as the seam on Figure's lobes suggests) and the column belongs to the torso (it turns with the
# waist yaw inside the pelvis's centre block).
# Iteration 42 (user: "all squares right now"). The joint template (reports/joint_template.md) from the hips to the neck. A real pelvis: one machined T, a round bar across holding both hip pitch RS04s, a round
# housing on top holding the waist's RS04; each motor bolted through its real rear face, each output on a cup in a CRBS 508 in a bolted ring, so the motor's own bearing isn't asked. y from the centre, hip motor at 0.092:
# rear cover 0.064 | plate 0.0723..0.0773 | motor | ring from 0.1159, its race 0.1185..0.1265 | carrier's eye from 0.129. Same stack up the waist, from its motor at z 0.99. The bar's roof is open inside, its underside a window for keys.
HOUSING_BORE, HOUSING_OD = 0.06005, 0.0627   # RS04: 2.65 mm wall; walking loads leave it far over (reports/strength.md). Lobed at the rim's M3
PITCH_ACT_Y = 0.092                       # the hip pitch actuator's centre on its axis (iteration 21: 6 mm inboard)
HIP_RIM_ANGLES = (30.0, 90.0, 150.0, 210.0, 270.0, 330.0)
WAIST_RIM_ANGLES = (15.0, 75.0, 135.0, 195.0, 255.0, 315.0)
RIM_R, RIM_LOBE = 0.0631, 0.0043


def pelvis_tree():
    y_end = PITCH_ACT_Y + 0.0241
    bar = HW.s_rev((0, 0, HZ), 'y', HW.rounded_poly([(0.0, -y_end), (HOUSING_OD, -y_end, 0.0008), (HOUSING_OD, y_end, 0.0008), (0.0, y_end)]))
    inside = HW.s_cyl((0, 0, HZ), 'y', 0.0615, 2 * (PITCH_ACT_Y - 0.01935))
    window = HW.s_box((0, 0, HZ - 0.06), (0.09, 0.124, 0.08), e=0.012)
    waist = HW.s_rev((0, 0, 0.0), 'z', HW.rounded_poly([(0.0, WY - 0.01965), (HOUSING_OD, WY - 0.01965, 0.003), (HOUSING_OD, WY + 0.0241, 0.0008), (0.0, WY + 0.0241)]))
    lobes = []
    for sg in (1, -1):
        for a in HIP_RIM_ANGLES:
            t = math.radians(a)
            lobes.append(HW.s_cyl((RIM_R * math.cos(t), sg * (y_end - 0.0042), HZ + RIM_R * math.sin(t)), 'y', RIM_LOBE, 0.0085, e=0.002))
    for a in WAIST_RIM_ANGLES:
        t = math.radians(a)
        lobes.append(HW.s_cyl((RIM_R * math.cos(t), RIM_R * math.sin(t), WY + 0.0241 - 0.0042), 'z', RIM_LOBE, 0.0085, e=0.002))
    side_windows = [HW.s_cyl((xx, sg * 0.032, HZ - 0.004), 'x', 0.021, 0.03) for sg in (-1, 1) for xx in (-0.055, 0.055)]
    t = HW.s_union(0.007, HW.s_cut(0.0, bar, inside, window, *side_windows), waist)
    return HW.s_union(0.0015, t, *lobes)


link('pelvis', 6.5, [
    # centre block; its underside is the arch between the legs (high enough for a hip flexed 60 degrees)
    # (iteration 4: shallower at the back, clear of the forward-facing hip roll outputs; the lobes cover the rest)
    loft('pelvis', [(HZ - 0.01, 0.005, 0, 0.065, 0.058, 3.0), (HZ + 0.03, 0.005, 0, 0.075, 0.085, 3.2),
                    (HZ + 0.088, 0.005, 0, 0.072, 0.08, 3.0)], GLOSS, open=()),
    # it carries the power board and BMS (0.35 kg)
    HW.machined('pelvis frame', pelvis_tree(), BLACK, voxel=0.0008, voxel_hi=0.0006, exposed=True, add_mass=0.35, add_what='electronics', finish='silver', zones=[dict(c=[0.0, 0.0, HZ], axis='y', r=0.08, t0=0.0705, t1=0.2, finish='black'), dict(c=[0.0, 0.0, HZ], axis='y', r=0.08, t0=-0.2, t1=-0.0705, finish='black'), dict(c=[0.0, 0.0, 0.0], axis='z', r=0.07, t0=HZ + 0.0385, t1=1.2, finish='black')]),
    dict(cut(cyl('waist register', (0, 0, WY + 0.00543), 'z', HOUSING_BORE, 0.03955), 'pelvis frame'), segs=256),
    cut(cyl('waist plate hole', (0, 0, WY - 0.02), 'z', 0.0475, 0.016), 'pelvis frame'),
    dict(cut(cyl('L pelvis hip register', (0, PITCH_ACT_Y + 0.00533, HZ), 'y', HOUSING_BORE, 0.03955), 'pelvis frame'), segs=256),
    cut(cyl('L pelvis hip plate hole', (0, PITCH_ACT_Y - 0.02, HZ), 'y', 0.0475, 0.016), 'pelvis frame'),
    # clearance (iteration 17, every joint swept with the covers across it): the waist column turns in a round
    # opening, 4 mm clear
    cut(cyl('pelvis waist opening', (0, 0, 1.019), 'z', 0.0755, 0.042), 'pelvis'),   # from 0.998, over the frame
    # iteration 19: its side walls cut into the hip pitch motors; round openings 3 mm clear of them (mirrored)
    dict(cut(cyl('L pelvis hip opening', (0, 0.085, HZ), 'y', 0.0665, 0.05), 'pelvis'), internal=True),
], 'hip pitch actuators, waist yaw actuator, PDB')

# Waist: yaw only, like Asimov 1. A roll/pitch waist (A3, H2) cannot hide under a one-piece fabric torso: the
# hem has to clear the hips by ~5 cm for 20 degrees of tilt (2026-10-31 sweep). Lifting uses hips and knees.
joint('waist_yaw', 'pelvis', 'torso', (0, 0, WY), 'z', (-2.6, 2.6), 'RS04', flange=HW.machined('waist_yaw flange', HW.output_cup((0, 0, WY), 'z', 1), BLACK, voxel=0.0004, voxel_hi=0.0003, exposed=True))

# ---------------------------------------------------------------- torso, neck, head

# iteration 11 (user: "upper body looks skinnier than the lower"; measured on Figure's front frame at 1:50 and side
# frame): the chest is right (0.30 x 0.235) but Figure's torso tapers to 0.25 x 0.20 at the hem, a V that makes the
# chest read broad; ours stayed 0.27 x 0.22 down to the hem.
TORSO = [(HEM, 0.000, 0, 0.100, 0.124, 3.0),
         (HEM + 0.045, -0.002, 0, 0.104, 0.130, 3.1),
         (HEM + 0.10, -0.004, 0, 0.111, 0.137, 3.3),
         (HEM + 0.185, 0.000, 0, 0.115, 0.140, 3.4),
         (SZ - 0.035, -0.004, 0, 0.112, 0.148, 3.4),
         (SZ + 0.03, -0.010, 0, 0.100, 0.150, 3.4),
         # iterations 6, 9: one slope from the neck onto the shoulder cap (Figure has no shoulder corner): the
         # torso holds its full width up to the cap's top line, then rolls into the neck
         # iteration 11: a higher, straighter slope (Figure's shoulder line rises ~18 deg from the deltoid to the
         # neck, and the head sits lower between the shoulders); ours dropped steeply near the neck. The top stays
         # at 1.498 m and leans back (iteration 18: the neck now tilts at its base; nodding fully down, 0.45 rad, the
         # chin stays 2 cm in front of the chest)
         # iteration 28 (row scan of the 3/4 frame at 1.46 m: Figure 0.149 m, ours 0.259; side: 0.13, ours 0.168): the top
         # holds its width only to the shoulder motors' tops (1.45 m), then one slope to the collar: no corner above the cap
         # iteration 39: one straight line from the collar to the shoulder, about 21 deg (the old note: Figure ~18; the top at 1.445 m as targets.json): y 0.056 at +0.107,
         # 0.075 at +0.0995, 0.10 at +0.09, 0.14 at +0.0745, 0.19 at +0.055, ending in the yoke below
         (SZ + 0.052, -0.012, 0, 0.090, 0.149, 3.0),
         (SZ + 0.0745, -0.016, 0, 0.071, 0.139, 2.6),
         (SZ + 0.09, -0.018, 0, 0.063, 0.10, 2.3),
         (SZ + 0.0995, -0.020, 0, 0.058, 0.075, 2.2),
         (SZ + 0.107, -0.021, 0, 0.051, 0.056, 2.1)]
# Iteration 39 (user: Figure keeps its actuators further in, ours sit outside the torso; numbers agreed: the RS06 roll spanned y 0.161-0.249 and the torso ended at 0.165). Pitch
# 30 mm in (0.115 to 0.085), roll axis 30 mm in (0.205 to 0.175), the arm's centre line stays at SY through an offset in the roll link. The motor now lies under the skin.
PITCH_Y, ROLL_Y = 0.085, 0.175
# The yoke's outline from the front: its outer edge at each height, on the line above, round into the arm's outside (0.249), its inner edge hidden in the torso; depth from the arm's
# (Figure about 11 cm) to the top. (z from SZ, outer y, inner y, half depth.) RS06 turns at r 0.051: at rest it has the allowance; turned by pitch the skin must give, not tested
YOKE = [(-0.072, 0.212, 0.172, 0.02), (-0.06, 0.232, 0.155, 0.042), (-0.03, 0.249, 0.12, 0.062), (0.0, 0.2485, 0.11, 0.066), (0.02, 0.243, 0.11, 0.0645),
        (0.04, 0.225, 0.11, 0.06), (0.055, 0.19, 0.11, 0.05), (0.0655, 0.16, 0.11, 0.035), (0.0745, 0.14, 0.11, 0.018)]
CHEST_END = 0.1089            # the bar's ends, y, where the shoulder motors' front faces are
CHEST_PLATE = PITCH_Y - 0.01935


def chest_frame():
    """The chest's frame in one piece: a disc on the waist's output, two closed rails up the sides, past the pack, and a round bar across holding both shoulder pitch motors, the neck's motor inside its top. Battery and compute are passengers on pads, out through the back and front."""
    rail = lambda sg, w: HW.s_loft([(1.031, -0.008, sg * 0.040, 0.024 - w, 0.016 - w, 3.0), (1.055, -0.009, sg * 0.058, 0.0235 - w, 0.0145 - w, 3.0), (1.09, -0.010, sg * 0.082, 0.023 - w, 0.0125 - w, 3.0), (1.13, -0.010, sg * 0.0985, 0.022 - w, 0.012 - w, 3.0),
                                    (1.20, -0.012, sg * 0.1045, 0.022 - w, 0.012 - w, 3.0), (1.29, -0.012, sg * 0.1045, 0.024 - w, 0.0125 - w, 3.0), (SZ - 0.045, -0.006, sg * 0.1, 0.03 - w, 0.013 - w, 3.0)])
    disc = HW.s_rev((0, 0, 0.0), 'z', HW.rounded_poly([(0.0, WY + 0.037), (0.058, WY + 0.037, 0.001), (0.058, WY + 0.0405, 0.0015), (0.030, WY + 0.0435, 0.004), (0.0, WY + 0.0435)]))
    bar = HW.s_rev((0, 0, SZ), 'y', HW.rounded_poly([(0.0, -CHEST_END), (HOUSING_OD, -CHEST_END, 0.0008), (HOUSING_OD, CHEST_END, 0.0008), (0.0, CHEST_END)]))
    hollow = HW.s_cyl((0, 0, SZ), 'y', 0.0617, 2 * CHEST_PLATE)
    windows = [HW.s_cyl((xx, sg * 0.031, SZ - 0.006), 'x', 0.025, 0.03, e=0.0) for sg in (-1, 1) for xx in (-0.055, 0.055)] + [HW.s_cyl((0.0, sg * 0.031, SZ - 0.055), 'z', 0.023, 0.03) for sg in (-1, 1)] + [HW.s_cyl((0.0, sg * 0.031, SZ + 0.05), 'z', 0.016, 0.03) for sg in (-1, 1)]
    land = HW.s_cyl((NX, 0, 1.44665), 'z', 0.033, 0.0037)
    pads = [HW.s_box((-0.03, sg * 0.045, SZ - 0.0645), (0.05, 0.03, 0.006), e=0.002) for sg in (-1, 1)]
    seats = []
    for sg in (-1, 1):
        seats += [HW.s_box((-0.02, sg * 0.0745, 1.115), (0.04, 0.022, 0.010), e=0.002), HW.s_box((-0.04, sg * 0.0945, 1.26), (0.03, 0.022, 0.010), e=0.002), HW.s_box((0.045, sg * 0.0925, 1.2297), (0.08, 0.040, 0.0035), e=0.0012)]
    rails = [HW.s_cut(0.0, rail(sg, 0.0), rail(sg, 0.0014)) for sg in (-1, 1)]
    frame = HW.s_cut(0.0, bar, hollow, *windows)
    frame = HW.s_union(0.004, frame, land, *pads)
    frame = HW.s_union(0.006, frame, *rails)
    disc = HW.s_cut(0.0, disc, *[HW.s_cyl((0.043 * math.cos(math.radians(a)), 0.043 * math.sin(math.radians(a)), WY + 0.04), 'z', 0.0105, 0.02) for a in (45, 135, 225, 315)])
    frame = HW.s_union(0.003, frame, disc, *seats)
    return HW.machined('chest frame', frame, BLACK, voxel=0.0009, voxel_hi=0.0007, finish='silver', zones=[dict(c=[0.0, 0.0, SZ], axis='y', r=0.07, t0=CHEST_PLATE - 0.001, t1=0.2, finish='black'), dict(c=[0.0, 0.0, SZ], axis='y', r=0.07, t0=-0.2, t1=-CHEST_PLATE + 0.001, finish='black')])


BATT_TOP = SZ - 0.065           # level with the shoulder pitch actuators' bottoms (they sit outboard of it)
link('torso', 19.96, [
    chest_frame(),
    cut(cyl('chest waist opening', (0, 0, WY + 0.0405), 'z', 0.0266, 0.008), 'chest frame'),
    dict(cut(cyl('L chest shoulder register', (0, PITCH_Y + 0.00503, SZ), 'y', HOUSING_BORE, 0.03895), 'chest frame'), segs=256),
    dict(cut(cyl('R chest shoulder register', (0, -PITCH_Y - 0.00503, SZ), 'y', HOUSING_BORE, 0.03895), 'chest frame'), segs=256),
    cut(cyl('L chest shoulder plate hole', (0, PITCH_Y - 0.02, SZ), 'y', 0.0475, 0.016), 'chest frame'),
    cut(cyl('R chest shoulder plate hole', (0, -PITCH_Y + 0.02, SZ), 'y', 0.0475, 0.016), 'chest frame'),
    cut(cyl('chest neck hole', (NX, 0, 1.452), 'z', 0.0205, 0.03), 'chest frame'),
    cut(cyl('chest neck pocket', (NX, 0, 1.4373), 'z', 0.0292, 0.0148), 'chest frame'),
    # round (iteration 17): turning, it sweeps nothing; 5 mm round the waist actuator inside its wall
    # open below (a closed end crossed the actuator), reaching down into the pelvis opening to 2 mm over the frame
    dict(loft('waist column', [(0.997, 0, 0, 0.0715, 0.0715, 2.0), (HEM - 0.03, 0, 0, 0.0715, 0.0715, 2.0)], GLOSS,
         open=('bottom',))),
    dict(loft('torso rim', [(HEM - 0.035, 0, 0, 0.080, 0.094, 2.6), (HEM - 0.015, 0, 0, 0.095, 0.116, 2.9),
                       (HEM + 0.005, 0, 0, 0.0985, 0.1225, 3.0)], GLOSS, open=('top',))),     # iteration 42: 2 mm polycarbonate, not 3: the check's 3 mm stays as its room
    dict(loft('torso', TORSO, GREY, 'knit'), wall_fill=0.42),     # iteration 42: the pack and frame fill most of the torso; ribs off the frame where they don't, not a former: assumptions.json garment_former
    # Figure 03's pack is 2.3 kWh (figure.ai F.03 battery, 5 h at peak); at 175 Wh/kg for a cell pack with BMS and
    # case, 13.1 kg (it was 12.1 kg, 2 kWh: the hand mass budget's balance)
    # iteration 36: 13.1 kg at 2500 kg/m3 is 5.24 L; the box was 3.3 L (3970 kg/m3, denser than a cell). Two boxes, as the
    # torso narrows to the hem: the upper as it was (3.31 L), a lower 1.93 L sliding forward and narrower, so no cover changes.
    # Mass shared by volume. A single box needs the back of the torso out or a 6 mm wider hem
    # iteration 42: a shaped case, not a box: same volumes, rounded; a passenger on the frame's pads
    span('battery', BATT_TOP - 0.13, BATT_TOP, x=-0.03, sx=0.15, sy=0.17, f=0.016, color=CELL, mass=8.27),
    span('battery lower', BATT_TOP - 0.235, BATT_TOP - 0.13 + 0.001, x=-0.01, sx=0.14, sy=0.131, f=0.016, color=CELL, mass=4.83),
    # compute (a Jetson-class module, carrier and heat sink), iteration 36: 1.6 kg at 1500 kg/m3 is 1.07 L; the box was
    # 0.42 L. In front of the battery, behind the chest: above it the torso's slope and the neck motor leave no room
    # iteration 39: 2.2 cm lower: its top corner met the pitch motors where they moved in (same link, so no check sees it; computed by hand: at x 44 the motor's circle
    # reaches z 1.349, the box now tops out at 1.346)
    rbox('compute', (0.071, 0, 1.288), (0.054, 0.17, 0.116), 0.012, '#2A2C30', mass=1.6, material='electronics'),
    rbox('chest light', (front_x(TORSO, 0, SZ - 0.01) + 0.0006 - 0.002, 0, SZ - 0.01), (0.004, 0.04, 0.006),
         0.0015, ACCENT, 'decal'),
    # our mark on the chest, below the light bar (dark grey on the grey knit); not Figure's logo
    *tri_mark('chest mark', TORSO, SZ - 0.055, 0.034, '#34373B'),
    # the yokes: where the torso's skin runs out over the shoulder motors. Round about the pitch axis, so the roll motor turns inside them
    *[loft(f'{p} shoulder yoke', [(SZ + z, -0.005, k * (o + n) / 2, d, (o - n) / 2, 2.25) for z, o, n, d in YOKE], GREY, 'knit', open=()) | dict(mass=0.14) for p, k in (('L', 1), ('R', -1))],
], 'battery 13.1 kg (2.3 kWh, as Figure 03), compute, shoulder pitch and neck yaw actuators')

# iteration 37 (close-up audit: Figure's neck runs down in one concave line into the shoulders; ours stood on a shelf in a collar). A knit
# flare on the torso, from inside its top up round the neck. Role decal: soft, the rigid checks do not see it; the neck turns inside it.
LINKS['torso']['geoms'].append(loft('neck flare', [
    (SZ + 0.092, -0.019, 0, 0.057, 0.083, 2.3), (SZ + 0.100, -0.020, 0, 0.0555, 0.073, 2.2), (SZ + 0.112, -0.021, 0, 0.0505, 0.060, 2.1),
    (SZ + 0.125, -0.021, 0, 0.0475, 0.0505, 2.0), (SZ + 0.14, -0.020, 0, 0.046, 0.046, 2.0)], GREY, 'knit', open=()))
joint('neck_yaw', 'torso', 'neck_yaw_link', (NX, 0, NZ), 'z', (-1.4, 1.4), 'RS00', out=+1)
link('neck_yaw_link', 0.45, [
    # iteration 18: only a post from the yaw flange up 2.5 mm into the pitch actuator; slim, because the neck's
    # base tilts round it
    dict(cyl('neck pitch post', (NX, 0, (NZ + 0.0295 + NP - 0.0275) / 2), 'z', 0.0115, NP - 0.0275 - NZ - 0.0295, BLACK, 'struct'), exposed=False, fill=0.8),
])
# iteration 18 (neck pitch swept with the covers): with the axis at the jaw the hood's underside crossed the pitch
# actuator and the neck at every angle, and the chin swung into the neck. Now the neck tilts with the head, as
# Figure's knit neck bends: one piece with the hood
joint('neck_pitch', 'neck_yaw_link', 'head', (NX, 0, NP), 'y', (-0.4, 0.45), 'RS00', out=+1)
# slim at the base, flaring into the hood, leaning a little forward, rising behind the head's sloped underside as on
# Figure (iterations 4, 7); its back line runs up into the hood's nape without a notch (iteration 8). Below the
# pitch axis its base is a zone of a sphere about the axes' crossing (round about both, so neither sweeps anything in
# the torso's opening), open below for the post
NECK_R = 0.042
NECK = [(NP + dz, NX, 0, (NECK_R ** 2 - dz ** 2) ** 0.5, (NECK_R ** 2 - dz ** 2) ** 0.5, 2.0)
        for dz in (-0.0235, -0.018, -0.012, -0.006, 0.0)] + \
       [(NP + 0.015, NX, 0, NECK_R, NECK_R, 2.0), (1.528, -0.014, 0, 0.051, 0.045, 2.0),
        (1.55, -0.012, 0, 0.052, 0.047, 2.0), (1.585, -0.006, 0, 0.058, 0.048, 2.0)]
# iteration 27 (row scan of the side frame: Figure's neck 0.118 deep at 1.50 m, ours 0.08): deeper than wide from just
# above the collar, where the tilt's 26 deg still leaves it 1 cm over the torso's top
# Head (figure.ai close-ups and reveal video): a glossy black helmet-like egg, deeper than wide, widest above the
# middle, the chin narrower and set forward, a round crown; a small camera port on the centre line at the brow and
# another low on the chin; the white three-block mark glowing in the middle of the face.
# Iteration 5 (user review of the side view): a helmet. The face stays nearly vertical from chin to brow and the
# crown rounds over the top; only the underside slopes, rising from the chin up to the back of the head, where the
# fabric neck shows behind it. (Iteration 4 tilted the whole egg back ~25 deg and it read as a bean facing up.)
# (z, front x, back x, half width): sections are built from the profile so the outline is what we specify.
# Iteration 7 (user review): the head leans forward, forehead ahead of a tucked chin (iteration 5-6 had the face
# vertical with the chin forward, which read as glancing up).
HEAD_PROFILE = [(CHIN, 0.060, 0.025, 0.016), (CHIN + 0.013, 0.074, 0.000, 0.032),
                (CHIN + 0.028, 0.083, -0.030, 0.046), (CHIN + 0.048, 0.090, -0.058, 0.058),
                (CHIN + 0.078, 0.096, -0.080, 0.067), (CHIN + 0.113, 0.100, -0.090, 0.070),
                (CHIN + 0.143, 0.098, -0.090, 0.069), (CHIN + 0.168, 0.088, -0.083, 0.062),
                (H - 0.019, 0.070, -0.070, 0.050), (H - 0.007, 0.046, -0.050, 0.034),
                (H - 0.0005, 0.018, -0.022, 0.014)]
HEAD = [(z, (f + b) / 2, 0, (f - b) / 2, w, 2.1) for z, f, b, w in HEAD_PROFILE]
# Iteration 6 (close-up of the reveal video): the head is a grey knit HOOD (it carries on from the neck over the
# back and crown) with a glossy black VISOR over the front, its rear edge running from the crown down to the jaw.
# The visor is the head shape 1.5 mm proud, cut by that slanted plane.
VISOR = [(z, cx, cy, a + 0.0015, b + 0.0015, n) for z, cx, cy, a, b, n in HEAD]
# iteration 27 (side frame, measured on the visor's corners: chin tip, rear low corner, rear top): two edges, not one: a
# lower edge from the chin rising 29 deg to the back of the head, and a near-vertical rear edge there, with a crescent
# of hood behind it. One plane from the crown to the jaw (iterations 6, 7) kept black under the back of the head, where
# Figure's grey neck flows up into the hood
VISOR_EDGE = [above((0.07, CHIN + 0.004), (-0.08, CHIN + 0.088)), ahead(-0.078)]
link('head', 1.95, [
    loft('head', HEAD, GREY, open=()),
    loft('visor', VISOR, GLOSS, 'decal', open=(), clip=VISOR_EDGE),
    *camera_port('brow camera', VISOR, CHIN + 0.143, 0.0085),
    *camera_port('chin camera', VISOR, CHIN + 0.025, 0.0075),
    *tri_mark('face mark', VISOR, CHIN + 0.083, 0.03, '#F4F6F8'),
    loft('neck', NECK, GREY, 'knit'),
    # cameras, mics, the face display: behind the visor, on the head frame
    rbox('head electronics', (0.035, 0, CHIN + 0.09), (0.07, 0.08, 0.07), 0.024, '#2A2C30', mass=0.3,
         material='electronics'),
    # frame: from the pitch flange (beside the pitch actuator's +y face) up into the head
    HW.machined('head frame', HW.s_cut(0.0, HW.s_ext('y', 0.0302, 0.0342, HW.lug((NX, NP), 0.016, (0.018, CHIN + 0.088), 0.020), e=0.0015), HW.s_ext('y', 0.0, 0.1, HW.lug((NX + 0.006, NP + 0.03), 0.007, (0.016, CHIN + 0.085), 0.012))), BLACK, voxel=0.0006, finish='silver'),
], 'cameras, mics')

# ---------------------------------------------------------------- left arm (mirrored below)

def finger(name, x, z, r=0.0105):
    """A relaxed finger below the palm at x, reaching z below the wrist axis: a base and a tip curling toward the palm."""
    zk = WZ - 0.16 - 0.6 * (z - 0.16)
    return [cap(name, (x, SY, WZ - 0.16), (x, SY - 0.007, zk), r), cap(name + ' tip', (x, SY - 0.007, zk), (x, SY - 0.022, WZ - z), r)]


def oval_pad(name, zc, y, side, w, h, n=2.3, t=0.0075, dome=0.0):
    """An oval pad standing in the side plane at y (w x h half-sizes, t half-thick) centred at height zc, rim black, a
    glossy centre (3/4 size) 1 mm prouder. side = +1 / -1: which way is out. dome: how much it thins toward its top
    and bottom (0 a slab, 1 a lens), its inner face flat."""
    dz = [h * d / 0.07 for d in (-0.069, -0.064, -0.055, -0.04, -0.02, 0.0, 0.02, 0.04, 0.055, 0.064, 0.069)]
    half = lambda d, s: s * w * max(1 - abs(d / (s * h)) ** n, 0.0) ** (1 / n)
    th = lambda d: t * (1 - dome * (1 - max(1 - (d / h) ** 2, 0.0) ** 0.5))
    pad = loft(name, [(zc + d, 0.0, y + side * (th(d) - t), half(d, 1), th(d), 4.0) for d in dz], BLACK, open=())
    eye = loft(name + ' centre', [(zc + d * 0.75, 0.0, y + side * (th(d) - t + 0.001), half(d * 0.75, 0.75), th(d),
                                   4.0) for d in dz], GLOSS, 'decal', open=())
    return [pad, eye]


# Iteration 38 (review: the pick shape, a round face on a long tail, was drawn, not measured). Figure's black on a grid, across its
# long axis (side frame, both knees; 3/4 far knee): 150 long x 103 wide, widths 63, 84, 94, 102, 103, 98, 89, 78, 62 at 10 to 90 % of its
# length: a plump oval with a short chin. Elbow: 125 x 83, widths 56, 70, 77, 78, 82, 82, 79, 70, 54.
HUB_T = (0.005, 0.03, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.97, 0.995)
KNEE_HUB_HALF = (0.010, 0.021, 0.0315, 0.042, 0.047, 0.051, 0.0515, 0.0495, 0.0445, 0.039, 0.031, 0.018, 0.009)
ELBOW_HUB_HALF = (0.008, 0.017, 0.028, 0.035, 0.0385, 0.0395, 0.041, 0.041, 0.0395, 0.035, 0.027, 0.014, 0.007)


def joint_hub(name, side, zc, yc, top, bot, halves, y_in, y_out):
    """A black hub: the measured outline in the side view (top and bot from the axis), a leaf in the front view, thickest at its middle."""
    mid, b0 = (top + bot) / 2, (y_out - y_in) / 2
    secs = []
    for t, a in zip(HUB_T, halves):
        z = top + t * (bot - top)
        u = abs(z - mid) / ((top - bot) / 2)
        b = b0 * (1 - 0.42 * u ** 5)
        secs.append((zc + z, 0.0, yc + side * (y_in + b), a, b, 4.0))
    return loft(name, secs, GLOSS, open=())


def hub_clearance(name, zc, yc, top, bot, halves, grow):
    """The hub's outline grown by `grow`, through the whole limb: a cover's edge round it."""
    return loft(name, [(zc + top + t * (bot - top), 0.0, yc, a + grow, 0.09, 4.0) for t, a in zip(HUB_T, halves)] +
                [(zc + top + grow, 0.0, yc, 0.004, 0.09, 4.0), (zc + bot - grow, 0.0, yc, 0.004, 0.09, 4.0)], open=())


def dome_pad(name, zc, y, side, w, h, t):
    """A domed oval pad on the side plane y (side = +1 / -1: which way is out), w x h half-sizes, t proud at its centre,
    thinning to nothing at its whole rim (iteration 23: an edge of full thickness left and right, as oval_pad has, reads
    as a puck from a 3/4 view). One glossy piece, as Figure's."""
    ds = [h * d for d in (-0.995, -0.97, -0.9, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 0.9, 0.97, 0.995)]
    k = lambda d: max(1 - (d / h) ** 2, 0.0) ** 0.5
    return [loft(name, [(zc + d, 0.0, y, w * k(d), t * k(d), 2.0) for d in ds], GLOSS, open=(),
                 clip=((0.0, y, 0.0), (0.0, float(side), 0.0)))]


# Elbow cups (iteration 20) gave way to a hinge (iteration 23, user, circling the elbow bent about 30 deg in the walk:
# "You see how the elbow joint is not realistic?"; the upper arm ended in a flat wedge over a slab-topped forearm with
# a mouth between them showing a recessed sleeve, and a round puck stood off the side). Figure's elbow, close-ups of
# the side frame (arm bent ~35 deg) and front frame: a domed black oval, about 8.6 x 12.3 cm, each side of a narrow grey
# forearm neck, the upper arm ending above it in a hem. So, as the knee (joint-covers.md, "hinge"): the forearm's top
# is a DRUM round about the axis (ELBOW_R, 3 mm round the actuator), which maps onto itself at every bend. The upper
# arm's end is a socket 4 mm clear of it behind the top, and in front one plane tangent to the circle the forearm's
# front face rolls on at full flexion (FOLD, 3.5 mm clear at 1.9 rad), so hem and socket are one smooth line. The
# stator is carried from outside (a cheek plate under the pad) rather than from a bracket behind it, the sector the
# forearm's drum now needs: output on the inner face, stator held on the outer, as a fork holds a hinge.
# the arm's yaw seams (iteration 24): spheres on the yaw axes, their tops 3.7 mm under each yaw actuator
ARM_YAW_R, WRIST_YAW_R = 0.08, 0.07
ARM_YAW_C = (0, SY, SZ - 0.1165 - ARM_YAW_R)
WRIST_YAW_C = (0, SY, EZ - 0.1215 - WRIST_YAW_R)
up = lambda line, dz: tuple((x, z + dz) for x, z in line)
ELBOW_R, ELBOW_A = 0.050, 0.054           # drum radius; the upper arm's half depth at the elbow, = the socket's radius
# Iteration 37 (the knee's lesson, same family; close-up audit: pads stuck on, a dark slit behind the hem). As the knee: hubs on the
# forearm round the RS06's stator, the upper arm's fork holds the output, a knit panel and hem instead of the fold's V. Figure's
# front frame: about 89 mm overall; the RS06 (50.5 thick) with flange and walls forces a 65 mm neck, so 107 overall here.
ELBOW_W = 0.0325                          # the neck's half width, between the hubs
ELBOW_HUB_IN, ELBOW_HUB_OUT = 0.0375, 0.0535
ELBOW_CHEEK_Y = 0.035
ELBOW_HEM = 0.063
ELBOW_HUB_TOP, ELBOW_HUB_BOT = 0.059, -0.066
FOLD = ((-0.1, EZ + 0.0544 - 0.03425), (0.1, EZ + 0.0544 + 0.03425))


def elbow_ray(deg, r=0.1):
    """Side-view point (x, z) at angle deg about the elbow axis."""
    import math
    return r * math.cos(math.radians(deg)), EZ + r * math.sin(math.radians(deg))


# exposed (iteration 17): the actuator's face shows dark in the gap round the cap, in the torso's side opening
joint('L_shoulder_pitch', 'torso', 'L_shoulder_pitch_link', (0, PITCH_Y, SZ), 'y', (-3.0, 1.0), 'RS04', out=+1)
link('L_shoulder_pitch_link', 0.99, [
    # iteration 39: no cap: from the pitch flange (y 0.113-0.119) a block to the side of the RS06, which now lies under the yoke; and a post on its front face, for the roll
    # fork's second bearing
    rbox('L shoulder bracket', (0, 0.1245, SZ), (0.036, 0.023, 0.036), 0.004),
    dict(cyl('L shoulder roll post', (0.0325, ROLL_Y, SZ), 'x', 0.012, 0.015, BLACK)),
])
# shoulder roll: RS06 (88 mm) rather than RS03 (106 mm) so the shoulder cap can be small (user, 2026-10-31: smaller
# actuators at the cost of holding capacity). An arm held straight out sideways is a short move, not a pose to hold.
# iteration 18: 150 deg up (2.8 swept the arm's outer top into the cap's top at full height; AgiBot A3 stops at 150,
# Asimov 1 at 90); output on the back face, so the bracket runs behind the cap instead of through it
joint('L_shoulder_roll', 'L_shoulder_pitch_link', 'L_shoulder_roll_link', (0, ROLL_Y, SZ), 'x', (-0.05, 2.6), 'RS06',
      out=-1)
# Arms (iteration 11): oval like Figure's, deeper than wide (side frame: upper arm 0.11 deep, elbow disc 0.09;
# front frame: 0.085 wide). Round 0.096 arms read thin from the side and wide from the front.
link('L_shoulder_roll_link', 1.0, [
    # iteration 39: the sleeve. Figure's arm in the side frame about 11 cm deep; this one 11, then it leans in under the yoke toward the roll axis, its head a dome that stays there as
    # the arm rolls (the skin between them is what stretches). The crease where sleeve and yoke cross is the raglan seam's line.
    loft('L upper arm', [(SZ - 0.15, 0, SY, 0.053, 0.044, 2.2), (SZ - 0.114, 0, SY, 0.053, 0.044, 2.2), (SZ - 0.075, 0, SY, 0.055, 0.045, 2.2),
                         (SZ - 0.04, -0.001, SY - 0.001, 0.0555, 0.046, 2.2), (SZ - 0.015, -0.001, SY - 0.006, 0.053, 0.047, 2.2),
                         (SZ + 0.005, 0, SY - 0.01, 0.04, 0.036, 2.1), (SZ + 0.015, 0, SY - 0.012, 0.015, 0.015, 2.0)],
         GREY, 'knit', open=()),
    # the fork round the RS06, a plate each side and a bridge under it: rear on the output flange, front on a bearing round the post. The offset from the roll axis to the arm's line
    # is in these plates, 75 wide x 95 tall x 6 (a hand estimate is still to do). They carry the yaw motor, which they run onto
    dict(cyl('L shoulder roll axle', (-0.0325, ROLL_Y, SZ), 'x', 0.011, 0.011, BLACK)),
    rbox('L shoulder roll plate back', (-0.034, 0.19, SZ - 0.011), (0.006, 0.064, 0.058), 0.002, BLACK),
    rbox('L shoulder roll plate front', (0.034, 0.19, SZ - 0.011), (0.006, 0.064, 0.058), 0.002, BLACK),
    rbox('L shoulder roll plate back low', (-0.034, 0.197, SZ - 0.0575), (0.006, 0.05, 0.035), 0.002, BLACK),
    rbox('L shoulder roll plate front low', (0.034, 0.197, SZ - 0.0575), (0.006, 0.05, 0.035), 0.002, BLACK),
    cut(cyl('L shoulder roll bearing bore', (0.034, ROLL_Y, SZ), 'x', 0.0135, 0.01), 'L shoulder roll plate front'),
    rbox('L shoulder roll bridge', (0, 0.2, SZ - 0.055), (0.074, 0.05, 0.01), 0.002, BLACK),
])
joint('L_shoulder_yaw', 'L_shoulder_roll_link', 'L_shoulder_yaw_link', (0, SY, SZ - 0.12), 'z', (-2.6, 2.6),
      'RS02', act_c=(0, SY, SZ - 0.09), out=-1, exposed='flange')   # its flange shows in the seam
# iteration 39 (garment): the skin narrows smoothly to the hub's inner face and back; no socket, drum, hem or fold plane
ELBOW_NECK = 0.036                        # iteration 42: 1.5 mm short of the hub's inner face, a shadow line
ARM_LOW = [(EZ - 0.02, 0, SY, 0.052, ELBOW_NECK, 6.0), (EZ + 0.04, 0, SY, 0.0525, ELBOW_NECK, 5.0), (EZ + 0.059, 0, SY, 0.053, 0.039, 4.0),
           (EZ + 0.072, 0, SY, 0.0535, 0.0415, 3.0), (EZ + 0.085, 0, SY, 0.053, 0.043, 2.2), (SZ - 0.090, 0, SY, 0.053, 0.044, 2.2)]
link('L_shoulder_yaw_link', 1.05, [
    loft('L upper arm lower', ARM_LOW, GREY, 'knit', open=(), clip=above((-1.0, EZ + 0.003), (1.0, EZ + 0.003))),
    cut(loft('L upper arm fold', [(EZ - 0.06, 0, SY, 0.1, 0.06, 8.0), (EZ + 0.12, 0, SY, 0.1, 0.06, 8.0)],
             clip=[below(*FOLD), above(elbow_ray(271.4), elbow_ray(91.4))]), 'L upper arm lower'),
    HW.seam_groove('L elbow seam groove', ARM_LOW, EZ + 0.0045, 0.0015, 0.0025, ['L upper arm lower', 'L elbow gaiter']),
    dict(loft('L elbow seam', [(EZ + 0.0046, 0, SY, 0.052 - 0.0026, ELBOW_NECK - 0.0026, 6.0), (EZ + 0.0059, 0, SY, 0.052 - 0.0026, ELBOW_NECK - 0.0026, 6.0)], '#2A2C30', 'decal', open=()), material='pa12', mass=0.002),
    loft('L elbow gaiter', ARM_LOW, GREY, 'knit', open=(), clip=[below(*up(FOLD, 0.003)), above(*up((elbow_ray(271.4), elbow_ray(91.4)), -0.003)), above((-1.0, EZ + 0.003), (1.0, EZ + 0.003))]),
    span('L upper arm frame', EZ + 0.066, SZ - 0.115, y=SY, sx=0.03, sy=0.03, exposed=True),
    # the output: a bridge from the frame to a cheek each side, in the band between neck and hub (3 mm 7075, 4.6 cm wide round the bore: at the
    # actuator's peak 36 N m about 127 MPa, a hand estimate, scaling the 4 mm figure; 7075-T6 yields near 500). The inner is bolted to
    # the axle on the motor's output; the outer runs on a bearing round the post
    rbox('L elbow bridge', (0, SY, EZ + 0.063), (0.024, 2 * (ELBOW_CHEEK_Y + 0.0015), 0.010), 0.002, BLACK, exposed=True),
    rbox('L elbow cheek in', (0, SY - ELBOW_CHEEK_Y, EZ + 0.0425), (0.046, 0.003, 0.045), 0.002, GREY, exposed=True),
    rbox('L elbow cheek out', (0, SY + ELBOW_CHEEK_Y, EZ + 0.0425), (0.046, 0.003, 0.045), 0.002, GREY, exposed=True),
    dict(cyl('L elbow cheek disc in', (0, SY - ELBOW_CHEEK_Y, EZ), 'y', 0.040, 0.003, BLACK), exposed=True),
    dict(cyl('L elbow cheek disc out', (0, SY + ELBOW_CHEEK_Y, EZ), 'y', 0.040, 0.003, BLACK), exposed=True),
    cut(cyl('L elbow cheek bearing bore', (0, SY + ELBOW_CHEEK_Y, EZ), 'y', 0.0195, 0.01), 'L elbow cheek disc out'),
    dict(cyl('L elbow axle', (0, SY - 0.0313, EZ), 'y', 0.017, 0.0048, BLACK), exposed=True),
    cut(cyl('L elbow cheek idler bore', (0, SY - ELBOW_CHEEK_Y, EZ), 'y', 0.0115, 0.01), 'L elbow cheek disc in'),
])
# Elbow: RS06 (88 mm), the size of Figure's elbow disc. User decision 2026-10-31: smaller elbows at the cost of
# holding capacity (RS04 120 mm carried 20 kg short / 10 kg continuous; RS06 carries about 10 kg short / 4 kg
# continuous with forearms level). The 1.6 kg saved went into the battery. Output on the inner face (iteration 4):
# from outside the elbow is a clean black disc.
# flexion 129 deg (iteration 11: the deeper oval forearm meets the upper arm at 135); 109 deg (iteration 17: with
# the covers swept too, the forearm meets the upper arm's frame in the elbow zone at 113)
joint('L_elbow', 'L_shoulder_yaw_link', 'L_elbow_link', (0, SY, EZ), 'y', (-1.9, 0.0), 'RS06', act_c=(0, SY + 0.00075, EZ), out=-1,
      stator_on_child=True)
link('L_elbow_link', 0.85, [
    # iteration 39 (garment): one surface to the hub's inner face; a disc round the axis, so the outside of the bend stays round. The skin overlaps the lower forearm's, no seam line
    loft('L forearm', [(EZ - 0.16, 0, SY, 0.0445, 0.0415, 2.2), (EZ - 0.119, 0, SY, 0.045, 0.0425, 2.2), (EZ - 0.072, 0, SY, 0.048, 0.0428, 2.2),
                       (EZ - 0.062, 0, SY, 0.048, 0.0405, 3.0), (EZ - 0.05, 0, SY, 0.0485, 0.0385, 4.0), (EZ - 0.04, 0, SY, 0.0485, ELBOW_NECK, 5.0),
                       (EZ - 0.022, 0, SY, 0.049, ELBOW_NECK, 6.0), (EZ - 0.008, 0, SY, 0.05, ELBOW_NECK, 12.0), (EZ, 0, SY, 0.051, ELBOW_NECK, 16.0)], GREY, 'knit', open=()),
    cyl('L forearm disc', (0, SY, EZ), 'y', 0.0515, 2 * ELBOW_NECK, GREY, 'knit'),
    # the hubs: on the outside a cap on the stator's back, by the post; inside the idler, carried from the frame under the motor
    dict(joint_hub('L elbow hub in', -1, EZ, SY, ELBOW_HUB_TOP, ELBOW_HUB_BOT, ELBOW_HUB_HALF, ELBOW_HUB_IN, ELBOW_HUB_OUT), material='al7075'),
    dict(joint_hub('L elbow hub out', +1, EZ, SY, ELBOW_HUB_TOP, ELBOW_HUB_BOT, ELBOW_HUB_HALF, ELBOW_HUB_IN, ELBOW_HUB_OUT), material='al7075'),
    dict(cyl('L elbow idler stub', (0, SY - 0.0375, EZ), 'y', 0.010, 0.007, BLACK), exposed=True),
    rbox('L elbow idler arm', (0, SY - 0.0395, EZ - 0.028), (0.012, 0.004, 0.05), 0.001, BLACK, exposed=True),
    *[g for sd, y in (('in', -1), ('out', +1)) for g in (
        cyl(f'L elbow hub ring {sd}', (0, SY + y * ELBOW_HUB_OUT, EZ), 'y', 0.034, 0.002, '#2A2C30', 'decal'),
        cyl(f'L elbow hub face {sd}', (0, SY + y * (ELBOW_HUB_OUT + 0.0003), EZ), 'y', 0.032, 0.002, GLOSS, 'decal'))],
    dict(cyl('L elbow post', (0, SY + 0.03325, EZ), 'y', 0.017, 0.0145, BLACK), exposed=True),
    rbox('L elbow under motor', (0, SY - 0.005, EZ - 0.052), (0.03, 0.071, 0.008), 0.002, BLACK, exposed=True),
    # side plate from the stator's back down onto the wrist yaw actuator
    rbox('L forearm frame', (0, SY + 0.0285, EZ - 0.035), (0.04, 0.005, 0.10), 0.003, BLACK, exposed=True),
])
joint('L_wrist_yaw', 'L_elbow_link', 'L_wrist_yaw_link', (0, SY, EZ - 0.125), 'z', (-2.6, 2.6), 'RS02',
      act_c=(0, SY, EZ - 0.095), out=-1, exposed='flange')   # its flange shows in the seam
link('L_wrist_yaw_link', 0.6, [
    loft('L forearm lower', [(WZ + 0.025, 0, SY, 0.034, 0.032, 2.2), (WZ + 0.045, 0, SY, 0.037, 0.034, 2.2),
                             (EZ - 0.125, 0, SY, 0.044, 0.041, 2.2), (EZ - 0.112, 0, SY, 0.0445, 0.042, 2.2)],
         GREY, 'knit', open=('bottom',)),
    # from just over the wrist pitch actuator up into the wrist yaw flange (clear of the yaw actuator)
    span('L forearm lower frame', WZ + 0.027, EZ - 0.1185, y=SY, sx=0.03, sy=0.024, exposed=True),   # 24 wide: clear of the wrist housing at 0.9 rad
    # Figure's black wrist cuff, over the wrist pitch actuator (iteration 4)
    # iteration 17: wider (clear of the wrist cheek) and above the pitch axis (clear of the wrist roll actuator as
    # the hand pitches 0.9 rad); iteration 29: squarer, as Figure's block, its inner corners clear of the housing's as
    # the housing's rim rises into it (only the pitch moves one against the other: the yaw turns both)
    # iteration 35 (user: bulkier than Figure's slim wrist; about 65 mm on the frames): 78 x 78, was 92 x 86. Along x nothing
    # sweeps it, so it only needs the forearm's end inside it. Across, the housing's rim needs 3 mm as it rises
    loft('L wrist cuff', [(WZ + 0.008, 0, SY, 0.039, 0.039, 5.0), (WZ + 0.040, 0, SY, 0.039, 0.039, 5.0)], BLACK,
         open=('bottom',)),
])
joint('L_wrist_pitch', 'L_wrist_yaw_link', 'L_wrist_pitch_link', (0, SY, WZ), 'x', (-0.9, 0.9), 'RS00',
      out=+1, exposed=True)
# iteration 29 (same family as the elbow: bent, the wrist showed a bare actuator drum, a cheek plate and a hand on a
# post, with daylight through them; Figure's reads as a block, a round black wrist and the palm): one black housing on the
# pitch link round both actuators. Its top is open and low (+9 mm), rising inside the cuff as the pitch turns it, clear
# of the frame that carries the pitch stator; its lower end is round about the roll axis (the palm's top 3 mm clear),
# and on the hand's side a round opening 3 mm round the roll actuator, where the output leaves for the hand's bracket
# iteration 35: the maker's STEP (2026-10-30): RS00 is round d57, 52.1 long with its hub, as drawn near enough (d57 x 51). So a
# 3 mm wall allows 63 mm, and the housing is 65 x 65 (it was 68 x 66), its section squarer (n 10) to reach the motors' end
# faces. Below the pitch motor its sides draw in, so the points that would swing up into the cuff at 0.9 rad stay under it.
WRIST_HOUSING_R = 0.0325                  # the lower end, round about the roll axis (the palm stays where it is, now 5.5 mm off)
WRIST_HOUSING = [(WZ + 0.009, 0, SY, 0.0325, 0.0325, 10.0), (WZ - 0.012, 0, SY, 0.0325, 0.0325, 10.0),
                 (WZ - 0.02, 0, SY + 0.0003, 0.0325, 0.0300, 11.0), (WZ - 0.025, 0, SY + 0.0008, 0.0325, 0.0285, 12.0),
                 (WZ - 0.045, 0, SY + 0.0015, 0.0325, 0.0293, 12.0),
                 (WZ - 0.06, 0, SY + 0.0015, 0.0325, 0.0293, 12.0)] + \
                [(WZ - 0.06 - d, 0, SY + 0.0015, (WRIST_HOUSING_R ** 2 - d ** 2) ** 0.5, 0.0293, 12.0)
                 for d in (0.01, 0.02, 0.026, 0.03, 0.032)]
link('L_wrist_pitch_link', 0.45, [
    rbox('L wrist cheek', (0.0285, SY, WZ - 0.0275), (0.006, 0.04, 0.055), 0.002, BLACK, exposed=True),
    loft('L wrist housing', WRIST_HOUSING, BLACK, open=('top',)),
    cut(cyl('L wrist housing opening', (0, SY - 0.025, WZ - 0.06), 'y', 0.0315, 0.006), 'L wrist housing'),
])
joint('L_wrist_roll', 'L_wrist_pitch_link', 'L_hand', (0, SY, WZ - 0.06), 'y', (-0.6, 0.6), 'RS00', out=-1,
      exposed=True)
link('L_hand', 1.0, [
    # iteration 29: its top 38 mm from the roll axis, 3 mm clear of the wrist housing's round end (no change to the axes)
    # the hand's motors, tendons and palm: about 0.85 kg in the palm, the rest in the fingers' covers and bracket
    rbox('L palm', (0, SY - 0.005, WZ - 0.1315), (0.092, 0.06, 0.067), 0.018, BLACK, 'shell', mass=0.85,
         material='hand'),
    # from the wrist roll flange down into the palm, outside the roll actuator's length
    rbox('L hand bracket', (0, SY - 0.030, WZ - 0.085), (0.02, 0.006, 0.042), 0.002, BLACK, exposed=True),
    # iteration 30 (Figure's hands hang relaxed: fingers curling toward the palm, the thumb in front; straight capsules
    # read as a paddle): two capsules a finger, the tip turning in. No joints: the curl stays clear of the thigh by 2 mm
    *finger('L index', 0.033, 0.219), *finger('L middle', 0.011, 0.229), *finger('L ring', -0.011, 0.224),
    *finger('L little', -0.033, 0.209, r=0.0098),
    cap('L thumb', (0.042, SY - 0.012, WZ - 0.105), (0.056, SY - 0.022, WZ - 0.14), 0.0115),
    cap('L thumb tip', (0.056, SY - 0.022, WZ - 0.14), (0.052, SY - 0.026, WZ - 0.172), 0.0105),
], 'dexterous hand placeholder (about 1 kg)')

# The knee (iteration 19, user: "real knees with actuators won't look like this"), a hinge; iteration 26: bent 35 deg
# in the walk, as the elbow was (user's elbow screenshot, same family), it still showed a flat-topped shank, a black
# drum in the mouth above it, and pads on posts with daylight behind them. As the elbow now: the shank's top is a DRUM
# (grey, the shank's own depth) round about the axis; the thigh ends in a socket 4 mm clear of it, running into one plane
# behind, tangent to the circle the calf's back face rolls on at full flexion (4 mm clear at 2.15 rad); the fork moves
# outside the drum: cheeks at both ends of the axis, each with a stub axle through a round hole in the drum's end wall
# to the knuckle inside (a clevis, as Unitree's and Asimov's knees); the pads are domes on the thigh's flat sides.
# Figure's side frame shows just this: oval domes on a narrow neck under a hem.
# Iteration 33 (DECISIONS 4: the spec's 2:1 linkage cannot work, a 4-bar over this range averages about 1.05:1; X2 at 120
# N m and G1 drive the knee directly): an RS04 on the axis, inside the drum, its stator on the outer cheek, its output
# on a plate of the shank that rides on the inner stub axle. So the drum is 6 mm larger in radius (3 mm round the d120
# motor), the shank's top that much deeper, the thigh's end and the fold plane with it. It FAILS the stair check on
# purpose (134 N m static against 120): the v5d run decides, and the check stands.
# Iteration 36 (user, circling both, beside Figure: "anything black is just the actuator around which the white body parts
# move", and "no humanoid I saw has such an odd gap at knees"). The pads were 12 mm domes stuck on the sides, with the RS04
# hidden in a 123 mm grey drum; the V behind the knee was the fold plane cut from a hard thigh. Measured on Figure's front frame
# (2 mm per px across the 140 mm thigh): black about 28 mm each side of a 60 mm grey neck, 116 overall; ours were 12, 123, 147.
# Side frame (3.2 mm per px): the black about 100 wide x 138 tall, a round face with a ring seam and a tail that follows
# the SHANK at any bend (compare its two knees). So now:
# - the black is a hub riding on the shank: a round face r 50 about the axis running into a tail down the shank. The RS04's
#   stator is on the shank too, carried by a plate on the motor's back; a post on the same back carries the outer hub (an
#   aluminium cap, a heat path the motor needs). The inner hub holds the idler. The RS04 stays inside the grey, wider than the
#   round face (d120 against d100): so the hub is on the motor, not the motor itself, and that is the one difference left;
# - the neck between the hubs is the motor's 55.7 mm plus plate, flange and walls: 72 mm;
# - the thigh holds the output: a cheek each side, in the 5 mm between neck and hub; the inner one is bolted to the axle on the
#   motor's output, and the axle runs on into the inner hub on the idler bearing; the outer one runs on a bearing round the
#   post. So the joint is held on both sides, the fork as before, with stator and output swapped;
# - thigh and shank are full width only above and below the hubs: the thigh's sides end 10 mm over the hub's top, the shank's
#   sides start under the tail, as both swing past the hub's band at full flexion;
# - the V is closed by a knit panel on the thigh, soft, as Figure's covers are textile, running the thigh's back line straight
#   down to the drum; and a knit sleeve hem carries the thigh's sides on down round the hub's top to 2 cm over the axis, as on
#   Figure, where its grey edge wraps the black. Both are role 'decal': the rigid checks do not see them, they are not
#   structure, the shank folds into the panel and the tail pushes the hem aside past about 1.4 rad. What they do in a real
#   knee (bunching, wear, snagging) is untested.
KNEE_R, KNEE_SOCKET = 0.066, 0.0715
KNEE_W = 0.036                            # the neck's half width, between the hubs
KNEE_WIDE = 0.056                         # the shank's half width below the hub
KNEE_HEM = 0.07                           # above the axis, where the thigh's full width ends
HUB_IN, HUB_OUT = 0.043, 0.0615           # the hub's faces, from the leg's centre plane (iteration 41: inner face 41 to 43, the fork's eyes and CRBS 508s need the room; overall width kept)
HUB_TOP, HUB_BOT = 0.066, -0.084           # the hub, from the axis (fig_hub.py: 150 long; the front frame's black centred about 1 cm low)
import math as _m                                                      # noqa: E402
KNEE_TAB = (_m.cos(_m.radians(75 + MOTOR_CLOCK['L_knee'])), _m.sin(_m.radians(75 + MOTOR_CLOCK['L_knee'])))     # the RS04's cable tab, in the side view (x, z)
KNEE_RIM_R, KNEE_RIM_LOBE = 0.0631, 0.0043
KNEE_RIM_ANGLES = (22.5, 67.5, 112.5, 157.5, 202.5, 337.5)   # M3 round the rim; none under the motor, where the barrel runs into the body
KNEE_CHEEK_SCREWS = ((-0.025, 0.079), (0.025, 0.079), (-0.019, 0.107), (0.019, 0.107))
KNEE_WALL = 0.0017                        # the closed section's wall (1.4 past the RS03)
KNEE_XS = -0.0505                         # where the back panel splits off, x
KNEE_PANEL_SCREWS = ((1, 0.285), (1, 0.325), (1, 0.355), (-1, 0.285), (-1, 0.325), (-1, 0.355))
KNEE_FOLD_RAD, KNEE_FOLD_D = 2.15, 0.0699   # the fold plane: the calf's back face at this flexion, this far from the axis


def knee_fold(rad=KNEE_FOLD_RAD, d=KNEE_FOLD_D):
    """The fold plane's line in the side view, (x, z) rear point then front point."""
    import math
    nx, nz, tx, tz = -math.cos(rad), math.sin(rad), math.sin(rad), math.cos(rad)
    return ((d * nx - 0.12 * tx, KZ + d * nz - 0.12 * tz), (d * nx + 0.12 * tx, KZ + d * nz + 0.12 * tz))


KNEE_FOLD = knee_fold()
KNEE_FOLD_DEG = 180 - KNEE_FOLD_RAD * 57.29578   # the plane's normal, in the side view


def knee_hub(name, side):
    return joint_hub(name, side, KZ, HY, HUB_TOP, HUB_BOT, KNEE_HUB_HALF, HUB_IN, HUB_OUT)


def knee_ray(deg, r=0.1):
    """Side-view point (x, z) at angle deg about the knee axis."""
    import math
    return r * math.cos(math.radians(deg)), KZ + r * math.sin(math.radians(deg))


# ---------------------------------------------------------------- left leg (mirrored below)

# Hip (iteration 21, user, of the thigh hanging from the hip seen from below: "that connection looks so weak for
# such a big robot"; it hung on a thin plate and a grey actuator rim in a 2.5 cm gap under the lobe). The thigh now
# meets the black hip in a 3 mm seam, as Figure's thighs tuck in under its black hips: the lobe rides on the hip
# ROLL link (it rolls with the thigh), so only the hip yaw turns the thigh under it, and a seam square to the yaw
# axis never sweeps. The roll parts and the yaw actuator sit inside the lobe and the thigh. The roll moves the lobe
# against the pelvis instead: its inner face is flat (the pitch and the inward roll keep it 3 mm clear of the
# pelvis block), bevelled above the axis where the outward roll tips its top toward the block, and hollow round
# what turns inside it (the pitch actuator, its flange and the bracket to the roll actuator, swept through the roll
# range). The joint order stays pitch, roll, yaw (Asimov 1, Unitree G1 and H2): yaw first (H1) puts a motor above
# each hip, where our torso's hem turns, and Figure's black hips (0.86-1.0 m, under the waist column) have no room
# for one either.
HIP_SEAM = HZ - 0.073                     # the lobe's underside on the yaw axis; the thigh's top is 3 mm under it
# iteration 22 (user, of white lines in the side view): a flat seam lined up with the other hip's showed the
# background through both, and so would any flat seam seen level. The seam is a dome round the yaw axis instead
# (the thigh's top inside a sphere, the lobe's underside 3 mm outside it), so it still never sweeps as the thigh
# yaws, but no straight line runs along it. It drops 9 mm to the sides and 3.6 cm to the back of the thigh, as
# Figure's thigh top falls toward the back under its black hip.
# iteration 25 (side frames against the replica's: black down to 0.82 m at the back, a box behind the thigh, where
# Figure's grey thigh rises to its black hip, and it was the dome that dropped it): a shallow BOWL about the yaw axis
# instead: the thigh's top dished to a sphere centred above, the lobe's underside 3 mm inside the same sphere. Round about
# the yaw axis as the dome was, front 0.852 m, back 0.863 m (it cannot rise further: the roll actuator's underside is
# 0.87 m, 8 mm above). Radius 0.7: across the thigh's 14 cm width the seam rises 3.5 mm against its 3 mm gap, so no
# level line runs through it (lesson: a flat seam shows light in a level side view), checked on green.
# Known: a hairline of light remains over the seam's rear 5 cm, in an exact level side view with both hips aligned, where the
# thigh's chord is too short for the bowl to close it. A lip round about the yaw axis blocked it but stood out as a
# black tab behind the thigh, worse than the hairline (iteration 32, tried and removed).
HIP_DOME_R = 0.70
HIP_DOME_C = (0, HY, HIP_SEAM - 0.007 + 0.003 + HIP_DOME_R + 0.0)
HIP_LOBE_CLIP = [((0, 0.088, 0), (0, 1, 0))]
# the bevel: over the pelvis block's length the outward roll (0.6 rad) tips the lobe's top inward toward the block, so
# above the axis the lobe stays outboard of a slanted plane (3 mm clear at the full roll); behind the block it stays
# full over the roll actuator, so from the front the lobe still reads round (a plane over its whole length left a fin)
HIP_BEVEL = [(0.0, HZ + 0.004), (0.088, HZ + 0.004), (0.088 + 0.0685, HZ + 0.104), (0.0, HZ + 0.104)]
HIP_ROLL = (-0.35, 0.6)                   # outward 0.6 rad (iteration 17)


def hull2d(pts):
    """Convex hull of 2-D points, counter-clockwise (monotone chain)."""
    pts = sorted(set(pts))

    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) -
                                   (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0])) <= 0:
                h.pop()
            h.append(p)
        return h
    return half(pts)[:-1] + half(pts[::-1])[:-1]


def roll_sweep(name, rects, x0, x1, gap=0.003):
    """Cut: what the hip roll turns past the lobe. rects: (dy0, dy1, dz0, dz1) about the roll axis, fixed to the pitch
    link or the pelvis; seen from the lobe, they turn through the roll range. One convex prism from x0 to x1, gap
    clear."""
    import math
    pts = []
    for y0, y1, z0, z1 in rects:
        for k in range(21):
            t = -(HIP_ROLL[0] + (HIP_ROLL[1] - HIP_ROLL[0]) * k / 20)    # the lobe turns +t: they turn -t in it
            c, s = math.cos(t), math.sin(t)
            for y, z in ((y0, z0), (y0, z1), (y1, z0), (y1, z1)):
                for a in range(8):
                    yy, zz = y + gap * math.cos(a * math.pi / 4), z + gap * math.sin(a * math.pi / 4)
                    pts.append((round(HY + yy * c - zz * s, 5), round(HZ + yy * s + zz * c, 5)))
    return prism(name, x0, x1 - x0, hull2d(pts), BLACK, 'cut')


def pitch_sweep(name):
    """Cuts: the hip pitch actuator (round about y, on the pelvis) and its flange, swept by the roll; in slices along
    x, each as tall as the actuator is at the slice's inner edge (a cylinder about y is lower away from x = 0)."""
    # iteration 42: the pelvis's barrel and ring round the motor, the eye of the carrier beyond them
    r, h, rf = HOUSING_OD, ACT['RS04']['h'], 0.0385
    y0, y1 = PITCH_ACT_Y - h / 2 - HY, PITCH_ACT_Y + 0.0285 - HY
    f0 = y1 - 0.001
    out, edges = [], [0.0, 0.015, 0.025, 0.035, 0.045, 0.055, r + 0.003]
    for i in range(len(edges) - 1):
        xa, xb = edges[i], edges[i + 1]
        a, af = (max(r ** 2 - xa ** 2, 0) ** 0.5), (max(rf ** 2 - xa ** 2, 0) ** 0.5)
        rects = [(y0, y1, -a, a)] + ([(f0, f0 + 0.008, -af, af)] if af else [])
        out += [dict(cut(roll_sweep(f'{name} {i}f', rects, xa, xb), 'L hip lobe'), internal=True),
                dict(cut(roll_sweep(f'{name} {i}b', rects, -xb, -xa), 'L hip lobe'), internal=True)]
    return out


# limits from the sweep: flexion stops at 97 deg (hip yaw actuator meets the pelvis block), extension at 0.30 rad
# (the rounded hip lobe meets the waist column; walking needs ~0.2): a little range traded for Figure's hip shape
# iteration 17: extension 0.24 (the hip roll actuator met the pelvis block at 0.257 once the lobe stopped hiding it);
# the pitch actuator shows black between the pelvis block and the lobe, as on Figure
# iteration 39 (garment): from the full thigh the skin narrows smoothly to the hub's inner face (82 wide: motor stack, walls, cheeks; Figure's about 66). No socket, fold plane or hem
KNEE_NECK = 0.0415                        # iteration 41: the machined stack. Iteration 42 (user: gaps between sliding parts don't read): 1.5 mm short of the hub, a shadow line round it


# Iteration 41 (user, 2026-11-05: "our connections are flat square ones while industry goes with strong cylindrical style", "my main goal is the looks for the skeleton structure as well"). The knee is the template for
# the other joints (reports/joint_template.md). Our own shapes round RobStride's real faces, same principles as Figure 01 (round, closed, held both sides), no part traced. All y from the leg's centre, left leg, motor centred on it:
#   y -42 inner cheek -37 | ring -36.6 | outer race 34.6..26.5 (CRBS 508, centre -30.5) in the barrel's collar, inner race on the cup | -24.85 hub face | -24.35 stator's front face, 10 M4 through the barrel's plate -29.6 |
#   motor | +27.85 its rear cover | rear cap 28.45..31.4 with its post | outer cheek 31.65..42.6 with the second CRBS 508, centre +37.4 | hub +43 on, 61.5 end.
# Load goes foot, shank body, barrel, two bearings, fork, tube: not through the motor's case. The motor hangs on its front face for torque, and its output on the cup.
KNEE_BORE = 0.06005                       # the motor's d120, 0.05 over on the radius: a registered fit, also the heat path (with paste)
KNEE_OD = 0.0635                          # 127 mm: 3.45 mm wall. Load wants far less (reports/strength.md: SF over 10); kept for the bore's roundness and the heat path, lobed at the rim's M3
KNEE_PLATE, KNEE_COLLAR_END = -0.0296, -0.0366
KNEE_REAR = 0.02845                       # the barrel's open end, and the rear cap's seat
KNEE_IN_BEARING, KNEE_OUT_BEARING = -0.0305, 0.0374
KNEE_RACE_R, KNEE_RACE_OD = 0.025, 0.033  # CRBS 508: d50, D66, 8 wide
SLOT_IN, SLOT_OUT = -0.037, 0.03165       # the fork's slot, y from the leg's centre
ARCH = 0.0677                             # the slot's roof about the axis: the barrel and its boss turn under it
TUBE_TOP_C = (0.0, KZ + 0.15)             # where the fork has become the round tube
TUBE_R, TUBE_WALL = 0.033, 0.002          # d66 x 2: the load leaves a lot in hand (reports/strength.md); what limits it is dents and the process, not stress
THIGH_TOP = 0.7822                        # the flange on the hip yaw's hub ends here


def knee_y(c=0.0):
    return HY + c


def knee_fork():
    """The thigh's end: fork (inner cheek and tube in one piece) and the bolted outer cheek. Same outside, split on the slot's far face."""
    front = [(KZ - 0.05, 0.0003, HY + 0.0003, 0.056, 0.0423, 7.0), (KZ + 0.066, 0.0003, HY + 0.0003, 0.056, 0.0423, 7.0), (KZ + 0.082, 0.0, HY + 0.0002, 0.052, 0.0412, 6.0), (KZ + 0.098, 0.0, HY + 0.0001, 0.046, 0.0385, 5.0),
             (KZ + 0.114, 0.0, HY, 0.0405, 0.0358, 3.8), (KZ + 0.132, 0.0, HY, 0.0355, 0.0338, 2.6), (KZ + 0.15, 0.0, HY, TUBE_R, TUBE_R, 2.0), (KZ + 0.19, 0.0, HY, TUBE_R, TUBE_R, 2.0)]
    bosses = [HW.s_cyl((x, HY + 0.0225, KZ + z), 'y', 0.0058, 0.02, e=0.0) for x, z in KNEE_CHEEK_SCREWS]
    # the tube's bore runs on down into the crown, wide, leaving walls and a boss at each cheek screw: a closed box over the barrel
    pocket = HW.s_cut(0.004, HW.s_loft([(KZ + 0.042, 0.0, HY - 0.0045, 0.0375, 0.0325, 4.5), (KZ + 0.075, 0.0, HY - 0.0045, 0.0370, 0.0325, 4.0), (KZ + 0.098, 0.0, HY - 0.0045, 0.0345, 0.0325, 3.2), (KZ + 0.118, 0.0, HY - 0.003, 0.0325, 0.0300, 2.6), (KZ + 0.135, 0.0, HY - 0.001, TUBE_R - TUBE_WALL, 0.0295, 2.3), (KZ + 0.155, 0.0, HY, TUBE_R - TUBE_WALL, TUBE_R - TUBE_WALL, 2.0), (THIGH_TOP + 0.001, 0.0, HY, TUBE_R - TUBE_WALL, TUBE_R - TUBE_WALL, 2.0)], e=0.0), *bosses, HW.s_cyl((0.0, HY, KZ), 'y', ARCH + 0.0045, 0.2))
    yoke = HW.fork_yoke((0.0, HY, KZ), 0.048, TUBE_TOP_C, TUBE_R + 0.002, front, HY + SLOT_IN, HY + SLOT_OUT, ARCH, fillet=0.006, pocket=pocket)
    tube, _ = HW.closed_tube(KZ + 0.15, THIGH_TOP + 0.0002, 0.0, HY, TUBE_R, TUBE_WALL, top_collar=(0.0445, 0.0042, 0.002))
    tube = HW.s_cut(0.0, tube, pocket)
    fork = HW.s_cut(0.0, HW.s_union(0.002, yoke, tube), HW.s_box((0.0, HY + SLOT_OUT + 0.05, KZ + 0.0375), (0.3, 0.1, 0.175)))     # the slot's far face, up to where the cheek ends: KZ + 0.125
    outer = HW.s_inter(0.0, HW.s_inter(0.006, HW.s_ext('y', HY - 0.2, HY + 0.2, HW.lug((0.0, KZ), 0.048, TUBE_TOP_C, TUBE_R + 0.002)), HW.s_loft(front)),
                       HW.s_box((0.0, HY + SLOT_OUT - 0.0002 + 0.05, KZ + 0.0375), (0.3, 0.1, 0.175)))
    # pocketed from outside, a rim all round, a ring round the bearing, a land at each screw: all the metal is on a load path
    web = HW.s_ext('y', HY + 0.0355, HY + 0.06, HW.lug((0.0, KZ + 0.012), 0.0415, (0.0, KZ + 0.118), 0.024))
    keep = [HW.s_cyl((0, HY + 0.05, KZ), 'y', 0.0395, 0.04)] + [HW.s_cyl((x, HY + 0.05, KZ + z), 'y', 0.0072, 0.04) for x, z in KNEE_CHEEK_SCREWS]
    outer = HW.s_cut(0.002, outer, HW.s_cut(0.002, web, *keep))
    # the inner eye's opening, round the cup's neck: a key reaches the hub's M5s through the cup
    return [
        HW.machined('L thigh fork', fork, BLACK, voxel=0.0008, voxel_hi=0.0006, exposed=True, finish='silver'),
        cut(cyl('L thigh fork opening', (0, HY - 0.0395, KZ), 'y', 0.0266, 0.0065), 'L thigh fork'),
        HW.machined('L knee outer cheek', outer, BLACK, voxel=0.0005, voxel_hi=0.0004, exposed=True, finish='silver'),
        cut(cyl('L knee outer cheek bearing bore', (0, HY + KNEE_OUT_BEARING - 0.00002, KZ), 'y', KNEE_RACE_OD, 0.00804), 'L knee outer cheek'),
        cut(cyl('L knee outer cheek thread', (0, HY + (SLOT_OUT + KNEE_OUT_BEARING - 0.004) / 2, KZ), 'y', 0.034, KNEE_OUT_BEARING - 0.004 - SLOT_OUT + 0.0004), 'L knee outer cheek'),
        cut(cyl('L knee outer cheek lip', (0, HY + 0.0425, KZ), 'y', 0.0312, 0.003), 'L knee outer cheek'),
        *[cut(cyl(f'L knee cheek head way {k + 1}', (x, HY + (0.0413 if k < 2 else 0.0379), KZ + z), 'y', 0.0050, 0.0077), 'L knee outer cheek') for k, (x, z) in enumerate(KNEE_CHEEK_SCREWS)],
    ]


def knee_cup():
    """The motor's output: floor on the hub, thin wall round the M5 heads, seat for the inner race, into the fork's inner cheek. One piece with the cheek when made (the fork and it are drawn apart for the checks' sake, as the flange of its joint).
    Inside, scallops round the heads: the wall is 4 mm between heads and 2.45 at them, the key gets to each. Outside, the seat."""
    hub, floor = -0.02485, -0.03065
    poly = HW.rounded_poly([(0.0, hub + 0.0002), (KNEE_RACE_R - 0.00004, hub + 0.0002), (KNEE_RACE_R - 0.00004, -0.0342), (0.0247, -0.0342), (0.0247, -0.0349), (0.0253, -0.0349), (0.0253, -0.03455), (0.0264, -0.03455), (0.0264, -0.03695), (0.040, -0.03695), (0.040, -0.0372), (0.0264, -0.0372), (0.0264, -0.0423), (0.0210, -0.0423), (0.0210, floor), (0.0, floor)])
    tree = HW.s_rev((0, HY, KZ), 'y', [[r, t] for r, t in poly])
    return HW.machined('L_knee flange', tree, ALU, voxel=0.0004, voxel_hi=0.0003, exposed=True)


def knee_secs(wall=0.0):
    """The shank's closed section, (z, cx, cy, a, b, n), as an outside (wall 0) or a cavity. Front line tangent to the barrel, then the skin's, 3 mm in; the back line scooped behind the knee, so thigh and shank fold to 2.15 rad: their half angles at any radius have
    to add up under 57 deg. At the RS03 the skin and the section square up (n 4.4): the motor is as deep as the shank (98.5 of 110), and the wall has to pass it (1.4 mm there)."""
    top = [(KZ - 0.020, 0.058, -0.026, 0.025, 3.0), (KZ - 0.045, 0.0540, -0.0240, 0.026, 3.0), (KZ - 0.066, 0.0502, -0.0225, 0.0272, 3.0), (KZ - 0.075, 0.0497, -0.0240, 0.0335, 3.0),
           (KZ - 0.088, 0.0490, -0.0300, 0.0415, 3.2), (KZ - 0.100, 0.0485, -0.0360, 0.0445, 3.4), (KZ - 0.115, 0.0480, -0.0445, 0.0465, 3.5), (KZ - 0.135, 0.0473, -0.0525, 0.0475, 3.6), (KZ - 0.160, 0.0464, -0.0585, 0.0480, 3.6)]
    out = [(z, (f + b) / 2, HY, (f - b) / 2 - wall, y - wall, n) for z, f, b, y, n in top]
    for z, ncav, w in ((0.355, 3.7, wall), (0.34, 4.1, min(wall, 0.0014)), (PITCH_Z, 4.6, min(wall, 0.0014)), (0.265, 4.1, min(wall, 0.0014)), (0.25, 3.7, wall), (0.2325, 3.6, wall)):
        q = shank_section(z)
        f, b, y = q[1] + q[3] - 0.0033, q[1] - q[3] + 0.0033, min(q[4] - 0.0033, 0.0505)
        out.append((z, (f + b) / 2, HY, (f - b) / 2 - w, y - w, ncav if wall else q[5]))
    return out


def knee_body():
    """The shank's top in one piece: the barrel round the RS04, a waist, then a closed section past both ankle motors, with the motors' seats. Open behind from z 0.238 up to the waist, closed by the back panel."""
    c = (0.0, HY, KZ)
    barrel = HW.round_housing(c, 'y', KNEE_OD, KNEE_COLLAR_END, KNEE_REAR, face_t=KNEE_PLATE, collar_r=0.0385)
    barrel = HW.s_cut(0.0, barrel, HW.s_rev(c, 'y', HW.rounded_poly([(0.0405, KNEE_PLATE - 0.001), (0.0405, KNEE_PLATE + 0.0025, 0.001), (0.0475, KNEE_PLATE + 0.0025, 0.001), (0.0475, KNEE_PLATE - 0.001)])))      # plate 2.8 mm thin inside its ring of screws
    bosses = [HW.s_cyl((KNEE_RIM_R * math.cos(math.radians(a)), HY + 0.0242, KZ + KNEE_RIM_R * math.sin(math.radians(a))), 'y', KNEE_RIM_LOBE, 0.0085, e=0.002) for a in KNEE_RIM_ANGLES]
    outside = HW.s_loft(knee_secs())
    outer = HW.s_union(0.009, barrel, outside)
    outer = HW.s_union(0.002, outer, *bosses)
    # the hubs' feet: under the motor, outside the cheeks' reach (the cheek's edge stops 16 deg behind straight down at 2.15 rad)
    feet = [HW.s_ext('y', HY + y * 0.0300, HY + y * 0.0431, HW.lug((-0.001, KZ - 0.0735), 0.0072, (0.0125, KZ - 0.0735), 0.0072), e=0.001) for y in (-1, 1)]
    outer = HW.s_cut(0.0, outer, HW.s_box((0.0, HY + KNEE_REAR + 0.03, KZ + 0.0075), (0.17, 0.06, 0.149)), HW.s_box((0.0, HY + KNEE_COLLAR_END - 0.03, KZ + 0.0075), (0.17, 0.06, 0.149)))      # the blend's bulge off the two ends: seats stay flat
    cav = sorted(knee_secs(KNEE_WALL))
    cav = [q for q in cav if q[0] > 0.2325]
    cav = [(0.2355, *cav[0][1:])] + [q for q in cav if q[0] > 0.2355]                   # the floor, 3 mm: a bulkhead, with pads the lower plates lap into
    # the hollow runs up round the barrel: its wall is the box's top, and no metal sits between them
    cavity = HW.s_cut(0.004, HW.s_loft(cav, e=0.003), HW.s_cyl((0.0, HY, KZ), 'y', KNEE_OD + 0.0003, 0.2), HW.s_box((0.0, HY, KZ + 0.05), (0.2, 0.2, 0.1)))
    rear = HW.s_box((KNEE_XS - 0.05, HY, (0.238 + 0.447) / 2), (0.1, 0.13, 0.447 - 0.238))
    slots = [HW.s_box((-0.0348, HY - 0.0312, 0.238), (0.041, 0.016, 0.02), e=0.002), HW.s_box((-0.0268, HY + 0.0362, 0.238), (0.0285, 0.0135, 0.02), e=0.002)]
    # lands for the panel's screws, and the ankle motors' seats on their front faces, outside the cranks' sweep. Not the motors' real patterns yet: the ankle's round
    lands = [HW.s_box((KNEE_XS + 0.0035, HY + sg * 0.036, z), (0.0072, 0.015, 0.014), e=0.001) for sg, z in KNEE_PANEL_SCREWS]
    seats = [HW.s_box((0.0315, HY - 0.03515, PITCH_Z), (0.011, 0.0187, 0.024), e=0.002), HW.s_box((-0.0468, HY - 0.03515, PITCH_Z), (0.0056, 0.0187, 0.024), e=0.001),
             HW.s_box((0.0255, HY + 0.0356, ROLL_Z), (0.011, 0.0238, 0.024), e=0.002), HW.s_box((-0.010, HY + 0.0356, ROLL_Z + 0.037), (0.024, 0.0238, 0.012), e=0.002)]
    inside = HW.s_inter(0.0, HW.s_union(0.0015, *lands, *seats), HW.s_loft(knee_secs(0.0009)))      # they end inside the wall, so none shows outside
    shell = HW.s_union(0.0, HW.s_cut(0.0, outer, cavity), inside)      # no blend: one would ghost through the thin wall
    shell = HW.s_union(0.004, shell, *feet)
    pads = [HW.s_box((0.0035, HY + y * 0.026, 0.2405), (0.033, 0.0085, 0.013), e=0.001) for y in (-1, 1)]
    shell = HW.s_union(0.0, shell, HW.s_inter(0.0, HW.s_union(0.0, *pads), HW.s_loft(knee_secs(0.0009))))
    shell = HW.s_cut(0.0, shell, rear, *slots, HW.s_cyl((CALF_X, HY - 0.0271, PITCH_Z), 'y', 0.0365, 0.0026))      # the RS03's hub turns in that
    # user: "the dark color of the actuators doesn't show up". A round window each side of each ankle motor, so the black shows inside the silver. What it costs at walking load: reports/strength.md, 'closed section' rows
    shell = HW.s_cut(0.002, shell, *[HW.s_cyl((CALF_X, HY + sg * 0.05, z), 'y', r, 0.03) for sg in (-1, 1) for z, r in ((PITCH_Z, 0.031), (ROLL_Z, 0.027))])
    return HW.machined('L shank body', shell, BLACK, voxel=0.0007, voxel_hi=0.0007, finish='silver', zones=[dict(c=[0.0, HY, KZ], axis='y', r=KNEE_OD + 0.006, finish='black')])


def knee_panel():
    """The back of the closed section: the same outside and wall, behind the split, with a pad at each screw."""
    outside = HW.s_loft(knee_secs())
    cav = [q for q in sorted(knee_secs(KNEE_WALL)) if q[0] <= KZ - 0.1 and q[0] >= 0.2325]
    tree = HW.s_cut(0.0, outside, HW.s_loft(cav))
    pads = [HW.s_box((KNEE_XS - 0.002, HY + sg * 0.036, z), (0.0048, 0.015, 0.014), e=0.001) for sg, z in KNEE_PANEL_SCREWS]
    tree = HW.s_union(0.0, tree, HW.s_inter(0.0, HW.s_union(0.0, *pads), HW.s_loft(knee_secs(0.0009))))
    tree = HW.s_inter(0.0, tree, HW.s_box((KNEE_XS + 0.0002 - 0.05, HY, (0.238 + 0.447) / 2), (0.1, 0.13, 0.447 - 0.238)))      # the split, seated 0.2 mm on the lands
    return HW.machined('L shank back panel', tree, BLACK, voxel=0.0006, voxel_hi=0.0005, finish='silver')


def knee_rear_cap():
    """Closes the barrel, carries the outer bearing's post, and takes heat off the motor's back through a skirt on its rear face. Lobed at its screws: M3 want 4.1 mm of edge."""
    c = (0.0, HY, KZ)
    cap = HW.end_cap(c, 'y', 0.060, 0.001, KNEE_OD, KNEE_REAR - 0.0013, 0.0314, post=(KNEE_RACE_R - 0.00004, 0.0414, 0.0263, 0.0333))
    lobes = [HW.s_cyl((KNEE_RIM_R * math.cos(math.radians(a)), HY + 0.0299, KZ + KNEE_RIM_R * math.sin(math.radians(a))), 'y', KNEE_RIM_LOBE, 0.003, e=0.0006) for a in KNEE_RIM_ANGLES]
    cap = HW.s_cut(0.0, cap, HW.s_rev(c, 'y', HW.rounded_poly([(0.0305, 0.0296, 0.001), (0.0555, 0.0296, 0.001), (0.0555, 0.033), (0.0305, 0.033)])))      # a dished plate, 1.6 thick inside the rim
    skirt = HW.s_rev(c, 'y', HW.rounded_poly([(0.0485, 0.01475), (0.0598, 0.01475), (0.0598, KNEE_REAR + 0.001), (0.0580, KNEE_REAR + 0.001), (0.0580, 0.0167, 0.001), (0.0485, 0.0167)]))
    return HW.machined('L knee rear cap', HW.s_union(0.0, HW.s_union(0.0006, cap, *lobes), skirt), BLACK, voxel=0.0005, voxel_hi=0.0004, exposed=True, finish='black')


THIGH_LOW = [(KZ - 0.04, 0.0, HY, KNEE_SOCKET, KNEE_NECK, 6.0), (KZ + 0.04, 0.0, HY, KNEE_SOCKET, KNEE_NECK, 5.0), (KZ + 0.052, -0.001, HY, 0.0720, 0.0450, 4.5),
             (KZ + 0.066, -0.002, HY, 0.0725, 0.0485, 3.6), (KZ + 0.082, -0.0035, HY, 0.0730, 0.055, 3.2),
             (KZ + 0.10, -0.0051, HY, 0.0721, 0.0615, 3.0), (KZ + 0.125, -0.0060, HY, 0.0735, 0.0645, 2.9),
             (KZ + 0.16, -0.0071, HY, 0.0756, 0.066, 2.6)]
# iteration 38: one list, for the thigh and the knit parts cut from the same surface
THIGH_SECS = THIGH_LOW + [(KZ + 0.19, -0.008, HY, 0.0775, 0.070, 2.4), (HZ - 0.15, -0.024, HY, 0.090, 0.070, 2.4),
                          (HZ - 0.095, -0.036, HY, 0.099, 0.070, 2.4), (HIP_SEAM + 0.004, -0.041, HY, 0.095, 0.070, 2.4),
                          (HIP_SEAM + 0.022, -0.041, HY, 0.095, 0.070, 2.4)]
joint('L_hip_pitch', 'pelvis', 'L_hip_pitch_link', (0, HY, HZ), 'y', (-1.65, 0.24), 'RS04',
      act_c=(0, PITCH_ACT_Y, HZ), out=+1, exposed=True, flange=HW.machined('L_hip_pitch flange', HW.output_cup((0, PITCH_ACT_Y, HZ), 'y', 1), BLACK, voxel=0.0004, voxel_hi=0.0003, exposed=True))
def hip_carrier():
    """The roll motor's housing and the arm to the pitch output, one piece: a round barrel on the roll axis, bolted to the motor's rear face, and a closed arm into an eye on the pitch cup. The arm runs over the top: below, the roll output's plate sweeps."""
    c = (-0.10, HY, HZ)
    barrel = HW.s_rev(c, 'x', HW.rounded_poly([(0.0, -0.01935), (HOUSING_OD, -0.01935, 0.005), (HOUSING_OD, 0.0238, 0.0008), (0.0, 0.0238)]))
    side = HW.s_ext('y', PITCH_ACT_Y + 0.037, HY + 0.0395, HW.lug((0.0, HZ), 0.0425, (-0.0625, HZ + 0.006), 0.057), e=0.002)
    clear = HW.s_ext('y', 0.0, 0.3, HW.rounded_poly([(-0.0775, HZ + 0.006, 0.003), (0.0, HZ + 0.006, 0.0), (0.0, HZ - 0.2), (-0.0775, HZ - 0.2)]))                # below the roll output's reach, the arm is cut away near the plate
    keep = HW.s_cyl((0.0, HY, HZ), 'y', 0.0425, 0.3)
    arm = HW.s_cut(0.002, side, HW.s_cut(0.0, clear, keep))
    arm = HW.s_cut(0.002, arm, HW.s_inter(0.0, HW.s_cut(0.0, HW.s_ext('y', PITCH_ACT_Y + 0.0393, HY + 0.0375, HW.lug((-0.004, HZ + 0.004), 0.0355, (-0.06, HZ + 0.008), 0.049)), HW.s_cyl((0.0, HY, HZ), 'y', 0.044, 0.3)), HW.s_box((-0.05, HY, HZ + 0.0095 + 0.05), (0.2, 0.1, 0.1))))      # closed: a floor over the cut-away below      # a closed box: walls 2.5
    t = HW.s_union(0.005, barrel, arm)
    return HW.machined('L hip carrier', t, BLACK, voxel=0.0007, voxel_hi=0.0005, exposed=True, finish='silver', zones=[dict(c=[-0.10, HY, HZ], axis='x', r=0.0675, t0=-0.03, t1=0.0245, finish='black')])


def hip_yoke():
    """The roll output down to the yaw motor, one piece: a boss on the roll hub, a plate in the 9 mm behind the pitch motor, a band concentric with the pitch axis under it, and a round housing on the yaw RS03, closed on top."""
    plate = HW.s_ext('x', -0.0750, -0.0686, HW.lug((HY, HZ), 0.0285, (HY, HZ - 0.066), 0.030), e=0.001)
    band = HW.s_rev((0, HY, HZ), 'y', HW.rounded_poly([(0.0708, -0.0105, 0.002), (0.0795, -0.0105, 0.003), (0.0795, 0.031, 0.003), (0.0708, 0.031, 0.002)]))      # from y 0.0895 out: inside that the pelvis's cover folds past
    band = HW.s_cut(0.002, band, HW.s_rev((0, HY, HZ), 'y', [[0.0726, -0.0065], [0.0777, -0.0065], [0.0777, 0.027], [0.0726, 0.027]]))      # a channel open at its ends
    sector = HW.s_box((-0.0165, HY, HZ - 0.075), (0.1175, 0.08, 0.11))
    band = HW.s_inter(0.0, band, sector)
    housing = HW.s_rev((0, HY, 0.0), 'z', HW.rounded_poly([(0.0, 0.8502), (0.0525, 0.8502, 0.004), (0.0553, 0.8452, 0.002), (0.0553, 0.7942), (0.0, 0.7942)]))
    lobes = [HW.s_cyl((0.0562 * math.cos(math.radians(a)), HY + 0.0562 * math.sin(math.radians(a)), 0.7985), 'z', 0.0042, 0.0086, e=0.0015) for a in (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)]
    t = HW.s_union(0.008, HW.s_union(0.006, plate, band), HW.s_union(0.002, housing, *lobes))
    t = HW.s_cut(0.0, t, HW.s_box((-0.1, HY, HZ - 0.05), (0.05, 0.2, 0.3)))      # the blend's bulge off the plate's back: the roll motor is there
    return HW.machined('L hip yoke', t, BLACK, voxel=0.0006, voxel_hi=0.0005, exposed=True, finish='silver', zones=[dict(c=[0.0, HY, 0.0], axis='z', r=0.0605, t0=0.78, t1=0.8455, finish='black')])


link('L_hip_pitch_link', 1.6, [
    # iteration 42: the eccentric bar gone: see hip_carrier()
    hip_carrier(),
    cut(cyl('L hip carrier eye', (0, PITCH_ACT_Y + 0.0425, HZ), 'y', 0.0266, 0.0175), 'L hip carrier'),
    dict(cut(cyl('L hip carrier register', (-0.10 + 0.0073, HY, HZ), 'x', HOUSING_BORE, 0.0434), 'L hip carrier'), segs=256),
    cut(cyl('L hip carrier plate hole', (-0.12, HY, HZ), 'x', 0.0475, 0.012), 'L hip carrier'),
])
# The roll actuator sits behind the pitch actuator and drives FORWARD: its output plate runs straight down the gap
# between the two actuators (it turns in its own plane, so it never meets the pitch actuator), and two tabs
# under the pitch actuator carry it to a crown on the hip yaw actuator, inside the thigh's top. Nothing reaches
# behind the roll actuator, so the lobe can be round at the back (iteration 4).
joint('L_hip_roll', 'L_hip_pitch_link', 'L_hip_roll_link', (0, HY, HZ), 'x', HIP_ROLL, 'RS04',
      act_c=(-0.10, HY, HZ), out=+1, exposed=True, flange=HW.machined('L_hip_roll flange', HW.s_rev((-0.10, HY, HZ), 'x', HW.rounded_poly([(0.0, 0.0247), (0.0285, 0.0247), (0.0285, 0.0330), (0.036, 0.0330), (0.036, 0.0334), (0.0, 0.0334)])), BLACK, voxel=0.0004, voxel_hi=0.0003, exposed=True))
link('L_hip_roll_link', 2.0, [
    # Figure's hip lobe: a smooth rounded glossy pod over the hip mechanism (iteration 4: the boxy lobe with the
    # roll bracket's corner behind it read as a square butt, user review 2026-10-31)
    # (iteration 6: full at the bottom-back so the back line runs on into the thigh, user review)
    # iteration 21: on the roll link, its underside the seam; its outer side 3 mm in from the thigh's at the seam
    # and 3 mm from the hanging wrist above
    # iteration 22: down over the thigh's domed top, its outline the thigh's (3 mm in on the outer side)
    loft('L hip lobe', [(HZ - 0.130, -0.028, 0.105, 0.093, 0.062, 2.4),
                        (HZ - 0.095, -0.036, 0.105, 0.099, 0.062, 2.4),
                        (HZ - 0.070, -0.040, 0.105, 0.097, 0.062, 2.6),
                        (HZ - 0.035, -0.033, 0.105, 0.103, 0.0615, 2.8),
                        (HZ + 0.005, -0.035, 0.105, 0.103, 0.058, 3.2),
                        (HZ + 0.035, -0.040, 0.1025, 0.100, 0.0565, 3.4),
                        (HZ + 0.065, -0.045, 0.100, 0.085, 0.050, 2.6),
                        (HZ + 0.080, -0.050, 0.100, 0.055, 0.036, 2.4)], GLOSS, open=(), clip=HIP_LOBE_CLIP,
         keep=(HIP_DOME_C, HIP_DOME_R)),
    # iteration 22: the roll actuator stood out of the lobe's slim back (its rim showed in a slot, and the slot's
    # 3 mm behind it let the light through): a cap round it, on the outer side, flush with the lobe's back
    loft('L hip roll cap', [(HZ + dz, -0.1045, HY, 0.0325, (0.0695 ** 2 - dz ** 2) ** 0.5, 6.0)
                            for dz in (-0.0675, -0.058, -0.04, -0.02, 0.0, 0.02, 0.04, 0.058, 0.0675)], GLOSS, open=(),
         clip=HIP_LOBE_CLIP, keep=(HIP_DOME_C, HIP_DOME_R)),
    cut(prism('L hip lobe bevel', -0.075, 0.14, HIP_BEVEL, BLACK), 'L hip lobe'),
    # hollow round the pitch actuator (pelvis) and its flange, and round the bracket, as the roll turns them
    *pitch_sweep('L hip lobe pitch sweep'),
    dict(cut(roll_sweep('L hip lobe arm sweep', [(0.029, 0.0395, -0.0425, 0.064)], -0.125, 0.0425), 'L hip lobe', 'L hip roll cap'), internal=True),
    # the roll actuator is round about the roll axis: a round opening, 3 mm clear (iteration 19)
    dict(cut(cyl('L hip lobe roll opening', (-0.10, HY, HZ), 'x', 0.0665, 0.062), 'L hip lobe', 'L hip roll cap'), internal=True),
    # iteration 42: plate, tabs and crown become one yoke, on the roll hub
    hip_yoke(),
    cut(cyl('L hip yoke hub hole', (-0.0712, HY, HZ), 'x', 0.0287, 0.0092), 'L hip yoke'),
    cut(cyl('L hip yoke yaw pocket', (0, HY, 0.82135), 'z', 0.0531, 0.0544), 'L hip yoke'),
    HW.machined('L hip yaw ring', HW.s_union(0.0008, HW.s_rev((0, HY, 0.0), 'z', HW.rounded_poly([(0.0456, 0.7890), (0.0562, 0.7890, 0.0008), (0.0562, 0.7944), (0.0456, 0.7944)])), *[HW.s_cyl((0.0562 * math.cos(math.radians(a)), HY + 0.0562 * math.sin(math.radians(a)), 0.7917), 'z', 0.0042, 0.0054, e=0.0008) for a in (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)]), BLACK, voxel=0.0004, voxel_hi=0.0003, exposed=True, finish='black'),
])
# iteration 41: a real flange, wide enough for the thigh tube's collar screws, with the hub's own 6 M4 inside it
joint('L_hip_yaw', 'L_hip_roll_link', 'L_hip_yaw_link', (0, HY, HZ - 0.11), 'z', (-1.2, 1.2), 'RS03', out=-1,
      exposed=True, flange=HW.machined('L_hip_yaw flange', HW.s_rev((0, HY, 0.0), 'z', HW.rounded_poly([(0.0, THIGH_TOP), (0.0205, THIGH_TOP, 0.0005), (0.0205, 0.7872, 0.001), (0.0335, 0.7872, 0.001), (0.0335, THIGH_TOP, 0.0005), (0.0445, THIGH_TOP, 0.0005), (0.0445, HZ - 0.11 - 0.0283 + 0.0002, 0.0005), (0.0, HZ - 0.11 - 0.0283 + 0.0002)])), BLACK, voxel=0.0004, voxel_hi=0.0003))
link('L_hip_yaw_link', 3.6, [
    # iteration 6 (user review of the side view): Figure's thigh is full at the back under the hip and tapers to
    # the knee, so the back line runs on from the hip in one curve; the front stays nearly straight
    # iteration 9 (user review): fuller at the knee (13 cm wide) so the knee discs sit flush in the leg, as on
    # Figure; no wider than before at the top, where the hands hang beside the thigh
    # iteration 11 (side frame): Figure's thigh is 0.14 deep at 0.65 m and 0.15 at 0.70; ours was ~1 cm deeper.
    # The knee RS04 (centre 0.72 m) keeps 9 mm inside the front.
    # iteration 20: the loft ran down only to 3 cm above the knee axis, so the clip below never cut it and the knee
    # showed a 3.7 cm black band in front (Figure's thigh meets the shank in a seam): now down past the axis
    # iteration 35 (user: the thigh's back shows a corner and a notch, from holding 2.15 rad). The fold plane cannot move
    # at this range: it is where the calf's back face lands. What can go is the corner: the back line bends in over its
    # last 7 cm, so the plane meets it at about 30 deg, not 49, and the corner sits 1 cm lower; 13 cm deep there
    # iteration 41: the former under the knit (2.5 of the 3.3 kg/m2 the garment's density stands for) was a placeholder for what holds the cloth out. Thigh: ribs off the tube, 1.5 assumed (wall_fill). Gaiter and shank: knit and foam only (0.8), as the shank's own body fills the skin to 3 mm, so a former there counted the same volume twice
    dict(loft('L thigh', THIGH_SECS, GREY, 'knit',
         # iteration 21: up to 3 mm under the hip lobe (the yaw turns it under the lobe); iteration 25: dished
         # iteration 26: flat sides at the knee (n 4) for the pads; the socket's tail stops where it would thin out
         clip=above((-1.0, KZ + 0.003), (1.0, KZ + 0.003)), open=()), wall_fill=0.70),
    cut(ell('L thigh bowl', HIP_DOME_C, (HIP_DOME_R + 0.003,) * 3), 'L thigh'),
    # round the hip yaw actuator and the crown on it (they turn in it), 3 mm clear
    cut(cyl('L thigh top opening', (0, HY, HIP_SEAM - 0.018), 'z', 0.0635, 0.056), 'L thigh'),
    # panels, not one jumpsuit (reports/garment_options.md): the thigh's panel stops at the fold plane behind the knee; a small gaiter closes from there to the shank's
    cut(loft('L thigh fold', [(KZ - 0.06, -0.05, HY, 0.1, 0.09, 8.0), (KZ + 0.2, -0.05, HY, 0.1, 0.09, 8.0)],
             clip=[below(*KNEE_FOLD), above(knee_ray(KNEE_FOLD_DEG + 180), knee_ray(KNEE_FOLD_DEG))]), 'L thigh'),
    dict(loft('L knee gaiter', THIGH_SECS, GREY, 'knit', open=(), clip=[below(*up(KNEE_FOLD, 0.003)), above(*up((knee_ray(KNEE_FOLD_DEG + 180), knee_ray(KNEE_FOLD_DEG)), -0.003)), above((-1.0, KZ + 0.003), (1.0, KZ + 0.003))]), wall_fill=0.24),
    # the sewn lines (decals): across the knee's front, as Figure's seam; a crease over each hub
    HW.seam_groove('L knee seam groove', THIGH_SECS, KZ + 0.0045, 0.0015, 0.0025, ['L thigh', 'L knee gaiter']),
    dict(loft('L knee seam', [(KZ + 0.0046, 0.0, HY, KNEE_SOCKET - 0.0026, KNEE_NECK - 0.0026, 6.0), (KZ + 0.0059, 0.0, HY, KNEE_SOCKET - 0.0026, KNEE_NECK - 0.0026, 6.0)], '#2A2C30', 'decal', open=()), material='pa12', mass=0.002),
    *[dict(loft(f'L thigh crease {sd}', [(KZ + dz, -0.004, HY + y * w, 0.0011, 0.0011, 2.0) for dz, w in ((0.068, 0.0478), (0.076, 0.0518), (0.084, 0.0560), (0.094, 0.0598), (0.104, 0.0620))],
                  '#8E9297', 'decal', open=()), material='pa12', mass=0.001) for sd, y in (('in', -1), ('out', +1))],
    *knee_fork(),
], 'the knee RS04 output: fork and axle; its stator rides on the shank')
# flexion 2.15 rad (iteration 17: through the knee zone the calf meets the thigh at 2.2; the deep squat uses 2.0)
joint('L_knee', 'L_hip_yaw_link', 'L_knee_link', (0, HY, KZ), 'y', (0.0, 2.15), 'RS04', act_c=(0, HY, KZ), out=-1,
      stator_on_child=True, flange=knee_cup())
SHANK = lambda f: AZ + 0.045 + f * (KZ - AZ - 0.01)      # height at fraction f of the shank skin
# Ankle drive, iteration 34 (DECISIONS 3, 4a; the solved numbers in the paper report of 2026-11-04, paper only: no published
# example of this exact layout, a warning sign to keep in view). No shared parallel pair: each motor has its own crank and
# rod. Pitch: RS03 low in the calf, crank 22.7, rod 212, to an arm 32.9 behind the ankle on the cross, on the roll axis' line
# so roll does not change its length; 1.44:1 at the middle of its range, modelled as 1.5 (a table of ratios is the second pass).
# Roll: RS06 above it, crank 26, rod 312, to a pin on the foot at y 34 on the pitch axis' line; 1.32:1 at mid range.
CALF_X = -0.01                            # the motors' axes
PITCH_Z, ROLL_Z = 0.300, 0.3945
PITCH_CRANK, PITCH_ROD_END = (-0.0320, HY - 0.0398, 0.3054), (-0.0318, HY - 0.012, 0.0937)
ROLL_CRANK, ROLL_ROD_END = (-0.0360, HY + 0.0368, ROLL_Z), (0.0, HY + 0.034, AZ)


def ankle_drive(s):
    """The cranks and rods, at the zero pose: they move with the linkage, not with the shank or the foot, so they are drawn
    ('linkage') and kept out of the collision checks."""
    crank = lambda n, z, c: cap(f'{s} ankle {n} crank', (CALF_X, c[1], z), c, 0.008, '#3A3D42', 'linkage')
    return [crank('pitch', PITCH_Z, PITCH_CRANK), cap(f'{s} ankle pitch rod', PITCH_CRANK, PITCH_ROD_END, 0.005, ALU, 'linkage'),
            crank('roll', ROLL_Z, ROLL_CRANK), cap(f'{s} ankle roll rod', ROLL_CRANK, ROLL_ROD_END, 0.005, ALU, 'linkage')]


def SHANK_FRONT(z):
    """The shank's front line in the side view: one straight line from the collar to above the motors."""
    return 0.0455 + 0.045 * (z - PITCH_Z)


def SHANK_BACK(z):
    """The back line: straight from the collar to the RS03, then easing to the drum (it is what folds into the thigh)."""
    import numpy as np
    return float(np.interp(z, (PITCH_Z, 0.35, 0.40, KZ - 0.075), (-0.0645, -0.0655, -0.0660, -0.0660))) if z > PITCH_Z else \
        -0.0645 + 0.121 * (PITCH_Z - z)


def shank_section(z):
    """Loft section from the two lines, the width's straight taper, and one section shape (rounder toward the ankle)."""
    import numpy as np
    f, b = SHANK_FRONT(z), SHANK_BACK(z)
    return (z, (f + b) / 2, HY, (f - b) / 2, min(KNEE_WIDE, 0.0445 + 0.0604 * (z - 0.15)),
            float(np.interp(z, (0.13, 0.25, 0.27, 0.30, 0.33, 0.37, 0.40, KZ - 0.075), (2.8, 3.6, 4.0, 4.4, 4.0, 3.6, 3.6, 3.0))))


# Shoe lines in the side view, (x, z) pairs (iteration 12, user: "feet so off"; Figure's shoe from the reveal frames:
# a low sneaker whose white upper rises from the toe to the ankle, a thick black outsole, a black stripe rising from
# the midfoot to the heel, and an angled black collar, high at the back). What turns with the ankle cannot be on the
# foot: the shoe's collar line drops from the heel to the front of the ankle, and the white above it (the 'sock', up
# to the angled black collar) is on the shank, as on a knit sock-shoe. The lines are placed against the shank's
# sweep over the ankle's range (dorsiflexion 0.8 rad, plantarflexion 0.55, roll 0.3); scanned every 0.01 rad, the
# worst contact is a 0.04 cm3 graze of the collar at full dorsiflexion (collision threshold 0.3 cm3).
COLLAR = ((-0.10, 0.125), (0.06, 0.088))                  # top of the shoe around the ankle (heel 0.122)
SOCK_LOW = ((-0.10, 0.121), (0.06, 0.084))                # bottom of the sock, 4 mm down inside the shoe
# the black collar: 2.8 cm at the front, 4.5 cm at the back (Figure's is a crescent, deepest behind the ankle)
TRIM_LOW, TRIM_HIGH = ((-0.04, 0.158), (0.04, 0.118)), ((-0.04, 0.203), (0.04, 0.146))
# pieces cut apart along a plane overlap by 2 mm, hidden inside the bigger one (pieces that only touch fall apart
# in the solid union: 'covers mounted' counted 156 slivers)
link('L_knee_link', 2.08, [
    # iteration 11 (side frame): Figure's calf is ~0.105 deep; ours was 0.135, then 0.121 round the RS03.
    # iteration 16: with RS06, 0.105 like Figure's, 4.5 mm clear of them inside the wall
    # iteration 13: slim above the ankle like Figure's (~8 cm), below the ankle actuators
    # iteration 26: narrowing to the neck under the pads (within the cheeks' reach), up to the drum, inside the depth the
    # thigh's fold plane clears; the ankle motors keep their room below
    # iteration 34: 107 mm deep at the RS03 (Figure about 105), the section squared there (n 4) so its depth reaches the
    # motor's end faces; 110 at the plain section. The motor's ears turned, as the maker's drawing allows (98.5 deep)
    # iteration 35 (user: the iteration 34 shank read curvy and muscular, a bulge at the calf and a pinch above the ankle;
    # Figure's is long, straight and tapered). Figure's standing leg on the side frame (reports grid, perpendicular to the
    # knee-ankle line): 94 mm deep at z 0.20, 113 at 0.29, 132 at 0.40; its bent leg 86, 100, 124. So the depth the motors
    # force is not against Figure's line at all: only at z 0.30 does the RS03 need its 104.5, and its round side falls away
    # above and below. Now two straight lines from the collar to the motor (SHANK_FRONT, SHANK_BACK), straight on above it
    # in front, easing into the drum behind, one section shape all the way (the old n 2.2 to 4 to 2.6 was the muscle's
    # shading). 110 mm at the RS03, 3 mm wall to the corners of its end faces with n 3.6.
    # iteration 38 (review: an hourglass, daylight between chin and neck; Figure's front frame: full grey between the hubs, black ending about
    # 4.4 cm under the axis as the shank's side comes round it): full width up beside the chin, which sits in its side; the band closed from
    # there down, the cheeks fill it above
    dict(loft('L shank', [shank_section(z) for z in (SHANK(0), 0.175, 0.214, 0.25, PITCH_Z, 0.35, 0.40)] +
         [(*shank_section(KZ - 0.10)[:4], 0.053, 3.2), (*shank_section(KZ - 0.084)[:4], 0.0485, 3.8), (*shank_section(KZ - 0.07)[:4], 0.0465, 4.5), (*shank_section(KZ - 0.055)[:4], KNEE_NECK, 5.0),
          (KZ - 0.03, -0.001, HY, 0.0665, KNEE_NECK, 6.0),
          (KZ - 0.008, 0.0, HY, 0.0695, KNEE_NECK, 12.0), (KZ, 0.0, HY, 0.0702, KNEE_NECK, 16.0)],
         GREY, 'knit', open=(), clip=above(*up(TRIM_HIGH, -0.002))), wall_fill=0.30),     # 0.24 over the body; below it the old former runs on to the collar: 0.30 overall, a hand figure
    loft('L ankle collar', [(0.11, -0.004, HY, 0.0395, 0.0465, 2.6), (0.15, -0.006, HY, 0.0455, 0.0470, 3.0),
                            (0.21, -0.0065, HY, 0.0505, 0.0505, 3.3)], BLACK,
         clip=[above(*TRIM_LOW), below(*TRIM_HIGH)]),
    # flaring into the shoe toward the heel and the sides, where the ankle's range leaves room (not in front)
    loft('L ankle sock', [(0.08, -0.013, HY, 0.048, 0.0470, 2.2), (0.105, -0.010, HY, 0.0455, 0.0460, 2.2),
                          (0.135, -0.005, HY, 0.0410, 0.0445, 2.2), (0.17, -0.004, HY, 0.0405, 0.0445, 2.2)], WHITE,
         clip=[above(*SOCK_LOW), below(*up(TRIM_LOW, 0.002))]),
    # a disc round the axis, so the outside of the bend stays round
    dict(cyl('L knee disc', (0, HY, KZ), 'y', 0.0702, 2 * KNEE_NECK, GREY, 'knit'), mass=0.02),   # iteration 41: the band round the bend only, 0.02, knit and foam: its end discs are not cloth, and half the band was counted in the thigh and shank panels (it read 0.21)

    # the hubs. A foot under the motor carries each to the shank's upper plates; the cheeks swing above it
    knee_hub('L knee hub in', -1),                                      # iteration 41: a cover with its own bracket under it: polycarbonate, 0.25 kg to 0.11

    dict(knee_hub('L knee hub out', +1), material='al7075', wall_fill=0.67),     # the heat's last stop: 2 mm of 7075, the post's washer bears on its inner wall

    *[g for sd, y in (('in', -1), ('out', +1)) for g in (
        cyl(f'L knee hub ring {sd}', (0, HY + y * HUB_OUT, KZ), 'y', 0.040, 0.002, '#2A2C30', 'decal'),
        cyl(f'L knee hub face {sd}', (0, HY + y * (HUB_OUT + 0.0003), KZ), 'y', 0.038, 0.002, GLOSS, 'decal'))],
    *[rbox(f'L knee hub mark {sd}', (0, HY + y * (HUB_OUT - 0.0048), KZ - 0.0735), (0.014, 0.004, 0.0025), 0.0005, '#F4F6F8', 'decal')
      for sd, y in (('in', -1), ('out', +1))],
    *[dict(ell(f'L knee hub lip {sd}', (0, HY + y * (HUB_OUT - 0.0066), KZ - 0.0705), (0.030, 0.0048, 0.0115), GLOSS, 'decal'), material='pa12')
      for sd, y in (('in', -1), ('out', +1))],
    # iteration 41: the machined shank: barrel round the motor, closed section down past both ankle motors. The upper plates, stator plate, post, disc, rim and bridge are gone
    knee_body(),
    cut(dict(cyl('L knee register', (0, HY + 0.0026, KZ), 'y', KNEE_BORE, 0.0538), segs=256), 'L shank body'),
    cut(cyl('L knee barrel lip', (0, HY - 0.0254, KZ), 'y', 0.0306, 0.0024), 'L shank body'),
    cut(cyl('L knee barrel bearing bore', (0, HY + KNEE_IN_BEARING, KZ), 'y', KNEE_RACE_OD, 0.0080), 'L shank body'),
    cut(cyl('L knee barrel thread', (0, HY - 0.0357, KZ), 'y', 0.034, 0.0024), 'L shank body'),
    cut(cyl('L knee cable window', (0.054 * KNEE_TAB[0], HY + 0.0195, KZ + 0.054 * KNEE_TAB[1]), [KNEE_TAB[0], 0.0, KNEE_TAB[1]], 0.009, 0.03), 'L shank body'),
    knee_panel(),
    knee_rear_cap(),
    cut(cyl('L knee rear cap hollow', (0, HY + 0.0278, KZ), 'y', 0.0479, 0.00125), 'L knee rear cap'),
    cut(cyl('L knee post hollow', (0, HY + 0.0375, KZ), 'y', 0.0125, 0.0082), 'L knee rear cap'),
    cut(cyl('L knee rear cap tab', (0.0545 * KNEE_TAB[0], HY + 0.0195, KZ + 0.0545 * KNEE_TAB[1]), [KNEE_TAB[0], 0.0, KNEE_TAB[1]], 0.0095, 0.012), 'L knee rear cap'),
    # iteration 34: the lower plates inside the roll rod's plane (y 29.5). Iteration 41: they lap into the body's floor; the upper plates and pads are gone
    span('L shank plate in', SHANK(0), 0.2448, y=HY - 0.026, sy=0.004, f=0.0015, color=BLACK, exposed=True),
    span('L shank plate out', SHANK(0), 0.2448, y=HY + 0.026, sy=0.004, f=0.0015, color=BLACK, exposed=True),
    motor('L ankle pitch actuator (buy) RS03', 'RS03', (CALF_X, HY, PITCH_Z), 'y', -1, MOTOR_CLOCK.get('L_ankle_pitch', 0.0), r=ACT['RS03']['depth'] / 2),
    motor('L ankle roll actuator (buy) RS06', 'RS06', (CALF_X, HY, ROLL_Z), 'y', +1, MOTOR_CLOCK.get('L_ankle_roll', 0.0), r=ACT['RS06']['depth'] / 2),
    *ankle_drive('L'),
    # 32 mm deep (iteration 12: inside the slimmer ankle sock); iteration 34: outer face at 28.5 mm, inside the roll rod
    rbox('L ankle fork cheek in', (0, HY - 0.0255, AZ + 0.025), (0.032, 0.006, 0.07), 0.003, BLACK, exposed=True),
    rbox('L ankle fork cheek out', (0, HY + 0.0255, AZ + 0.025), (0.032, 0.006, 0.07), 0.003, BLACK, exposed=True),
    rbox('L ankle fork bridge', (0, HY, AZ + 0.057), (0.032, 0.057, 0.012), 0.003, BLACK, exposed=True),
], 'RS03 pitch and RS06 roll in the calf, each with a crank and rod')
# dorsiflexion 0.8 rad (iteration 12, was 0.9): the deepest pose uses 0.77 (90 deg squat); the 0.1 rad buys the
# shoe's instep its height
joint('L_ankle_pitch', 'L_knee_link', 'L_ankle_pitch_link', (0, HY, AZ), 'y', (-0.8, 0.55), None, remote=True,
      capacity=dict(model='RS03', count=1, ratio=1.5))
link('L_ankle_pitch_link', 0.25, [
    # iteration 34: 43 mm across, inside the narrower fork; its tail through the back clevis (a round hole) to the pitch arm
    rbox('L ankle cross', (0, HY, AZ), (0.028, 0.043, 0.026), 0.004, BLACK, exposed=True),
    dict(cyl('L ankle cross tail', (-0.0215, HY, AZ), 'x', 0.007, 0.015, BLACK), exposed=True),
    dict(cap('L ankle pitch arm', (-0.029, HY, AZ), PITCH_ROD_END, 0.005, BLACK, 'struct'), exposed=True),
])
joint('L_ankle_roll', 'L_ankle_pitch_link', 'L_foot', (0, HY, AZ), 'x', (-0.3, 0.3), None, remote=True,
      capacity=dict(model='RS06', count=1, ratio=1.32))
# Foot: the outsole (3 cm black rubber, rounded in plan, rocker toe) and the shoe's lower upper. The upper is one
# loft cut in two: the toe (in front of x = 0.06, its own top: a low toe box and an instep rising to the ankle) and
# the heel (behind, its top cut along COLLAR). A black stripe 1.2 mm proud runs from the midfoot sole up to the heel.
# The outsole's 'contact' box (inside its rounded outline) is what the balance check and the simulator stand on.
# (iteration 13: toe box 8 mm lower, the instep one slope from the toe to the ankle as on Figure's)
# (iteration 14: rounder in plan low down, so the toe tapers inside the outsole's lip)
SHOE = [(0.022, 0.056, HY, 0.140, 0.0550, 2.45), (0.036, 0.0525, HY, 0.1395, 0.0560, 2.4),
        (0.048, 0.046, HY, 0.133, 0.0560, 2.35), (0.058, 0.0345, HY, 0.1215, 0.0560, 2.3),
        (0.067, 0.0195, HY, 0.1065, 0.0560, 2.3), (0.075, 0.003, HY, 0.090, 0.0560, 2.35),
        (0.081, -0.0115, HY, 0.0755, 0.0560, 2.35), (0.100, -0.0145, HY, 0.0705, 0.0560, 2.3),
        (0.145, -0.017, HY, 0.0640, 0.0555, 2.3)]                   # the heel rounds in at the top
# Iteration 15 (user: "that white thing is a hollow space"): Figure's black outsole rises into a frame round a
# window that goes right through the shoe. Black midsole up to RAIL (rising to the heel), a slanted slot through it.
RAIL = ((-0.09, 0.082), (0.12, 0.048))
# the slot: within 10 mm of the line (0.065, 0.040) - (-0.035, 0.053), sitting on the outsole; the rail above it is
# 5 mm thick at the front, 12 mm at the heel. A 1.5 cm deep pocket in each side, not a hole right through: through,
# it showed the ankle clevis plates as bars and read as three windows (iteration 15)
SLOT = [(0.0305, 0.0643, 0.0038), (0.0320, 0.0600, 0.0110), (0.0350, 0.0498, 0.0239), (0.0390, 0.0350, 0.0399),
        (0.0430, 0.0194, 0.0551), (0.0470, 0.0146, 0.0576), (0.0510, 0.0066, 0.0514), (0.0550, -0.0088, 0.0360),
        (0.0590, -0.0233, 0.0197), (0.0615, -0.0315, 0.0087), (0.0625, -0.0343, 0.0038)]
WINDOW = lambda side: [(z, cx, HY + side * 0.0615, a, 0.0195, 6.0) for z, cx, a in SLOT]
# the black midsole layer's pocket: 0.5 mm wider and 2 mm deeper, so its floor is not in the same plane as the
# upper's (coincident faces flickered in the render, iteration 17)
WINDOW_DEEP = lambda side: [(z, cx, HY + side * 0.0605, a + 0.0005, 0.0205, 6.0) for z, cx, a in SLOT]
link('L_foot', 1.1, [
    # iteration 14: toe spring (the toe lifts 2 cm over its last 4 cm, as Figure's does) and a rounded heel edge;
    # the contact box is the flat part that stands on the floor
    dict(loft('L sole', [(0.0, 0.040, HY, 0.120, 0.050, 2.6), (0.005, 0.0475, HY, 0.1325, 0.0545, 2.65),
                         (0.012, 0.0545, HY, 0.1415, 0.0570, 2.7), (0.020, 0.0565, HY, 0.1440, 0.0575, 2.75),
                         (0.030, 0.058, HY, 0.1435, 0.0575, 2.8)], SOLE, open=()),
         contact=dict(c=[0.040, HY, 0.015], size=[0.24, 0.10, 0.03])),
    # it carries the wireless charging coil (Figure 03 charges at 2 kW through its feet): Litz coil, ferrite, rectifier
    rbox('L foot plate', (0.055, HY, 0.017), (0.21, 0.07, 0.016), 0.004, add_mass=0.3, add_what='electronics'),
    loft('L shoe', SHOE, WHITE, open=('top',), clip=[ahead(0.058)]),
    # the collar rim rounded (user: "curve this edge"; Figure's heel counter rolls over into the opening)
    loft('L shoe heel', SHOE, WHITE, open=('top',), clip=[behind(0.062), rounded(below(*COLLAR), 0.008)]),
    loft('L shoe midsole', [(z, cx, cy, a + 0.0012, b + 0.0012, n) for z, cx, cy, a, b, n in SHOE[:8]], SOLE,
         'decal', open=(), clip=below(*RAIL)),
    cut(loft('L shoe window out', WINDOW(+1), WHITE, open=()), 'L shoe', 'L shoe heel'),
    cut(loft('L shoe window in', WINDOW(-1), WHITE, open=()), 'L shoe', 'L shoe heel'),
    cut(loft('L shoe window out rim', WINDOW_DEEP(+1), WHITE, open=()), 'L shoe midsole'),
    cut(loft('L shoe window in rim', WINDOW_DEEP(-1), WHITE, open=()), 'L shoe midsole'),
    rbox('L ankle clevis front', (0.02, HY, 0.06), (0.006, 0.04, 0.08), 0.002, BLACK, exposed=True),
    rbox('L ankle clevis back', (-0.02, HY, 0.06), (0.006, 0.04, 0.08), 0.002, BLACK, exposed=True),
    cut(cyl('L ankle clevis hole', (-0.02, HY, AZ), 'x', 0.010, 0.012), 'L ankle clevis back'),
    # iteration 34: where the roll rod pushes, on the pitch axis' line, 34 mm out (8 mm wide: inside the shoe's window pocket)
    rbox('L ankle roll pin', (ROLL_ROD_END[0], ROLL_ROD_END[1], (0.022 + AZ + 0.004) / 2), (0.014, 0.008, AZ + 0.004 - 0.022), 0.003),
])


# ---------------------------------------------------------------- real connections (iteration 40, user, 2026-11-05: "start real connections: screws, holes, bearings")

HW.setup(LINKS, JOINTS, ACT, motor_frame)


def knee_hardware():
    """Every race stopped both ways, every screw in reach in its turn. The build order is at the end: reports/joint_template.md explains it."""
    t = lambda y_mm: -y_mm                                       # y from the leg's centre, in mm, to t along the output (out is -y)
    HW.pattern('L knee hub', 'L_knee', 'hub', 'L_hip_yaw_link', t(-30.65), 12, clamps=['L_knee flange'], note='9 M5 through the cup floor into the drawn 6.5 mm')
    HW.pin_holes('L knee hub pin hole', 'L_knee', 'L_knee flange')
    for k, g in enumerate(x for x in list(LINKS['L_hip_yaw_link']['geoms']) if x.get('hw', {}).get('pattern') == 'L knee hub' and x['hw']['type'] == 'screw'):
        LINKS['L_hip_yaw_link']['geoms'].append(cut(cyl(f'L knee hub head way {k + 1}', (g['hw']['pos'][0], HY - 0.0367, g['hw']['pos'][2]), 'y', 0.00455, 0.0121), 'L_knee flange'))
    HW.pattern('L knee stator', 'L_knee', 'stator_front', 'L_knee_link', t(-29.6), 10, clamps=['L shank body'], note='10 M4 through the barrel plate into the drawn front 5 mm')
    for k, g in enumerate(x for x in list(LINKS['L_knee_link']['geoms']) if x.get('hw', {}).get('pattern') == 'L knee stator' and x['hw']['type'] == 'screw'):      # the waist's fillet is spot-faced where a head and key come
        LINKS['L_knee_link']['geoms'].append(cut(cyl(f'L knee stator spot face {k + 1}', (g['hw']['pos'][0], HY - 0.0361, g['hw']['pos'][2]), 'y', 0.0040, 0.013), 'L shank body'))
    # the bearings: inner race on the cup, stopped by a nut on the hub side and the cup's shoulder; outer race in the barrel, stopped by the lip and a threaded ring. Outer side the other way round
    HW.bearing('L knee inner bearing', 'CRBS 508', (0, HY + KNEE_IN_BEARING, KZ), 'y', 'L_hip_yaw_link', ['L_knee flange'], 'L_knee_link', ['L shank body'], 'L_knee', partner='L knee outer bearing')
    HW.bearing('L knee outer bearing', 'CRBS 508', (0, HY + KNEE_OUT_BEARING, KZ), 'y', 'L_knee_link', ['L knee rear cap'], 'L_hip_yaw_link', ['L knee outer cheek'], 'L_knee', partner='L knee inner bearing')
    HW.retainer('L knee inner nut', 'L_hip_yaw_link', (0, HY - 0.0257, KZ), 'y', 0.0281, KNEE_RACE_R - 0.00005, 0.0015, 'L knee inner bearing', kind='thin nut, M50 x 1, on the cup (made, not bought)')
    HW.retainer('L knee inner ring', 'L_knee_link', (0, HY - 0.03555, KZ), 'y', 0.03405, 0.0305, 0.002, 'L knee inner bearing', kind='threaded ring, M68 x 1, in the barrel\'s collar (made, not bought)')
    HW.retainer('L knee outer ring', 'L_hip_yaw_link', (0, HY + 0.0325, KZ), 'y', 0.03405, 0.0305, 0.0017, 'L knee outer bearing', kind='threaded ring, M68 x 1, in the cheek (made, not bought)')
    HW.retainer('L knee post washer', 'L_knee_link', (0, HY + 0.04225, KZ), 'y', 0.0265, 0.0125, 0.0017, 'L knee outer bearing', kind='washer plate on the post (made, not bought)', exposed=True)
    # the rear cap on the barrel: M3 countersunk, flush, under the cheek
    th = [math.radians(a) for a in KNEE_RIM_ANGLES]
    HW.bolt_group('L knee rim', 'L_knee_link', 'M3', 10, 'iso10642', [(KNEE_RIM_R * math.cos(a), HY + 0.0314, KZ + KNEE_RIM_R * math.sin(a)) for a in th], 'y', -1, ['L knee rear cap'], 'L shank body', 2.95,
                  load=dict(V=0.0, Fa=1614.0, T=0.0, Mt=0.0, r=0.063), note='the side force and the post\'s tilt; the cap takes no torque')
    # the outer cheek on the fork: M5 from outside, 2 d into the crown
    xz = KNEE_CHEEK_SCREWS
    HW.bolt_group('L knee cheek top', 'L_hip_yaw_link', 'M5', 14, 'iso7380', [(x, HY + 0.0342, KZ + z) for x, z in xz[2:]], 'y', -1, ['L knee outer cheek'], 'L thigh fork', 2.55,
                  load=dict(V=1287.0, Fa=0.0, T=0.0, Mt=80.0, r=0.025), note='half of the outer support, 2,574 N, and half of the moment: the side force 1,614 N at 90 mm over the row, and the bearing 5.75 mm off the seam: an upper bound')
    HW.bolt_group('L knee cheek', 'L_hip_yaw_link', 'M5', 16, 'iso7380', [(x, HY + 0.0376, KZ + z) for x, z in xz[:2]], 'y', -1, ['L knee outer cheek'], 'L thigh fork', 5.95,
                  load=dict(V=1287.0, Fa=0.0, T=0.0, Mt=80.0, r=0.025), note='the other half, as the top pair')
    # the washer on the post
    HW.bolt_group('L knee post', 'L_knee_link', 'M4', 10, 'iso7380', [(0.0185 * math.cos(a), HY + 0.0431, KZ + 0.0185 * math.sin(a)) for a in (0.785, 2.356, 3.927, 5.498)], 'y', -1, ['L knee post washer'], 'L knee rear cap', 1.6,
                  load=dict(V=0.0, Fa=1614.0, T=0.0, Mt=0.0, r=0.0185), note='the side force, on the inner race')
    # the hubs' two screws each, into their feet: the head inside the hub, a key through a hole under the white mark
    for sd, sg in (('in', -1), ('out', 1)):
        pos = [(-0.001, HY + sg * 0.046, KZ - 0.0735), (0.0125, HY + sg * 0.046, KZ - 0.0735)]
        HW.bolt_group(f'L knee hub {sd} foot', 'L_knee_link', 'M4', 12, 'iso4762', pos, 'y', -sg, [f'L knee hub {sd}'], 'L shank body', 2.8,
                      load=dict(V=30.0, Fa=30.0, T=0.0, Mt=0.0, r=0.01), note='a cover: its own weight at 10 g')
        for k, p in enumerate(pos):
            LINKS['L_knee_link']['geoms'].append(cut(cyl(f'L knee hub {sd} key hole {k + 1}', (p[0], HY + sg * 0.0562, p[2]), 'y', 0.0042, 0.0108), f'L knee hub {sd}'))
    LINKS['L_knee_link']['geoms'].append(dict(cut(cyl('L knee hub out post hole', (0, HY + 0.046, KZ), 'y', 0.023, 0.008), 'L knee hub out'), internal=True))
    # the back panel
    HW.bolt_group('L shank panel', 'L_knee_link', 'M3', 10, 'iso10642', [(KNEE_XS - 0.003, HY + sg * 0.036, z) for sg, z in KNEE_PANEL_SCREWS], 'x', 1, ['L shank back panel'], 'L shank body', 3.0,
                  load=dict(V=200.0, Fa=0.0, T=0.0, Mt=0.0, r=0.0375), note='shear flow in the closed section, a hand figure from the motor\'s torque on the section')
    LINKS['L_knee_link']['geoms'].append(dict(motor('L ankle pitch relief', 'RS03', (CALF_X, HY, PITCH_Z), 'y', -1, MOTOR_CLOCK.get('L_ankle_pitch', 0.0), r=ACT['RS03']['depth'] / 2 * 1.012), role='cut', targets=['L shank back panel'], color='#000000', internal=True))      # the RS03's ears: machined into the panel's inner face
    for k, (sg, z) in enumerate(KNEE_PANEL_SCREWS):                # a way in for head and key, through the wall behind the pad
        LINKS['L_knee_link']['geoms'].append(dict(cut(cyl(f'L shank panel head way {k + 1}', (KNEE_XS - 0.009, HY + sg * 0.036, z), 'x', 0.00365, 0.012), 'L shank back panel'), internal=True))
    # the thigh's top: the flange on the hip yaw's hub, and the tube's collar on the flange
    HW.pattern('L hip yaw hub', 'L_hip_yaw', 'hub', 'L_hip_yaw_link', 28.3 + 4.5, 10, clamps=['L_hip_yaw flange'], counterbore_to=28.3 + 9.5, note='6 M4 into the drawn 6 mm')
    a6 = [math.radians(30 + 60 * k) for k in range(6)]
    HW.bolt_group('L thigh collar', 'L_hip_yaw_link', 'M4', 12, 'iso4762', [(0.039 * math.cos(a), HY + 0.039 * math.sin(a), THIGH_TOP - 0.004) for a in a6], 'z', 1, ['L thigh fork'], 'L_hip_yaw flange', 4.0,
                  load=dict(V=810.0, Fa=3400.0, T=60.0, Mt=120.0, r=0.039), note='the whole leg, and the yaw motor\'s and knee motor\'s peak')
    # ---- iteration 42: hip, waist, chest
    HW.tab_notch('L hip pitch notch', 'L_hip_pitch', 'pelvis frame', 'pelvis')
    HW.tab_notch('waist notch', 'waist_yaw', 'pelvis frame', 'pelvis')
    HW.tab_notch('L hip roll notch', 'L_hip_roll', 'L hip carrier', 'L_hip_pitch_link')
    HW.tab_notch('L shoulder stator notch', 'L_shoulder_pitch', 'chest frame', 'torso')
    LINKS['L_knee_link']['geoms'].append(dict(motor('L ankle roll relief', 'RS06', (CALF_X, HY, ROLL_Z), 'y', +1, MOTOR_CLOCK.get('L_ankle_roll', 0.0), r=ACT['RS06']['depth'] / 2 * 1.012), role='cut', targets=['L shank back panel'], color='#000000', internal=True))
    # the pelvis: motors on their rear faces, from inside the bar; rings on the barrels; cups on the hubs
    HW.pattern('L hip pitch stator', 'L_hip_pitch', 'stator_rear', 'pelvis', -19.35, 10, clamps=['pelvis frame'], exposed=True, note='10 M4 through the bar\'s plate into the drawn rear 6 mm')
    HW.pattern('waist stator', 'waist_yaw', 'stator_rear', 'pelvis', -19.35, 10, clamps=['pelvis frame'], note='10 M4 up through the housing\'s plate')
    HW.pattern('L hip roll stator', 'L_hip_roll', 'stator_rear', 'L_hip_pitch_link', -19.35, 10, clamps=['L hip carrier'], exposed=True, note='10 M4 from behind')
    HW.pattern('L hip pitch hub', 'L_hip_pitch', 'hub', 'L_hip_pitch_link', 30.65, 12, clamps=['L_hip_pitch flange'], exposed=True, note='9 M5 through the cup')
    HW.pattern('waist hub', 'waist_yaw', 'hub', 'torso', 30.65, 12, clamps=['waist_yaw flange'], note='9 M5 through the cup')
    HW.pattern('L hip roll hub', 'L_hip_roll', 'hub', 'L_hip_roll_link', 30.65, 12, clamps=['L_hip_roll flange'], exposed=True, note='9 M5 through the boss, heads in the 9 mm behind the pitch motor')
    for j, ln, part in (('L_hip_pitch', 'L_hip_pitch_link', 'L_hip_pitch flange'), ('waist_yaw', 'torso', 'waist_yaw flange'), ('L_hip_roll', 'L_hip_roll_link', 'L_hip_roll flange')):
        HW.pin_holes(f"{part} pin hole", j, part)
    HW.output_stack('L hip pitch', 'L_hip_pitch', 'L_hip_pitch flange', 'L_hip_pitch_link', 'pelvis', 'pelvis frame', 'L hip ring', (0, PITCH_ACT_Y, HZ), 'y', HOUSING_OD, HIP_RIM_ANGLES, RIM_R, RIM_LOBE, BLACK, thrust=1614.0)
    HW.output_stack('waist', 'waist_yaw', 'waist_yaw flange', 'torso', 'pelvis', 'pelvis frame', 'waist ring', (0, 0, WY), 'z', HOUSING_OD, WAIST_RIM_ANGLES, RIM_R, RIM_LOBE, BLACK, thrust=1000.0)
    for nm, part, ln, c, ax in (('L hip pitch hub', 'L_hip_pitch flange', 'L_hip_pitch_link', (0, PITCH_ACT_Y, HZ), 'y'), ('waist hub', 'waist_yaw flange', 'torso', (0, 0, WY), 'z'), ('L hip roll hub', 'L_hip_roll flange', 'L_hip_roll_link', (-0.10, HY, HZ), 'x')):
        for k, g in enumerate(x for x in list(LINKS[ln]['geoms']) if x.get('hw', {}).get('pattern') == nm and x['hw']['type'] == 'screw'):
            p = list(g['hw']['pos'])
            i = 'xyz'.index(ax)
            p[i] = c[i] + 0.0367
            LINKS[ln]['geoms'].append(cut(cyl(f'{nm} head way {k + 1}', p, ax, 0.00455, 0.0121), part, *(['L hip yoke'] if 'roll' in nm else [])))
    # keys: a way through the bar's wall for screws its curve would hide
    for pat, ln, part, ax in (('waist stator', 'pelvis', 'pelvis frame', 'z'), ('L hip pitch stator', 'pelvis', 'pelvis frame', 'y'), ('L shoulder stator', 'torso', 'chest frame', 'y')):
        i = 'xyz'.index(ax)
        for k, g in enumerate(x for x in list(LINKS[ln]['geoms']) if x.get('hw', {}).get('pattern') == pat and x['hw']['type'] == 'screw'):
            p = list(g['hw']['pos'])
            p[i] -= g['hw']['dir'][i] * 0.030
            LINKS[ln]['geoms'].append(cut(cyl(f'{pat} way {k + 1}', p, ax, 0.0041, 0.06), part))
    # the carrier's eye on the cup: it's one with the cup when made; the yaw motor's front face
    HW.pattern('L hip yaw stator', 'L_hip_yaw', 'stator_front', 'L_hip_roll_link', 25.8 + 5.2, 10, clamps=['L hip yaw ring'], note='8 M4 up through the ring')
    pos = [(0.0562 * math.cos(math.radians(a)), HY + 0.0562 * math.sin(math.radians(a)), 0.7890) for a in (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)]
    HW.bolt_group('L hip yaw rim', 'L_hip_roll_link', 'M3', 12, 'iso10642', pos, 'z', 1, ['L hip yaw ring'], 'L hip yoke', 5.2, load=dict(V=810.0, Fa=810.0, T=60.0, Mt=60.0, r=0.0555), note='the leg, at 810 N, from the motor\'s face into the housing, and the yaw motor\'s peak')
    # the chest: both shoulder pitch motors on rear faces, the neck's on its front
    HW.pattern('L shoulder stator', 'L_shoulder_pitch', 'stator_rear', 'torso', -19.35, 10, clamps=['chest frame'], note='10 M4 through the bar\'s plate')
    HW.pattern('neck stator', 'neck_yaw', 'stator_front', 'torso', 24.5 + 3.5, 8, clamps=['chest frame'], note='6 M3 down through the bar\'s land')
    for k, g in enumerate(x for x in list(LINKS['torso']['geoms']) if x.get('hw', {}).get('pattern') == 'neck stator' and x['hw']['type'] == 'screw'):
        LINKS['torso']['geoms'].append(cut(cyl(f'neck stator way {k + 1}', (g['hw']['pos'][0], g['hw']['pos'][1], g['hw']['pos'][2] + 0.008), 'z', 0.0037, 0.0158), 'chest frame'))
    # the build order, each step checked by group 'assembly' (tools/hw_checks.py)
    py, my, up, back = (0, 1, 0), (0, -1, 0), (0, 0, 1), (-1, 0, 0)
    HW.step(['L_knee actuator (buy) RS04'], path=py, distance=0.08, screws=['L knee stator'], why='the motor into the barrel from the open end, hub first; 10 M4 from the inboard side, nothing there yet')
    HW.step(['L knee rear cap'], path=py, distance=0.03, screws=['L knee rim'], why='the cap, skirt to the motor\'s back, post out; 6 flush M3')
    HW.step(['L thigh fork', 'L_knee flange', 'L knee inner bearing', 'L knee inner nut', 'L knee inner ring'], path=my, distance=0.03, screws=['L knee hub'], why='fork and cup, bearing and nut on the cup, ring loose behind them, from inboard: the slot passes over the barrel, the cup into the bore. 9 M5 through the eye; the ring turned through the opening with a pin spanner')
    HW.step(['L knee outer cheek', 'L knee outer bearing', 'L knee outer ring', 'L knee post washer'], path=py, distance=0.02, screws=['L knee cheek', 'L knee cheek top', 'L knee post'], why='the outer cheek, bearing and ring in it, from outboard onto the post. 4 M5 into the crown, then the washer')
    HW.step(['L knee hub in'], path=my, distance=0.02, screws=['L knee hub in foot'], why='the hubs last: 2 M4 each, a key through the hole under the mark')
    HW.step(['L knee hub out'], path=py, distance=0.02, screws=['L knee hub out foot'], why='')
    HW.step(['L shank back panel'], path=back, distance=0.02, screws=['L shank panel'], why='the ankle motors and rods are in from behind by now (their own round); the panel closes the section with 6 flush M3')
    # the pelvis
    HW.step(['L_hip_pitch actuator (buy) RS04', 'R_hip_pitch actuator (buy) RS04'], path=None, distance=0.08, screws=['L hip pitch stator', 'R hip pitch stator'], why='the hip motors into the bar from outboard, 10 M4 from inside the bar, in through the window underneath')
    HW.step(['waist_yaw actuator (buy) RS04'], path=up, distance=0.08, screws=['waist stator'], why='the waist motor from above, 10 M4 up from inside the bar')
    HW.step(['L hip ring', 'R hip ring'], screws=['L hip pitch rim', 'R hip pitch rim'], why='the hip ring on the barrel: 6 flush M3')
    HW.step(['waist ring'], path=up, distance=0.02, screws=['waist rim'], why='the same on the waist')
    # the chest on the waist, then what it carries
    HW.step(['chest frame', 'waist_yaw flange', 'waist bearing', 'waist nut', 'waist retaining ring'], path=up, distance=0.03, screws=['waist hub'], why='the chest frame and cup from above, 9 M5 down through the eye, before anything is in it')
    HW.step(['L_shoulder_pitch actuator (buy) RS04', 'R_shoulder_pitch actuator (buy) RS04'], screws=['L shoulder stator', 'R shoulder stator'], why='each shoulder motor from outboard, 10 M4 from inside the bar')
    HW.step(['neck_yaw actuator (buy) RS00'], screws=['neck stator'], why='the neck motor up into the bar, through its top: 6 M3 down onto its face')
    HW.step(['battery', 'battery lower', 'compute'], why='pack and computer last, from behind and from the front, on their shelves. No path: they slide on the 0.2 mm the seats are drawn into them')
    # the leg's hip, on the bench
    HW.step(['L_hip_roll actuator (buy) RS04'], path=(1, 0, 0), distance=0.08, screws=['L hip roll stator'], bench=['L hip carrier'], why='on the bench: the roll motor into its barrel from the front, 10 M4 from behind')
    HW.step(['L hip yoke', 'L_hip_roll flange'], path=(1, 0, 0), distance=0.03, screws=['L hip roll hub'], bench=['L hip carrier', 'L_hip_roll actuator (buy) RS04'], why='the yoke onto the roll hub, 9 M5 from the front: there is no pitch motor on the bench')
    HW.step(['L_hip_yaw actuator (buy) RS03'], path=(0, 0, -1), distance=0.08, bench=['L hip yoke', 'L_hip_roll flange', 'L hip carrier', 'L_hip_roll actuator (buy) RS04'], why='the yaw motor up into its housing')
    HW.step(['L hip yaw ring'], path=(0, 0, -1), distance=0.02, screws=['L hip yaw stator', 'L hip yaw rim'], bench=['L hip yoke', 'L_hip_roll flange', 'L hip carrier', 'L_hip_roll actuator (buy) RS04', 'L_hip_yaw actuator (buy) RS03'], why='the ring on: 8 M4 into the motor, 4 M3 into the housing, from below')
    HW.step(['L hip carrier', 'L_hip_pitch flange', 'L hip pitch bearing', 'L hip pitch nut', 'L hip pitch retaining ring'], path=py, distance=0.03, screws=['L hip pitch hub'], with_parts=['L_hip_roll actuator (buy) RS04', 'L hip yoke', 'L_hip_roll flange', 'L hip yaw ring', 'L_hip_yaw actuator (buy) RS03'], with_screws=['L hip roll stator', 'L hip roll hub', 'L hip yaw stator', 'L hip yaw rim'], why='the whole hip onto the pitch hub from outboard: 9 M5 through the eye')
    HW.step(['L_hip_yaw flange'], path=(0, 0, -1), distance=0.02, screws=['L hip yaw hub'], bench=['L_hip_yaw actuator (buy) RS03', 'L hip yaw ring'], why='the flange on the yaw hub from below')
    HW.step([], screws=['L thigh collar'], why='the built leg up to the flange: 6 M4 from below, up beside the tube')


knee_hardware()
# iteration 41: bearings and every fastener in these two links are parts now, not the 6 % allowance's: it stays for their cables, thermal paste and small parts (a hand figure)
LINKS['L_knee_link']['wiring'] = LINKS['L_hip_yaw_link']['wiring'] = 0.03
for _ln in ('pelvis', 'torso', 'L_hip_pitch_link', 'L_hip_roll_link'):
    LINKS[_ln]['wiring'] = 0.03


# ---------------------------------------------------------------- mirror, actuators, flanges

def mirror_geom(g):
    g = copy.deepcopy(g)
    g['name'] = g['name'].replace('L ', 'R ', 1) if g['name'].startswith('L ') else g['name']
    if 'targets' in g:
        g['targets'] = [mirror_part(t) for t in g['targets']]
    if 'hw' in g:
        g['hw'] = mirror_record(g['hw'])
    for k in ('c', 'p0', 'p1'):
        if k in g:
            g[k][1] = -g[k][1]
    if 'contact' in g:
        g['contact']['c'][1] = -g['contact']['c'][1]
    if 'keep' in g:
        g['keep']['c'][1] = -g['keep']['c'][1]
    for c in ([g['clip']] if isinstance(g.get('clip'), dict) else g.get('clip') or []):
        c['p'][1], c['n'][1] = -c['p'][1], -c['n'][1]
    if g['kind'] == 'cyl' and not isinstance(g['axis'], str):
        g['axis'][1] = -g['axis'][1]
    if g.get('outline'):                                  # a mirror image of the real shape. Is it buildable? check 'pattern' asks whether the same motor, turned, has it
        for k in ('x', 'y', 'out'):
            g['outline'][k][1] = -g['outline'][k][1]
    if g['kind'] == 'revolve' and g['axis'] == 'y':
        g['profile'] = [[-t, r] for t, r in reversed(g['profile'])]
    if g['kind'] == 'sdf':                                # its tree stays as it is: the mesh and the inside test reflect it
        g['flip_y'] = not g.get('flip_y', False)
    if g['kind'] == 'prism':
        g['pts'] = [[-y, z] for y, z in g['pts']]
    if g['kind'] == 'loft':
        for s in g['sections']:
            s['cy'] = -s['cy']
    return g


def mirror_name(n):
    return 'R_' + n[2:] if n.startswith('L_') else n


def mirror_part(n):
    """'L knee post' and 'L_knee flange' both have a right side; 'motor:L_knee ...' too."""
    if n.startswith('motor:'):
        return 'motor:' + mirror_part(n[6:])
    return 'R ' + n[2:] if n.startswith('L ') else 'R_' + n[2:] if n.startswith('L_') else n


def mirror_record(h):
    h = copy.deepcopy(h)
    for k, x in h.items():
        if k in ('pos', 'dir', 'face'):
            x[1] = -x[1]
        elif isinstance(x, str):
            h[k] = mirror_part(x)
        elif isinstance(x, list) and x and isinstance(x[0], str):
            h[k] = [mirror_part(p) for p in x]
    return h


def build():
    links, joints = {}, []
    for n, l in LINKS.items():
        links[n] = l
        if n.startswith('L_'):
            m = copy.deepcopy(l)
            m['name'] = mirror_name(n)
            m['geoms'] = [mirror_geom(g) for g in l['geoms']]
            links[m['name']] = m
    # the pelvis carries both hip pods: mirror its left-side geoms in place
    p = links['pelvis']
    p['geoms'] += [mirror_geom(g) for g in p['geoms'] if g['name'].startswith('L ')]
    t = links['torso']                                   # the chest bar's bolts, for both shoulders
    t['geoms'] += [mirror_geom(g) for g in t['geoms'] if g['name'].startswith('L shoulder stator')]
    for j in JOINTS:
        joints.append(j)
        if j['name'].startswith('L_'):
            m = copy.deepcopy(j)
            m['name'], m['parent'], m['child'] = mirror_name(j['name']), mirror_name(j['parent']), mirror_name(j['child'])
            m['origin'][1] = -m['origin'][1]
            m['act_c'][1] = -m['act_c'][1]
            if m['axis'] in ('x', 'z'):                    # roll and yaw limits flip with the mirror
                m['lower'], m['upper'] = -j['upper'], -j['lower']
            if j['axis'] == 'y':                          # a flange facing +y on the left faces -y on the right
                m['out'] = -j['out']
            joints.append(m)

    # actuator stators into the parent link, flanges onto the child. Remote drives (knee linkage) get the stator
    # only; the ankle's two RS03 are drawn in the calf directly (their joints have act=None).
    for j in joints:
        if not j['act']:
            continue
        a = ACT[j['act']]
        color = BLACK if j['exposed'] is True else '#3A3D42'
        stator, output = (j['child'], j['parent']) if j.get('stator_on_child') else (j['parent'], j['child'])
        left = next((x for x in joints if x['name'] == 'L_' + j['name'][2:]), j) if j['name'].startswith('R_') else j
        m = motor(f"{j['name']} actuator (buy) {j['act']}", j['act'], left['act_c'], j['axis'], left['out'], left['clock'], color)
        if left is not j:                                 # the right side's motor is the left's, mirrored
            m = mirror_geom(m)
        links[stator]['geoms'].append(m)
        if j['remote']:
            continue
        ax = {'x': 0, 'y': 1, 'z': 2}[j['axis']]
        c = list(j['act_c'])
        # the real hub (RS04's stands 3 mm under its pin tips, so a flange placed from the old cylinder's end floated on the pins): from the hub's face, 0.2 mm in so
        # they read as seated, to where the flange's outer face always was. Pin holes are cut by the hardware that seats it
        t_hub = OUTLINES[j['act']]['t_hub_face'] / 1000
        t_out = a['h'] / 2 + FLANGE_T - 0.001
        c[ax] += j['out'] * (t_hub - 0.0002 + t_out) / 2
        own = j.pop('flange', None)
        if own:                                           # a real part stands for the flange
            own = dict(mirror_geom(dict(own, name=f"{j['name']} flange")) if left is not j else dict(own, name=f"{j['name']} flange"), name=f"{j['name']} flange")
            links[output]['geoms'].append(own)
            continue
        # iteration 29: the wrists' flanges black (still 7075): bare metal showed as a bright chip in the gaps when bent
        links[output]['geoms'].append(cyl(f"{j['name']} flange", c, j['axis'], 0.6 * a['r'], t_out - t_hub + 0.0002,
                                              BLACK if 'wrist' in j['name'] else ALU, 'struct'))
    for l in links.values():                              # every part's material (tools/mass_model.py)
        for g in l['geoms']:
            if g['role'] != 'cut':
                g['material'] = material_of(g)
    rec = HW.records()
    for k in ('patterns', 'bearings', 'singles', 'groups'):         # the right side's, as the geoms are
        rec[k] = rec[k] + [mirror_record(r) for r in rec[k] if r['name'].startswith('L ')]
    return dict(frame='metres; X forward, Y left, Z up; floor z=0; zero pose standing, arms down, palms in',
                actuators=ACT, materials=MATERIALS, links=links, joints=joints, root='pelvis', hardware=rec)


if __name__ == '__main__':
    robot = build()
    out = os.path.join(HERE, 'robot.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(robot, f, indent=1)
    print(f"{len(robot['links'])} links, {len(robot['joints'])} joints -> {out} (masses: python -I tools/mass_model.py)")
