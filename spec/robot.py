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
  decal     cosmetic surface detail sunk into a skin (camera ports, logo, light bar). Must touch its link.
  struct    rigid internal part (7075 bracket, fork, battery). Must not overlap rigid parts of any other link.
  actuator  bought actuator, stator side. Rigid. Lives in the joint's PARENT link, centred on the joint axis.
"""
import copy
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ACT = json.load(open(os.path.join(HERE, 'actuators.json'), encoding='utf-8'))

GREY, BLACK, GLOSS, WHITE, SOLE = '#BCC0C4', '#1B1C1F', '#08090B', '#EDEDED', '#2B2B2E'
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
    'hand': dict(density=1500, note="the hand's motors, tendons and palm: explicit mass"),
}


def material_of(g):
    """Default material by role and colour (a geom may name its own)."""
    if g.get('material'):
        return g['material']
    role, col = g['role'], g['color']
    if role in ('struct', 'linkage'):
        return 'cells' if col == CELL else 'al7075'
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


def link(name, budget, geoms, note=''):
    """budget: the hand estimate of the link's mass (kg), kept for comparison; its real mass, centre of mass and
    inertia come from its parts' materials (tools/mass_model.py)."""
    LINKS[name] = dict(name=name, budget=budget, geoms=list(geoms), note=note)


def joint(name, parent, child, origin, axis, lim, act=None, act_c=None, out=+1, exposed=False, remote=False,
          capacity=None):
    """act: actuator model whose stator is added to `parent`, centred at act_c (default: the joint origin) and
    aligned with `axis`. out = +1/-1: which end of the actuator faces the child (where its flange goes).
    capacity: what drives the joint, {'model', 'count', 'ratio'} (default: one `act`, ratio 1)."""
    j = dict(name=name, parent=parent, child=child, origin=list(origin), axis=axis, lower=lim[0], upper=lim[1],
             act=act, act_c=list(act_c or origin), out=out, exposed=exposed, remote=remote,
             capacity=capacity or (dict(model=act, count=1, ratio=1.0) if act else None))
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
link('pelvis', 6.5, [
    # centre block; its underside is the arch between the legs (high enough for a hip flexed 60 degrees)
    # (iteration 4: shallower at the back, clear of the forward-facing hip roll outputs; the lobes cover the rest)
    loft('pelvis', [(HZ - 0.01, 0.005, 0, 0.065, 0.058, 3.0), (HZ + 0.03, 0.005, 0, 0.075, 0.085, 3.2),
                    (HZ + 0.088, 0.005, 0, 0.072, 0.08, 3.0)], GLOSS, open=()),
    # its ends show by the lobes; it carries the power board and BMS (0.35 kg)
    rbox('pelvis frame', (0, 0, HZ + 0.03), (0.07, 0.15, 0.07), 0.008, exposed=True, add_mass=0.35,
         add_what='electronics'),
    # clearance (iteration 17, every joint swept with the covers across it): the waist column turns in a round
    # opening, 4 mm clear
    cut(cyl('pelvis waist opening', (0, 0, 1.019), 'z', 0.072, 0.042), 'pelvis'),   # from 0.998, over the frame
    # iteration 19: its side walls cut into the hip pitch motors; round openings 3 mm clear of them (mirrored)
    cut(cyl('L pelvis hip opening', (0, 0.085, HZ), 'y', 0.063, 0.05), 'pelvis'),
], 'hip pitch actuators, waist yaw actuator, PDB')

# Waist: yaw only, like Asimov 1. A roll/pitch waist (A3, H2) cannot hide under a one-piece fabric torso: the
# hem has to clear the hips by ~5 cm for 20 degrees of tilt (2026-10-31 sweep). Lifting uses hips and knees.
joint('waist_yaw', 'pelvis', 'torso', (0, 0, WY), 'z', (-2.6, 2.6), 'RS04')

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
         (SZ + 0.052, -0.012, 0, 0.094, 0.150, 3.2),
         (SZ + 0.075, -0.016, 0, 0.082, 0.140, 3.0),
         (SZ + 0.093, -0.018, 0, 0.068, 0.106, 2.6),
         (SZ + 0.103, -0.020, 0, 0.057, 0.074, 2.3),
         (SZ + 0.108, -0.021, 0, 0.051, 0.056, 2.1)]
BATT_TOP = SZ - 0.065           # level with the shoulder pitch actuators' bottoms (they sit outboard of it)
link('torso', 19.96, [
    span('waist hub', WY + 0.03, HEM - 0.025, sx=0.10, sy=0.10, f=0.008),
    # round (iteration 17): turning, it sweeps nothing; 5 mm round the waist actuator inside its wall
    # open below (a closed end crossed the actuator), reaching down into the pelvis opening to 2 mm over the frame
    loft('waist column', [(0.997, 0, 0, 0.068, 0.068, 2.0), (HEM - 0.03, 0, 0, 0.068, 0.068, 2.0)], GLOSS,
         open=('bottom',)),
    loft('torso rim', [(HEM - 0.035, 0, 0, 0.080, 0.094, 2.6), (HEM - 0.015, 0, 0, 0.095, 0.116, 2.9),
                       (HEM + 0.005, 0, 0, 0.1005, 0.1245, 3.0)], GLOSS, open=('top',)),
    loft('torso', TORSO),
    # Figure 03's pack is 2.3 kWh (figure.ai F.03 battery, 5 h at peak); at 175 Wh/kg for a cell pack with BMS and
    # case, 13.1 kg (it was 12.1 kg, 2 kWh: the hand mass budget's balance)
    span('battery', BATT_TOP - 0.13, BATT_TOP, x=-0.03, sx=0.15, sy=0.17, f=0.01, color=CELL, mass=13.1),
    # 7075 frame: hub -> spine -> battery (stressed enclosure) -> shoulder beam -> neck post
    span('spine', HEM - 0.035, BATT_TOP - 0.12, sx=0.06, sy=0.06, f=0.006),
    rbox('shoulder beam', (-0.02, 0, SZ - 0.045), (0.05, 0.20, 0.05), 0.006),
    span('neck post', SZ - 0.035, NZ - 0.02, x=NX, f=0.006),
    # compute (a Jetson-class module, carrier and heat sink), on the shoulder beam's front, between the shoulder
    # actuators
    rbox('compute', (0.034, 0, SZ - 0.035), (0.06, 0.14, 0.05), 0.004, '#2A2C30', mass=1.6, material='electronics'),
    rbox('chest light', (front_x(TORSO, 0, SZ - 0.01) + 0.0006 - 0.002, 0, SZ - 0.01), (0.004, 0.04, 0.006),
         0.0015, ACCENT, 'decal'),
    # our mark on the chest, below the light bar (dark grey on the grey knit); not Figure's logo
    *tri_mark('chest mark', TORSO, SZ - 0.055, 0.034, '#34373B'),
    # clearance (iteration 17): openings the neck and the shoulder caps turn in, 3-5 mm clear of them
    # iteration 18: the neck opening is 3 mm round the neck tilted through its range (it tilts at its base now),
    # computed from the swept neck; the ring round the neck is 9 mm at the collar
    cut(revolve('torso neck opening', 'z', (NX, 0, 0),
                [(1.4725, 0.0), (1.4727, 0.0386), (1.475, 0.0401), (1.48, 0.0425), (1.485, 0.0449), (1.49, 0.0472),
                 (1.495, 0.0503), (1.50, 0.0539), (1.505, 0.0576), (1.5052, 0.0)]), 'torso'),
    cut(revolve('L torso shoulder opening', 'y', (0, 0, SZ), [(0.118, 0.0), (0.1182, 0.0615), (0.165, 0.0615),
                                                            (0.1652, 0.0)]), 'torso'),
    cut(revolve('R torso shoulder opening', 'y', (0, 0, SZ), [(-0.1652, 0.0), (-0.165, 0.0615), (-0.1182, 0.0615),
                                                            (-0.118, 0.0)]), 'torso'),
], 'battery 13.1 kg (2.3 kWh, as Figure 03), compute, shoulder pitch and neck yaw actuators')

joint('neck_yaw', 'torso', 'neck_yaw_link', (NX, 0, NZ), 'z', (-1.4, 1.4), 'RS00', out=+1)
link('neck_yaw_link', 0.45, [
    # iteration 18: only a post from the yaw flange up 2.5 mm into the pitch actuator; slim, because the neck's
    # base tilts round it
    span('neck pitch post', NZ + 0.0295, NP - 0.026, x=NX, sx=0.02, sy=0.02, f=0.003),
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
       [(NP + 0.015, NX, 0, NECK_R, NECK_R, 2.0), (1.53, -0.012, 0, 0.045, 0.047, 2.0),
        (1.575, -0.010, 0, 0.050, 0.048, 2.0)]
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
# through the back of the jaw, leaning back over the crown: only a crescent of hood shows at the back (iteration 7)
VISOR_EDGE = ((0.0, 0, CHIN + 0.008), (0.915, 0, 0.404))
link('head', 1.95, [
    loft('head', HEAD, GREY, open=()),
    loft('visor', VISOR, GLOSS, 'decal', open=(), clip=VISOR_EDGE),
    *camera_port('brow camera', VISOR, CHIN + 0.143, 0.0085),
    *camera_port('chin camera', VISOR, CHIN + 0.025, 0.0075),
    *tri_mark('face mark', VISOR, CHIN + 0.083, 0.03, '#F4F6F8'),
    loft('neck', NECK),
    # cameras, mics, the face display: behind the visor, on the head frame
    rbox('head electronics', (0.035, 0, CHIN + 0.09), (0.07, 0.08, 0.07), 0.006, '#2A2C30', mass=0.3,
         material='electronics'),
    # frame: from the pitch flange (beside the pitch actuator's +y face) up into the head
    rbox('head frame', (NX, 0.033, NP + 0.05), (0.03, 0.006, 0.11), 0.002),
], 'cameras, mics')

# ---------------------------------------------------------------- left arm (mirrored below)

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
ELBOW_W = 0.0405                          # the neck's half width, as Figure's narrow neck between the pads
FOLD = ((-0.1, EZ + 0.0544 - 0.03425), (0.1, EZ + 0.0544 + 0.03425))


def elbow_ray(deg, r=0.1):
    """Side-view point (x, z) at angle deg about the elbow axis."""
    import math
    return r * math.cos(math.radians(deg)), EZ + r * math.sin(math.radians(deg))


# exposed (iteration 17): the actuator's face shows dark in the gap round the cap, in the torso's side opening
joint('L_shoulder_pitch', 'torso', 'L_shoulder_pitch_link', (0, 0.115, SZ), 'y', (-3.0, 1.0), 'RS04', out=+1,
      exposed=True)
link('L_shoulder_pitch_link', 0.99, [
    # iteration 9 (user review): a deltoid, not a ball: round about the pitch axis (so it never sweeps), full from
    # the torso side, flat on top, rounding off at the outer end; just big enough for the roll actuator inside.
    # Iteration 10: with the RS06 roll actuator it is only 7 mm proud of the upper arm.
    # Iteration 11 (still read as a drum): no waist at the inner end (it starts at full radius, hidden in the
    # torso's side), as deep as the new upper arm, so from the side the arm runs up into it in one line
    # iteration 17: starts 7 mm past the pitch actuator's face (it overlapped it), in the torso's side opening
    # iteration 18 (shoulder roll swept with the covers): the outer end is a hemisphere about the shoulder centre,
    # where the roll axis crosses the pitch axis, so the arm's top can turn round it
    revolve('L shoulder cap', 'y', (0, 0, SZ),
            [(0.150, 0.0), (0.1502, 0.054), (0.156, 0.0565), (0.172, 0.0575), (SY, CAP_R)] +
            [(SY + d, (CAP_R ** 2 - d ** 2) ** 0.5) for d in (0.01, 0.02, 0.03, 0.04, 0.047, 0.052, 0.0555)] +
            [(SY + CAP_R, 0.0)]),
    rbox('L shoulder bracket', (0, 0.155, SZ), (0.036, 0.018, 0.036), 0.004, exposed=True),
    # the roll output's axle leaves through the back of the cap on the roll axis (a hole there never sweeps)
    cut(cyl('L shoulder cap hole', (-0.0575, SY, SZ), 'x', 0.016, 0.014), 'L shoulder cap'),
])
# shoulder roll: RS06 (88 mm) rather than RS03 (106 mm) so the shoulder cap can be small (user, 2026-10-31: smaller
# actuators at the cost of holding capacity). An arm held straight out sideways is a short move, not a pose to hold.
# iteration 18: 150 deg up (2.8 swept the arm's outer top into the cap's top at full height; AgiBot A3 stops at 150,
# Asimov 1 at 90); output on the back face, so the bracket runs behind the cap instead of through it
joint('L_shoulder_roll', 'L_shoulder_pitch_link', 'L_shoulder_roll_link', (0, SY, SZ), 'x', (-0.05, 2.6), 'RS06',
      out=-1)
# Arms (iteration 11): oval like Figure's, deeper than wide (side frame: upper arm 0.11 deep, elbow disc 0.09;
# front frame: 0.085 wide). Round 0.096 arms read thin from the side and wide from the front.
link('L_shoulder_roll_link', 1.0, [
    # flaring at the top into the shoulder cap (the deltoid), under the cap's outer end
    # iteration 17: ends 6 mm above the lower upper arm (they overlapped), a seam at the shoulder yaw joint
    # iteration 24 (round the elbow's lesson, bends seen from the same angle): a flat 6 mm seam here and at the wrist yaw
    # let the background through in level views and read as a separate ring. The seam is now 3 mm, curved about the
    # yaw axis: the lower arm's top a closed dome inside a sphere, this cover's end 3 mm outside it (a sphere on the
    # axis maps onto itself as the arm yaws, and no straight line runs through a curved gap into a closed dome)
    loft('L upper arm', [(SZ - 0.142, 0, SY, 0.053, 0.044, 2.2), (SZ - 0.114, 0, SY, 0.053, 0.044, 2.2),
                         (SZ - 0.075, 0, SY, 0.055, 0.045, 2.2),
                         (SZ - 0.035, 0, SY, 0.0545, 0.047, 2.2), (SZ - 0.008, 0, SY, 0.0505, 0.052, 2.3)]),
    # iteration 18: its top is a socket round the cap, 4 mm clear of it: a sphere about the shoulder centre under
    # the hemispherical end, a cylinder about the pitch axis under the rest
    cut(ell('L upper arm socket', (0, SY, SZ), (CAP_R + 0.004,) * 3), 'L upper arm'),
    cut(ell('L upper arm seam', ARM_YAW_C, (ARM_YAW_R + 0.003,) * 3), 'L upper arm'),
    cut(cyl('L upper arm socket in', (0, 0.17, SZ), 'y', CAP_R + 0.004, 0.07), 'L upper arm'),
    # from the roll flange on the back face: an axle out through the cap's hole, a short lug down behind the cap,
    # a step into the sleeve just under the cap (16 deg clear of the cap's sweep), a foot onto the yaw actuator
    dict(cyl('L shoulder roll axle', (-0.0465, SY, SZ), 'x', 0.011, 0.036, BLACK), exposed=True),
    rbox('L shoulder roll lug', (-0.064, SY, SZ - 0.0195), (0.006, 0.02, 0.065), 0.002, BLACK, exposed=True),
    rbox('L shoulder roll step', (-0.0555, SY, SZ - 0.048), (0.023, 0.02, 0.008), 0.002, BLACK, exposed=True),
    rbox('L shoulder roll foot', (-0.0435, SY, SZ - 0.07), (0.013, 0.02, 0.04), 0.002, BLACK),
])
joint('L_shoulder_yaw', 'L_shoulder_roll_link', 'L_shoulder_yaw_link', (0, SY, SZ - 0.12), 'z', (-2.6, 2.6),
      'RS02', act_c=(0, SY, SZ - 0.09), out=-1, exposed='flange')   # its flange shows in the seam
link('L_shoulder_yaw_link', 1.05, [
    # squarer at the elbow (n 4): flat sides under the pad, as Figure's. Iteration 23: ends in a hem and a socket round
    # the forearm's drum; a plane stops the socket's tail where it would thin to nothing behind the elbow
    loft('L upper arm lower', [(EZ - 0.02, 0, SY, ELBOW_A, 0.0425, 4.0), (EZ + 0.05, 0, SY, ELBOW_A, 0.043, 4.0),
                               (EZ + 0.085, 0, SY, 0.053, 0.043, 2.2), (SZ - 0.110, 0, SY, 0.053, 0.044, 2.2)],
         open=('bottom',), clip=above((-1.0, EZ + 0.02), (1.0, EZ + 0.02)), keep=(ARM_YAW_C, ARM_YAW_R)),
    cut(cyl('L upper arm elbow socket', (0, SY, EZ), 'y', ELBOW_A, 0.12), 'L upper arm lower'),
    # in front of where the tangent plane meets the socket (91 deg), everything below the plane
    cut(loft('L upper arm fold', [(EZ - 0.06, 0, SY, 0.1, 0.06, 8.0), (EZ + 0.12, 0, SY, 0.1, 0.06, 8.0)],
             clip=[below(*FOLD), above(elbow_ray(271.4), elbow_ray(91.4))]), 'L upper arm lower'),
    span('L upper arm frame', EZ + 0.066, SZ - 0.115, y=SY, sx=0.03, sy=0.03, exposed=True),
    # the hem's underside dark, as Figure's hem has a shadow line under it, the grey wall's 3 mm edge round it: a light
    # flat face there read as a cut with a lip
    loft('L upper arm hem', [(EZ + 0.02, 0, SY, ELBOW_A - 0.003, 0.040, 4.0), (EZ + 0.085, 0, SY, 0.050, 0.040, 4.0)],
         BLACK, 'decal', open=(), clip=[below(*up(FOLD, 0.004)), above(*up(FOLD, -0.0005)),
                                        ahead(ELBOW_A * -0.03)]),
    # the stator hangs from the frame by a bridge inside the hem, then a cheek under the pad (4 mm 7075, 2.4 cm wide: at
    # the actuator's peak 36 N m about 95 MPa, 7075-T6 yields near 500), its mount through a round hole in the
    # forearm's outer wall
    rbox('L elbow bridge', (-0.006, SY + 0.027, EZ + 0.063), (0.012, 0.031, 0.010), 0.002, BLACK, exposed=True),
    rbox('L elbow cheek', (0, SY + 0.0455, EZ + 0.016), (0.024, 0.004, 0.076), 0.002, BLACK, exposed=True),
    # (the cheek's top, thicker inward and under the hem, is what meets the bridge: above it the pad is too thin)
    rbox('L elbow cheek top', (0, SY + 0.0445, EZ + 0.056), (0.024, 0.006, 0.008), 0.001, BLACK, exposed=True),
    dict(cyl('L elbow mount', (0, SY + 0.0345, EZ), 'y', 0.02, 0.0225, BLACK), exposed=True),
    # Figure's oval (8.6 x 12.3 cm), a dome on the arm's flat side: 1.1 cm proud at its centre, thinning to its rim
    *dome_pad('L elbow disc', EZ + 0.007, SY + 0.0435, +1, 0.043, 0.0615, 0.011),
])
# Elbow: RS06 (88 mm), the size of Figure's elbow disc. User decision 2026-10-31: smaller elbows at the cost of
# holding capacity (RS04 120 mm carried 20 kg short / 10 kg continuous; RS06 carries about 10 kg short / 4 kg
# continuous with forearms level). The 1.6 kg saved went into the battery. Output on the inner face (iteration 4):
# from outside the elbow is a clean black disc.
# flexion 129 deg (iteration 11: the deeper oval forearm meets the upper arm at 135); 109 deg (iteration 17: with
# the covers swept too, the forearm meets the upper arm's frame in the elbow zone at 113)
joint('L_elbow', 'L_shoulder_yaw_link', 'L_elbow_link', (0, SY, EZ), 'y', (-1.9, 0.0), 'RS06', out=-1)
link('L_elbow_link', 0.85, [
    # no narrower than 0.085 across: the wrist yaw RS02 (79 mm) sits in the lower forearm
    # iteration 17: ends 6 mm above the lower forearm (a seam at the wrist yaw joint)
    # iteration 23: narrowing to the neck under the pads, up to the drum; the wrist yaw RS02 keeps its 3 mm at -0.07
    loft('L forearm', [(EZ - 0.146, 0, SY, 0.0445, 0.0415, 2.2), (EZ - 0.119, 0, SY, 0.045, 0.0425, 2.2),
                       (EZ - 0.07, 0, SY, 0.048, 0.0428, 2.2),
                       (EZ - 0.045, 0, SY, 0.048, ELBOW_W + 0.0005, 2.6), (EZ - 0.022, 0, SY, 0.0485, ELBOW_W, 5.0),
                       (EZ - 0.008, 0, SY, 0.0488, ELBOW_W, 12.0), (EZ, 0, SY, ELBOW_R - 0.0005, ELBOW_W, 16.0)]),   # 0.5 mm inside the drum: no coincident faces
    cut(ell('L forearm seam', WRIST_YAW_C, (WRIST_YAW_R + 0.003,) * 3), 'L forearm'),   # iteration 24: as the upper arm's
    # the upper half of the drum (the lower half would only be a wall inside the forearm)
    cyl('L forearm drum', (0, SY, EZ), 'y', ELBOW_R, 2 * ELBOW_W, GREY, 'shell'),
    cut(rbox('L forearm drum lower', (0, SY, EZ - 0.056), (0.12, 0.1, 0.1), 0.001), 'L forearm drum'),
    cut(cyl('L forearm mount hole', (0, SY + ELBOW_W - 0.002, EZ), 'y', 0.023, 0.012), 'L forearm drum', 'L forearm'),
    # the pad on the inner side: round, as it rides on the forearm and turns with it against the upper arm
    dict(cyl('L elbow disc in mount', (0, SY - 0.039, EZ), 'y', 0.012, 0.012, BLACK), exposed=True),
    *dome_pad('L elbow disc in', EZ, SY - 0.0435, -1, 0.043, 0.043, 0.011),
    # side plate from the elbow's output flange (inner face) down onto the wrist yaw actuator
    rbox('L forearm frame', (0, SY - 0.031, EZ - 0.035), (0.04, 0.008, 0.10), 0.003, BLACK, exposed=True),
])
joint('L_wrist_yaw', 'L_elbow_link', 'L_wrist_yaw_link', (0, SY, EZ - 0.125), 'z', (-2.6, 2.6), 'RS02',
      act_c=(0, SY, EZ - 0.095), out=-1, exposed='flange')   # its flange shows in the seam
link('L_wrist_yaw_link', 0.6, [
    loft('L forearm lower', [(WZ + 0.025, 0, SY, 0.034, 0.032, 2.2), (WZ + 0.045, 0, SY, 0.037, 0.034, 2.2),
                             (EZ - 0.125, 0, SY, 0.044, 0.041, 2.2), (EZ - 0.112, 0, SY, 0.044, 0.041, 2.2)],
         open=('bottom',), keep=(WRIST_YAW_C, WRIST_YAW_R)),
    # from just over the wrist pitch actuator up into the wrist yaw flange (clear of the yaw actuator)
    span('L forearm lower frame', WZ + 0.027, EZ - 0.1185, y=SY, sx=0.03, sy=0.03, exposed=True),
    # Figure's black wrist cuff, over the wrist pitch actuator (iteration 4)
    # iteration 17: wider (clear of the wrist cheek) and above the pitch axis (clear of the wrist roll actuator as
    # the hand pitches 0.9 rad); the wrist pitch actuator's lower half shows black below it
    loft('L wrist cuff', [(WZ + 0.002, 0, SY, 0.043, 0.043, 2.0), (WZ + 0.040, 0, SY, 0.043, 0.043, 2.0)], BLACK,
         open=('bottom',)),                       # open below: the wrist cheek's corners rise into it
])
joint('L_wrist_pitch', 'L_wrist_yaw_link', 'L_wrist_pitch_link', (0, SY, WZ), 'x', (-0.9, 0.9), 'RS00',
      out=+1, exposed=True)
link('L_wrist_pitch_link', 0.45, [
    rbox('L wrist cheek', (0.0285, SY, WZ - 0.0275), (0.006, 0.04, 0.055), 0.002, BLACK, exposed=True),
])
joint('L_wrist_roll', 'L_wrist_pitch_link', 'L_hand', (0, SY, WZ - 0.06), 'y', (-0.6, 0.6), 'RS00', out=-1,
      exposed=True)
link('L_hand', 1.0, [
    # iteration 17: its top 5 mm below the wrist roll actuator, 34 mm from the roll axis (round about it)
    # the hand's motors, tendons and palm: about 0.85 kg in the palm, the rest in the fingers' covers and bracket
    rbox('L palm', (0, SY - 0.005, WZ - 0.1295), (0.092, 0.06, 0.071), 0.018, BLACK, 'shell', mass=0.85,
         material='hand'),
    # from the wrist roll flange down into the palm, outside the roll actuator's length
    rbox('L hand bracket', (0, SY - 0.030, WZ - 0.085), (0.02, 0.006, 0.042), 0.002, BLACK, exposed=True),
    cap('L index', (0.033, SY, WZ - 0.16), (0.033, SY, WZ - 0.215), 0.0105),
    cap('L middle', (0.011, SY, WZ - 0.16), (0.011, SY, WZ - 0.225), 0.0105),
    cap('L ring', (-0.011, SY, WZ - 0.16), (-0.011, SY, WZ - 0.22), 0.0105),
    cap('L little', (-0.033, SY, WZ - 0.16), (-0.033, SY, WZ - 0.205), 0.0100),
    cap('L thumb', (0.042, SY - 0.012, WZ - 0.10), (0.058, SY - 0.032, WZ - 0.157), 0.0115),
], 'dexterous hand placeholder (about 1 kg)')

# The knee (iteration 19, user: "real knees with actuators won't look like this"), a hinge; iteration 26: bent 35 deg
# in the walk, as the elbow was (user's elbow screenshot, same family), it still showed a flat-topped shank, a black
# drum in the mouth above it, and pads on posts with daylight behind them. As the elbow now: the shank's top is a DRUM
# (grey, the shank's own depth) round about the axis; the thigh ends in a socket 4 mm clear of it, running into one plane
# behind, tangent to the circle the calf's back face rolls on at full flexion (4 mm clear at 2.15 rad); the fork moves
# outside the drum: cheeks at both ends of the axis, each with a stub axle through a round hole in the drum's end wall
# to the knuckle inside (a clevis, as Unitree's and Asimov's knees); the pads are domes on the thigh's flat sides.
# Figure's side frame shows just this: oval domes on a narrow neck under a hem.
KNEE_R, KNEE_SOCKET = 0.060, 0.064
KNEE_W = 0.0615                           # the neck's half width, 2 mm inside the cheeks
KNEE_FOLD = ((-0.1, KZ + 0.0763 + 0.06494), (0.1, KZ + 0.0763 - 0.06494))


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
# iteration 25 (side frames against the replica's: black down to 0.80 m at the back, a box behind the thigh, where
# Figure's grey thigh rises to its black hip, and it was the dome that dropped it): a shallow BOWL about the yaw axis
# instead: the thigh's top dished to a sphere centred above, the lobe's underside 3 mm inside the same sphere. Round about
# the yaw axis as the dome was, front 0.852 m, back 0.863 m (it cannot rise further: the roll actuator's underside is
# 0.87 m, 8 mm above). Radius 0.7: across the thigh's 14 cm width the seam rises 3.5 mm against its 3 mm gap, so no
# level line runs through it (lesson: a flat seam shows light in a level side view), checked on green.
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


PITCH_ACT_Y = 0.092                       # the hip pitch actuator's centre on its axis (iteration 21: 6 mm inboard)


def pitch_sweep(name):
    """Cuts: the hip pitch actuator (round about y, on the pelvis) and its flange, swept by the roll; in slices along
    x, each as tall as the actuator is at the slice's inner edge (a cylinder about y is lower away from x = 0)."""
    r, h, rf = ACT['RS04']['r'], ACT['RS04']['h'], 0.6 * ACT['RS04']['r']
    y0, y1 = PITCH_ACT_Y - h / 2 - HY, PITCH_ACT_Y + h / 2 - HY
    f0 = y1 - 0.001                                           # the flange, FLANGE_T on the actuator's outer face
    out, edges = [], [0.0, 0.015, 0.025, 0.035, 0.045, 0.055, r + 0.003]
    for i in range(len(edges) - 1):
        xa, xb = edges[i], edges[i + 1]
        a, af = (max(r ** 2 - xa ** 2, 0) ** 0.5), (max(rf ** 2 - xa ** 2, 0) ** 0.5)
        rects = [(y0, y1, -a, a)] + ([(f0, f0 + FLANGE_T, -af, af)] if af else [])
        out += [cut(roll_sweep(f'{name} {i}f', rects, xa, xb), 'L hip lobe'),
                cut(roll_sweep(f'{name} {i}b', rects, -xb, -xa), 'L hip lobe')]
    return out


# limits from the sweep: flexion stops at 97 deg (hip yaw actuator meets the pelvis block), extension at 0.30 rad
# (the rounded hip lobe meets the waist column; walking needs ~0.2): a little range traded for Figure's hip shape
# iteration 17: extension 0.24 (the hip roll actuator met the pelvis block at 0.257 once the lobe stopped hiding it);
# the pitch actuator shows black between the pelvis block and the lobe, as on Figure
joint('L_hip_pitch', 'pelvis', 'L_hip_pitch_link', (0, HY, HZ), 'y', (-1.65, 0.24), 'RS04',
      act_c=(0, PITCH_ACT_Y, HZ), out=+1, exposed=True)
link('L_hip_pitch_link', 1.6, [
    # from the pitch flange back over the roll output (its plate turns below the axis, its flange is a disc) onto
    # the outer side of the roll actuator; within the roll actuator's radius of the roll axis, so the lobe rolling
    # round it stays slim (iteration 21)
    rbox('L hip bracket', (-0.045, 0.1295, HZ + 0.0325), (0.09, 0.013, 0.015), 0.003, BLACK, exposed=True),
])
# The roll actuator sits behind the pitch actuator and drives FORWARD: its output plate runs straight down the gap
# between the two actuators (it turns in its own plane, so it never meets the pitch actuator), and two tabs
# under the pitch actuator carry it to a crown on the hip yaw actuator, inside the thigh's top. Nothing reaches
# behind the roll actuator, so the lobe can be round at the back (iteration 4).
joint('L_hip_roll', 'L_hip_pitch_link', 'L_hip_roll_link', (0, HY, HZ), 'x', HIP_ROLL, 'RS04',
      act_c=(-0.10, HY, HZ), out=+1, exposed=True)
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
    loft('L hip roll cap', [(HZ + dz, -0.1045, HY, 0.0325, (0.066 ** 2 - dz ** 2) ** 0.5, 6.0)
                            for dz in (-0.064, -0.055, -0.04, -0.02, 0.0, 0.02, 0.04, 0.055, 0.064)], GLOSS, open=(),
         clip=HIP_LOBE_CLIP, keep=(HIP_DOME_C, HIP_DOME_R)),
    cut(prism('L hip lobe bevel', -0.075, 0.14, HIP_BEVEL, BLACK), 'L hip lobe'),
    # hollow round the pitch actuator (pelvis) and its flange, and round the bracket, as the roll turns them
    *pitch_sweep('L hip lobe pitch sweep'),
    cut(roll_sweep('L hip lobe bracket sweep', [(0.023, 0.036, 0.025, 0.040)], -0.093, 0.003), 'L hip lobe'),
    # the roll actuator is round about the roll axis: a round opening, 3 mm clear (iteration 19)
    cut(cyl('L hip lobe roll opening', (-0.10, HY, HZ), 'x', 0.063, 0.062), 'L hip lobe', 'L hip roll cap'),
    rbox('L hip front plate', (-0.0665, HY, (HIP_SEAM + 0.001 + HZ - 0.02) / 2),
         (0.007, 0.104, HZ - 0.02 - HIP_SEAM - 0.001), 0.002, BLACK, exposed=True),
    # under the pitch actuator, outside its sweep (3 mm), above the thigh
    rbox('L hip tab in', (-0.045, HY - 0.035, HIP_SEAM + 0.005), (0.042, 0.010, 0.010), 0.002, BLACK, exposed=True),
    rbox('L hip tab out', (-0.045, HY + 0.035, HIP_SEAM + 0.005), (0.042, 0.010, 0.010), 0.002, BLACK, exposed=True),
    # on the yaw actuator's back face, through a round hole in the thigh's top (it turns round it)
    dict(cyl('L hip crown', (0, HY, (HZ - 0.083 + HZ - 0.070) / 2), 'z', 0.045, 0.013, BLACK), exposed=True),
])
joint('L_hip_yaw', 'L_hip_roll_link', 'L_hip_yaw_link', (0, HY, HZ - 0.11), 'z', (-1.2, 1.2), 'RS03', out=-1,
      exposed=True)
link('L_hip_yaw_link', 3.6, [
    # iteration 6 (user review of the side view): Figure's thigh is full at the back under the hip and tapers to
    # the knee, so the back line runs on from the hip in one curve; the front stays nearly straight
    # iteration 9 (user review): fuller at the knee (13 cm wide) so the knee discs sit flush in the leg, as on
    # Figure; no wider than before at the top, where the hands hang beside the thigh
    # iteration 11 (side frame): Figure's thigh is 0.14 deep at 0.65 m and 0.15 at 0.70; ours was ~1 cm deeper.
    # The knee RS04 (centre 0.72 m) keeps 9 mm inside the front.
    # iteration 20: the loft ran down only to 3 cm above the knee axis, so the clip below never cut it and the knee
    # showed a 3.7 cm black band in front (Figure's thigh meets the shank in a seam): now down past the axis
    loft('L thigh', [(KZ - 0.04, 0.0, HY, KNEE_SOCKET, 0.066, 4.0), (KZ + 0.03, 0.0, HY, KNEE_SOCKET, 0.066, 4.0),
                     (KZ + 0.10, -0.003, HY, 0.070, 0.068, 3.0),
                     (KZ + 0.19, -0.008, HY, 0.0775, 0.070, 2.4), (HZ - 0.15, -0.024, HY, 0.090, 0.070, 2.4),
                     (HZ - 0.095, -0.036, HY, 0.099, 0.070, 2.4), (HIP_SEAM + 0.004, -0.041, HY, 0.095, 0.070, 2.4),
                     (HIP_SEAM + 0.022, -0.041, HY, 0.095, 0.070, 2.4)],
         # iteration 21: up to 3 mm under the hip lobe (the yaw turns it under the lobe); iteration 25: dished
         # iteration 26: flat sides at the knee (n 4) for the pads; the socket's tail stops where it would thin out
         clip=above((-1.0, KZ + 0.02), (1.0, KZ + 0.02)), open=('bottom',)),
    cut(ell('L thigh bowl', HIP_DOME_C, (HIP_DOME_R + 0.003,) * 3), 'L thigh'),
    # round the hip yaw actuator and the crown on it (they turn in it), 3 mm clear
    cut(cyl('L thigh top opening', (0, HY, HIP_SEAM - 0.006), 'z', 0.056, 0.032), 'L thigh'),
    cut(cyl('L thigh knee socket', (0, HY, KZ), 'y', KNEE_SOCKET, 0.16), 'L thigh'),
    cut(loft('L thigh fold', [(KZ - 0.06, -0.05, HY, 0.1, 0.09, 8.0), (KZ + 0.2, -0.05, HY, 0.1, 0.09, 8.0)],
             clip=[below(*KNEE_FOLD), above(knee_ray(236.8), knee_ray(56.8))]), 'L thigh'),
    # its underside dark, as the elbow's hem
    loft('L thigh hem', [(KZ + 0.02, -0.002, HY, 0.062, 0.063, 4.0), (KZ + 0.17, -0.008, HY, 0.070, 0.066, 3.0)],
         BLACK, 'decal', open=(), clip=[below(*up(KNEE_FOLD, 0.004)), above(*up(KNEE_FOLD, -0.0005)),
                                        above(knee_ray(236.8), knee_ray(56.8))]),
    # to the knee drum's crank (inside the drum); it stops 1 cm above the drum
    dict(cap('L knee pushrod', (0.035, HY, KZ + 0.15), (0.035, HY, KZ + 0.062), 0.008, BLACK, 'struct'),
         exposed=True),
    # in front, beside the pushrod (behind it the calf's ankle actuators fold up at deep knee bends); from the
    # yoke, 1 cm above the drum, up into the hip yaw flange
    span('L thigh frame', KZ + 0.068, HZ - 0.14, x=0.024, y=HY, sx=0.024, sy=0.08, color=BLACK, exposed=True),
    rbox('L knee yoke', (0.028, HY, KZ + 0.072), (0.024, 0.127, 0.012), 0.002, BLACK, exposed=True),
    # the fork: a cheek each side, outside the drum (3 mm 7075, inside the thigh's wall above the hem), and a stub axle through the drum's end wall to the
    # knuckle
    rbox('L knee cheek in', (0.006, HY - 0.064, KZ + 0.0275), (0.024, 0.003, 0.095), 0.002, BLACK, exposed=True),
    rbox('L knee cheek out', (0.006, HY + 0.064, KZ + 0.0275), (0.024, 0.003, 0.095), 0.002, BLACK, exposed=True),
    dict(cyl('L knee axle in', (0, HY - 0.0488, KZ), 'y', 0.02, 0.0295, BLACK), exposed=True),
    dict(cyl('L knee axle out', (0, HY + 0.0488, KZ), 'y', 0.02, 0.0295, BLACK), exposed=True),
    # Figure's knees: black pads on both sides of the joint (front video frame 1:50), not a front kneecap; 10 cm wide x
    # 14 cm tall, centred 1 cm above the axis (iteration 14). Iteration 26: domes on the thigh's flat sides, 1.2 cm proud
    *dome_pad('L knee pad in', KZ + 0.01, HY - 0.0665, -1, 0.05, 0.07, 0.012),
    *dome_pad('L knee pad out', KZ + 0.01, HY + 0.0665, +1, 0.05, 0.07, 0.012),
], 'knee RS04 drives the shank through a 2:1 linkage')
# flexion 2.15 rad (iteration 17: through the knee zone the calf meets the thigh at 2.2; the deep squat uses 2.0)
joint('L_knee', 'L_hip_yaw_link', 'L_knee_link', (0, HY, KZ), 'y', (0.0, 2.15), 'RS04', act_c=None, remote=True,
      capacity=dict(model='RS04', count=1, ratio=2.0))
JOINTS[-1]['act_c'] = [0, HY, KZ + 0.185]
SHANK = lambda f: AZ + 0.045 + f * (KZ - AZ - 0.01)      # height at fraction f of the shank skin
# rods behind the ankle, either side; their foot ends (iteration 17: 3 cm behind, inside the slim lower calf; at
# 4.5 cm they showed behind it; at 3.7 cm out they showed through the sock)
ROD_X, ROD_Y, ROD_Z = -0.025, 0.030, 0.065


def ankle_drive(s, z, side):
    """Crank on an ankle motor's output (side +1 outboard, -1 inboard), outside the shank plate its shaft passes
    through, and its rod down to the foot."""
    y = HY + side * 0.0315
    return [rbox(f'{s} ankle crank {"out" if side > 0 else "in"}', ((-0.01 + ROD_X) / 2, y, z),
                 (abs(ROD_X + 0.01) + 0.016, 0.006, 0.016),
                 0.003, '#3A3D42', 'linkage'),
            cap(f'{s} ankle rod {"out" if side > 0 else "in"}', (ROD_X, HY + side * ROD_Y, z),
                (ROD_X, HY + side * ROD_Y, ROD_Z), 0.005, ALU, 'linkage')]


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
# Ankle drive (parallel ankle, as Unitree G1 and Asimov 1): two motors high in the calf (a light lower leg), each
# turning a crank that pushes a rod down to the heel of the foot plate, one left and one right of the ankle. Both
# rods the same way: pitch; opposite: roll. Iteration 16 (user: "they look too big for ankles"): 2 x RS06 (88 mm)
# instead of 2 x RS03 (106 mm); the static poses need 23.5 N m (x 1.5 peak margin = 35 of 72), but fast walking
# push-off (60-90 N m) has less headroom. The 1 kg saved went into the battery.
link('L_knee_link', 2.08, [
    # iteration 11 (side frame): Figure's calf is ~0.105 deep; ours was 0.135, then 0.121 round the RS03.
    # iteration 16: with RS06, 0.105 like Figure's, 4.5 mm clear of them inside the wall
    # iteration 13: slim above the ankle like Figure's (~8 cm), below the ankle actuators
    # iteration 26: narrowing to the neck under the pads (within the cheeks' reach), up to the drum, inside the depth the
    # thigh's fold plane clears; the ankle motors keep their room below
    loft('L shank', [(SHANK(0), 0.0, HY, 0.040, 0.042, 2.2), (SHANK(0.159), -0.006, HY, 0.046, 0.044, 2.2),
                     (SHANK(0.27), -0.010, HY, 0.0500, 0.052, 2.4), (SHANK(0.386), -0.010, HY, 0.0525, 0.056, 2.5),
                     (SHANK(0.49), -0.010, HY, 0.0525, 0.058, 2.5), (SHANK(0.60), -0.010, HY, 0.0525, 0.060, 2.4),
                     (KZ - 0.075, -0.004, HY, 0.056, KNEE_W, 2.8), (KZ - 0.03, -0.001, HY, 0.0585, KNEE_W, 5.0),
                     (KZ - 0.008, 0.0, HY, KNEE_R - 0.001, KNEE_W, 12.0), (KZ, 0.0, HY, KNEE_R - 0.0005, KNEE_W, 16.0)],
         clip=above(*up(TRIM_HIGH, -0.002))),
    loft('L ankle collar', [(0.11, -0.004, HY, 0.0395, 0.0465, 2.2), (0.15, -0.006, HY, 0.0455, 0.0470, 2.2),
                            (0.21, -0.007, HY, 0.0520, 0.0505, 2.2)], BLACK,
         clip=[above(*TRIM_LOW), below(*TRIM_HIGH)]),
    # flaring into the shoe toward the heel and the sides, where the ankle's range leaves room (not in front)
    loft('L ankle sock', [(0.08, -0.013, HY, 0.048, 0.0470, 2.2), (0.105, -0.010, HY, 0.0455, 0.0460, 2.2),
                          (0.135, -0.005, HY, 0.0410, 0.0445, 2.2), (0.17, -0.004, HY, 0.0405, 0.0445, 2.2)], WHITE,
         clip=[above(*SOCK_LOW), below(*up(TRIM_LOW, 0.002))]),
    # the upper half of the drum (below it would only be a wall inside the shank); the axles turn in round holes
    cyl('L knee drum', (0, HY, KZ), 'y', KNEE_R, 2 * KNEE_W, GREY, 'shell'),
    cut(rbox('L knee drum lower', (0, HY, KZ - 0.064), (0.14, 0.14, 0.1), 0.001), 'L knee drum'),
    cut(cyl('L shank axle hole in', (0, HY - KNEE_W + 0.002, KZ), 'y', 0.023, 0.012), 'L shank', 'L knee drum'),
    cut(cyl('L shank axle hole out', (0, HY + KNEE_W - 0.002, KZ), 'y', 0.023, 0.012), 'L shank', 'L knee drum'),
    # the knuckle inside it carries the shank plates and the pushrod's crank
    cyl('L knee knuckle', (0, HY, KZ), 'y', 0.03, 0.068, BLACK),
    # clamping the ankle motors between them (iteration 16: the narrower RS06 floated 2.5 mm clear of plates at 29 mm)
    span('L shank plate in', SHANK(0), KZ - 0.01, y=HY - 0.026, sy=0.004, f=0.0015, color=BLACK, exposed=True),
    span('L shank plate out', SHANK(0), KZ - 0.01, y=HY + 0.026, sy=0.004, f=0.0015, color=BLACK, exposed=True),
    # iteration 17: 5 cm lower, below the knee zone's slanted bottom
    cyl('L ankle actuator A (buy) RS06', (-0.01, HY, SHANK(0.60)), 'y', ACT['RS06']['r'], ACT['RS06']['h'],
        '#3A3D42', 'actuator'),
    cyl('L ankle actuator B (buy) RS06', (-0.01, HY, SHANK(0.386)), 'y', ACT['RS06']['r'], ACT['RS06']['h'],
        '#3A3D42', 'actuator'),
    # cranks on the outputs (A outboard, B inboard) and the rods down to the foot: they move with the linkage, not
    # with the shank or the foot, so they are drawn ('linkage') but kept out of the collision checks
    *ankle_drive('L', SHANK(0.60), +1),
    *ankle_drive('L', SHANK(0.386), -1),
    # 32 mm deep (iteration 12: inside the slimmer ankle sock)
    rbox('L ankle fork cheek in', (0, HY - 0.033, AZ + 0.025), (0.032, 0.006, 0.07), 0.003, BLACK, exposed=True),
    rbox('L ankle fork cheek out', (0, HY + 0.033, AZ + 0.025), (0.032, 0.006, 0.07), 0.003, BLACK, exposed=True),
    rbox('L ankle fork bridge', (0, HY, AZ + 0.057), (0.032, 0.072, 0.012), 0.003, BLACK, exposed=True),
], 'two RS06 drive the ankle through cranks and rods (parallel ankle)')
# dorsiflexion 0.8 rad (iteration 12, was 0.9): the deepest pose uses 0.77 (90 deg squat); the 0.1 rad buys the
# shoe's instep its height
joint('L_ankle_pitch', 'L_knee_link', 'L_ankle_pitch_link', (0, HY, AZ), 'y', (-0.8, 0.55), None, remote=True,
      capacity=dict(model='RS06', count=2, ratio=1.0))
link('L_ankle_pitch_link', 0.25, [
    rbox('L ankle cross', (0, HY, AZ), (0.028, 0.058, 0.026), 0.004, BLACK, exposed=True),
])
joint('L_ankle_roll', 'L_ankle_pitch_link', 'L_foot', (0, HY, AZ), 'x', (-0.3, 0.3), None, remote=True,
      capacity=dict(model='RS06', count=2, ratio=0.6))
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
    # where the ankle rods push on the foot, either side behind the ankle, on the foot plate (8 mm wide: inside the
    # shoe's window pockets)
    rbox('L ankle rod mount out', (ROD_X, HY + ROD_Y, (0.022 + ROD_Z) / 2), (0.014, 0.008, ROD_Z - 0.022), 0.003),
    rbox('L ankle rod mount in', (ROD_X, HY - ROD_Y, (0.022 + ROD_Z) / 2), (0.014, 0.008, ROD_Z - 0.022), 0.003),
])


# ---------------------------------------------------------------- mirror, actuators, flanges

def mirror_geom(g):
    g = copy.deepcopy(g)
    g['name'] = g['name'].replace('L ', 'R ', 1) if g['name'].startswith('L ') else g['name']
    if 'targets' in g:
        g['targets'] = [t.replace('L ', 'R ', 1) if t.startswith('L ') else t for t in g['targets']]
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
    if g['kind'] == 'revolve' and g['axis'] == 'y':
        g['profile'] = [[-t, r] for t, r in reversed(g['profile'])]
    if g['kind'] == 'prism':
        g['pts'] = [[-y, z] for y, z in g['pts']]
    if g['kind'] == 'loft':
        for s in g['sections']:
            s['cy'] = -s['cy']
    return g


def mirror_name(n):
    return 'R_' + n[2:] if n.startswith('L_') else n


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
        links[j['parent']]['geoms'].append(cyl(f"{j['name']} actuator (buy) {j['act']}", j['act_c'], j['axis'],
                                               a['r'], a['h'], color, 'actuator'))
        if j['remote']:
            continue
        ax = {'x': 0, 'y': 1, 'z': 2}[j['axis']]
        c = list(j['act_c'])
        c[ax] += j['out'] * (a['h'] / 2 + FLANGE_T / 2 - 0.001)   # 1 mm into the face so they touch
        links[j['child']]['geoms'].append(cyl(f"{j['name']} flange", c, j['axis'], 0.6 * a['r'], FLANGE_T, ALU,
                                              'struct'))
    for l in links.values():                              # every part's material (tools/mass_model.py)
        for g in l['geoms']:
            if g['role'] != 'cut':
                g['material'] = material_of(g)
    return dict(frame='metres; X forward, Y left, Z up; floor z=0; zero pose standing, arms down, palms in',
                actuators=ACT, materials=MATERIALS, links=links, joints=joints, root='pelvis')


if __name__ == '__main__':
    robot = build()
    out = os.path.join(HERE, 'robot.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(robot, f, indent=1)
    print(f"{len(robot['links'])} links, {len(robot['joints'])} joints -> {out} (masses: python -I tools/mass_model.py)")
