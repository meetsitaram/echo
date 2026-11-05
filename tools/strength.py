"""Hand estimates of strength along echo's load paths, from the live spec's solids. Not FEA.

  python -I tools/strength.py        -> reports/strength.md (every section, worst first), prints the worst ten

tools/checks.py runs the same tables as its groups 'loadpath' and 'contact' (see THRESHOLDS below), so the report and
the check cannot disagree. What this file holds:

  SECTIONS    named sections along named load paths: a plane through named parts, cut from the real solids (with their
              cuts), so shrinking, moving, deleting or re-roling a part changes the answer. A plane that misses a part is
              zero area, which fails: it cannot pass by accident.
  INTERFACES  the contacts the same paths cross, each with the force it passes.
  NOT_ON_PATH rigid parts that carry no path load, each with its reason. A new part in neither list fails 'coverage'.

Method per section: slice the parts, get area and second moments from the polygons; add the load case's forces on the
far side of the cut and its free moments at the centroid; axial plus biaxial bending at every vertex, torsion from Saint-Venant's
J = A^4 / (40 Ip), shear 1.5 V / A; von Mises of the worst sigma and worst tau. That stacks peaks that may not coincide
(conservative) and ignores stress concentration, fasteners and local buckling (not conservative). Stated in the report.

Load cases. Geometry at the zero pose, left side (the right is built as its mirror).
  peak     single-leg 3,400 N at the foot. Largest vertical foot force seen in the shoved sim: 3,432 N, clip
           neutral_walk_180_R_002__A122 with a 1.48 m/s pelvis push ('mv6/2_shove15.npz', 50 Hz, weight 569 N: the v6 plant).
           Its line runs up the straight leg, off the axes by what the ankle motors can hold (dx = 90/3400, dy = 47.5/3400), because
           a joint back-drives past its peak torque. That is what limits the moment; a joint on its hard stop does not limit it.
           The body above takes it as inertia: a = F / total mass.
  side     'peak' plus the 1,614 N side force recorded in that same frame, on a line through the ankle, for parts at or
           below the knee only: at that frame hip roll held 43 N m, so most of it went into the leg's inertia, and a
           static estimate can't split it. An upper bound.
  walk     810 N, the same recorder's peak on Relaxed_walk_forward_001__A057 (force_v6.npz; frame 1, settling; the steps
           themselves peak near 670 N, 99th percentile 612 N). Levers from the same rule, inside the sole's size.
  motor    each motor beside the section at its peak torque, one at a time, as a free moment (a motor must not break its own
           bracket); ankle motors as the rod force they make.
  payload  5 kg per hand ('carry 10 kg at the chest' in tools/checks.py POSES) plus the links' weight, static, in the worst
           direction relative to the arm, as the arm can point anywhere.
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom as G  # noqa: E402

ROOT = G.ROOT

# ---------------------------------------------------------------- THRESHOLDS (each with its reason)

# Safety factor 2 to yield under every case, 3,400 N first. Why 2 and not the 1.25 to 1.5 usual on limit loads:
#  - those factors assume FEA and a test. These are hand sections with no stress concentration: a fillet or bore in 7075
#    typically has Kt 1.5 to 2.5, and at yield the part has already taken a set, so a joint is out of true;
#  - 3,400 N is one 20 ms sample (4 of 526 frames pass 2,000 N) whose size depends on the sim's contact stiffness, and no fall
#    was recorded at all: it is not a ceiling;
#  - fatigue: 7075-T6 keeps about 160 MPa at 5e8 cycles, reversed (MIL-HDBK-5, from memory), a third of its yield. At
#    factor 2 on the 3,400 N case the same part sits near 60 MPa on an 810 N step (load ratio alone), which leaves room for a Kt of 2.5.
# Not lowered to pass: failing rows are the design's open items (reports/strength.md).
SF_MIN = 2.0
# User, 2026-11-05: "only go as far as it looks right on camera and something that is buildable in real life that can walk around. It's ok if it breaks." So what's required is calm walking and each motor's own peak, at this factor, a hand estimate.
# The 3,400 N shove and the side force are still read and printed, as a report: nothing is sized for them.
SF_WALK = 1.5

# Contact. One M5 with two diameters of edge distance all round needs a 10 x 10 pad. M5 is what the RS04's output takes
# (retailer drawing: 9 x M5 on d36); nothing that carries a limb should be held by less. So a contact under 100 mm2 can't
# be bolted at all, whatever its shape.
CONTACT_MIN_MM2 = 100.0
# Slip load of that M5 (12.9 into 7075): preload about 10 kN (75 % of proof on 14.2 mm2), friction 0.15, from memory. A path
# interface passing force V needs ceil(SF_MIN x V / BOLT_SLIP_N) bolts, each with its pad.
BOLT_SLIP_N = 1500.0
# The knee's fork since it has real bearings: a force line 14 mm off the leg's centre (what the ankle roll holds at 3,400 N), on supports at y -30.5 and +37.4 (iteration 41), puts (37.4 + 14) / 67.9 of it on the inner one.
KNEE_SPLIT = 0.757
# A drawn overlap thinner than this can't be told from a gap once made: general machining tolerance on parts this size is
# +-0.1 mm (ISO 2768-m). So a 0.02 mm contact counts as none. Exactly coplanar faces (within FLUSH, float32 rounding at 1 m)
# are a different thing, a drawn face-to-face seat, and count by their shared area.
CONTACT_MIN_DEPTH = 0.0001
FLUSH = 2e-6

# ---------------------------------------------------------------- materials and load magnitudes

# Yield or, for PA12, the tensile strength it fails at (MPa), and modulus (MPa). 7075-T6 typical plate; PA12 MJF, the maker's
# data sheet as remembered, about the same across layers. A material not listed cannot carry load. Here rather than in
# spec/robot.py so this round leaves robot.json untouched: move it to MATERIALS when the spec is next opened.
STRENGTH = {'al7075': (503.0, 71700.0), 'pa12': (48.0, 1700.0), 'pc': (60.0, 2300.0)}
CARRYING_ROLES = ('struct', 'linkage')

F_PEAK, F_WALK, F_SIDE = 3400.0, 810.0, 1614.0
HAND_KG = 5.0
G_ACC = 9.81
SOLE_X, SOLE_Y = (-0.08, 0.16), 0.05    # the sole's contact box about the ankle (spec: c 0.040, size 0.24 x 0.10)


def yield_of(g):
    s = STRENGTH.get(g.get('material'))
    return s[0] if s else None


def why_not_carrying(g):
    """None if g can carry load, else the reason it cannot."""
    if g['role'] not in CARRYING_ROLES:
        return f"role {g['role']}: a {g['role']} carries no load"
    if yield_of(g) is None:
        return f"material {g.get('material')}: no structural strength on record"
    return None


# ---------------------------------------------------------------- sections

def sec(name, path, link, parts, axis, at, **kw):
    """One row. parts: names, or '*' for every carrying part of the link (so a new rail counts as soon as it exists).
    axis: 'x', 'y', 'z' or ('along', part). at: the plane's position: a number, or (part or 'joint:NAME', 'lo' / 'hi' / 'c' / 'p0' /
    'mid', offset), so it moves with the part.
    load: 'leg' (the foot's line), 'upper' or 'limb' (inertia of what is beyond the cut), with distal (roots of what is beyond),
    own (+1 / -1: this link's own parts on that side of the plane count too), dirs. cases: which load cases apply. motors: [(joint, factor)];
    default every motor beside the link. rods: rod forces in 'motor'. local: [(fraction of F, point)] replaces the line in peak and walk,
    where a part sees only its share. toe: the line at the sole's ends (the ankle on its stop). hip_moment: the hips' peak
    moments pass through. solo: pieces do not act together (no parallel-axis help). buckle: (K, L in m, or part for its length)."""
    return dict(name=name, path=path, link=link, parts=parts, axis=axis, at=at, **kw)


LEG = 'foot to pelvis'
BODY = 'pelvis to shoulder'
ARM = 'shoulder to hand'
NECK = 'neck to head'
ROD = 'ankle rods and cranks'
BELOW_KNEE = ('peak', 'side', 'walk', 'motor')
ABOVE_KNEE = ('peak', 'walk', 'motor')
FALLS = ('peak', 'walk', 'motor')
LIFTS = ('peak', 'walk', 'motor', 'payload')

HAT = ['L_shoulder_pitch_link', 'R_shoulder_pitch_link', 'neck_yaw_link']

SECTIONS = [
    # ------------------------------------------------------------ foot to pelvis
    sec('Foot plate at the ankle (all load at toe or heel)', LEG, 'L_foot', ['L foot plate'], 'x', ('L ankle clevis front', 'c', 0.0),
        load='leg', toe=True, cases=('peak', 'walk'), note='ankle on its stop, so the motors do not limit the lever'),
    sec('Clevis plates above the foot plate', LEG, 'L_foot', ['L ankle clevis front', 'L ankle clevis back'], 'z', ('L foot plate', 'hi', 0.004),
        load='leg', cases=BELOW_KNEE, motors=[('L_ankle_roll', 1)]),
    sec('Clevis at the pivot height (rear plate net of its hole)', LEG, 'L_foot', ['L ankle clevis front', 'L ankle clevis back'], 'z', ('joint:L_ankle_roll', 'c', 0.0),
        load='leg', cases=BELOW_KNEE, motors=[('L_ankle_roll', 1)]),
    sec('Ankle cross at its middle', LEG, 'L_ankle_pitch_link', ['L ankle cross'], 'y', ('L ankle cross', 'c', 0.0),
        load='leg', cases=('peak', 'walk'), local=[(0.5, ('L ankle fork cheek out', 'c', (0, 0, -0.025)))],
        note='half the load from each fork cheek, if a pin existed'),
    sec('Cross tail', LEG, 'L_ankle_pitch_link', ['L ankle cross tail'], 'x', ('L ankle cross tail', 'c', 0.0),
        load='leg', cases=('motor',), motors=[], rods=['pitch']),
    sec('Fork cheeks', LEG, 'L_knee_link', ['L ankle fork cheek in', 'L ankle fork cheek out'], 'z', ('L ankle fork cheek in', 'c', 0.0),
        load='leg', cases=BELOW_KNEE, motors=[], rods=['pitch', 'roll']),
    sec('Shank lower plates', LEG, 'L_knee_link', ['L shank plate in', 'L shank plate out'], 'z', ('L ankle fork bridge', 'hi', 0.05),
        load='leg', cases=BELOW_KNEE, motors=[], rods=['pitch', 'roll'], buckle=(0.7, 'L shank plate in'), note='buckling: K 0.7 over the plate length'),
    sec('Shank closed section at the ankle pitch motor, panel on', LEG, 'L_knee_link', ['L shank body', 'L shank back panel'], 'z', 0.30,
        load='leg', cases=BELOW_KNEE, motors=[('L_knee', 1)], rods=['pitch', 'roll'], note='counts the back panel: its 7 M3 hold it (Bolted joints). With the panel off it is an open section, SF about 0.4: the screws are not optional'),
    sec('Shank closed section at the ankle roll motor, panel on', LEG, 'L_knee_link', ['L shank body', 'L shank back panel'], 'z', 0.3945,
        load='leg', cases=BELOW_KNEE, motors=[('L_knee', 1)], rods=['pitch', 'roll']),
    sec('Shank waist behind the knee', LEG, 'L_knee_link', ['L shank body'], 'z', ('joint:L_knee', 'c', -0.075),
        load='leg', cases=BELOW_KNEE, motors=[('L_knee', 1)], note='closed; where the back line is scooped for the fold'),
    sec('Barrel where it meets the waist', LEG, 'L_knee_link', ['L shank body'], 'z', ('joint:L_knee', 'c', -0.058),
        load='leg', cases=BELOW_KNEE, motors=[('L_knee', 1)]),
    sec('Barrel at the axis (walls, net of the bore)', LEG, 'L_knee_link', ['L shank body'], 'z', ('joint:L_knee', 'c', 0.0),
        load='leg', cases=('peak', 'walk', 'motor'), motors=[('L_knee', 1)], local=[(KNEE_SPLIT, ('L knee inner bearing', 'c', (0, 0, 0)))], note='the inner bearing\'s share through the plate and walls, and the motor\'s torque'),
    sec('Barrel plate round the inner bearing', LEG, 'L_knee_link', ['L shank body'], 'y', ('L knee inner bearing', 'c', 0.003),
        load='leg', cases=('peak', 'walk'), local=[(KNEE_SPLIT, ('L knee inner bearing', 'c', (0, 0, 0)))]),
    sec('Knee post at its root', LEG, 'L_knee_link', ['L knee rear cap'], 'y', ('L knee outer bearing', 'c', -0.0042),
        load='leg', cases=('peak', 'walk'), local=[(KNEE_SPLIT, ('L knee outer bearing', 'c', (0, 0, 0)))], note='the outer support taking the larger share, an upper bound for this side'),
    sec('Rear cap at the axis', LEG, 'L_knee_link', ['L knee rear cap'], 'z', ('joint:L_knee', 'c', 0.0),
        load='leg', cases=('peak', 'walk'), local=[(KNEE_SPLIT, ('L knee outer bearing', 'c', (0, 0, 0)))]),
    sec('Cup neck between bearing and cheek', LEG, 'L_hip_yaw_link', ['L_knee flange'], 'y', ('L knee inner bearing', 'c', -0.0075),
        load='leg', cases=('peak', 'motor'), motors=[('L_knee', 1)], local=[(KNEE_SPLIT, ('L knee inner bearing', 'c', (0, 0, 0)))], note='the wall round the M5 heads: the motor\'s torque and the nearer support\'s share'),
    sec('Fork inner eye, net of its opening', LEG, 'L_hip_yaw_link', ['L thigh fork'], 'z', ('joint:L_knee', 'c', 0.0),
        load='leg', cases=('peak', 'walk', 'motor'), motors=[('L_knee', 1)], local=[(KNEE_SPLIT, ('L knee inner bearing', 'c', (0, 0, 0)))]),
    sec('Outer cheek at its bearing', LEG, 'L_hip_yaw_link', ['L knee outer cheek'], 'z', ('joint:L_knee', 'c', 0.0),
        load='leg', cases=('peak', 'walk'), local=[(KNEE_SPLIT, ('L knee outer bearing', 'c', (0, 0, 0)))]),
    sec('Fork cheeks over the barrel, both', LEG, 'L_hip_yaw_link', ['L thigh fork', 'L knee outer cheek'], 'z', ('joint:L_knee', 'c', 0.055),
        load='leg', cases=ABOVE_KNEE, motors=[('L_knee', 1)]),
    sec('Fork inner cheek alone (the only one that takes motor torque)', LEG, 'L_hip_yaw_link', ['L thigh fork'], 'z', ('joint:L_knee', 'c', 0.055),
        load='leg', cases=('peak', 'motor'), motors=[('L_knee', 1)], local=[(KNEE_SPLIT, ('L knee inner bearing', 'c', (0, 0, 0)))]),
    sec('Fork cheeks out of plane, under the recorded side force', LEG, 'L_hip_yaw_link', ['L thigh fork', 'L knee outer cheek'], 'z', ('joint:L_knee', 'c', 0.055),
        load='leg', cases=('side',), solo=True, side_local=('joint:L_knee', 'c', (0, 0, 0)), note='upper bound: all of the 1,614 N reaches the knee; each cheek bending on its thin side'),
    sec('Fork crown', LEG, 'L_hip_yaw_link', ['L thigh fork', 'L knee outer cheek'], 'z', ('joint:L_knee', 'c', 0.09),
        load='leg', cases=ABOVE_KNEE, motors=[('L_knee', 1), ('L_hip_yaw', 1)]),
    sec('Thigh tube', LEG, 'L_hip_yaw_link', ['L thigh fork'], 'z', ('joint:L_knee', 'c', 0.19),
        load='leg', cases=ABOVE_KNEE, motors=[('L_knee', 1), ('L_hip_yaw', 1)], note='d66 x 2.5; no local buckling in this figure (D / t 26, plain 7075 tube: not close)'),
    sec('Thigh tube under its collar', LEG, 'L_hip_yaw_link', ['L thigh fork'], 'z', ('L_hip_yaw flange', 'lo', -0.012),
        load='leg', cases=ABOVE_KNEE, motors=[('L_knee', 1), ('L_hip_yaw', 1)]),
    sec('Yaw housing at the motor', LEG, 'L_hip_roll_link', ['L hip yoke'], 'z', ('L_hip_yaw actuator (buy) RS03', 'c', 0.0),
        load='leg', cases=ABOVE_KNEE, motors=[('L_hip_yaw', 1)]),
    sec('Yoke band under the pitch motor', LEG, 'L_hip_roll_link', ['L hip yoke'], 'x', -0.03,
        load='leg', cases=ABOVE_KNEE, motors=[('L_hip_yaw', 1)]),
    sec('Yoke plate behind the pitch motor', LEG, 'L_hip_roll_link', ['L hip yoke'], 'z', ('joint:L_hip_pitch', 'c', -0.05),
        load='leg', cases=ABOVE_KNEE, motors=[('L_hip_roll', 1), ('L_hip_yaw', 1)]),
    sec('Carrier arm over the roll output', LEG, 'L_hip_pitch_link', ['L hip carrier'], 'x', -0.071,
        load='leg', cases=ABOVE_KNEE, motors=[('L_hip_pitch', 1), ('L_hip_roll', 1)]),
    sec('Carrier arm at the eye', LEG, 'L_hip_pitch_link', ['L hip carrier'], 'x', -0.045,
        load='leg', cases=ABOVE_KNEE, motors=[('L_hip_pitch', 1), ('L_hip_roll', 1)]),
    sec('Roll barrel', LEG, 'L_hip_pitch_link', ['L hip carrier'], 'x', -0.09,
        load='leg', cases=ABOVE_KNEE, motors=[('L_hip_pitch', 1), ('L_hip_roll', 1)]),
    sec('Pelvis bar between the hips', LEG, 'pelvis', ['pelvis frame'], 'y', 0.03,
        load='leg', cases=ABOVE_KNEE, motors=[('L_hip_pitch', 1)]),
    sec('Hip barrel on the pelvis', LEG, 'pelvis', ['pelvis frame'], 'y', 0.095,
        load='leg', cases=ABOVE_KNEE, motors=[('L_hip_pitch', 1)]),

    # ------------------------------------------------------------ pelvis to shoulder
    sec('Waist housing on the pelvis', BODY, 'pelvis', ['pelvis frame'], 'z', ('waist_yaw actuator (buy) RS04', 'c', 0.0),
        load='upper', distal=['torso'], cases=FALLS, hip_moment=True, motors=[('waist_yaw', 1)]),
    sec('Waist cup neck', BODY, 'torso', ['waist_yaw flange'], 'z', ('joint:waist_yaw', 'c', 0.0367),
        load='upper', distal=HAT, own=1, cases=LIFTS, hip_moment=True, motors=[('waist_yaw', 1), ('L_shoulder_pitch', 2)]),
    sec('Chest rails over the waist', BODY, 'torso', ['chest frame'], 'z', 1.09,
        load='upper', distal=HAT, own=1, cases=LIFTS, hip_moment=True, motors=[('waist_yaw', 1), ('L_shoulder_pitch', 2)], note='two closed rails, tied at both ends'),
    sec('Frame between waist and shoulder bar (the pack is not frame)', BODY, 'torso', '*', 'z', ('battery lower', 'hi', 0.0),
        load='upper', distal=HAT, own=1, cases=LIFTS, hip_moment=True, motors=[('waist_yaw', 1), ('L_shoulder_pitch', 2)],
        note='slices every carrying part of the torso'),
    sec('Chest bar under the neck', BODY, 'torso', ['chest frame'], 'y', 0.0,
        load='upper', distal=HAT, own=1, cases=LIFTS, motors=[('L_shoulder_pitch', 2), ('neck_yaw', 1)]),
    sec('Chest bar at the shoulder motor', BODY, 'torso', ['chest frame'], 'y', ('L_shoulder_pitch actuator (buy) RS04', 'lo', -0.005),
        load='limb', distal=['L_shoulder_pitch_link'], cases=LIFTS, motors=[('L_shoulder_pitch', 1)]),

    # ------------------------------------------------------------ shoulder to hand
    sec('Shoulder bracket', ARM, 'L_shoulder_pitch_link', ['L shoulder bracket'], 'y', ('L shoulder bracket', 'c', 0.0),
        load='limb', distal=['L_shoulder_roll_link'], cases=LIFTS, motors=[('L_shoulder_pitch', 1), ('L_shoulder_roll', 1)]),
    sec('Shoulder roll post (front of the fork)', ARM, 'L_shoulder_pitch_link', ['L shoulder roll post'], 'x', ('L shoulder roll post', 'c', 0.0),
        load='limb', distal=['L_shoulder_roll_link'], fork=('L shoulder roll post', 'L shoulder roll axle', 'L_shoulder_pitch'), cases=LIFTS, motors=[],
        note='a pitch moment reaches the fork as a couple over the bearing spacing'),
    sec('Shoulder roll axle (back of the fork)', ARM, 'L_shoulder_roll_link', ['L shoulder roll axle'], 'x', ('L shoulder roll axle', 'c', 0.0),
        load='limb', distal=['L_shoulder_yaw_link'], fork=('L shoulder roll axle', 'L shoulder roll post', 'L_shoulder_pitch'), cases=LIFTS, motors=[('L_shoulder_roll', 1)]),
    sec('Fork plates, upper, solid', ARM, 'L_shoulder_roll_link', ['L shoulder roll plate back', 'L shoulder roll plate front'], 'z', ('L shoulder roll plate back', 'lo', 0.004),
        load='limb', distal=['L_shoulder_yaw_link'], own=-1, cases=LIFTS, motors=[('L_shoulder_pitch', 1), ('L_shoulder_roll', 1)]),
    sec('Fork plates at the roll axis (front net of its bore)', ARM, 'L_shoulder_roll_link', ['L shoulder roll plate back', 'L shoulder roll plate front'], 'z', ('joint:L_shoulder_roll', 'c', 0.0),
        load='limb', distal=['L_shoulder_yaw_link'], own=-1, cases=LIFTS, motors=[('L_shoulder_pitch', 1), ('L_shoulder_roll', 1)]),
    sec('Fork front plate alone, net of its bore', ARM, 'L_shoulder_roll_link', ['L shoulder roll plate front'], 'z', ('joint:L_shoulder_roll', 'c', 0.0),
        load='limb', distal=[], cases=('motor',), motors=[], fork=('L shoulder roll post', 'L shoulder roll axle', 'L_shoulder_pitch'), note='takes the pitch couple alone'),
    sec('Fork plates, lower', ARM, 'L_shoulder_roll_link', ['L shoulder roll plate back low', 'L shoulder roll plate front low'], 'z', ('L shoulder roll plate back low', 'hi', -0.004),
        load='limb', distal=['L_shoulder_yaw_link'], cases=LIFTS, motors=[('L_shoulder_pitch', 1), ('L_shoulder_roll', 1), ('L_shoulder_yaw', 1)]),
    sec('Fork bridge', ARM, 'L_shoulder_roll_link', ['L shoulder roll bridge'], 'x', ('L shoulder roll bridge', 'c', 0.0),
        load='limb', distal=['L_shoulder_yaw_link'], cases=('peak', 'payload'), note='ties the plates; read as a beam across them'),
    sec('Upper arm frame', ARM, 'L_shoulder_yaw_link', ['L upper arm frame'], 'z', ('L upper arm frame', 'c', 0.0),
        load='limb', distal=['L_elbow_link'], cases=LIFTS, motors=[('L_shoulder_yaw', 1), ('L_elbow', 1)]),
    sec('Elbow bridge beside the frame', ARM, 'L_shoulder_yaw_link', ['L elbow bridge'], 'y', ('L upper arm frame', 'lo', -0.0),
        load='limb', distal=['L_elbow_link'], cases=LIFTS, motors=[('L_elbow', 1)], note='all the arm load through one side, as drawn'),
    sec('Elbow cheeks over the discs, both', ARM, 'L_shoulder_yaw_link', ['L elbow cheek in', 'L elbow cheek out'], 'z', ('L elbow cheek disc in', 'hi', 0.002),
        load='limb', distal=['L_elbow_link'], cases=('peak', 'walk', 'payload')),
    sec('Elbow inner cheek alone (the only one that takes motor torque)', ARM, 'L_shoulder_yaw_link', ['L elbow cheek in'], 'z', ('L elbow cheek disc in', 'hi', 0.002),
        load='limb', distal=['L_elbow_link'], cases=('peak', 'walk', 'motor', 'payload'), motors=[('L_elbow', 1)], note='all the arm load as drawn: the outer cheek has air at its bearing'),
    sec('Elbow cheek discs at the bores (net of them)', ARM, 'L_shoulder_yaw_link', ['L elbow cheek disc in', 'L elbow cheek disc out'], 'z', ('joint:L_elbow', 'c', 0.0),
        load='limb', distal=['L_elbow_link'], cases=('peak', 'walk', 'motor', 'payload'), motors=[('L_elbow', 1)]),
    sec('Elbow axle', ARM, 'L_shoulder_yaw_link', ['L elbow axle'], 'y', ('L elbow axle', 'c', 0.0),
        load='limb', distal=['L_elbow_link'], cases=('peak', 'motor', 'payload'), motors=[('L_elbow', 1)]),
    sec('Elbow post', ARM, 'L_elbow_link', ['L elbow post'], 'y', ('L elbow cheek disc out', 'c', 0.0),
        load='limb', distal=['L_wrist_yaw_link'], own=-1, cases=('peak', 'payload')),
    sec('Forearm frame', ARM, 'L_elbow_link', ['L forearm frame'], 'z', ('L_wrist_yaw actuator (buy) RS02', 'hi', 0.005),
        load='limb', distal=['L_wrist_yaw_link'], cases=LIFTS, motors=[('L_elbow', 1), ('L_wrist_yaw', 1)]),
    sec('Elbow idler arm', ARM, 'L_elbow_link', ['L elbow idler arm'], 'z', ('L elbow idler arm', 'c', 0.0),
        load='limb', distal=['L_wrist_yaw_link'], cases=('peak', 'payload'), pin=('L elbow idler stub', 'c', (0, 0, 0)), share=0.5, note='half the forearm load as a force at the stub, if a bearing existed'),
    sec('Under the elbow motor', ARM, 'L_elbow_link', ['L elbow under motor'], 'y', ('joint:L_elbow', 'c', -0.02),
        load='limb', distal=['L_wrist_yaw_link'], cases=('peak', 'payload'), pin=('L elbow idler stub', 'c', (0, 0, 0)), share=0.5, note='half the forearm load as a force at the stub, if a bearing existed'),
    sec('Forearm lower frame', ARM, 'L_wrist_yaw_link', ['L forearm lower frame'], 'z', ('L forearm lower frame', 'c', 0.0),
        load='limb', distal=['L_wrist_pitch_link'], cases=LIFTS, motors=[('L_wrist_yaw', 1), ('L_wrist_pitch', 1)]),
    sec('Wrist cheek', ARM, 'L_wrist_pitch_link', ['L wrist cheek'], 'z', ('L wrist cheek', 'c', 0.0),
        load='limb', distal=['L_hand'], cases=LIFTS, motors=[('L_wrist_pitch', 1), ('L_wrist_roll', 1)]),
    sec('Hand bracket', ARM, 'L_hand', ['L hand bracket'], 'z', ('L hand bracket', 'c', 0.0),
        load='limb', distal=[], own=-1, cases=LIFTS, motors=[('L_wrist_roll', 1)]),

    # ------------------------------------------------------------ neck to head
    sec('Neck pitch post', NECK, 'neck_yaw_link', ['neck pitch post'], 'z', ('neck pitch post', 'c', 0.0),
        load='limb', distal=['head'], cases=FALLS, motors=[('neck_yaw', 1), ('neck_pitch', 1)]),
    sec('Head frame', NECK, 'head', ['head frame'], 'z', ('neck_pitch flange', 'hi', -0.0),
        load='limb', distal=[], own=1, cases=FALLS, motors=[('neck_pitch', 1)]),

    # ------------------------------------------------------------ ankle rods and cranks: motor peak, through each
    sec('Pitch rod', ROD, 'L_knee_link', ['L ankle pitch rod'], ('along', 'L ankle pitch rod'), ('L ankle pitch rod', 'mid', 0.0),
        load='leg', cases=('motor',), motors=[], rods=['pitch'], buckle=(1.0, 'L ankle pitch rod'), note='Euler, pinned both ends, solid'),
    sec('Roll rod', ROD, 'L_knee_link', ['L ankle roll rod'], ('along', 'L ankle roll rod'), ('L ankle roll rod', 'mid', 0.0),
        load='leg', cases=('motor',), motors=[], rods=['roll'], buckle=(1.0, 'L ankle roll rod'), note='Euler, pinned both ends, solid'),
    sec('Pitch crank at the motor', ROD, 'L_knee_link', ['L ankle pitch crank'], ('along', 'L ankle pitch crank'), ('L ankle pitch crank', 'p0', 0.002),
        load='leg', cases=('motor',), motors=[], rods=['pitch']),
    sec('Roll crank at the motor', ROD, 'L_knee_link', ['L ankle roll crank'], ('along', 'L ankle roll crank'), ('L ankle roll crank', 'p0', 0.002),
        load='leg', cases=('motor',), motors=[], rods=['roll']),
    sec('Ankle pitch arm', ROD, 'L_ankle_pitch_link', ['L ankle pitch arm'], ('along', 'L ankle pitch arm'), ('L ankle pitch arm', 'p0', 0.002),
        load='leg', cases=('motor',), motors=[], rods=['pitch']),
    sec('Roll pin block', ROD, 'L_foot', ['L ankle roll pin'], 'z', ('L ankle roll pin', 'c', 0.0),
        load='leg', cases=('motor',), motors=[], rods=['roll']),
]

# The contacts each path crosses: (path, part a, part b, the force through it). A number scales F_PEAK; 'arm', 'head', 'rod pitch',
# 'rod roll' or 'body' take it from the masses or the motors.
INTERFACES = [
    (LEG, 'L foot plate', 'L ankle clevis front', 0.5), (LEG, 'L foot plate', 'L ankle clevis back', 0.5),
    (LEG, 'L ankle fork cheek in', 'L shank plate in', 0.5), (LEG, 'L ankle fork cheek out', 'L shank plate out', 0.5),
    (LEG, 'L shank plate in', 'L shank body', 0.5), (LEG, 'L shank plate out', 'L shank body', 0.5),
    (LEG, 'L shank body', 'L shank back panel', 0.1), (LEG, 'L shank body', 'L knee rear cap', KNEE_SPLIT), (LEG, 'L shank body', 'L_knee actuator (buy) RS04', 0.67),
    (LEG, 'L thigh fork', 'L_knee flange', KNEE_SPLIT), (LEG, 'L thigh fork', 'L knee outer cheek', KNEE_SPLIT), (LEG, 'L thigh fork', 'L_hip_yaw flange', 1.0),
    (LEG, 'L hip yoke', 'L hip yaw ring', 1.0), (LEG, 'L hip yaw ring', 'L_hip_yaw actuator (buy) RS03', 1.0), (LEG, 'L hip yoke', 'L_hip_roll flange', 1.0),
    (LEG, 'L hip carrier', 'L_hip_roll actuator (buy) RS04', 1.0), (LEG, 'L hip carrier', 'L_hip_pitch flange', 1.0),
    (LEG, 'pelvis frame', 'L_hip_pitch actuator (buy) RS04', 0.67), (LEG, 'pelvis frame', 'L hip ring', 1.0),
    (BODY, 'pelvis frame', 'waist_yaw actuator (buy) RS04', 'body'), (BODY, 'pelvis frame', 'waist ring', 'body'), (BODY, 'chest frame', 'waist_yaw flange', 'body'),
    (BODY, 'chest frame', 'L_shoulder_pitch actuator (buy) RS04', 'arm'),
    (ARM, 'L shoulder bracket', 'L_shoulder_pitch flange', 'arm'), (ARM, 'L shoulder bracket', 'L_shoulder_roll actuator (buy) RS06', 'arm'),
    (ARM, 'L shoulder roll axle', 'L_shoulder_roll flange', 'arm'), (ARM, 'L shoulder roll axle', 'L shoulder roll plate back', 'arm'),
    (ARM, 'L shoulder roll plate back', 'L shoulder roll plate back low', 'arm'), (ARM, 'L shoulder roll plate front', 'L shoulder roll plate front low', 'arm'),
    (ARM, 'L shoulder roll plate back low', 'L shoulder roll bridge', 'arm'), (ARM, 'L shoulder roll plate front low', 'L shoulder roll bridge', 'arm'),
    (ARM, 'L shoulder roll plate back low', 'L_shoulder_yaw actuator (buy) RS02', 'arm'), (ARM, 'L shoulder roll plate front low', 'L_shoulder_yaw actuator (buy) RS02', 'arm'),
    (ARM, 'L upper arm frame', 'L_shoulder_yaw flange', 'arm'), (ARM, 'L upper arm frame', 'L elbow bridge', 'arm'),
    (ARM, 'L elbow bridge', 'L elbow cheek in', 'arm'), (ARM, 'L elbow bridge', 'L elbow cheek out', 'arm'),
    (ARM, 'L elbow cheek in', 'L elbow cheek disc in', 'arm'), (ARM, 'L elbow cheek out', 'L elbow cheek disc out', 'arm'),
    (ARM, 'L elbow cheek disc in', 'L elbow axle', 'arm'), (ARM, 'L elbow axle', 'L_elbow flange', 'arm'),
    (ARM, 'L elbow post', 'L forearm frame', 'arm'), (ARM, 'L forearm frame', 'L_elbow actuator (buy) RS06', 'arm'), (ARM, 'L elbow idler stub', 'L elbow idler arm', 'arm'),
    (ARM, 'L elbow idler arm', 'L elbow under motor', 'arm'), (ARM, 'L elbow under motor', 'L forearm frame', 'arm'),
    (ARM, 'L forearm frame', 'L_wrist_yaw actuator (buy) RS02', 'arm'), (ARM, 'L forearm lower frame', 'L_wrist_yaw flange', 'arm'),
    (ARM, 'L forearm lower frame', 'L_wrist_pitch actuator (buy) RS00', 'arm'), (ARM, 'L wrist cheek', 'L_wrist_pitch flange', 'arm'),
    (ARM, 'L wrist cheek', 'L_wrist_roll actuator (buy) RS00', 'arm'), (ARM, 'L hand bracket', 'L_wrist_roll flange', 'arm'),
    (NECK, 'chest frame', 'neck_yaw actuator (buy) RS00', 'head'),
    (NECK, 'neck pitch post', 'neck_yaw flange', 'head'), (NECK, 'neck pitch post', 'neck_pitch actuator (buy) RS00', 'head'), (NECK, 'head frame', 'neck_pitch flange', 'head'),
    (ROD, 'L ankle cross', 'L ankle cross tail', 'rod pitch'), (ROD, 'L ankle cross tail', 'L ankle pitch arm', 'rod pitch'),
    (ROD, 'L foot plate', 'L ankle roll pin', 'rod roll'),
]

# Rigid parts on no path above, each with its reason. A new part must go in a table: see check_coverage.
NOT_ON_PATH = {
    'R hip ring': 'the right hip\'s ring: its left twin is in INTERFACES', 'R hip pitch retaining ring': 'hardware', 
    'L ankle fork bridge': 'ties the fork cheeks; the plates land in line with the cheeks, so the path passes beside it',
    'battery': 'a passenger: role struct, but cells are not frame', 'battery lower': 'a passenger: role struct, but cells are not frame',
    'compute': 'a passenger', 'head electronics': 'a passenger',
}


# ---------------------------------------------------------------- the model: masses, parts, planes

class Model:
    def __init__(self, spec, S):
        import mass_model as MM
        self.spec, self.S, self.MM = spec, S, MM
        self.R = G.Robot(spec)
        self.part = {}
        for ln, l in spec['links'].items():
            for i, g in enumerate(l['geoms']):
                if g['role'] != 'cut':
                    self.part[g['name']] = (ln, i, g)
        self.joint = {j['name']: j for j in spec['joints']}
        self.children = self.R.children
        inert = MM.link_inertials(spec, S)
        self.link_mass = {ln: (v[0], np.array(v[1], float)) for ln, v in inert.items()}
        self.total = sum(v[0] for v in self.link_mass.values())

    def geom_man(self, name):
        p = self.part.get(name)
        return None if p is None else self.S.geom.get((p[0], p[1]))

    def own_parts(self, link):
        k = 1 + self.MM.WIRING
        return [(p[0], p[2] * k, np.array(p[3], float)) for p in self.MM.parts(self.spec, self.S, link)]

    def cap(self, jname):
        c = self.joint[jname]['capacity']
        return self.spec['actuators'][c['model']]['peak_nm'] * c['count'] * c['ratio'] if c else None

    def point(self, ref):
        """A world point: ('joint:NAME' or part name, 'c', (dx, dy, dz))."""
        name, _, off = ref
        if name.startswith('joint:'):
            return np.array(self.joint[name[6:]]['origin'], float) + off
        p = self.part.get(name)
        if p is None:
            return None
        g = p[2]
        if 'c' in g:
            return np.array(g['c'], float) + off
        lo, hi = bounds(self.geom_man(name))
        return (lo + hi) / 2 + off

    def rod(self, which):
        """(force N, unit direction, point at the far end of the rod), or None when the rod or crank is gone."""
        crank, rod = self.part.get(f'L ankle {which} crank'), self.part.get(f'L ankle {which} rod')
        j = self.joint[f'L_ankle_{which}']
        if crank is None or rod is None:
            return None
        c = crank[2]
        arm = np.linalg.norm(np.array(c['p1'], float) - np.array(c['p0'], float))
        p0, p1 = np.array(rod[2]['p0'], float), np.array(rod[2]['p1'], float)
        peak = self.spec['actuators'][j['capacity']['model']]['peak_nm']
        return peak / arm, (p1 - p0) / np.linalg.norm(p1 - p0), p1


def bounds(m):
    b = m.bounding_box()
    return np.array(b[:3]), np.array(b[3:])


def frame_of(axis, M):
    if isinstance(axis, str):
        n = G.AXIS[axis]
    else:
        g = M.part[axis[1]][2]
        n = np.array(g['p1'], float) - np.array(g['p0'], float)
        n = n / np.linalg.norm(n)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(a, n)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u), n


def plane_at(at, n, M):
    """Distance along n of the plane, or None if the part it hangs on is gone."""
    if not isinstance(at, tuple):
        return float(at)
    name, where, off = at
    if name.startswith('joint:'):
        return float(np.array(M.joint[name[6:]]['origin'], float) @ n) + off
    p = M.part.get(name)
    if p is None:
        return None
    g = p[2]
    if where in ('p0', 'mid'):
        p0, p1 = np.array(g['p0'], float), np.array(g['p1'], float)
        return float((p0 if where == 'p0' else (p0 + p1) / 2) @ n) + off
    m = M.geom_man(name)
    v = np.asarray(m.to_mesh().vert_properties[:, :3], float) @ n
    return {'lo': float(v.min()), 'hi': float(v.max()), 'c': float((v.min() + v.max()) / 2)}[where] + off


# ---------------------------------------------------------------- section properties

def polygon_props(polys):
    """Area, centroid, second moments about the origin (mm), from signed polygons: holes come back the other way round."""
    A = cx = cy = Iuu = Ivv = Iuv = 0.0
    for p in polys:
        x, y = p[:, 0], p[:, 1]
        x1, y1 = np.roll(x, -1), np.roll(y, -1)
        c = x * y1 - x1 * y
        A += c.sum() / 2
        cx += ((x + x1) * c).sum() / 6
        cy += ((y + y1) * c).sum() / 6
        Iuu += ((y * y + y * y1 + y1 * y1) * c).sum() / 12      # int v^2
        Ivv += ((x * x + x * x1 + x1 * x1) * c).sum() / 12      # int u^2
        Iuv += ((x * y1 + 2 * x * y + 2 * x1 * y1 + x1 * y) * c).sum() / 24
    return A, cx, cy, Iuu, Ivv, Iuv


def _seg_dist(p, poly):
    a, b = poly, np.roll(poly, -1, axis=0)
    ab = b - a
    t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    return float(np.min(np.linalg.norm(a + ab * t[:, None] - p, axis=1)))


class Piece:
    """One connected island of a slice."""

    def __init__(self, polys):
        self.polys = polys
        A, sx, sy, Iuu, Ivv, Iuv = polygon_props(polys)
        self.A = A
        self.c = np.array([sx / A, sy / A])
        self.Iuu = Iuu - A * self.c[1] ** 2
        self.Ivv = Ivv - A * self.c[0] ** 2
        self.Iuv = Iuv - A * self.c[0] * self.c[1]
        Ip = self.Iuu + self.Ivv
        d = math.hypot((self.Iuu - self.Ivv) / 2, self.Iuv)
        self.Imin, self.Imax = Ip / 2 - d, Ip / 2 + d
        self.t = math.sqrt(max(12 * self.Imin / A, 1e-12))       # the thickness and width of the rectangle with this A and I
        self.b = math.sqrt(max(12 * self.Imax / A, 1e-12))
        self.J = A ** 4 / (40 * Ip)                              # Saint-Venant: within 2 % for a circle, 10 % for a thin strip
        # distance giving tau = T c / J: 0.5 d on a circle, 0.68 a on a square, t on a thin strip
        self.cJ = self.t * (0.68 + 0.32 * (1 - self.t / self.b))
        # A closed section (a hole taking a real share of its outline): Saint-Venant's formula is for solid shapes, and read a d66 x 2.5 tube at 470 x its real shear stress (iteration 41). Bredt's: tau = T / (2 Am t), with the
        # thinnest wall found, J = 4 Am^2 t_mean / perimeter. A small screw hole in a plate isn't this: the plate still twists as a solid.
        areas = sorted(((abs(0.5 * (np.dot(q[:, 0], np.roll(q[:, 1], -1)) - np.dot(q[:, 1], np.roll(q[:, 0], -1)))), q) for q in polys), key=lambda x: -x[0])
        if len(areas) >= 2 and areas[1][0] > 0.25 * areas[0][0]:
            Ao, qo = areas[0]
            Ai, qi = areas[1]
            Am = (Ao + Ai) / 2
            per = lambda q: float(np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1).sum())
            pm = (per(qo) + per(qi)) / 2
            tmean = A / pm
            step = max(len(qi) // 200, 1)
            tmin = float(min(min(_seg_dist(p, qo) for p in qi[::step]), tmean))
            self.J = 4 * Am * Am * tmean / pm
            self.cJ = self.J / (2 * Am * max(tmin, 0.1))


def slice_section(man, u, v, n, at):
    """Pieces of the solid at the plane, in mm, in the plane's own axes."""
    R = np.vstack([u, v, n]) * 1000.0
    T = np.column_stack([R, np.zeros(3)])
    cs = man.transform(T.astype(np.float64)).slice(at * 1000.0)
    out = []
    for part in cs.decompose():
        polys = [np.asarray(p, float) for p in part.to_polygons()]
        if polys and abs(part.area()) > 1e-3:
            out.append(Piece(polys))
    return out


def size_text(pieces):
    """'2 x 40.0 x 4.0' style: the measured box of each piece, along its own principal axes."""
    txt = {}
    for p in pieces:
        pts = np.vstack(p.polys) - p.c
        th = 0.5 * math.atan2(2 * -p.Iuv, p.Iuu - p.Ivv) if abs(p.Iuv) > 1e-6 else 0.0
        c, s = math.cos(th), math.sin(th)
        du, dv = pts[:, 0] * c + pts[:, 1] * s, -pts[:, 0] * s + pts[:, 1] * c
        a, b = sorted((du.max() - du.min(), dv.max() - dv.min()))
        k = f'{b:.1f} x {a:.1f}'
        txt[k] = txt.get(k, 0) + 1
    return ' + '.join(f'{v} x {k}' if v > 1 else k for k, v in txt.items())


# ---------------------------------------------------------------- load elements

def F(f, p, link=None):
    return ('F', np.array(f, float), np.array(p, float), link)


def Mo(m):
    return ('M', np.array(m, float))


def wrench(elems, c):
    """Force and moment (N, N m) of the elements about c (m)."""
    f, m = np.zeros(3), np.zeros(3)
    for e in elems:
        if e[0] == 'F':
            f += e[1]
            m += np.cross(e[2] - c, e[1])
        else:
            m += e[1]
    return f, m


def apply_caps(M, elems, link):
    """A joint back-drives past its peak: scale what lies beyond it so its moment about its axis stays at the peak. Only for
    the joints between the section's link (with its own proximal joint) and each load."""
    joints = {}
    for e in elems:
        if e[0] == 'F' and e[3]:
            chain = M.R.chain(e[3])
            start = next((i for i, j in enumerate(chain) if j['child'] == link), None)
            for j in (chain if start is None else chain[start:]):
                joints[j['name']] = j
    out = list(elems)
    for j in sorted(joints.values(), key=lambda j: -len(M.R.chain(j['child']))):
        cap = M.cap(j['name'])
        if cap is None:
            continue
        beyond = set(M.R.subtree(j['child']))
        a, o = G.AXIS[j['axis']], np.array(j['origin'], float)
        idx = [i for i, e in enumerate(out) if e[0] == 'F' and e[3] in beyond]
        tq = sum(np.cross(out[i][2] - o, out[i][1]) @ a for i in idx)
        if abs(tq) > cap:
            k = cap / abs(tq)
            for i in idx:
                out[i] = ('F', out[i][1] * k, out[i][2], out[i][3])
    return out


def mass_points(M, s, sign_plane=None):
    """[(mass, com, link)] of what is beyond the cut: each distal root's subtree, and this link's parts on the far side."""
    pts = []
    for root in s.get('distal', []):
        for ln in M.R.subtree(root):
            m, c = M.link_mass[ln]
            pts.append((m, c, ln))
    if s.get('own') and sign_plane is not None:
        n, at = sign_plane
        for name, m, c in M.own_parts(s['link']):
            if s['own'] * (c @ n - at) > 0:
                pts.append((m, c, s['link']))
    return pts


def hands_in(M, s):
    out = []
    for root in s.get('distal', []) + ([s['link']] if s.get('own') else []):
        for ln in M.R.subtree(root):
            if ln.endswith('_hand'):
                palm = M.part.get(f'{ln[0]} palm')
                if palm:
                    out.append((np.array(palm[2]['c'], float), ln))
    return out


def inertial_alts(M, s, a, dirs, plane, payload=0.0):
    alts = []
    pts = mass_points(M, s, plane)
    hands = hands_in(M, s) if payload else []
    for d in dirs:
        d = np.array(d, float)
        el = [F(m * a * d, c, ln) for m, c, ln in pts] + [F(payload * a * d, p, ln) for p, ln in hands]
        el = apply_caps(M, el, s['link'])
        k = s.get('share', 1.0)
        el = [('F', e[1] * k, e[2], e[3]) for e in el]
        if s.get('pin'):                                  # a bearing passes force, not moment: the resultant, at the bearing
            p = M.point(s['pin'])
            el = [('F', e[1], p, e[3]) for e in el]
        if s.get('hip_moment'):
            mx, my = M.cap('L_hip_roll'), M.cap('L_hip_pitch')
            alts += [el + [Mo((sx * mx, sy * my, 0))] for sx in (1, -1) for sy in (1, -1)]
        else:
            alts.append(el)
    return alts


AXES6 = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
Z2 = [(0, 0, 1), (0, 0, -1)]


def leg_line(M, f):
    """The force line, off the axes by what the ankle motors can hold; never outside the sole."""
    o = np.array(M.joint['L_ankle_pitch']['origin'], float)
    dx = M.cap('L_ankle_pitch') / f
    dy = min(M.cap('L_ankle_roll') / f, SOLE_Y)
    pts = [o + (sx, sy * dy, 0) for sx in (min(dx, SOLE_X[1]), -min(dx, -SOLE_X[0])) for sy in (1, -1)]
    return [[F((0, 0, f), (p[0], p[1], 0.0), 'L_foot')] for p in pts]


def alts_for(M, s, case, plane):
    load = s.get('load', 'leg')
    if case == 'motor':
        return motor_alts(M, s)
    if load == 'leg':
        if case == 'side':
            o = np.array(M.joint['L_ankle_pitch']['origin'], float)
            if s.get('side_local'):
                p = M.point(s['side_local'])
                return [[F((0, sy * F_SIDE, 0), p)] for sy in (1, -1)]
            dx = M.cap('L_ankle_pitch') / F_PEAK
            return [[F((0, sy * F_SIDE, F_PEAK), o + (sx * dx, 0, 0))] for sx in (1, -1) for sy in (1, -1)]
        f = F_PEAK if case == 'peak' else F_WALK
        if s.get('local'):
            out = []
            for frac, ref in s['local']:
                p = M.point(ref)
                if p is None:
                    return None
                out.append(F((0, 0, f * frac), p))
            return [out]
        if s.get('toe'):
            o = np.array(M.joint['L_ankle_pitch']['origin'], float)
            return [[F((0, 0, f), (o[0] + x, o[1], 0.0))] for x in SOLE_X]
        return leg_line(M, f)
    dirs = s.get('dirs') or (Z2 if load == 'upper' else AXES6)
    if case == 'peak':
        return inertial_alts(M, s, F_PEAK / M.total, dirs, plane)
    if case == 'walk':
        return inertial_alts(M, s, F_WALK / M.total, dirs, plane)
    if case == 'payload':
        return inertial_alts(M, s, G_ACC, s.get('dirs') or ([(1, 0, 0), (-1, 0, 0), (0, 0, -1)] if load == 'upper' else AXES6), plane, payload=HAND_KG)
    raise ValueError(case)


def motor_alts(M, s):
    alts = []
    if s.get('fork'):
        a, b, jn = s['fork']
        pa, pb = M.point((a, 'c', (0, 0, 0))), M.point((b, 'c', (0, 0, 0)))
        if pa is None or pb is None:
            return None
        lev = np.linalg.norm(pa - pb)
        v = M.cap(jn) / lev
        alts += [[F((0, 0, sg * v), pa)] for sg in (1, -1)]
    motors = s.get('motors')
    if motors is None:
        motors = [(j['name'], 1) for j in M.spec['joints'] if j['capacity'] and not j['name'].startswith('R_') and s['link'] in (j['parent'], j['child'])]
    for jn, k in motors:
        a = G.AXIS[M.joint[jn]['axis']]
        alts += [[Mo(sg * k * M.cap(jn) * a)] for sg in (1, -1)]
    for r in s.get('rods', []):
        got = M.rod(r)
        if got is None:
            return None
        f, d, p = got
        alts += [[F(sg * f * d, p)] for sg in (1, -1)]
    return alts


# ---------------------------------------------------------------- stress at a section

def stress(pieces, elems, u, v, n, at, solo=False):
    """Worst sigma and tau (MPa) from the elements (forces in N at points in m). Returns a dict of the parts of the load."""
    A = sum(p.A for p in pieces)
    c2 = sum(p.A * p.c for p in pieces) / A
    c3 = u * c2[0] / 1000 + v * c2[1] / 1000 + n * at
    f, m = wrench(elems, c3)
    Nf = f @ n
    Vu, Vv = f @ u, f @ v
    V = math.hypot(Vu, Vv)
    T = m @ n * 1000.0                                     # N mm
    Mu, Mv = m @ u * 1000.0, m @ v * 1000.0
    Mb = math.hypot(Mu, Mv)
    if solo:
        # each piece on its own: load shared by its stiffness in that direction, no help from the other's distance
        Iuu_t = sum(p.Iuu for p in pieces)
        Ivv_t = sum(p.Ivv for p in pieces)
        worst = s_axial = s_bend = 0.0
        for p in pieces:
            Na = Nf * p.A / A
            mu, mv = Mu * p.Iuu / Iuu_t, Mv * p.Ivv / Ivv_t
            det = p.Iuu * p.Ivv - p.Iuv ** 2
            a_ = (-mv * p.Iuu - mu * p.Iuv) / det
            b_ = (mu * p.Ivv + mv * p.Iuv) / det
            pts = np.vstack(p.polys) - p.c
            sig = a_ * pts[:, 0] + b_ * pts[:, 1]
            if abs(Na / p.A) + np.abs(sig).max() > worst:
                worst = abs(Na / p.A) + np.abs(sig).max()
                s_axial, s_bend = abs(Na / p.A), np.abs(sig).max()
        tau_t = max(abs(T) * p.J / sum(q.J for q in pieces) * p.cJ / p.J for p in pieces)
        tau_v = 1.5 * math.hypot(Vu * 1.0, Vv * 1.0) / A
    else:
        Iuu = sum(p.Iuu + p.A * (p.c[1] - c2[1]) ** 2 for p in pieces)
        Ivv = sum(p.Ivv + p.A * (p.c[0] - c2[0]) ** 2 for p in pieces)
        Iuv = sum(p.Iuv + p.A * (p.c[0] - c2[0]) * (p.c[1] - c2[1]) for p in pieces)
        det = Iuu * Ivv - Iuv ** 2
        # sigma = N/A + a du + b dv; Mu = int dv sigma, Mv = -int du sigma
        a_ = (-Mv * Iuu - Mu * Iuv) / det
        b_ = (Mu * Ivv + Mv * Iuv) / det
        pts = np.vstack([np.vstack(p.polys) for p in pieces]) - c2
        sig = a_ * pts[:, 0] + b_ * pts[:, 1]
        s_axial = abs(Nf) / A
        s_bend = np.abs(sig).max()
        worst = np.abs(Nf / A + sig).max()
        s_axial, s_bend = (s_axial, worst - s_axial) if worst > s_axial else (worst, 0.0)
        # torsion: each piece twists on its own and the pieces pass a couple between them, as far as they are tied
        Jt = sum(p.J + p.A * ((p.c - c2) @ (p.c - c2)) for p in pieces)
        tau_t = max(abs(T) * (p.cJ / Jt + (1.5 * math.sqrt((p.c - c2) @ (p.c - c2)) / Jt)) for p in pieces)
        tau_v = 1.5 * V / A
    tau = tau_t + tau_v
    vm = math.sqrt(worst ** 2 + 3 * tau ** 2)
    return dict(vm=vm, sigma=worst, tau=tau, s_axial=s_axial, s_bend=s_bend, N=Nf, V=V, Mb=Mb / 1000.0, T=T / 1000.0)


# ---------------------------------------------------------------- real screws and bearings (tools/hw_checks.py reads these tables; assumptions: spec/assumptions.json)

# Screw tension at the top of its load. Preload alone is 0.75 of proof, so the ceiling is 1.33: asking for more than 1.1 would only forbid preloading. It's proof, not break, so it does yield margin of its own.
SF_TENSION = 1.1


def assumed(spec, key):
    return next(r['value'] for r in spec['hardware']['assumptions'] if r['id'] == key)


def screw_load(spec, h, le):
    """(preload N, strip force N, proof N, stress area mm2, what limits the preload). Preload: the smaller of the assumed fraction of proof, and the thread's strip force over its factor."""
    sys.path.insert(0, os.path.join(ROOT, 'spec'))
    import hardware as HW
    data = HW.screw_data(h['thread'], h['head'])
    tau = assumed(spec, 'shear_al_motor') if str(h['into']).startswith('motor:') else assumed(spec, 'shear_al_part')
    strip = HW.strip_area(h['d'], data['pitch'], max(le, 0.0)) * tau
    p_proof = assumed(spec, 'preload_fraction') * data['proof']
    p_strip = strip / assumed(spec, 'strip_factor')
    return min(p_proof, p_strip), strip, data['proof'], data['As'], 'thread strip' if p_strip < p_proof else f"{assumed(spec, 'preload_fraction'):.2f} x proof"


def engagement(h):
    """Thread engaged and the hole's depth, mm, from the record: the tip's distance past the face it enters."""
    tip = np.array(h['pos'], float) + np.array(h['dir'], float) * h['length'] / 1000
    return float((tip - np.array(h['face'], float)) @ np.array(h['dir'], float) * 1000), h['hole_depth']


def pattern_screws(spec, name):
    return [g['hw'] for l in spec['links'].values() for g in l['geoms'] if g.get('hw', {}).get('type') == 'screw' and g['hw'].get('pattern') == name]


def bolted_rows(M):
    """One row per screw pattern on the left side and centre: T the motor's peak; V the force across the joint; Mt the tilting moment. Formulas, per pattern of n screws on radius r:
        slip        SF = mu n Fp / (V + T / r)              the joint holds by friction: mu, Fp from the assumptions
        separation  SF = Fp / (Fa / n + 2 Mt / (n r))       the most loaded screw's share against its preload (2 M / (n r): a circle of screws)
        shear       SF = 0.6 Rm As / (V / n + T / (n r))    if it slipped, each screw in shear: 0.6 x its breaking stress (from memory, an estimate), with the threads in the plane. Pins not counted
        tension     SF = proof / (Fp + phi Ft)              phi the joint stiffness factor
        strip       SF = F_strip / (Fp + phi Ft)            the thread in the aluminium"""
    spec = M.spec
    mu, phi = assumed(spec, 'friction_joint'), assumed(spec, 'joint_stiffness_factor')
    rows = []
    for p in spec['hardware']['patterns']:
        if p['name'].startswith('R '):
            continue
        hw = pattern_screws(spec, p['name'])
        if not hw:
            rows.append(dict(name=p['name'], joint=p['joint'], n=0, sf=0.0, why='no screws placed'))
            continue
        h = hw[0]
        n = len(hw)
        r = p['r_bc'] / 1000
        T = M.cap(p['joint'])
        # The design: the two supports take the force, so the screws take the motor's torque (assumptions.json support_shared_by_levers). The what-if: nothing else takes it, the wrench of 'Output bearings' below.
        wrench_row = next((b for b in bearing_table(M) if b['joint'] == p['joint'] and b['case'] == 'peak'), None)
        for label, V, Mt, Fa in (('the design, the supports take the force', 0.0, 0.0, 0.0), ('what if the motor carried it all', wrench_row['Fr'] if wrench_row else F_PEAK, wrench_row['Mt'] if wrench_row else 0.0, F_SIDE)):
            le = min(engagement(x)[0] for x in hw)
            Fp, strip, proof, As, limit = screw_load(spec, h, le)
            Ft = Fa / n + 2 * Mt / (n * r)
            rm = 1220.0                    # 12.9: nominal breaking stress (ISO 898-1)
            sf = dict(slip=mu * n * Fp / (V + T / r), separation=Fp / Ft if Ft > 0 else float('inf'), shear=0.6 * rm * As / (V / n + T / (n * r)), tension=proof / (Fp + phi * Ft), strip=strip / (Fp + phi * Ft))
            ok = min(sf['slip'], sf['separation'], sf['shear'], sf['strip']) >= SF_WALK and sf['tension'] >= SF_TENSION
            torque = assumed(spec, 'torque_coefficient') * Fp * h['d'] / 1000
            rows.append(dict(name=p['name'], joint=p['joint'], label=label, whatif=label.startswith('what if'), n=n, count=p['count'], thread=h['thread'], r=r * 1000, le=le, Fp=Fp, limit=limit, strip=strip, torque=torque, T=T, V=V, Mt=Mt, Fa=Fa, sf_each=sf, sf=min(sf['slip'], sf['separation'], sf['shear'], sf['strip'], sf['tension'] * SF_MIN / SF_TENSION), ok=ok))
    # our own screws in our own parts: each group says what it must pass (spec/robot.py, HW.bolt_group)
    for p in spec['hardware'].get('groups', []):
        if p['name'].startswith('R '):
            continue
        hw = pattern_screws(spec, p['name'])
        if not hw:
            rows.append(dict(name=p['name'], joint='', n=0, sf=0.0, why='no screws placed'))
            continue
        h, n, ld = hw[0], len(hw), p['load']
        r = ld.get('r', 0.03)
        T, V, Mt, Fa = ld.get('T', 0.0), ld.get('V', 0.0), ld.get('Mt', 0.0), ld.get('Fa', 0.0)
        le = min(engagement(x)[0] for x in hw)
        Fp, strip, proof, As, limit = screw_load(spec, h, le)
        Ft = Fa / n + 2 * Mt / (n * r)
        rm = 1220.0
        sf = dict(slip=mu * n * Fp / max(V + T / r, 1e-6), separation=Fp / Ft if Ft > 0 else float('inf'), shear=0.6 * rm * As / max(V / n + T / (n * r), 1e-6), tension=proof / (Fp + phi * Ft), strip=strip / (Fp + phi * Ft))
        ok = min(sf['slip'], sf['separation'], sf['shear'], sf['strip']) >= SF_WALK and sf['tension'] >= SF_TENSION
        torque = assumed(spec, 'torque_coefficient') * Fp * h['d'] / 1000
        rows.append(dict(name=p['name'], joint='', label=p.get('note') or 'its load', whatif=False, n=n, count=p['count'], thread=h['thread'], r=r * 1000, le=le, Fp=Fp, limit=limit, strip=strip, torque=torque, T=T, V=V, Mt=Mt, Fa=Fa, sf_each=sf, sf=min(sf['slip'], sf['separation'], sf['shear'], sf['strip'], sf['tension'] * SF_MIN / SF_TENSION), ok=ok))
    return rows


def thermal_rows(M):
    """The RS04 at the knee: how much metal it conducts into, against the plate its rating was measured on. Areas from the design's numbers, the exposed one from the solids. A hand estimate: no thermal model behind it, and the knit over it is not counted."""
    bore, length = 120.0, 38.7                                        # the motor's d120 from its front face to its rear step
    outside = math.pi * bore * length
    front = math.pi * (60.0 ** 2 - 33.0 ** 2) - 10 * math.pi * 2.25 ** 2     # the plate on the front face, less its holes and the bearing's bore
    rear = math.pi * (59.8 ** 2 - 48.5 ** 2)                          # the skirt on the rear face, with a gap pad
    plate_big, plate_small = 345.0 * 345.0, 220.0 * 200.0
    # what faces out, from the design's numbers: barrel 127 x 58 and its two end faces. The closed body below it counts for less, through 1.7 mm of wall, so it isn't added
    outward = math.pi * 127.0 * 58.0 + 2 * math.pi * 63.5 ** 2
    hub_contact = math.pi * (26.5 ** 2 - 23.0 ** 2)              # the post's washer on the outer hub's inner wall
    return dict(outside=outside, front=front, rear=rear, contact=outside + front + rear, plate_big=plate_big, plate_small=plate_small, outward=outward, hub_contact=hub_contact)


def bearing_rows(M):
    """Each support bearing against the load cases, with its partner. A straight bar on two supports at the bearings' centres; the force line at each corner the ankle motors allow, worst kept.
    Static: P0r = Fr + 2 M / Dpw + 0.44 Fa, fs = C0 / P0r (IKO CAT-57166 eq. 5, 6), for roller; for a deep groove ball bearing P0 = max(Fr, 0.6 Fr + 0.5 Fa) (ISO 76, from memory, an estimate), s0 = C0 / P0.
    Dynamic: 810 N with the lever the ankle's rated torque allows; P = fw (Fr + 0.45 Fa); cycles = (90 / theta) (C / P)^e x 1e6, e 10/3 for rollers and 3 for balls; invalid over P = C / 2 (IKO eq. 1, 3, 4; same form for balls).
    No tilting moment on a single bearing: the pair takes it as this couple, which is why the lever rule and not a bearing's moment rating is what's read."""
    spec = M.spec
    rows = []
    ox = np.array(M.joint['L_ankle_pitch']['origin'], float)
    bears = {g['name']: g['hw'] for l in spec['links'].values() for g in l['geoms'] if g.get('hw', {}).get('type') == 'bearing'}
    for name, h in bears.items():
        if name.startswith('R '):
            continue
        rows.append(bearing_eval(M, name, h, bears.get(h.get('partner'))))
    return rows


def bearing_eval(M, name, h, partner):
    spec = M.spec
    a = G.AXIS[h['axis']]
    c = np.array(h['pos'], float)
    k = 'xyz'.index(h['axis'])
    other = np.array(partner['pos'], float) if partner else c
    span = abs((other - c) @ a)
    hip_y = M.joint['L_knee']['origin'][1]
    fs_min = assumed(spec, 'bearing_static_min')

    def share(line_y):                                     # the lever rule; alone, it takes everything
        return abs(line_y - other[k]) / span if span > 0 else 1.0
    dy_peak = min(M.cap('L_ankle_roll') / F_PEAK, SOLE_Y)
    Fr_peak = F_PEAK * max(share(hip_y + dy_peak), share(hip_y - dy_peak))
    Fa_side = F_SIDE
    rated = spec['actuators']['RS06']['rated_nm'] * M.joint['L_ankle_roll']['capacity']['ratio']
    dy_walk = min(rated / F_WALK, SOLE_Y)
    Fr_walk = F_WALK * max(share(hip_y + dy_walk), share(hip_y - dy_walk))
    C, C0 = h['C'], h['C0']
    roller = h['kind'] == 'roller'
    P0 = (lambda fr, fa: fr + 0.44 * fa) if roller else (lambda fr, fa: max(fr, 0.6 * fr + 0.5 * fa))
    fs_peak, fs_side = C0 / P0(Fr_peak, 0.0), C0 / P0(Fr_peak, Fa_side)
    fw = assumed(spec, 'bearing_load_factor')
    P = fw * Fr_walk
    e = 10 / 3 if roller else 3.0
    theta = assumed(spec, 'knee_oscillation') / 2
    cycles = 90 / theta * (C / P) ** e * 1e6
    need = assumed(spec, 'service_cycles')
    return dict(name=name, designation=h['designation'], d=h['d'], D=h['D'], B=h['B'], C=C, C0=C0, kind=h['kind'], source=h['source'], read_via=h['read_via'], Fr_peak=Fr_peak, Fa_side=Fa_side, fs_peak=fs_peak, fs_side=fs_side, fs_min=fs_min,
                Fr_walk=Fr_walk, P=P, cycles=cycles, need=need, life_valid=P <= C / 2, span=span * 1000)


def bearing_options(M):
    """Every catalogue bearing against the slot at the knee: which go in today, what each would rate, and what would have to grow. The slot: the d40 post and the 6.5 mm between stator disc and outer hub."""
    sys.path.insert(0, os.path.join(ROOT, 'spec'))
    import hardware as HW
    rec = {g['name']: g['hw'] for l in M.spec['links'].values() for g in l['geoms'] if g.get('hw', {}).get('type') == 'bearing'}
    h = rec.get('L knee outer bearing')
    if h is None:
        return []
    SLOT, SHAFT, HOUSING = 8.5, 50.0, 96.0     # mm: the room iteration 41 made; post d50; the eye is d96, a race OD can't reach its rim
    out = []
    for name, b in HW.bearing_table().items():
        probe = dict(h, designation=name, d=b['d'], D=b['D'], B=b['B'], C=b['C'], C0=b['C0'], kind=b['kind'], source=b['source'], read_via=b['read_via'])
        e = bearing_eval(M, name, probe, rec.get(h['partner']))
        grow = []
        if b['B'] + 2 * assumed(M.spec, 'machining_clearance') > SLOT:
            grow.append(f"band +{b['B'] + 2 * assumed(M.spec, 'machining_clearance') - SLOT:.1f} mm")
        if b['d'] > SHAFT:
            grow.append(f"post d{b['d']:.0f}")
        if b['d'] < SHAFT:
            grow.append(f"post down to d{b['d']:.0f} (fits; weaker post)")
        out.append(dict(name=name, b=b, e=e, fits=b['B'] + 2 * assumed(M.spec, 'machining_clearance') <= SLOT and b['d'] == SHAFT and b['D'] < HOUSING, grow='; '.join(grow) or 'nothing'))
    return sorted(out, key=lambda x: -x['e']['fs_peak'])


def verdict(sf):
    return 'FAIL' if sf < 1 else ('MARGINAL' if sf < SF_WALK else 'pass')


CASE_LABEL = {'peak': '3,400 N foot', 'side': '3,400 N + 1,614 N side', 'walk': '810 N walking', 'motor': 'motor peak', 'payload': '5 kg in the hand'}


def evaluate(M, s):
    """{name, material, area, size, results {case: {sf, ...}}, sf, case, why}."""
    u, v, n = frame_of(s['axis'], M) if not (isinstance(s['axis'], tuple) and s['axis'][1] not in M.part) else (None, None, None)
    row = dict(name=s['name'], path=s['path'], link=s['link'], note=s.get('note', ''), results={}, sf=0.0, case='', area=0.0, size='none', material='', why=None, sf_hollow=None, fill=None)
    if u is None:
        row['why'] = f"{s['axis'][1]} is gone"
        return row
    if s['parts'] == '*':
        names = [g['name'] for g in M.spec['links'][s['link']]['geoms'] if g['role'] != 'cut' and why_not_carrying(g) is None]
    else:
        names = s['parts']
    gs, mans = [], []
    for nm in names:
        p = M.part.get(nm)
        if p is None:
            row['why'] = f'{nm} is not in the spec'
            return row
        reason = why_not_carrying(p[2])
        if reason:
            row['why'] = f'{nm}: {reason}'
            return row
        gs.append(p)
        mans.append(M.S.geom[(p[0], p[1])])
    at = plane_at(s['at'], n, M)
    if at is None:
        row['why'] = f"{s['at'][0]}, where the plane is taken from, is gone"
        return row
    man = mans[0]
    for m in mans[1:]:
        man = man + m
    pieces = slice_section(man, u, v, n, at)
    if not pieces:
        row['why'] = 'no carrying material at this plane'
        row['size'] = '0 mm2'
        return row
    row['area'] = sum(p.A for p in pieces)
    row['size'] = size_text(pieces)
    sy = min(yield_of(p[2]) for p in gs)
    Es = [STRENGTH[p[2]['material']][1] for p in gs]
    row['material'] = '+'.join(sorted({p[2]['material'] for p in gs}))
    row['yield'] = sy
    # the mass model's fill: if the part is really that hollow, it has less section than the solid drawn here
    fills = [M.MM.fill_of(p[2], M.S.geom[(p[0], p[1])]) for p in gs]
    row['fill'] = float(np.mean(fills))
    best = None
    for case in s['cases']:
        alts = alts_for(M, s, case, (n, at))
        if alts is None:
            row['results'][case] = dict(sf=0.0, vm=float('inf'), why='a part this load comes through is gone')
            continue
        if not alts:
            continue
        worst = None
        for elems in alts:
            r = stress(pieces, elems, u, v, n, at, solo=s.get('solo', False))
            if worst is None or r['vm'] > worst['vm']:
                worst = r
        worst['sf'] = sy / worst['vm'] if worst['vm'] > 1e-9 else float('inf')
        if s.get('buckle'):
            K, ref = s['buckle']
            L = extent_along(M, ref, n) if isinstance(ref, str) else ref
            p = min(pieces, key=lambda q: q.Imin)
            cr = math.pi ** 2 * min(Es) * p.Imin / (p.A * (K * L * 1000.0) ** 2)
            sa = abs(worst['N']) / sum(q.A for q in pieces) + worst['s_bend'] if s.get('solo') else abs(worst['N']) / sum(q.A for q in pieces)
            worst['sf_buckle'] = cr / max(sa, 1e-9)
            if worst['sf_buckle'] < worst['sf']:
                worst['sf'] = worst['sf_buckle']
                worst['buckling'] = True
        f = row['fill']
        g = 1 - (1 - f) ** 2
        h = math.sqrt((worst['s_axial'] / f + worst['s_bend'] / g) ** 2 + 3 * (worst['tau'] / f) ** 2)
        worst['sf_hollow'] = min(sy / h, worst['sf']) if h > 1e-9 else float('inf')
        row['results'][case] = worst
        if best is None or worst['sf'] < best[0]:
            best = (worst['sf'], case, worst)
    if best is None and row['results']:
        row['why'] = 'a part this load comes through is gone'
    if best:
        row['sf'], row['case'] = best[0], best[1]
        need = [(x['sf'], c) for c, x in row['results'].items() if c not in ('peak', 'side')]
        row['sf_need'], row['case_need'] = min(need) if need else (best[0], best[1])
        row['sf_hollow'] = min(r.get('sf_hollow', float('inf')) for r in row['results'].values())
    return row


def extent_along(M, name, n):
    g = M.part[name][2]
    if 'p0' in g:
        return float(np.linalg.norm(np.array(g['p1'], float) - np.array(g['p0'], float)))
    m = M.geom_man(name)
    v = np.asarray(m.to_mesh().vert_properties[:, :3], float) @ n
    return float(v.max() - v.min())


def evaluate_all(M):
    return [evaluate(M, s) for s in SECTIONS]


# ---------------------------------------------------------------- contact

def contact(a, b):
    """How two solids meet: dict(kind, area mm2, depth mm, gap mm). kind: 'overlap' (a real seat, at least CONTACT_MIN_DEPTH),
    'flush' (exactly coplanar faces), 'token' (an overlap thinner than that) or 'gap'."""
    out = dict(kind='gap', area=0.0, depth=0.0, gap=None)
    x = a ^ b
    v = x.volume()
    if v > 1e-13:
        lo, hi = bounds(x)
        d = float((hi - lo).min())
        out['depth'], out['gap'] = d * 1000, 0.0
        if d >= CONTACT_MIN_DEPTH:
            out['kind'], out['area'] = 'overlap', v / d * 1e6
            return out
        if d > FLUSH:
            out['kind'], out['area'] = 'token', v / d * 1e6
            return out
    gap = a.min_gap(b, 0.5)
    out['gap'] = gap * 1000
    if gap <= FLUSH or (v > 0):
        # exactly coplanar faces share no volume: push one in by 0.1 mm each way and read the area from that
        step = 1e-4
        best = 0.0
        for ax in range(3):
            for sg in (1, -1):
                t = [0.0, 0.0, 0.0]
                t[ax] = sg * step
                best = max(best, (a.translate(t) ^ b).volume())
        if best > 0:
            out['kind'], out['area'] = 'flush', best / step * 1e6
    return out


def interface_force(M, spec_force):
    if isinstance(spec_force, (int, float)):
        return F_WALK * spec_force
    a_peak = F_WALK / M.total
    mass = lambda root: sum(M.link_mass[l][0] for l in M.R.subtree(root))
    if spec_force == 'arm':
        return max(mass('L_shoulder_pitch_link') * a_peak, (mass('L_shoulder_pitch_link') + HAND_KG) * G_ACC)
    if spec_force == 'head':
        return mass('neck_yaw_link') * a_peak
    if spec_force == 'body':
        return mass('torso') * a_peak
    if spec_force.startswith('rod '):
        r = M.rod(spec_force[4:])
        return r[0] if r else 0.0
    raise ValueError(spec_force)


def required_area(v):
    return CONTACT_MIN_MM2 * max(1, math.ceil(SF_WALK * v / BOLT_SLIP_N))


def evaluate_interfaces(M):
    rows = []
    for path, a, b, fs in INTERFACES:
        v = interface_force(M, fs)
        need = required_area(v)
        ma, mb = M.geom_man(a), M.geom_man(b)
        row = dict(path=path, a=a, b=b, v=v, need=need, area=0.0, kind='missing', depth=0.0, gap=None)
        if ma is not None and mb is not None:
            row.update(contact(ma, mb))
        row['ok'] = row['kind'] in ('overlap', 'flush') and row['area'] >= need
        row['tau'] = v / row['area'] if row['area'] else None
        rows.append(row)
    return rows


# ---------------------------------------------------------------- output bearings

def bearing_table(M):
    """What each output bearing must take, from the same load cases, as force along the axis, across it, and tilting moment."""
    rows = []
    for j in M.spec['joints']:
        if j['name'].startswith('R_') or not j['act']:
            continue
        a = M.spec['actuators'][j['act']]
        ax = G.AXIS[j['axis']]
        c = np.array(j['act_c'], float) + j['out'] * ax * a['h'] / 2
        side = j['child']
        leg = 'hip' in j['name'] or 'knee' in j['name']
        s = dict(link=side, distal=[side], load='limb', own=None)
        cases = []
        if leg:
            for case, f in (('peak', F_PEAK), ('walk', F_WALK)):
                alts = leg_line(M, f)
                cases.append((case, alts))
        elif j['name'] == 'waist_yaw':
            s2 = dict(s, load='upper', hip_moment=True)
            for case, f in (('peak', F_PEAK), ('walk', F_WALK)):
                cases.append((case, inertial_alts(M, s2, f / M.total, Z2, None)))
        else:
            for case, f in (('peak', F_PEAK), ('walk', F_WALK)):
                cases.append((case, inertial_alts(M, s, f / M.total, AXES6, None)))
            if 'neck' not in j['name']:
                cases.append(('payload', inertial_alts(M, s, G_ACC, AXES6, None, payload=HAND_KG)))
        pitch = 0.9 * a['r']
        for case, alts in cases:
            worst = None
            for el in alts:
                f, m = wrench(el, c)
                Fa = abs(f @ ax)
                Fr = np.linalg.norm(f - (f @ ax) * ax)
                Mt = np.linalg.norm(m - (m @ ax) * ax)
                eq = Fr + Fa + 2 * Mt / pitch
                if worst is None or eq > worst[3]:
                    worst = (Fa, Fr, Mt, eq)
            rows.append(dict(joint=j['name'], model=j['act'], case=case, Fa=worst[0], Fr=worst[1], Mt=worst[2], eq=worst[3], pitch=pitch))
    return rows


# ---------------------------------------------------------------- the report

def fmt_sf(x):
    return 'inf' if x == float('inf') else ('%.0f' % x if x >= 100 else ('%.1f' % x if x >= 10 else '%.2f' % x))


def hardware_report(M):
    """The screw and bearing tables, and the assumptions they rest on."""
    if not M.spec.get('hardware'):
        return []
    spec = M.spec
    L = ['', '## Bolted joints (hand estimates, not VDI 2230)', '',
         '12.9 screws (assumed), real patterns from the maker\'s drawing, placed by `spec/hardware.py`. Fp: preload, the smaller of 0.75 x proof and strip force / 2. Formulas per pattern of n screws on radius r: slip SF = mu n Fp / (V + T / r); '
         'separation SF = Fp / (Fa / n + 2 Mt / (n r)); shear if slipped SF = 0.6 Rm As / (V / n + T / (n r)); tension SF = proof / (Fp + phi Ft); strip SF = F_strip / (Fp + phi Ft). T: the motor\'s peak. V, Fa: below. Required: SF 2 except tension 1.1 (preload alone uses 0.75 of proof). '
         'Pins not counted in shear. Not modelled: fatigue, embedding, flange flex.', '',
         '| pattern | load | screws | thread, engaged mm | Fp kN (limit) | torque N m (estimate) | T, V, Fa, Mt | slip | separation | shear | tension | strip |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in bolted_rows(M):
        if r.get('why'):
            L.append(f"| {r['name']} | {r['why']} | 0 | | | | | | | | | |")
            continue
        sf = r['sf_each']
        f = lambda x: 'inf' if x == float('inf') else f'{x:.1f}'
        L.append(f"| {r['name']} | {r['label']} | {r['n']} of {r['count']} on d{2 * r['r']:.0f} | {r['thread']}, {r['le']:.1f} | {r['Fp'] / 1000:.1f} ({r['limit']}) | {r['torque']:.1f} | {r['T']:.0f} N m, {r['V']:.0f} N, {r['Fa']:.0f} N, {r['Mt']:.0f} N m | {f(sf['slip'])} | {f(sf['separation'])} | {f(sf['shear'])} | {f(sf['tension'])} | {f(sf['strip'])} |")
    L += ['', '## Bearings (catalogue ratings, hand estimates of load)', '',
          'Two supports at the knee, on the lever rule, the motor\'s own bearing taking nothing (assumed: its rating is unpublished). Static: fs = C0 / (Fr + 0.44 Fa) (IKO CAT-57166), needs 1.5 (its Table 4, normal operation). "With side" adds the recorded 1,614 N as axial, an upper bound as in the load cases. '
          'Dynamic: 810 N, the lever limited by the ankle\'s rated torque, fw 1.5; life in oscillating cycles of 60 deg; a hand figure for service life.', '',
          '| bearing | catalogue | d x D x B | C, C0 (N) | peak Fr (N) | fs at 3,400 N | fs with side | walk P (N) | life, millions of cycles | needs |', '|---|---|---|---|---|---|---|---|---|---|']
    for r in bearing_rows(M):
        L.append(f"| {r['name']} | {r['designation']} ({r['source']}, {r['read_via']}) | {r['d']:.0f} x {r['D']:.0f} x {r['B']:.0f} | {r['C']:.0f}, {r['C0']:.0f} | {r['Fr_peak']:.0f} | **{r['fs_peak']:.2f}** | {r['fs_side']:.2f} | {r['P']:.0f} | {r['cycles'] / 1e6:.0f}{'' if r['life_valid'] else ' (over C / 2: formula invalid)'} | {r['need'] / 1e6:.1f} |")
    opts = bearing_options(M)
    if opts:
        L += ['', '### Every catalogue bearing against the knee\'s slot', '',
              'The outer support: d50 post, 8.5 mm for the race between its shoulder and its lip, and the eye\'s d96. Same load as above. Rows from `spec/hardware_catalog.json`; `search_excerpt` rows want a check on the maker\'s page.', '',
              '| bearing | d x D x B | C0 (N) | fs at 3,400 N | fs with side | life, M cycles | goes in today | what would have to grow | read |', '|---|---|---|---|---|---|---|---|---|']
        for o in opts:
            b, e = o['b'], o['e']
            L.append(f"| {o['name']} | {b['d']:.0f} x {b['D']:.0f} x {b['B']:.0f} | {b['C0']:.0f} | {e['fs_peak']:.2f} | {e['fs_side']:.2f} | {e['cycles'] / 1e6:.0f} | {'**yes**' if o['fits'] else 'no'} | {o['grow']} | {b['read_via']} |")
    th = thermal_rows(M)
    L += ['', '## The knee motor\'s heat path (a hand estimate)', '',
          'RS04, 40 N m rated on a 345 x 345 mm plate, 35 on 220 x 200 (`spec/actuators.json`). What it conducts into now, with paste in the bore and a gap pad on the skirt, both assumed:', '',
          '| path | area (cm2) |', '|---|---|',
          f"| side wall in the barrel's bore, 0.05 mm gap | {th['outside'] / 100:.0f} |", f"| front face on the barrel's plate, bolted | {th['front'] / 100:.0f} |", f"| rear face on the rear cap's skirt | {th['rear'] / 100:.0f} |",
          f"| **total contact** | **{th['contact'] / 100:.0f}** |", f"| maker's plate, one side: 220 x 200 / 345 x 345 | {th['plate_small'] / 100:.0f} / {th['plate_big'] / 100:.0f} |",
          f"| of that, the washer on the outer hub cap, the hub's own path | {th['hub_contact'] / 100:.0f} |", f"| outward metal of the barrel and its ends | {th['outward'] / 100:.0f} |", '',
          f"**Ratio:** contact is {th['contact'] / th['plate_small']:.2f} of the small plate's one side, {th['contact'] / th['plate_big']:.2f} of the big one's. Outward metal is {th['outward'] / (2 * th['plate_small']):.2f} of the small plate's two sides and {th['outward'] / (2 * th['plate_big']):.2f} of the big one's, and it sits under knit, where the plate is in open air. So the rated torque isn't transferred by this: sustained torque under the garment wants a thermal model or test. The hub cap's own path is thin. Not modelled: paste resistance, airflow, the knit."]
    L += ['', '## Assumptions behind the two tables above', '',
          'From `spec/assumptions.json`, one row each: change a value there and run the checks. All status `assumed`. "estimate, from memory" means a hand figure no document here backs.', '',
          '| id | value | basis | source | what it risks | settled by |', '|---|---|---|---|---|---|']
    for r in spec['hardware']['assumptions']:
        L.append(f"| {r['id']} | {r['value']} {r['unit']} | {r['basis']} | {r['source']} | {r['risk']} | {r['settled_by']} |")
    return L


def report(M, rows, irows, brows, path):
    ranked = sorted(rows, key=lambda r: r.get('sf_need', r['sf']))
    n_fail = sum(r.get('sf_need', r['sf']) < 1 for r in rows)
    n_marg = sum(1 <= r.get('sf_need', r['sf']) < SF_WALK for r in rows)
    n_pass = len(rows) - n_fail - n_marg
    bad_if = [r for r in ranked if r['sf'] >= SF_MIN and r.get('sf_hollow') is not None and r['sf_hollow'] < SF_MIN]
    ibad = [r for r in irows if not r['ok']]
    L = ['# echo strength table: hand estimates, not FEA', '',
         f"Live spec (`spec/robot.json`), total mass {M.total:.2f} kg. Generated by `python -I tools/strength.py`; `tools/checks.py` groups `loadpath` and `contact` read the same tables. "
         "Left side at the zero pose; the right is built as its mirror.", '',
         '**How far to trust this.** Every number is a hand estimate on a section cut from the spec\'s solid primitives. No FEA, no test. '
         'It finds the weak parts and ranks them; it cannot sign anything off.', '',
         f'**Result:** {len(rows)} sections on what is required, calm walking at 810 N and each motor at peak: **{n_fail} fail** (under 1 to yield), **{n_marg} marginal** (under {SF_WALK}), {n_pass} pass. The 3,400 N shove is reported in the SF column, and nothing is sized for it. '
         f'{len(ibad)} of {len(irows)} interfaces fail. '
         f'Sections that pass only because they are read as solid: {len(bad_if)}: at the mass model\'s fill they drop under {SF_MIN:.0f}.', '',
         '## Load cases', '',
         '| case | size | where it comes from |', '|---|---|---|',
         '| 3,400 N foot | 3,400 N up the straight leg, off the axes by 26 mm fore-aft and 14 mm sideways, the most the ankle motors can hold (a joint back-drives past its peak). The body takes it as inertia, %.1f m/s2 | Largest vertical foot force in the recorded sims: 3,432 N, one 20 ms sample, `neutral_walk_180_R_002__A122` with a 1.48 m/s pelvis push (`mv6/2_shove15.npz`, 569 N weight: the v6 plant, not this exact mass). Only 4 of 526 frames pass 2,000 N |' % (F_PEAK / M.total),
         '| 3,400 N + 1,614 N side | the same frame\'s side force on a line through the ankle | Foot and shank only, plus one row for the knee cheeks. Upper bound: hip roll held only 43 N m at that frame, so most of the side force went into the leg\'s inertia |',
         '| 810 N walking | 810 N, levers to what the ankle can hold, inside the sole | `force_v6.npz`, `Relaxed_walk_forward_001__A057`. That peak is frame 1 (settling); later steps peak near 670 N, 99th percentile 612 N. A faster neutral walk with 300 N neck shoves reached 2,471 N, and jogging 1,660 to 2,220 N, so 810 N is a calm walk, not a limit |',
         '| motor peak | each motor beside the section at peak, one at a time, as a free moment; ankle motors as rod force | `spec/actuators.json`: a motor must not break its own bracket |',
         '| 5 kg in the hand | 5 kg per hand plus link weight, static, worst direction | `POSES` in `tools/checks.py`: "carry 10 kg at the chest" |', '',
         '## Sections, worst first (hand estimates)', '',
         '"SF" is to yield, worst of all cases, the 3,400 N shove included; the verdict reads calm walking and motor peak only (`other cases`).  "SF hollow" is the same at the fill the mass model charges for that part (`tools/mass_model.py`: 0.25 for anything thicker than 20 mm, 0.5 to 20 mm, 0.8 to 8 mm, beam 0.12), keeping the outline; '
         'the parts cannot be both as light as the mass model says and as strong as the SF says. Thin-wall buckling is not in it.', '',
         '| # | verdict | section | path | material | smallest section, mm | mm2 | governing case | N, V (N) | bending, torsion (N m) | stress (MPa) | SF | SF hollow | other cases (SF) | note |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for i, r in enumerate(ranked, 1):
        v = verdict(r.get('sf_need', r['sf']))
        if r['why'] or not r['case']:
            L.append(f"| {i} | **FAIL** | {r['name']} | {r['path']} | {r['material'] or 'n/a'} | {r['size']} | 0 | n/a | n/a | n/a | n/a | **0** | n/a | n/a | {r['why'] or ''} {r['note']} |")
            continue
        g = r['results'][r['case']]
        oth = '; '.join(f"{CASE_LABEL[c]} {fmt_sf(x['sf'])}" for c, x in r['results'].items() if c != r['case'])
        extra = ' (buckling)' if g.get('buckling') else ''
        mark = f'**{v}**' if v != 'pass' else v
        L.append(f"| {i} | {mark} | {r['name']} | {r['path']} | {r['material']} ({r['yield']:.0f}) | {r['size']} | {r['area']:.0f} | {CASE_LABEL[r['case']]} | {abs(g['N']):.0f}, {g['V']:.0f} | {g['Mb']:.0f}, {abs(g['T']):.0f} | {g['vm']:.0f} | **{fmt_sf(r['sf'])}**{extra} | {fmt_sf(r['sf_hollow'])} | {oth} | {r['note']} |")
    L += ['', '## Interfaces (hand estimates)', '',
          f'Contact area between two rigid parts: overlap volume over its thinnest dimension, or the shared area of exactly coplanar faces. Required: {CONTACT_MIN_MM2:.0f} mm2 (the pad of one M5) x the bolts needed to pass the calm-walking force at factor {SF_WALK} (slip load {BOLT_SLIP_N:.0f} N a bolt, from memory). '
          'This only asks whether a joint has room to exist. Screws are read in their own table below, at the knee so far.', '',
          '| verdict | interface | path | force (N) | area (mm2) | required | kind | depth (mm) |', '|---|---|---|---|---|---|---|---|']
    for r in sorted(irows, key=lambda r: (r['ok'], r['area'] / max(r['need'], 1))):
        gap = '' if r['kind'] != 'gap' or r['gap'] is None else f" ({r['gap']:.2f} mm of air)"
        L.append(f"| {'pass' if r['ok'] else '**FAIL**'} | {r['a']} / {r['b']} | {r['path']} | {r['v']:.0f} | {r['area']:.0f} | {r['need']:.0f} | {r['kind']}{gap} | {r['depth']:.2f} |")
    L += ['', '## Output bearings: required (hand estimates), rating unknown', '',
          'RobStride publishes no output-bearing rating that I could find, so nothing here is a verdict. This is what each must take, to put to the maker. Tilting moment is about the axes across the joint, at the output face. '
          '"Equivalent" is radial + axial + 2 x moment / pitch, with an assumed bearing pitch of 0.9 x motor radius: for comparing joints, not for selecting a bearing.', '',
          '| joint | motor | case | axial (N) | radial (N) | tilting moment (N m) | equivalent (kN) |', '|---|---|---|---|---|---|---|']
    for r in sorted(brows, key=lambda r: -r['eq']):
        L.append(f"| {r['joint']} | {r['model']} | {CASE_LABEL[r['case']]} | {r['Fa']:.0f} | {r['Fr']:.0f} | {r['Mt']:.0f} | {r['eq'] / 1000:.1f} |")
    hw = hardware_report(M)
    L += hw
    mb, mp = M.geom_man('shoulder beam'), M.geom_man('battery')
    gap_pack = contact(mp, mb) if mb is not None and mp is not None else None
    pack_txt = '' if gap_pack is None or gap_pack['kind'] != 'gap' else f" The upper pack does not touch the beam ({gap_pack['gap']:.1f} mm of air): the shoulders reach the pack only through the compute box's 1 mm overlap."
    L += ['', '## What this cannot assess, and what would settle it', '',
          'Nothing here waits on hardware: nothing is being bought soon, and the design has to stand up in a pitch and in sim. So each row names the analysis that could settle it now, and says plainly where only a physical test can. Any number in this section is a hand estimate too.', '',
          '| item | what is open | settles it now | only a test can settle |', '|---|---|---|---|',
          '| Fasteners | Real screws at the knee only (tables above); every other joint is still contact area standing in for room. As of 2026-11-05 the hip bracket and front plate touch their flanges only outside the d36 output bolt circle, so neither can bolt there as drawn. The bolt slip load in Interfaces is from memory | Place the remaining patterns with `spec/hardware.py`; VDI 2230 calculation per joint; FEA with preload on the worst few | Thread and preload scatter, if a margin ends up thin |',
          "| Output bearings | Required loads are above; I found no rating. Hip roll is the one to ask about. The knee no longer rests on its motor's bearing by intention, but only stiffness decides what that bearing really takes | Maker data: ask RobStride for the rating or the bearing's part number, then read the bearing maker's catalogue | If the maker gives neither: a moment-to-failure test on one module |",
          '| Motor housings as structure | Several rows pass load face to face through a motor | Maker data for an allowed load. Failing that, route frame past the motor so the question goes away, or FEA of the housing from the STEP with an assumed alloy (an estimate: the casting is unknown) | The housing\'s true strength, if the design keeps relying on it |',
          f"| Battery case | Undefined. No frame between spine and beam, and the pack is not counted as frame: its row has zero section.{pack_txt} | Design the case or rails, then FEA; cell limits from the cell maker's data sheet | Abuse and crush behaviour of the pack in a fall |",
          '| Thin-wall hollow sections | Solid is the best case. "SF hollow" keeps the outline at the mass model\'s fill; walls, pockets and corner radii are not drawn | Draw the walls; FEA with linear buckling | Nothing at this stage |',
          '| Stress concentration | None included: fillets and bores at Kt 1 | Linear static FEA of the top rows; handbook Kt values (Peterson) as a first pass | Nothing at this stage |',
          '| Fatigue | Static yield only. Bolted aluminium in a walking machine fatigues well below these stresses | Cycle counts from the recorded takes (rainflow on sim forces and torques), FEA stresses, published 7075 S-N data | Real scatter and fretting at the joints |',
          '| Impact and falls | 3,400 N is one 20 ms sample from a stiff sim. No fall was run | Run falls in sim with whole-body collision and read the forces; sweep contact stiffness to see how far the peak moves; explicit dynamic FEA on the worst | What foot, floor and covers really absorb: the sim\'s contact has no measured basis without a drop test |',
          '| Bent poses and hard stops | Zero pose only, each joint limited by its peak. Only the foot plate row allows for a stop | Rerun these sections on joint wrenches recorded from the sim takes (MuJoCo reports them per body), with stops modelled | Nothing at this stage |',
          "| Stiffness | The 3 mm cheeks, the 4 mm stator plate and the thigh frame's thin side look soft; none of that is here | FEA for compliance, fed into the sim model as joint compliance | Measured stiffness, only if the sim proves sensitive to it |",
          "| Thermal | Not in this table: a hot motor derates its peak, and every motor case here assumes the full peak. Audit item 4 stands | Thermal model from the maker's plate ratings and the real conduction paths; maker data for winding limits | Heat through the garment |",
          '| Hand, covers, knit | The hand is a placeholder, not assessed past its bracket. Shell and knit carry no load by rule. The knee and elbow hubs are 7075 but role shell, so they are not counted | Design the hand; decide whether the hubs are structure and re-role them | How a textile wears, if the claim depends on it |', '',
          '## Notes on method', '',
          f'- Yield: 7075-T6 503 MPa (typical plate); PA12 48 MPa as a strength.',
          '- Sections are slices of the real solids through a plane tied to a part, so they follow the part. A plane that misses is zero area and fails.',
          '- Stress: axial plus biaxial bending at every outline point, Saint-Venant torsion, shear 1.5 V / A, von Mises of the worst of each, whether or not they coincide.',
          '- Pieces in one plane act together unless the row says otherwise (they are assumed tied at both ends: a generous assumption for 3 mm plates).',
          '- The load line in 3,400 N and 810 N is taken at all four corners of what the ankle allows, worst result kept. Above the waist and in the limbs, inertia in the worst direction, each joint limited by its peak torque.',
          '']
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    return ranked


def main():
    import checks as C
    spec = G.load()
    S = C.Solids(spec)
    M = Model(spec, S)
    rows = evaluate_all(M)
    irows = evaluate_interfaces(M)
    brows = bearing_table(M)
    path = os.path.join(ROOT, 'reports', 'strength.md')
    ranked = report(M, rows, irows, brows, path)
    print(f"{'SF':>6} {'hollow':>6} {'A mm2':>7}  section")
    for r in ranked:
        print(f"{r['sf']:6.2f} {(r['sf_hollow'] if r['sf_hollow'] is not None else 0):6.2f} {r['area']:7.0f}  {r['name']} [{r['size']}] {r['material']} {CASE_LABEL.get(r['case'], '')} {r['why'] or ''}")
    print(f"-> {path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
