# Figure 01 mechanical study, for echo's phase 3

Our own measurements and descriptions, from one third-party video and one photo. No frames are kept in this repo; go back to a frame with its time code. Date: 2026-11-05.

## Sources and how far to trust them

- **Video:** youtu.be/dO830ROcXVs, 122.8 s, a third-party narrated re-upload of Figure's footage. Played at 1080p, scanned at 0.5 s steps, close-ups read at native pixels.
- **Photo:** the frontal standing photo the user sent (1170 x 1608 px once rotated). It is the only frame sharp and frontal enough for a whole-body scale.
- **Patents:** nine documents, assignee Figure AI Inc per Google Patents metadata (see "Patent documents used as technical sources"). The downloaded USPTO PDFs have no text layer, so wording comes from Google Patents' text; verify against the PDFs. **Patents never name a robot. Linking one to Figure 01, 02 or 03 is always my inference.** Patent ranges are embodiments, not the dimensions of a shipped robot.
- **Sharpening:** the pelvis close-up (46-49 s) has halos around cables and smeared textures, so it looks AI-upscaled or sharpened. The photo has similar halos when enlarged. **Fine features, tiny text and thread shapes in those are unreliable; layout is usable.** Caption overlays hide parts of many shots.
- **Not possible:** "mm to mm". Best honest accuracy is about ±10 % for large parts and ±15-25 % for small ones. Nothing here is a true dimension.
- **Unknown:** playback speed, lens, and whether any walking was slowed down.

**Tags:** VISIBLE = seen in the video or photo. MEASURED = scaled from pixels, with error. STATED = written in a patent, with number and figure. INFERRED = my reading, may be wrong.

## Answer

1. **VISIBLE:** the structure is round or closed everywhere: barrel motors, sculpted forks, tubes. No flat bar carries load in any shot.
2. **VISIBLE:** motors are large, ribbed barrels sitting directly on the joint, with bolt circles on their end faces.
3. **VISIBLE:** the waist is two motors, cranks, two rods with spherical ends, a central body and a round ring. STATED (US12583126): an articulation member with orthogonal pitch and roll axes, two upper actuators moving the torso about both through connecting rods, and a spinal yaw actuator below.
4. **VISIBLE:** two round vertical-axis housings stand on top of the pelvis. STATED (US12583126): hip order is yaw, then roll, then pitch. echo's is pitch, roll, yaw. STATED (US12611766, a later document): flex, roll, then twist below, the same order as echo, with angled axes.
5. **STATED (US12649246, figs. 7, 9A):** a foot-flexion motor at the top of the shin turns a drive shaft running down inside the shin; a slide and connector rods pitch the foot, and a separate finned roll actuator sits below the shin. The shaft isn't visible in the video, so tying this to Figure 01 is INFERRED, though the outline matches closely.
6. **STATED (US12611766, table 3):** peak torques, including 204-307 N-m at hip flex and knee, for a robot "approximately 1725 mm" tall. echo's knee is 120 N-m at 1.727 m. See the joint list below.
7. **For Figure 03:** the oval end caps in US12649246 fig. 3 have the same shape as Figure 03's black ovals. So those are probably covers over knee and hip motors (medium confidence, INFERRED).

## Scale

**Calibration:** head top to sole on the photo is about 1,530 px (y 30 to 1560). Height 168 cm is from secondary sources, and I could not confirm it from Figure. That gives **about 1.10 mm/px, ±10 %**. Knees are slightly bent, legs splayed and feet nearer the camera, so true lengths may be larger than measured at the legs and the scale drifts along the body.

**Side measurements** chain from head height 214 mm through the 24.0 s side frame: about 1.86 mm per screen px at the crop used, ±15 %.

| Item (photo unless noted) | Pixels | Estimate mm | ± |
|---|---|---|---|
| Head width x height | 127 x 195 | 140 x 214 | 10 % |
| Neck, narrowest | 60 | 66 | 15 % |
| Chest width at shell top | 160 | 176 | 10 % |
| Torso width, widest | 205 | 225 | 10 % |
| Torso width at hem | 185 | 203 | 10 % |
| Shell top to hem | 270 | 297 | 10 % |
| Over shoulder caps | 362 | 397 | 10 % |
| Shoulder cap width | 95 | 105 | 15 % |
| Upper-arm barrel diameter | 70 | 77 | 15 % |
| Lower arm-joint barrel width | 80 | 88 | 15 % |
| Forearm tube near elbow | 65 | 70 | 15 % |
| Forearm tube at wrist | 50 | 55 | 15 % |
| Forearm length, elbow to wrist band | 195 front, about 150 side | 215 front, 280 side | Unresolved: elbow centre is ambiguous and the arm foreshortens |
| Waist motors, overall width | 187 | 205 | 10 % |
| Rod centre spacing | 58 | 64 | 15 % |
| Rod pin to pin | 75 | 82 | 20 % |
| Grey box on pelvis front | 56 x 42 | 61 x 46 | 15 %, and sharpened |
| Pelvis plate width | 275 | 302 | 10 % |
| Hip barrel axial length | 100 | 110 | 15 % |
| Thigh tube diameter | 60 | 66 | 15 % |
| Thigh clamp-like band | 66 wide, 35 tall | 72, 38 | 20 % |
| Knee barrel axial length | 102 | 112 | 15 % |
| Hip barrel axis to knee barrel axis | 355 | 390 | 12 % |
| Knee barrel axis to ankle | 320 | 351 | 12 % |
| Ankle axis height | 105 | 115 | 20 % |
| Shank width, front view | 52 | 57 | 15 % |
| Shank flare above ankle | 77 | 85 | 15 % |
| Ankle round housing, front | 58 | 64 | 15 % |
| Foot width | 90 | 99 | 15 % |
| Harness bundle | 20-25 | 22-27 | 25 % |
| Backpack depth x height (side, 24.0 s) | 80 x 195 screen px | 150 x 360 | 15 % |
| Elbow round end (side, 24.0 s) | 50 screen px | 93 | 20 % |

**Direct counts:** shank bolt row, 5 per leg, clear on both legs. Hem screws, 2 clear at the front on the photo and 3 at 15.4 s, more may be hidden. Others below are partly hidden, so treat them as minimums.

## Region by region

### Head and neck
- VISIBLE: one-piece chrome shell, gloss black visor, a small display mark. Concave flared neck with no seam visible at the front. A black cable loops from behind the neck into the back (10.5 s).
- Not visible: neck motors, axes, support.
- INFERRED: at least two neck joints; head turns slightly in the walking shots. I can't separate the axes.

### Shoulder (16-18 s, 64-68 s, photo)
- VISIBLE: a thin quarter-sphere metal cap, held by a single visible screw near its edge, with a fine hatch pattern consistent with 3D-milled tool marks (65.5 s).
- VISIBLE: under it, a bolted plate with at least 9 socket screws in two arcs, heads blued (possibly thread-locker or finish; sharpened, so unsure).
- VISIBLE: the chest shell has a large round opening with a formed lip. Behind the lip, a large ring with pockets and bolts sits **inside the torso's outline** (65.5, 66.3 s).
- VISIBLE: wiring (coloured ribbon, multi-way connector) passes between the ring and the arm through this opening. Green light or board flash at the shoulder.
- INFERRED: shoulder pitch motor is inside the torso, as the user suspected; the ring is its output or bearing. Next axis is under the cap. The cap is a cosmetic cover moving with the arm.
- STATED (US12707556): arm actuators J1 are held in an "arm tube" of the front skeleton. STATED (US12583126): shoulder axes in a range of about 35-40 degrees. STATED (US12611766): J1 axis angled rearward 1-45 degrees. echo's shoulder pitch axis is horizontal and straight across.

### Upper arm (15.4 s, photo)
- VISIBLE: stacked coaxial cylinders with circumferential ribs (at least 4-5 ribs) and stepped flanges. No separate tube: the barrels are the arm.
- VISIBLE: black moulded winged clamp around the barrel, a small black box on the arm's outside with wires, a pocketed plate with a label below.
- INFERRED: upper barrel is shoulder yaw with its axis down the arm.

### Elbow (15.4 s, 27 s, 13.2 s)
- VISIBLE: below the upper-arm barrel, a block with ribs running across the arm, then a round end with dark centre and about 8 holes on its outer face (27 s, 13.2 s). Black seal-like ring where it meets the forearm.
- MEASURED: end about 93 mm across, ±20 %.
- Unresolved: one-sided or forked. Only the outer face is clear.

### Forearm and wrist (13.2 s, 15.4 s)
- VISIBLE: a smooth tapered closed tube, turned or machined finish with lengthwise reflection, a few small screw pockets. Black wrist band, about 10-15 mm wide.
- INFERRED: wrist motors inside the tube; no bulge is visible.

### Hand (13-16 s, 27 s)
- VISIBLE: 4 fingers and a thumb. Cream-coloured finger segments; dark gaps at knuckles. Back of hand shows parallel cylinders, one per finger, with small fixings (13.2 s right).
- INFERRED: a linear drive per finger on the back of the hand. Internal linkages aren't visible.

### Chest shell (9-18 s, 67.5 s)
- VISIBLE: thin metal or metal-finish panels with a seam down the chest, a lower belly panel overlapping with a curved edge, and a formed hem with a rim at the bottom and countersunk screws.
- VISIBLE: stamped logo and small text on the chest. Text is too sharpened to read reliably.
- Not visible: anything behind it. **Whether the shell carries load cannot be told from the video.**
- STATED (US12583126): actuator housings "provide support structure" with load paths through them, and the robot "lacks an exoskeleton". This fits the waist and pelvis as seen.
- STATED (US12611766): housings "to form an exoskeleton" that give structure and form, with an endoskeleton as an alternative. STATED (US12707556, figs. 4, 9, 10): front and rear skeletons around arm actuators, battery and computer; front skeleton thickness from under 1 mm to over 20 mm, thickened by region; torso volume about 17.4-21.3 L; side vents and fans. INFERRED: this is Figure's later approach, and a route for echo's chest.

### Waist (18.5-21.9 s; repeated 68.5-71.5 s; 46.5-49 s; photo)
- VISIBLE, top down:
  1. Two barrel motors in the bottom of the torso, one each side, with ribs on their outer ends and triangular gusset plates below them.
  2. A crank from each motor, ending in a clevis around an upper spherical rod end with a through-bolt.
  3. Two near-vertical rods, about 82 mm pin to pin, about 64 mm apart.
  4. Lower spherical ends on a crossbar or barrel with flats, between clevises.
  5. A central cylindrical body with a rectangular pocket, on the centreline between the rods, at mid rod height.
  6. A round stepped ring on the pelvis top, with a ribbed outer band.
- VISIBLE: red power wires pass down the centre behind the rods; a tube-like strut descends at one side (21.6 s); black blocks stand beside the rods.
- STATED (US12583126, figs. 3-5, 18-21): articulation member with orthogonal pitch and roll axes; left and right upper actuators on the torso; a connector assembly with connecting rods and an external linkage; a spinal actuator with a yaw axis, on front and rear support frames.
- INFERRED: the central body is the articulation member; rods together give pitch, opposed give roll; the ring is the spinal yaw actuator. Which of these the video's robot really has is INFERRED from the match.

### Pelvis (46.5-49 s, photo; sharpened)
- VISIBLE: thick machined plate with pockets and screws, about 302 mm wide, mating to a stepped round boss in the centre (underside).
- VISIBLE: on top, one each side, a round stepped housing with vertical axis (top disc), and below it a stack of circumferential ribs in a machined pocketed block.
- VISIBLE: exposed green boards with pin headers and ribbon wires on the left; a similar cluster on the right; rounded-rectangle grey box in the centre front, function unknown.
- INFERRED: the vertical-axis housings are hip yaw motors. STATED (US12583126, figs. 3-5): hip Z or yaw actuators with fins, spinal actuator between them; US12649246 fig. 3 shows the same two round housings with a central ring.

### Hip and upper thigh (15.4, 27, 28.5 s, photo)
- VISIBLE: below the plate, a large barrel with its axis across the body and bold radial ribs, about 110 mm axial length. Outer face: at least 12 socket screws on a circle, a central boss, and a ring.
- VISIBLE: a sculpted arm from the barrel down to the thigh, with lightening pockets and round holes, tapering. From the front it reads as a fork around the barrel with an open V-shaped window and a grey plate in it (photo y 830-900).
- Sections: fork walls are curved and pocketed; looks 5-axis machined, with a smooth surface.
- STATED (US12583126, figs. 3-6): hip yaw, then roll, then pitch. The roll actuators partly project rearward past a rear support frame; pitch axes are forward of the yaw axes. So the barrel visible here is INFERRED as hip pitch, with roll in the block behind it.
- STATED (same): rotation stops on the outside of actuator housings for yaw, pitch and roll.

### Thigh (photo)
- VISIBLE: round tube about 66 mm across, a wider band (about 72 mm) part-way down, then a tube continuing to the knee housing. Harness runs on the outside with a loop.
- US12649246 fig. 3 draws this band as a collar with screws. I didn't find its function stated. INFERRED: an assembly or adjustment joint.

### Knee (photo; 24.0-25.2 s side views)
- VISIBLE: barrel across the leg, about 112 mm axial length, with fine radial ribs in the middle, and a large round cap each end, screws along a flange.
- VISIBLE: below the barrel, a plate with two screws (a coloured rectangular plate), then a sculpted housing tapering in a shield shape into the shank. Round collar at the join; black foam-like or rubber piece at the back of the shield.
- STATED (US12649246, figs. 4-9): the shin's knee mount has opposed side portions forming a U, coupled to the knee actuator's output, and the text contrasts this with single-side mounts. So this leg's knee is held on both sides.
- STATED (US12611766, figs. 3A-3F, 43, 45-46): the knee is a rotary actuator "housed in the shin", not linear and "not driven by a linkage", with its bearing on the same side in both legs so identical parts fit either leg.
- STATED (US20260008178, figs. 9-10): the actuator sits in a receptacle in the lower thigh and the shin, with no separate knee housing, and neither part has a recess for the other at full flexion. Axis within about 10 mm of centre between their outer points.
- INFERRED: Figure moved from a U-mounted knee to a one-sided knee between generations.

### Shank (photo)
- VISIBLE: a slim closed section, about 57 mm wide from the front, with a raised strip carrying a row of 5 socket screws, plus small screws at two side heights, then flaring to a fork above the ankle.
- STATED (US12649246, figs. 7, 9A): shin housing has a frame and a rear access panel; a drive shaft on a vertical axis runs through the knee region and shin interior, with a slide and connector rod moving along it to pitch the foot.
- Consequence: mass stays near the knee, and the shank can be slimmer than echo's, which must hold motors at mid-shank.

### Ankle (photo; side walk frames)
- VISIBLE: wishbone fork from the shank flare, a side pivot with a screw each side (pitch axis across the leg), and at the front a round cup with a dark centre and about 8-10 small features on a circle, facing forward. A black arched band sits above it.
- STATED (US12649246): foot-roll actuator housing with heat-dissipation fins and rotation limiters, rotor extent below the shin, and a pitch axis and roll axis that do not intersect; loads pass from the shin into the roll actuator's housing. Gap of 0.25-10 mm to the heel. So the forward round unit is INFERRED to be this.
- VISIBLE: a small connector or board block at the outside of each ankle with thin wires.

### Foot (photo)
- VISIBLE: faceted metal toe, machined or folded look, about 99 mm wide; black patterned sole about 8-10 mm thick. Foot length isn't readable in this frontal view.
- From side frames, the foot stays roughly flat in stance and lifts low. No toe joint visible.

### Backpack (0-4, 22-26, 55-60, 95-103 s)
- VISIBLE: rectangular pack with a diamond dimple panel, a black top handle or latch, a black lower part. About 150 x 360 mm, ±15 %.
- Secondary source: Figure 02 moved the battery into the torso. The image search's notes quote Figure on Figure 03's battery as a torso member. So this is not a carry-over item.

### Cables and electronics
- VISIBLE: black braided sleeved bundles from pelvis to outer thigh to ankle, tied at intervals, about 22-27 mm, and thin lines to boards. Red pairs at the waist. Multi-coloured ribbon at the shoulder.
- VISIBLE: boards and connectors sit near joints: pelvis, upper arm, ankle.
- Not readable: connector types. Small white multi-way plastic connectors are plausible but can't be confirmed in sharpened frames.
- Secondary source: external cabling was for testing; Figure 02 moved it inside limbs.

### Motion (wide shots)
- VISIBLE: walks with knees bent throughout, hips flexed, short steps, feet low, torso nearly upright and slightly forward. Arms swing a little, elbows slightly bent. Head stays level.
- MEASURED roughly: knee flexion about 20-40° (by eye, ±10°); step period about 0.9-1.1 s, if played at real speed. Unknown playback rate.
- Foot landing looks flat or a small heel-first. No obvious toe-off push.
- It looks less natural than the relaxed, straighter walk echo's user wants.

## What echo should take, per joint

Figure's motors are custom; echo's are RobStride (`spec/actuator_interfaces.json`). So copy **the layout idea**, not shapes or sizes. Sizes below are first proposals, not designs: 7075 assumed, walls to be checked by the strength table and FEA.

| Joint | echo now | Take from Figure 01 | Proposal around the real motor | Can't transfer |
|---|---|---|---|---|
| Knee | RS04 d120 in a grey neck, cheeks and post | Barrel on axis, round cap each end, shield housing into a closed shank, both sides held | Round housing bolted to the 10 M4 d106 face, about 132 mm OD, 4-6 mm wall. Output hub 9 M5 d36 to a two-sided thigh fork, far side on a thin bearing | Figure's motor is wider (112 mm) and narrower in diameter. RS04 is 55.7 mm long, so width is easier for us, diameter harder |
| Thigh | 24 x 80 slab | Closed round tube, about 66 mm; clamp collar | Tube about d60-70 x 3 mm, or closed oval, inside the skin. Audit: d60 x 3 is 2.4x stiffer than the slab, with less material | Check it passes the yaw motor |
| Hip | Pitch, roll, yaw; roll stator 100 mm behind on one bar | Round housings on a real plate; forked, pocketed arm, motor on axis; maybe yaw first | Bring roll on axis in a two-sided yoke. Trial yaw-first in the hip round: RS03 or RS04 vertical at y about ±0.1 m in the pelvis | Changing order means new kinematics, retarget and retrain, and yaw motor must carry the whole leg's moment |
| Pelvis | Blocks and tabs | Thick pocketed plate as the leg and waist base | One machined plate about 300 mm wide, motors bolted through real faces | Figure's board layout |
| Waist | Yaw only, RS04 on a post | Two motors in the torso bottom, cranks, spherical rods, central joint, round ring | Phase two study: RS03 or RS04 pair with cranks at about 64 mm rod spacing, standard rod ends. Reserve this volume in the chest and pelvis now | Figure 03's waist is narrow and black; check it fits under the hem, whose clearance the skill's lessons warn about |
| Chest | No frame, battery is the path | Not visible on Figure 01 | Frame from hub to shoulder beam; battery a passenger. Audit items 3 and 19 | Figure 01's backpack |
| Shoulder | Pitch in, roll on a jog (new) | Pitch inside the torso, big ring behind a round opening | Confirms the 30 mm move. Bolt RS04 on d106 face; look at a large bearing ring for the arm's root | Cap: echo has fabric, not a cap |
| Upper arm | Plates and an RS02 | Coaxial barrels form the arm | Round housing on RS02 (9 M3 d73), tube above and below | RS02 is 78.5 mm dia vs Figure's about 77: a good match |
| Elbow | RS06 hubs, 65 mm neck | Round end, block, sealed join to tube | Round housing on RS06 (8 M3 d82), forearm a closed tube | Figure's elbow is narrower; our 88 mm over-ears motor sets width |
| Forearm | Flat frame | Tapered closed tube | Tube tapering from about 70 to 55 mm around RS02 and RS00s | The RS00 stack may not fit a 55 mm tube, since RS00 is 57 mm: check |
| Shank | Plates and motors, shank joined through a motor | Slim closed tube, ankle motor high | Keep motors higher and closer to the knee; closed U-channel or box continuous past motors, walls past each motor. Study a screw drive only if rods fail | Ball screw isn't off-the-shelf with our motor; RS03 is 106 mm over ears so the section stays near 110 mm |
| Ankle | Rods float, no pins | Wishbone fork, real side pivot, round roll unit | Fork with steel pins and needle or bronze bushes, rods actually on crank shafts | Figure's roll unit is custom |
| Foot | 1.51 kg | Faceted, light | Lighten first, audit item 6. STATED (US20260103253): a later document uses a single passive toe and a sole slit | Figure 03's shoe is the look target, not this |
| Cooling | None | Ribs on motor bodies in open air. STATED: fins on hip and ankle actuators (US12583126, US12649246); torso fans and vents (US12707556) | **Caution:** ribs on Figure 01 are in open air. Under fabric, the motor needs a conduction path or air path. Ribs alone won't transfer | RS motors' housings are fixed |
| Cabling | None | Bundles near joints, boards near motors | **Don't copy external routing.** No hollow RobStride shaft, so plan service loops under the skin at waist, shoulder, wrist | |

## Joint list stated in patents

US12611766 (figs. 3A-3F, 9, 35-36, 43, 45-46; table 3), for "robot 1": 62 DoF total, 42 rotary actuators in seven types, height about 1725 mm, arm span about 1800 mm, arm under 20 % shorter than leg, torso about 20 L. Names are stated where I read them; J6, J7, J8, J15 and J16's roles are INFERRED from position. Generation is INFERRED (later than the video's robot).

| Joint | Name | Peak N-m (preferred) | echo now |
|---|---|---|---|
| J1 | Arm, axis 1-45° rearward | 125-188 (141-172) | Shoulder pitch, RS04 120 N-m, horizontal |
| J2 | Shoulder | 81-122 (91-112) | Shoulder roll, RS06 |
| J3 | Upper arm twist | 81-122 (91-112) | Shoulder yaw, RS02 |
| J4 | Elbow | 81-122 (91-112) | Elbow, RS06 |
| J5 | Lower arm twist | 15-22 (17-20) | Wrist yaw, RS02 |
| J6, J7 | Wrist | 15-22 (17-20) | Wrist pitch and roll, RS00 |
| J8.1, J8.2 | Neck | 15-22 (17-20) | Neck yaw and pitch, RS00 |
| J9 | Torso lean, in the pelvis, axis 8-16° down | 204-307 (230-281) | None |
| J10 | Torso twist | 204-307 (230-281) | Waist yaw, RS04 120 N-m |
| J11 | Hip flex, axis at a downward angle | 204-307 (230-281) | Hip pitch, RS04 120 N-m, horizontal |
| J12 | Hip roll, axis about 10-20° rearward and down, about half of J11 by the text | 125-188 (141-172) | Hip roll, RS04 120 N-m |
| J13 | Leg twist, up to 90° each way | 125-188 (141-172) | Hip yaw, RS03 60 N-m |
| J14 | Knee, rotary, in the shin | 204-307 (230-281) | Knee, RS04 120 N-m |
| J15 | Foot flex | 125-188 (141-172) | Ankle pitch, RS03 through a rod |
| J16 | Ankle roll, lowest actuator | 45-68 (51-62) | Ankle roll, RS06 through a rod |
| Hands | | 2.5-6 (4-5.5) | Not modelled |

**Conflict:** US20260008178's table 3 gives different numbers for the same labels (J11 and J14 265-498 N-m, preferred 298-365). Treat these as ranges a patent chose to state, not the robot's real torques.

**Other stated numbers:** elbow axis offset rearward 1-50 mm, preferred 5-25 mm; upper-arm bearing radius 40-80 mm, lower-arm 25-50 mm; wrist pivot offset 10-100 mm (all US12611766). Lower torso width at least about 65-67 % of pelvis width (US12583126). Battery 1.5-5 kWh, preferred 2-3 (US12707556). Joint-safety limits including 130 N compressive force and 250 N/cm2 pressure (US20260008178). I found no thigh or shin lengths in the passages I read, and I searched rather than read all 220,000 characters of US12611766.

**For echo:** knee and hip flex at 120 N-m compare with a stated 204-307 N-m at the same height. The user has chosen RS04 and walking-only for first hardware; this is a stated, not measured, comparison.

## What Figure 03 probably kept

Evidence is the look of Figure 03's frames and Figure AI's patent drawings, which name no robot.

| Claim | Evidence | Confidence |
|---|---|---|
| Black ovals at hip and knee are end caps over barrel motors | Same oval-with-boss shape in fig. 3, at the same places | Medium |
| Wide black pelvis holds round yaw housings | Figure 03's wide T-shaped pelvis; fig. 3's pelvis | Medium to low |
| Slim shank from a high-mounted ankle motor | Figure 03's long straight shank | Medium to low |
| Round closed limb structure under fabric | Continuity of company style; cover patent mentions housings | Medium |
| Rod-driven waist | Only Figure 01; Figure 03's waist isn't visible | Low |

None of this is published as fact about Figure 03. State it as inference in any pitch.

## Still unknown

Thigh collar's purpose; elbow support on the video's robot; wrist motor arrangement; neck details; any patent-to-robot link; all materials (looks like machined aluminium, unconfirmed); gearbox types (patents list options); true torques; true walk speed.

## Patent documents used as technical sources

Read for how the mechanisms are arranged. They never name a robot.

- US12583126 B2: central region arrangement
- US12649246 B1: ankle region
- US12707556 B1: torso
- US12611766 B2: advanced kinematics
- US20250312911 A1: hip assembly and kinematics
- US20260008178 A1: advanced joint assemblies
- US20260103253 A1: leg
- US20250242500 A1: exterior covering
- US20260061625 A1: exterior covering

## Shot index (video time, s)

| Time | Subject | Use |
|---|---|---|
| 0-3.9 | Wide side walk with backpack | Gait, backpack |
| 4-6.9 | White four-legged machine on rocks | Not Figure 01; ignore |
| 7-9.9 | Upper body front, second robot behind | Chest, arm layout |
| 10-12.9 | Head and shoulders, 3/4 | Head, neck, cap, cable loop |
| 13-15.9 | Hands near camera, tilt to waist and hips | **Hand, forearm, elbow, upper arm, hip** (13.2, 14.0, 15.4) |
| 16-18.4 | Shoulder and chest close-up | **Cap, ring** (18.0) |
| 18.5-21.9 | Waist, tilt up | **Waist** (19.0, 20.5, 21.6) |
| 22-26 | Wide side walk | Gait, side proportions (24.0, 24.6, 25.2) |
| 26.5-28.9 | Hip and thigh, arms | **Hip outer face** (27.0, 28.5) |
| 29-30 | Head | |
| 30.5-33.9 | Walking, waist-up, front-side | Torso sway (31.5, 33.2) |
| 34-35.5 | Head | |
| 36-45.5 | Wide front, zooming in | Front proportions (38.0) |
| 46-49 | Pelvis (sharpened) | **Pelvis** (46.6, 48.9) |
| 49.5-51 | Head | |
| 51.5-55 | Repeat of 13-16 | |
| 55.5-59.5 | Wide side walk | |
| 60-62 | Repeat of 26.5-29 | |
| 62.5-63.5 | Head | |
| 64-66.5 | Shoulder, cap and bolted plate | **Best shoulder** (65.5, 66.3) |
| 67-68 | Shoulder and chest | 67.5 |
| 68.5-71.5 | Repeat of waist | **Sharpest waist** (69.2, 69.7, 71.0) |
| 72-75 | Head, zooming | |
| 75.5-78 | People seated | Not relevant |
| 78.5-81 | Repeat of 10-12 | |
| 81.5-122.8 | Wide walking in arcs and front, zoom to head | Gait (100-103.3, 106) |

About 34 s are mechanism close-ups, some of them repeats (about 10 distinct mechanical shots), and about 60 s are wide walking.
