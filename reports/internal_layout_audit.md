# echo internal layout audit

**Scope.** All findings apply to the frozen copy at `C:\Users\PC\AppData\Local\Temp\claude\714944801f6bb85e\092d1abd-f8dd-4f81-a7d0-79bb9f2082fb\scratchpad\frz2`, not to the live `spec/`, which may have moved on. Audit scripts and raw output are in the `audit` folder beside it. Stresses are hand estimates, not FEA.

## Verdict

1. The kinematics match common practice, and the mass distribution is in family except for one heavy foot and a torso that is unusually mass-heavy.
2. The internal structure does not match: leg torque is half of most peers, the hip load runs through one eccentric bracket, and the chest has no frame.
3. `check_connect` proves only that shapes touch, so its pass says little about strength; at 3,400 N the hip fails on paper (safety factor 0.85 to yield, on a solid section).

## How to read the verdicts

- I found no definition of "phase 3" in the frozen copy, `docs/`, `LESSONS.md` or the skill.
- So I used this test. **Architecture** changes a motor, an axis, a package volume, a cover or the mass budget. **Placeholder** can be redrawn inside the existing package.
- Normalised torque is N m / (M g L), with L = hip to knee + knee to ankle. Echo's M g L is 493 N m.
- Fork test: lines parallel to each joint axis, cast through the reference visual meshes, show which link spans the other.

## Real architecture problems, ranked by consequence

| # | Item | Echo | What references do (evidence) | Verdict | Consequence | Suggested change |
|---|---|---|---|---|---|---|
| 1 | Knee torque | RS04 direct, 120 N m peak, 0.24 | G1 0.66, A3 0.69, H1-2 0.57, H2 0.51, X2 0.45; only Asimov 1 matches at 0.24, and it is 35 kg with a 5 kg rated squat (URDF efforts, Asimov README) | **Wrong for this mass** | The stair check already fails: 138 N m x 1.5 = 173 % of peak. By hand, one knee at 120 N m holds about 1,100 N at 30 deg of flexion and about 570 N (1.0 BW) at 60 deg. So 3,400 N is only possible nearly straight | Target about 220-340 N m (0.45-0.69). A second reduction stage, or a motor high on the thigh driving through a rod at the ratio that range allows (H2 does this: `left_knee_motor_link` plus loop rod, 360 N m, `H2_loop.xml`) |
| 2 | Hip roll placement | Roll RS04 sits 100 mm behind the pitch axis; the only stator-to-output link is one 13 x 15 mm bar on the outer side. Pitch link CoM x = -91 mm | Reference pitch-link CoM within 8 mm of the pitch axis in x (Asimov 2 mm relative to its joint). They keep the roll stator under or outboard of the pitch axis, some with a 30-50 mm axis offset (G1, H2, A3) | **Unique to echo; wrong as a load path** | Every foot load bends through a 72-100 mm lever. Bracket about 590 MPa von Mises at 3,400 N (see estimates). Also contacts the flange over 160 mm², outside the Ø36 output bolt circle | Bring the roll stator to x about 0 with a small pitch/roll axis offset, or carry it in a two-sided yoke on its bolt face. Either changes the hip lobe, so do it now |
| 3 | Torso load path | Hub, 60 x 60 spine, then **no frame member from z 1.095 to 1.32 (225 mm)**, then shoulder beam. Connection is 5 mm into the lower pack, 1 mm between packs, 5 mm of beam into the upper pack | Reference torso meshes are closed bodies from waist to shoulder (A3, G1, H2, Asimov; cover vs frame not separable). Figure's pack is reported as load-bearing, in a die-cast housing with active cooling (secondary source) | **Concept has precedent; echo's version is unusual** | Checks only pass because the pack has role `struct`: re-roled to `decal`, the torso falls into 2 pieces. Removing the pack detaches the shoulders from the pelvis. Falls load the cells directly. Pack case not designed; 13.1 kg leaves no frame mass | Either two side rails or a rear plate from hub to beam, with the pack as a removable passenger; or a designed structural case. Reserve volume now. Budget mass for it |
| 4 | Thermal path | Stators inside a 3 mm PA12 + knit sleeve. No vents, heat pipes or air path. Knee: 85 cm² stator plate plus hub cap | G1: aluminium housings clamp the motors, copper heat pipe on the knee stator, two hip fans (Munro teardown via humanoid.guide; Unitree: "local air cooling") | **Unusual, unverified** | Ratings need plates: RS04 40 N m on 345 x 345 mm, 14x the plate echo gives it. `check_poses` uses `rated_nm` for static poses, but held still the stall figure applies: elbow 10.4 N m against about 8 at stall, so 3 kg "for as long as needed" is over | Add a thermal check using stall ratings and real conduction area. Provide an air path under the fabric, or conduct into the frame. Figure publishes nothing on its thermal design |
| 5 | Hip yaw sizing | RS03, 60 N m peak, 0.12 | G1 0.42, H1-2 0.38, X2 0.45, A3 0.48, H2 0.51; Asimov 0.15 | **Low** | 98 % used on the stair. It also carries the full leg load in its output bearing (rating unknown) | RS04, or accept a restricted gait. Changes the thigh top and seam |
| 6 | Foot mass | 1.51 kg | G1 0.61, A3 0.75, H1-2 0.75, X2 1.05, Asimov 0.47 (URDF; references have no shoe) | **Heaviest of all** | 45 % of leg inertia about the hip (1.23 of 2.73 kg m²). This is the largest inertia lever in the leg | At 0.75 kg, hip inertia drops 22 %. Lighten shoe, sole and coil before moving any motor |
| 7 | Shoulder roll and arm root | One-sided: axle, then a 6 x 20 mm lug behind the cap | Fork in 4 of 5 readable models (Asimov, G1, H2, A3); H1-2 one-sided | **Unusual** | RS04 pitch torque passes through the lug's weak axis: 120 N m gives about 1,000 MPa. Pitch RS04 is 0.24 against 0.08-0.22 on peers, so the motor can break its own arm | Fork the roll stator, or thicken the lug to at least 15 mm. Consider a smaller pitch motor |
| 8 | Planned 30 mm inboard shoulder move | Pitch at y 0.115, roll at y 0.205; spacing 90 mm, largest of all (peers 40-68) | Roll y/L: echo 0.243, range 0.189-0.314, so 0.207 stays in family. Arm jogged outboard of the roll axis: H1-2 33 mm, H2 26, X2 43 | **Fine if the arm is jogged** | A pitch axis has no y position: this really moves the pitch stator and the roll axis. With the arm on the new axis its inner face is at y about 0.128, inside the 0.148 torso by about 20 mm (10 mm clear today). Roll alone can move only 13 mm before the RS06 meets the pitch flange | Move both with a 25-35 mm arm jog, or narrow the torso. Moving pitch more than roll also tightens the 90 mm spacing |
| 9 | Hip range | Pitch -1.65..0.24 (1.89 rad), yaw +-1.2 | Pitch range 3.14 (Asimov) to 5.6 rad (H1-2); yaw +-2.7 on G1 and A3, but +-0.43 H1-2, +-0.79 Asimov | **Narrow** | Limited extension for stride, getting up and fall recovery. Set by covers | Decide what motions matter, then accept it or move the lobe |
| 10 | Cable routing and service | No wire or connector in the spec. 6 % mass allowance. Waist, shoulder yaw and wrist yaw at +-2.6 rad. No build-order check in the frozen `tools/` | G1 "full joint hollow electrical routing" and quick-release battery (Unitree). Asimov: modules are the interfaces. Figure: covers removable without tools | **Unaddressed** | External loops have nowhere to live under thin sleeves. Pack access under a one-piece torso is unresolved. Rework later will conflict with covers | Route one arm and one leg in CAD now; define pack removal with item 3 |

## Placeholders that phase 3 can fix, but must not inherit

| # | Item | Echo | References | Verdict | Consequence | Suggested change |
|---|---|---|---|---|---|---|
| 11 | Stator mounting | 18 of 29 stators attach only by overlapping the motor's side wall: waist, hip pitch, hip roll, shoulder pitch, roll and yaw, neck pitch, wrists | G1 clamps motors in aluminium housings. Retailers show housing bolt circles (RS04 Ø106 x 10 M4, output Ø36 x 9 M5; confirm on drawings) | **Wrong as drawn** | No fixing exists there. Real bolt faces may not be reachable in the current package, so this can become architecture | Model bolt circles; test each stator's access |
| 12 | Shank continuity | Lower and upper plates meet by a 20 mm² lap plus 41 mm² on the RS03, per side; the motor housing bridges them | G1 mid-shank section shows side walls past the motor (mesh) | **Wrong as drawn** | 3,400 N passes through a motor housing not rated for it | Continuous plates or a tube past the motor, with motors bolted inside |
| 13 | Ankle hardware | Cranks 3.5 mm short of the motor faces; clamp plates cover the output faces; roll crank and rod cut into the plate; 1 mm and 3 mm air at the pivots; no pin | G1 Cardan joint; Asimov moving ankle pivots to steel for wear | **Placeholder** | Details of items 12 and floating parts | Output through the plate; bronze or needle pivots; steel pins |
| 14 | Hip tabs and crown | Two 10 x 10 mm tabs, 36 mm² each onto the crown | n/a | **Placeholder** | Safety factor about 1.3, solid | Full plate to the yaw back face |
| 15 | Knee and elbow second support | Post r 20 in a bore r 22.5, axle r 20 in a bore r 21.5, cheek 3 mm | Forks common at the knee (Asimov, H1-2, H2, A3) | **Right idea, won't fit** | Those gaps are too small for a catalogue ball bearing (from memory, 6808 is 40 x 52 x 7): plain bush or more room. Bore leaves 7 mm of cheek width | Size for a real bearing; hub width may grow, so check the look early |
| 16 | Head and neck | Head on one 6 x 30 mm plate; 20 x 20 post on the pitch motor's side wall; head 1.0 kg | Pitch fork on Asimov, H2, X2; head 1.1-3.1 kg | **Unusual, low risk** | Stiffness and camera shake; the head mass looks light | Fork it; recheck mass |
| 17 | Thigh frame | 24 x 80 mm slab, weak axis sagittal | Thigh sections read as closed bodies (G1 75 x 72, H2 128 x 120, A3 136 x 141); X2 two C shells. Covers not separable | **Unusual** | Strength fine (26-52 MPa), stiffness weak | A Ø60 x 3 tube is 2.4x stiffer with less material |
| 18 | Rbox thickness | 5 plates have fillet > half thickness: 3 mm cheeks mesh at 5 mm, forearm frame at 7 mm | n/a | **Bug** | Mass and contact use 5 mm; spec's stress notes use 3 mm | Clamp the fillet. Connect still passes when clamped |
| 19 | Compute mount | Docs say on the shoulder beam. Geometry touches only the pack, 1 mm deep | n/a | **Mismatch** | 1.6 kg hung on the battery | Mount to the frame from item 3 |

## Both-side support

- The "23 of 29" note is about right on my test, but ankle roll is a real fork the test misses, because the tail runs through the clevis. So it is 21 one-sided, 8 forked.
- **14 of the 21 are standard.** Hip pitch, hip roll, hip yaw, waist yaw, shoulder pitch, shoulder yaw, wrist yaw and neck yaw are one-sided on nearly all readable references (H2 forks hip roll). Crossed-roller output bearings take the moment (Unitree).
- **7 are unusual:** shoulder roll x 2, wrist pitch x 2, wrist roll x 2, neck pitch. References fork these.
- Output-bearing moment ratings for RobStride are the unknown that decides whether the 14 are safe at this mass.

## What echo does that is fine or better

- **Joint order:** hip pitch-roll-yaw, knee, ankle pitch-roll matches G1, H2, A3, X2 and Asimov. Arm order matches when zero poses are aligned. Neck yaw-pitch matches Asimov, A3 and X2.
- **Yaw-only waist:** also on Asimov and H1-2, a fair trade for a one-piece torso.
- **Leg inertia:** I/(M L²) 0.064, within 0.046-0.079. Below-knee about the knee 0.046, within 0.024-0.061 (median 0.038), and most of the excess is the foot.
- **Arm inertia:** lowest of the set, 0.021 against 0.023-0.051 (arm straight, no hand). Forearm too.
- **CoM height:** 1.02 L, within 1.015-1.096, despite a torso at 35 % of mass (peers 20-29 %). Battery in the torso is standard.
- **Knee stator on the shank:** unusual (thigh/shank 1.02/4.85 kg, ratio 4.7 against 0.6-1.9) but about neutral. It is on the axis, so hip inertia is unchanged and knee inertia changes about 0.5 %. Moved back, the ratio is 1.4. H2's elbow looks similar. It is not a problem to fix.
- **Knee on the axis:** G1 and X2 do this too.
- **Ankle with two motors and separate rods:** unusual, not wrong. Most robots share two motors. H2 is a closer precedent than the spec's "no published example": `H2_loop.xml` has one crank and rod for pitch plus a separate roll drive. Torques 0.18 and 0.10 are in family; rods buckle at about 2.5-2.9x motor peak.
- **Materials:** 7075 and PA12 by CNC and MJF suit this size and quantity (Asimov 7075, PA12, 316L; G1 billet aluminium). Die-casting is a volume choice (Figure 03).
- **Narrow hips:** y/L 0.118 is the narrowest (0.147-0.225) but matches the target look. Its cost is item 2.

## Floating parts

**Method.** Each link's rigid parts checked pairwise with exact booleans on role struct and actuator, and linkage separately, ignoring shells. Contact means overlap, or a gap within 0.1 mm. Contact area = overlap volume / thinnest overlap dimension. "Structural" means at least 100 mm². Left side checked; right is a mirror.

**Results**

- **At any contact:** all 30 links are one body except for the linkages below. The picture's impression is partly right: nothing is truly loose among struct parts, but many joints are only touching.
- **Under 100 mm², the links split:**

| Link | Contact (mm²) | Note |
|---|---|---|
| L_hip_roll_link | tabs to crown 36 each; plate to tabs 78 each | **Every foot load passes here** |
| L_knee_link | lower to upper shank: lap 20 + RS03 41, per side | Shank is in two parts, joined by the motor |
| L_foot | roll pin to foot plate 38 | Pin overhangs the plate edge |
| L_ankle_pitch_link | tail to pitch arm 70 | Ø10 arm |
| L_shoulder_roll_link | step to foot 94; lug to step 145 | Whole arm |

- **Thin but over the limit:** bracket to pitch flange 160, wrist cheek to RS00 159, elbow bridge to cheeks 153, head electronics 112.
- **Truly floating linkage:** both ankle cranks and rods. Pitch crank 3.50 mm and roll crank 3.55 mm from their motors, with no shaft. Their only other contact is unintended overlap with the clamp plates (roll crank 0.49 cm³, roll rod 0.09 cm³, pitch crank 0.03 cm³), which no check sees.

**Joints (do the two links physically connect?)**

- **25 actuated joints:** yes, through a stator and 1 mm flange overlap (the check's assumed output connection). No bearing is modelled inside.
- **Knee and elbow:** yes on the inner side. The outer sides are bearing gaps: 2.5 mm radial at the cheek-post pair, 1 mm from axle to hub.
- **Ankle pitch and roll (4 joints):** **no.** 1.00 mm of air to the fork, 3.00 mm to the clevis, no pin. They pass because the axis line happens to enter both links.

**What `check_connect` actually accepts**

It unions role struct and actuator, counts pieces over 0.01 cm³, and tests that a ±180 mm line along each axis enters both links. Run in memory on copies:

| Change | Result |
|---|---|
| Hip bracket 13 x 15 reduced to 2.7 x 3.5 mm | Passes |
| Torso spine 60 x 60 reduced to 2 x 2 mm | Passes |
| Hip bracket made of PA12, role still struct | Passes |
| Bracket touching the flange by 0.02 mm | Passes |
| Loose chip of 0.008 cm³ | Passes (0.012 cm³ fails) |
| All four ankle cranks and rods deleted | Passes |
| Ankle as is, no pin | Passes |
| Same bracket shape, role shell (cosmetic cover) | Fails, correctly |
| Knee RS04 moved 0.02 mm off its plate | Fails: exact face contact passes only by luck of rounding |

So a cover cannot pass, but a token contact can. The knee's aluminium hubs are role `shell` with material al7075, so the second support is invisible to the check.

**Suggested check:** minimum contact area per link interface, minimum section along a named load path, and a pin or bearing part required at passive joints. Prove it with a `--bad` case.

## Hand estimates at 3,400 N (estimates, not analysis)

**Assumptions.** Solid sections, though the mass model assumes fill 0.25-0.8. 7075-T6 yield about 503 MPa. Load 5.8 g on the whole body. Knee and hip torques capped at 120 N m. No fatigue, buckling of thin walls or fastener checks.

**Hip to foot**

| Section | Size | Load | Stress | SF to yield |
|---|---|---|---|---|
| Hip bracket | 13 x 15 | 245 N m bending, 100 N m torsion | 590 MPa VM | **0.85** |
| Hip tabs | 10 x 10, two | 64 N m each | 380 MPa | 1.3 |
| Ankle pitch arm (at RS03 peak) | Ø10 | 33 N m | 330 MPa | 1.5 |
| Hip front plate | 7 x 104 | 226 N m | 266 MPa | 1.9 |
| Foot plate (all load at toe) | 70 x 16 | 408 N m | 137 MPa | 3.7 |
| Knee cheek tops | 2 x 3 x 40 | 188 N m + axial | 132 MPa | 3.8 |
| Shank lower plates | 2 x 4 x 40 | 120 N m + axial | 67 MPa | 7.5 |
| Thigh frame | 24 x 80 | 200 N m | 26 MPa (52 hollow) | 10-19 |

The smallest net section is the outer knee cheek at the bore, 7 x 3 mm = 21 mm², but the bearing pushes above it in compression.

**Shoulder to pelvis** (33.0 kg above the waist: about 1,880 N)

| Section | Size | Load | Stress | SF |
|---|---|---|---|---|
| Frame between spine and beam | **0 mm² of frame** | about 1,050 N entering the pack top, 1,820 N below | Depends on a pack case that is undefined | Unknown |
| Shoulder roll lug at RS04 peak | 6 x 20 | 120 N m | 1,000 MPa | **0.5** |
| Shoulder roll lug, arm at 5.8 g | 6 x 20 | 20 N m | 167 MPa | 3.0 |
| Spine | 60 x 60 | 188 N m + axial | Under 15 MPa | Over 30 |

**Bearings.** With a 100 mm lever, the waist RS04 output bearing sees 1,880 N and 188 N m, about 8.7 kN equivalent assuming a 55 mm bearing pitch. The hip roll bearing sees 3,400 N and 226 N m, about 11.6 kN. I could not find a rating. These may be the true limit.

## What I couldn't confirm

- **Internals of Figure 02/03, Optimus, NEO and T1:** no teardown or CAD found. Figure's load-bearing pack rests on a press article; its own post says only that structures are in-house. Optimus, NEO and T1 gave no usable layout evidence, so they support no finding here.
- **G1 teardown:** I read a written summary of the Munro video, not the video. It does not cover the torso, battery or cables. No teardown photos were seen.
- **RobStride:** output-bearing ratings, the housing bolt face and whether wiring can pass through the module. Bolt circles are retailer figures.
- **Reference meshes:** visual meshes mix covers, frames and motors, and many are not watertight. Several fork tests were unreadable: X2 (mostly), A3 hip and head, H2 hip yaw and wrist pitch. The X2 torso mesh has a stray part 3.96 m wide. Stator location in references is inferred from CoM and link mass.
- **Asimov:** only `sim-model` is cloned. CAD and its manual were not read; its structural docs are marked outdated. The manifest gives part counts by class only (43 7075, 39 PA12, 18 316L, 67 bought).
- **Berkeley Humanoid Lite:** a 16 kg printed robot, too different in class to set a standard here. Not used for scaling.
- **The 3,400 N figure:** taken from your brief; I did not rerun the sim, or its direction and foot position.
- **Live `spec/`:** not read.

## Sources

**Local, parsed:** `figrefs/unitree_ros/robots` (`g1_29dof_rev_1_0.urdf`, `h1_2_handless.urdf`, `H2.urdf`, `H2_loop.xml`), `figrefs/A3-A3U-robot-model/a3_t3d0/urdf/model.urdf`, `figrefs/agibot_x2_urdf/X2_URDF-v1.4.0/X2-Ultra.urdf`, `figrefs/asimov-1/sim-model/asimov_1_25dof`. Echo: `frz2/urdf/echo.urdf`, `frz2/spec/robot.json`, `frz2/spec/robot.py`, `frz2/tools/checks.py`, `frz2/tools/geom.py`. Picture: `reports/parts/skeleton_first_hub_round.png`.

**Web:**
- [Unitree G1 teardown summary, humanoid.guide](https://humanoid.guide/?p=17935)
- [Unitree G1 product page](https://www.unitree.com/g1)
- [Hip, knee, ankle budgets, humanoid.guide](https://humanoid.guide/?p=17104)
- [Parallel ankle mechanisms, humanoid.guide](https://humanoid.guide/?p=18750)
- [Asimov 1 mechanical tour](https://docs.menlo.ai/asimov/1/overview/system-tour/mechanical)
- [Asimov 1 frame and structure](https://docs.menlo.ai/asimov/1/hardware/frame-and-structural-components)
- [Menlo, humanoid legs in 100 days](https://menlo.ai/blog/humanoid-legs-100-days)
- [Asimov 1 fabrication manifest](https://raw.githubusercontent.com/menloresearch/asimov-1/main/mechanical/FABRICATION_MANIFEST.csv)
- [Introducing Figure 03](https://www.figure.ai/news/introducing-figure-03)
- [Figure battery, Interesting Engineering](https://interestingengineering.com/energy/figure-unveils-fireproof-humanoid-robot-battery)
- [Berkeley Humanoid Lite, arXiv](https://arxiv.org/abs/2504.17249)
- [RS03 listing](https://manuals.plus/ae/1005007705574359), [RS04 listing](https://rcdrone.top/products/robstride-04-qdd-120n-m-integrated-actuator-module-48v-9-1-ratio-dual-magnetic-encoder-ip52-700w-output)
