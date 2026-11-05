# The joint template

The knee (iteration 41) is the pattern for the hips, elbow, shoulders and ankle. Sizes below are the knee's; rules are for all. Hand estimates, offline solids, no FEA, not CAD. Picture: `reports/parts/knee_metal_0.60.png`, in section `reports/parts/knee_metal_section.png`.

## Rules

| Rule | At the knee | Checked by |
|---|---|---|
| A round housing round the motor, one piece with the link | Barrel: bore d120.1 on the RS04's d120, OD 127 (3.45 wall), `HW.round_housing` | `motorfit`, `stator`, `loadpath` rows "Barrel" |
| Bolted on the maker's own face, located by a register | 10 M4 on d106, front face (drawn 5 deep); the bore is the register | `pattern`, `fastener`, `bolted` |
| The output goes to a cup, not a plate | 9 M5 on d36 through a 5.8 mm floor; the wall scalloped round the heads so a key reaches each | `fastener`, `access` |
| Two real supports, with the motor's own bearing left out | Two CRBS 508 (d50 D66 B8, C0 6170 N), 67.9 mm apart, one each side | `bearing`, `pivot` |
| Every race stopped both ways, by a part you can name | Inner race: cup's shoulder and a thin nut. Outer: barrel's lip and a threaded ring. Far side: post shoulder and washer; cheek's lip and a ring | `bearing` "stopped toward" |
| Load goes round the motor's case, not through it | Foot, shank body, barrel, bearings, fork, tube. The motor passes torque only | `loadpath`, `contact` |
| A fork that tapers into a closed tube | Fork: side outline a lug from the eye (r 48) to the tube (r 33); sections go squared to round. `HW.fork_yoke`, `HW.closed_tube` | `loadpath` rows "Fork", "Thigh tube" |
| The slot's roof round about the joint's axis | r 67.7 over the barrel's 67.2 | `sweep` |
| One cheek bolts on, so the joint can be built | Outer cheek, 4 M5 button heads, at least 2 d into the crown's bosses | `assembly`, `fastener`, `bolted` |
| A cap closes the housing and carries the far bearing's post | Rear cap, dished, with a skirt on the motor's back for heat; `HW.end_cap` | `bearing`, `contact`, `bolted` |
| Closed sections past whatever sits in the limb | Shank body, wall 1.7 (1.4 past the RS03), rear panel on 7 flush M3 | `loadpath` "closed section"; Bredt for torsion, in `strength.py` |
| Fold room is a sum of angles | At any radius the two links' half angles add to under 180 deg less the range: 57 deg at 2.15 rad. So the shank's back is scooped behind the knee | `sweep` |
| Metal sits on load paths and nowhere else | Every part truly hollow, `fill` 1, mass from its solid: `HW.machined` | `mass`; `tools/mass_model.py` |
| Fillets are real | Distance trees (`tools/sdf.py`), blends a named size | pictures |

## Build order, as data

`HW.step(parts, path, distance, screws, why)`. Group `assembly` slides each part in along its path past what's on by then; `access` then reaches each pattern's screws at that moment. Parts not yet on aren't in the way; parts on are. Knee: flange on the hip; motor into barrel from outboard, 10 M4 from inboard; rear cap, 6 M3; fork with cup, bearing and nut from inboard, 9 M5 through the eye; outer cheek with its bearing from outboard, 4 M5, washer; hubs; back panel; leg up to the flange, 6 M4 from below.

## Sizes to start from on another joint

Bore motor radius + 0.05. Wall 3.5 to 4 at that diameter. Screws: M4 min on a stator; heads spot-faced where a blend runs near. Bearing by lever rule on the same force line as the rest of the table: `strength.py bearing_eval`. Tube wall 2 at d66. Cheek 5 mm at a torque side, 11 at a bearing side, pocketed to 3.5.

## What the template doesn't settle

- The maker's threads: M5 at 1.24 d, M4 at 1.19 d, against the 2 d rule. Drawn hole depth sets those, so nothing in this design can reach 2 d. Strip force still passes preload by 2.2; settled by RobStride question 4.
- Thermal under the knit (`reports/strength.md`, heat path).
- Thread forms, seals, cable clips and the service loop across the joint aren't drawn. The cable leaves the motor's tab straight down into the shank's cavity through a window in the barrel; how it crosses to the thigh is open.
- CAD: `kind: sdf` parts don't build through `cad/fusion` yet. They'd come in as meshes, or be redrawn as lofts and fillets from the same numbers.
