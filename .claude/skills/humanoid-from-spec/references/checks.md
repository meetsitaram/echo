# The checks

`python -I tools/checks.py` runs every group, about 2.5 minutes for echo's 246 checks, and writes
`reports/checks.md` and `checks.json`. `--only <groups>` runs some; `--bad <case>` breaks the design on purpose and
shows which check fails.

## What each group guards against

| group | fails when |
|---|---|
| targets | a measured landmark (heights, widths, head and chest size, foot length) is outside the reference's tolerance |
| mass | the bottom-up mass is more than 1.5 kg over the target, or more than 10 % under it |
| axis | an actuator is not on its joint's axis |
| static | at the zero pose a rigid part is inside another part, or a cover touches a link it is not next to |
| covers | a cover passes through a motor of its own or a neighbouring link, with its skin closed on its flat cut faces as Fusion draws them |
| connect | a link's rigid parts are not one structure (a bracket missing under a cover), a cover is not mounted on them, or a joint axis misses a link |
| skin | more than 0.5 % of an internal part pokes through the covers (dark marks in the render), unless it is meant to show |
| loadpath | a named section (`tools/strength.py`, cut from the real solids) is under factor 2 to yield in any load case, or is not a struct or linkage part of a material with a strength; or a rigid part is on no path and has no stated reason. Hand estimates, not FEA |
| contact | an interface on a path has less area than its force needs; a link's frame is not one piece at 100 mm2 a contact, motors not counting as frame; a passenger has no seat on the frame |
| pivot | a passive joint or drawn bearing seat has no modelled pin or bearing touching both links, or the pin is too small |
| linkage | motor, crank, rod and output do not actually meet, a rod end has no pivot, or a moving part runs through a fixed one |
| stator | a motor is held by its side wall and not by an end face |
| sweep | a joint cannot reach its limit: each joint alone, minimum to maximum, rigid parts and the covers' 3 mm walls, across the joint too |
| poses | in each task pose: static torque against rated (continuous tasks) and peak / 1.5 (short ones), centre of mass over the feet with a margin, joints inside their limits after balancing, collisions |
| motorfit | a motor at its STEP outline (`spec/actuator_outlines.json`) is inside any part, its own link's too, deeper than a seat |
| fastener | thread engaged under the rule (2 x d); the tip past its blind hole; a screw in anything but what it threads into, or through the joints beside it; a screw showing through the skin |
| holes | a hole's edge distance or ligament is short, read from a slice of the part with every hole cut |
| bolted | slip, separation, screw shear, tension or thread strip per pattern, against `tools/strength.py`'s load cases |
| bearing | catalogue ratings against the load cases (static, life); seat wall and land; a race not stopped both ways; a bearing that doesn't fit its slot |
| pattern | a motor's stator or hub pattern isn't fully used; a screw over a connector, cable exit or bump; warns where the right side's motor can't be the same part as a mirror |
| access | a straight key along a screw's axis from its head meets something, build order not considered |

Outside `checks.py`:
- `tools/sim_stand.py`: the MuJoCo model stands, squats to 90° knees and returns, every joint limited to its peak
  torque. It reports the maximum tilt, the return height and torque saturation.
- `tools/motion_collisions.py`: replays a recorded motion and tests each leg link against every link not jointed to
  it. This covers combined motions, which single-joint sweeps cannot.

## Severity

Collisions between links at most 3 joints apart are structural: the limit cannot be reached in any pose, so they
FAIL. Farther pairs (a hanging hand against the thigh as the waist turns) depend on the pose, so they WARN, and
self-collision avoidance handles them. Overlaps under 0.3 cm³ count as touching, not colliding.

## Prove every check on a broken design

A check that has never failed may not work. Each check has a `--bad` case in `break_spec()`:

| case | breaks | must fail |
|---|---|---|
| torque | knees on RS02 | poses (torque) |
| collide | left arm moved 7 cm into the body | static |
| offaxis | hip yaw actuator 2 cm off its axis | axis |
| floating | hip bracket deleted | connect |
| mass | 20 kg added to the battery | mass |
| poke | shoulder roll actuator enlarged through its cover | skin |
| window | a window cut over an actuator | skin |
| cover | the hip lobe's openings round its motors removed | covers |
| hipshrink | hip bracket 2.7 x 3.5 mm | loadpath, contact |
| spineshrink | spine 2 x 2 mm | loadpath |
| pa12 | hip bracket in PA12, role still struct | loadpath |
| rolebad | hip bracket as shell, thigh frame as knit | loadpath |
| newpart | a bracket on no load path | loadpath (coverage) |
| token, tokenedge | hip bracket touching its flange by 0.02 mm, on the face or at the rim | contact |
| norods | all ankle cranks and rods deleted | linkage, loadpath |
| nostator | knee motor's plate and post off its back | stator |
| thinpin | a pin that touches both links, d2 under 3,400 N | pivot |
| motorinto | a plate run into the knee motor | motorfit |
| shortthread, longscrew, protrude | hub screws 3.5 shorter or 4 longer; a block in their way | fastener |
| edge, ligament | stator rim down to r 54.5; a hub hole moved beside its neighbour | holes |
| fewscrews | 6 of the hub's 9 screws gone | bolted |
| smallbearing, thinseat, noshoulder | a third of the rating; a seat's wall at 1 mm; a race's collar gone | bearing |
| missingscrew, ontab | one screw gone; one at the cable tab's angle | pattern |
| blockkey | a block between a screw head and its key | access |

Those compare with the unbroken spec (a row newly fails or gets worse), because today's design already fails these groups. `--good pin`, `link`, `stator`, `frame`, `deepthread`, `bigbearing` and `keyclear` make one fix and show the row turn to PASS.

`python -I tools/motion_collisions.py --bad collide` checks the walk replay the same way.

To add a check:
1. Write it with a physical reason. echo dropped a "frame at least 15 % of actuator mass" rule that failed a neck
   post that is plainly strong enough.
2. Add a `break_spec` case and its expected group to `EXPECT`.
3. Run `--bad <case>` and watch it fail.
4. Rerun every calibration after touching any check: an "exposed" shortcut once silently disabled the 'poke' case.

## Reading a failure

- Sweep failures name the angle where the joint was blocked and the culprit pair, for example
  `blocked at +1.20 rad by L hip lobe / L wrist cuff (4 cm3)`. Render that pose and look before changing anything.
- Pose torque lines give the worst pose, the torque × margin, and the share of the rating used.
- A check that reads the wrong shape gives a confident wrong answer. The black-hip height once measured the lobe
  before its cuts. Make each measure read the finished shape, where the photo was measured.

## Make them yours

`measure()` holds the landmark definitions, and `spec/targets.json` holds the targets. `POSES` holds the task poses
and payloads. The mass target is in `check_mass`. The leg and arm keys decide which links count as a "leg" or "arm".
