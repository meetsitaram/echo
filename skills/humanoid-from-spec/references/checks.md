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
| sweep | a joint cannot reach its limit: each joint alone, minimum to maximum, rigid parts and the covers' 3 mm walls, across the joint too |
| poses | in each task pose: static torque against rated (continuous tasks) and peak / 1.5 (short ones), centre of mass over the feet with a margin, joints inside their limits after balancing, collisions |

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
