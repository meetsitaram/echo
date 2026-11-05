# Writing the spec (`spec/robot.py`)

The spec is plain Python that builds two lists, `LINKS` and `JOINTS`. `python -I spec/robot.py` writes them to
`spec/robot.json`, which the checks, the Fusion build, the URDF writer and the renderers all read. Keep it the only
place a dimension lives.

## Frame and units

- Metres. X forward, Y left, Z up, floor at z = 0.
- Zero pose: standing straight, arms hanging, palms in.
- All geometry is in world coordinates at the zero pose. Each link's frame is its parent joint's origin, with
  world-aligned axes. This keeps the URDF simple and lets you read every number against the picture.
- Fusion is set Z up and works in cm; only `cad/fusion/figlib.py` converts.

## Anchors first

Put the proportions in a handful of named numbers at the top, and place every part relative to them:

```python
H = 1.727                       # overall height
HY, HZ = 0.10, 0.93             # hip centre: half spacing, height
KZ, AZ = 0.535, 0.085           # knee and ankle axes
SY, SZ = 0.205, 1.39            # shoulder centre
EZ, WZ = 1.155, 0.955           # elbow and wrist pitch axes
```

When the comparison says the elbow is 2 cm low, you change `EZ`, and the arm's covers, frame, actuator and joint
all move together. echo's first re-proportioning then passed every check first time.

## Primitives

| primitive | shape | typical use |
|---|---|---|
| `loft(name, sections, color, role, open, clip, keep)` | superellipse sections `(z, cx, cy, a, b, n)` along z, interpolated (PCHIP); `n` 2 is an ellipse, higher is squarer | covers: torso, limbs, head, hip lobes, shoes |
| `cyl(name, c, axis, r, h, color, role)` | cylinder; `axis` 'x', 'y', 'z' or a vector | actuators, drums, flanges, openings |
| `rbox(name, c, size, fillet, color, role, exposed, ...)` | filleted box | brackets, plates, frames |
| `span(name, lo, hi, x, y, sx, sy, ...)` | a box filling the height lo..hi | frames joining two heights |
| `cap(name, p0, p1, r, ...)` | capsule | rods, fingers |
| `ell(name, c, r, ...)` | ellipsoid | spherical cuts, domes |
| `revolve(name, axis, c, profile, ...)` | solid of revolution, profile `(t, r)` | anything that turns about its own axis |
| `prism(name, x0, t, pts, color, role)` | convex polygon in the plane x = x0, extruded along +x | marks, swept-volume cuts |
| `HW.machined(name, tree, color)` | a distance tree (`tools/sdf.py`), blends with real fillets, truly hollow, fill 1 | housings, forks, tubes, caps: `reports/joint_template.md` |

Shaping lofts:
- `clip=` keeps the part in front of a plane, or of every plane in a list. Helpers: `below(p, q)` and
  `above(p, q)` for a line in the side view, `ahead(x)` and `behind(x)`, and `rounded(plane, r)` for a filleted
  rim.
- `keep=(centre, radius)` keeps the part inside a sphere: a domed top that a joint on the vertical through the
  centre turns in place.
- `open=` names the loft's ends that are sleeves (no cap).
- A clip only trims. A loft that stops 3 cm above a clip plane is never cut by it, so run the sections past every
  plane meant to shape them.

`cut(g, *targets)` removes shape `g` from the named geoms of the same link: windows, openings round motors, swept
volumes. In Fusion it is a combine-cut; offline it is a manifold difference. The skin check counts what a cut
uncovers as showing.

## Roles

| role | meaning | in the checks |
|---|---|---|
| `shell` | a cover (knit over PA12, polycarbonate, TPU) | a 3 mm wall; may touch neighbouring links' covers only where the sweep allows |
| `decal` | surface detail sunk into a cover (camera ports, marks, light bar) | must touch its link |
| `struct` | rigid internal part (7075 frame, bracket, battery) | may not overlap rigid parts of any other link; must be hidden unless `exposed` |
| `actuator` | a bought actuator's stator, added automatically from the joint | rigid |
| `linkage` | rods and cranks that move with a linkage, not with either link | drawn, kept out of the collision checks |
| `cut` | removed from its targets | no body of its own |
| `hardware` | a bought screw, pin, insert or retainer, placed by `spec/hardware.py` | out of the collision checks; read by groups fastener, holes, bolted, pattern, access |
| `bearing` | a catalogue bearing, placed by `spec/hardware.py` | read by groups bearing, pivot |

Mark a part that is meant to show with `exposed=True` (or `dict(cyl(...), exposed=True)`). For a joint whose
actuator shows black in a seam, use `exposed=True` on the joint; `exposed='flange'` exposes only its flange.

## Real connections (`spec/hardware.py`)

One call per connection, after the links and joints: `HW.pattern(name, joint, which, link, seat_t, length, clamps=..., counterbore_to=...)` reads a motor's pattern from `spec/actuator_interfaces.json`, puts a screw on every hole by the motor's clock, and cuts clearance holes and counterbores in the real parts. Also `fastener`, `pin_holes`, `dowel`, `register`, `insert`, `bearing`, `axle`, `retainer`; `python -I spec/hardware.py` self-tests all of them. Masses come from the catalogue (`spec/hardware_catalog.json`), or geometry times density. Anything unconfirmed is a row in `spec/assumptions.json`, read by id and printed in the reports. A motor's geom holds its clock through `motor()` and draws at the outline measured from the maker's STEP (`tools/motor_outlines.py`).

## Links and joints

```python
link('L_elbow_link', 0.85, [            # the budget (kg) is kept only for comparison
    loft('L forearm', [...], clip=below(...)),
    cut(cyl('L forearm socket', (0, SY, EZ), 'y', 0.051, 0.077), 'L forearm'),
    rbox('L elbow bracket', (-0.042, SY, EZ + 0.0375), (0.012, 0.04, 0.075), 0.003, BLACK, exposed=True),
])
joint('L_elbow', 'L_shoulder_yaw_link', 'L_elbow_link', (0, SY, EZ), 'y', (-1.9, 0.0), 'RS06', out=-1)
```

- `joint(name, parent, child, origin, axis, (lower, upper), actuator, act_c=..., out=±1, exposed=..., remote=...,
  capacity=...)`.
- The actuator's stator is added to the parent at `act_c` (default: the joint origin), aligned with the axis. An
  output flange (0.6 × the actuator radius) is added to the child on the side `out` points to. Every output then has
  a part of the child on it, and the axis check can see the actuator sits on its axis.
- Moving `act_c` along the axis changes nothing kinematically. In echo, moving the hip pitch actuator 6 mm inboard
  let the hip cover clear the wrist.
- `remote=True` with `capacity=dict(model, count, ratio)` is for joints driven through a linkage: the knee (one RS04
  at 2:1 through a pushrod) and the ankle (two RS06 through rods).
- Write the left side only. Links named `L_...` and geoms named `L ...` are mirrored to the right automatically,
  and roll and yaw limits flip.

## Masses from materials

Each geom gets a material from its role and colour (`material_of`):
- covers by their 3 mm wall;
- frames as 7075 times a fill factor, plus a per-link allowance for wiring, fasteners and bearings;
- actuators from the catalogue.

Give explicit masses with `mass=` (battery, hands) or `add_mass=`/`add_what=` (electronics on a frame).
`python -I tools/mass_model.py` reports each link's mass, centre of mass and inertia, and the total against the
target. Do not set a link's mass by hand: the bottom-up number tells you whether the design is realistic.

## Habits that paid off

- Comment each iteration's change where the number lives: the date, what the user said, and what was measured. The
  spec becomes the design history (`# iteration 21 (user: "that connection looks so weak") ...`).
- Overlap pieces that must join by 1-2 mm. Faces that only touch fall apart in the boolean union, and the connect
  check counts them as separate pieces.
- Use 3 mm for cover-to-motor clearance and 4 mm for cover-to-cover clearance at sockets.
- Start a closed spline's points at the back (t = pi), so Fusion's loft seam line is not on the front.
