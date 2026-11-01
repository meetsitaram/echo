# Joints and covers

Rigid covers are where a humanoid design gets hard. A real robot's covers must clear every joint through its whole
range, hide the mechanism, and still read as one body. Figure hides its folds under knit fabric; a rigid cover
cannot crease, so every joint needs a geometric answer.

## The one idea

When a joint turns, anything that is a body of revolution about its axis maps onto itself. So does a plane square
to the axis. Where several axes cross, a sphere about the crossing maps onto itself for all of them. So, at each
joint:

1. Decide which link each cover rides on, and list what moves against it: which joints lie between the two
   covers.
2. Make their interface round about those axes: a cylinder for one axis, a plane square to it, or a sphere for
   several axes through one point.
3. Keep 3 mm to motors and 4 mm between covers, and let the sweep check prove it.

## Patterns

**Opening round the axis.** Shoulder caps, the neck and the waist column turn inside round openings in the torso
or pelvis, 3-5 mm clear. For a part that tilts as well as turns (a neck pitching at its base), compute the opening
from the part's swept shape.

**Hinge (knee).** One link carries a drum, round about the axis, between fork cheeks of the other. Each cover ends
in a concave socket 4 mm clear of the drum. Leave a narrow seam on the outside of the fold (6° in front of the knee)
and an open notch on the inside, where the calf folds up (123°). Covers that stop short of the axis instead show a
black block in a gap: "real knees with actuators won't look like this".

**Cup (elbow).** Both covers run on round the actuator, hollow round it, side walls kept.
- The child ends on a line through the axis.
- The parent closes the outside of the fold down to 6° from where full flexion turns that line.
- In front, the parent is cut to the child's front face at full flexion, 4 mm clear: the fold's notch, which a
  rigid cover cannot avoid.
- Carry the actuator's stator from the sector the child never reaches.
- Set a disc into a flat side wall, with the other cover's wall wrapping it 3 mm clear. A disc standing off a
  rounded arm reads as a disc on a post.

**Socket round a crossing (shoulder roll over pitch).** Give the inner cover a hemisphere about the point where
the axes cross, and cut the next cover as a socket 4 mm round it. Take the output out on the axis: a round hole on
the axis never sweeps. Set the range from the sweep: echo stops at 150°, where the arm's outer top started sweeping
the cap's top.

**Neck.** Put the pitch axis at the neck's base, on the yaw axis, so the neck tilts with the head. Make the neck's
base a sphere zone about the crossing, and compute the torso's opening from the swept neck. The tilt range sets the
collar gap (26° needed 9 mm).

**Three axes through one point (hip: pitch, roll, yaw).**
- Put the visible cover (the black lobe) on the link before the last joint: the hip roll link. Then only the last
  joint (yaw) moves the thigh against it, and the seam can be round about the yaw axis. With the lobe on the pitch
  link, the thigh also rolled against it and needed a 2.5 cm gap, where thin plates showed.
- Make that seam curved, not flat: the thigh's top inside a sphere centred on the yaw axis (`keep=`), the lobe's
  underside 3 mm outside the same sphere. A flat seam lined up with the other hip's lets light through in a level
  side view. A curved one has no straight path, and dropping toward the back it also matched Figure's look.
- The lobe now rolls against the pelvis. Hollow it round what turns inside it: take the cross-sections of those
  parts (the pitch actuator, its flange, the bracket), sweep them through the roll range, and cut the convex hull
  as a prism. Slice the cut along the axis, each slice only as tall as the part is there.
- Bevel the lobe's top where the outward roll tips it toward the pelvis, but only over the length of what it must
  clear. A bevel along the whole lobe left a fin.
- Cap what a clearance cut reaches. A motor sticking out of a slim cover leaves a slot that light shows through.
- Before reordering joints to make a cup possible, check where the new first actuator goes. Yaw-first hips need a
  motor above each hip; echo had no room for one.

**Pads.** Put knee pads outside both covers, domed, on mounts that turn in a small round hole on the axis. Pads
inside the leg's width needed recesses that narrowed the shank.

**Ankle (sock-shoe).** A rigid shoe cannot rise in front of the ankle: at 0.9 rad of dorsiflexion the shin sweeps
down to 8 cm there. Drop the shoe's collar line from the heel to the front of the ankle, and put the upper above it
and the black collar on the shank. Then set dorsiflexion from what the deepest pose uses (0.8 rad; the 90° squat
uses 0.77).

**Waist.** A roll/pitch waist cannot hide under a one-piece torso: 20° of tilt needs about 5 cm of hem clearance. A
yaw-only waist (Asimov 1) keeps the look, and lifting uses the hips and knees.

## Ranges

Set each limit where the sweep finds contact, then compare it with what the tasks need: walking needs about 0.2 rad
of hip extension, and a 90° squat about 2.0 rad of knee. Trade range for looks only knowingly, and write the trade
next to the limit in the spec.

## How to verify a joint

1. `python -I tools/checks.py --only sweep covers skin connect`: the joint sweeps clear with the covers across it;
   no cover face passes through a motor; nothing internal shows; the parts still form one structure.
2. Render the joint at its limits and at rest, from the user's angle, before and after.
3. Render it on a green background: green inside the outline is light through a gap.
4. `python -I tools/motion_collisions.py`: combined motions over a recorded walk.
