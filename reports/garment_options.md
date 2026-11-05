# echo: options for the fabric-looking outer cover

Research only, 2026-11-05. Tags: **[P]** published by the maker or in its patent, **[R]** reported second-hand, **[I]** inferred, **[E]** my estimate. Prices are approximate unless cited. Numbers in brackets refer to the source list. A patent shows what a company described, not what it ships.

## Answer

1. **Try first: one pull-on stretch tube per link over an open printed former.** Stock double knit, one straight sewn seam, ends tucked into printed rim grooves beside the black hubs. This combines (a) and (c), follows the panel layout in Figure's patent [2], needs only beginner sewing, and costs under $200 to try.
2. **Heat decides where a cover may touch and stay closed, but thin fabric is rarely the main resistance.** By my estimate, trapped air and foam insulate far more, so a closed hard shell may be no cooler than fabric. Measure this before committing.
3. **Seamless knit (b) comes later**, with the proven pattern as its specification. **TPU (d)** suits formers, rims and wear patches, not the visible skin. **Flocked shells (e)** are the look-only fallback; they read as suede, not knit.

## What Figure 03 has actually published

- **Announcement [1] [P]:** soft textiles rather than hard machined parts; multi-density foam at pinch points; soft goods fully washable and removable without tools; cut-resistant options. Nothing on heat, fibre, gauge, thickness, supplier or closures.
- **Cover patent applications [2] [3] [P]:** 4-way stretch knit, 0.1-10 mm, preferably about 1.75-2.25 mm thick; elongation above 10 %, preferably above 50 %, so it stays in tension; jacquard, spacer or mesh options. Mostly separate panels, not "an oversized jumpsuit". Closures listed: zipper, hook-and-loop, interlocking strips, magnets, twist locks. Foam or elastomer pads beneath; accordion structures at the lower leg; textile inserts between moving parts. Heat: ventilating weaves and a thinner cover at the neck base so torso cooling air can leave. Covers do not surround every actuator; at least 20 degrees of freedom stay uncovered.
- **Torso-cooling patent [4] [P]:** blower fans, intake through a perforated shroud under the arms, exhaust at the waist. No limb cooling in the third of the text I read.
- **Unconfirmed:** whether Figure 03 uses these embodiments. Their priority date is 2023, and [2] describes no hood, yet Figure 03 has a knit hood. Limb actuator cooling under the cloth is unpublished. Robocouture's "low-friction liner" [5] is uncited; do not treat it as fact.

## Other precedents

| robot or source | cover | status |
|---|---|---|
| 1X NEO / NEO Gamma | Knit suit and shoes over a "3D lattice polymer" soft body [6] [P]. Machine-washable, knit in-house [7] [R]. Nylon on a Shima Seiki machine [8] [9] [R]; press calling it "3D-printed" probably means whole-garment knitting [I]. | Cooling under the suit unpublished. Tendon drive and 30 kg [6] make it a weak thermal guide for echo [I] |
| THK humanoid patent | Nylon and polyurethane foam garment; a back-mounted blower pushes air between garment and shell; posture can pinch the path [10] [P]. Read in excerpts only: full text fetch failed. | Closest precedent for forced air under cloth |
| SoftBank Pepper | Garments "specially designed to be flexible and disperse heat" [11] [R]. | Low-power robot; method not given |
| Sunday Memo | Soft silicone-like exterior [12] [R], not fabric. | Not a knit precedent |
| Unitree G1 | A jacket on a -47 C walk [13] [R], so no heat evidence. Overheating claims came only from an unverified clothing seller. | Nothing official found |
| Tesla Optimus, Figure 02 | Appear hard-shelled [I], from my general knowledge; no maker statement found. | Unconfirmed |
| KnitReal "RoboSkin" | Claimed seamless humanoid knit; no website, price or tests found [14]. | Unverified |
| Practitioner essay | "Textiles trap heat" and may snag or restrict joints [15]; judgement, not measurement. | Opinion |

## Heat: the numbers that frame the choice

**Sourced.** Rated torque needs a specified aluminium heat sink [16]. RS04: 40 N m on 345 x 345 mm, 35 N m on 220 x 200 mm, at 50 rpm; operating range -20 to 50 C; insulation class B [16]. A retailer's guide says to treat a closed housing or stagnant air as a different thermal test until measured [17]. In one open-source actuator paper, about $5 fans (30 x 10 mm) cut housing-to-air resistance from 6.68 to 1.44 C/W, nearly doubling continuous torque [18]; that is a ducted plastic actuator, so expect less on echo.

**My estimates [E], resistance in m2K/W; measurement should replace these.**

| layer | R | note |
|---|---|---|
| free surface to still air | 0.10-0.17 | lower for black anodised, higher for bare aluminium |
| thin knit touching metal | 0.02-0.04 | a T-shirt lists 0.014 as a garment value [19] |
| 3 mm PA12 wall | about 0.012 | negligible |
| 10 mm enclosed air gap | 0.15-0.18 | the real insulator, under fabric or hard shell alike |
| 5 mm foam | about 0.12 | keep Figure-style pads off housings |

**What follows [I].**
- Tight thin knit might add 15-40 % to surface resistance. A closed air gap can double or triple it, whether under fabric or under a shell.
- Frame area matters more than cloth. An RS04's own surface is about 0.044 m2, against 0.238 m2 for both faces of the big plate; yet the 0.088 m2 plate costs only 12.5 % of rated torque. If copper loss dominates, sustainable torque scales about with 1/sqrt(thermal ratio): 1.25x gives 0.89, 1.5x gives 0.82, 2x gives 0.71. This overstates the loss, since the cover does not change winding-to-housing resistance.
- The exposed black discs do not help the worst motors: the knee RS04 sits in the thigh and two RS06 in the calf (`docs/DESIGN.md`), all under grey. Thigh, calf and torso are the thermal cases.
- So: bolt motors to generous frames, keep air paths along the limb, use mesh in hidden spots (behind the knee, inner thigh, underarm), and allow for small fans, as THK and Figure describe.

## Options compared

Mass assumes about 1.5 m2 covered [E]. `reports/mass.md` now carries 4.28 kg knit-over-PA12 plus 0.59 kg knit. Costs are for one robot [E]. Option (f) is my addition.

| | (a) cut-and-sew, stock fabric | (b) seamless knit service | (c) shells, fabric at joints only | (d) TPU lattice, or TPU on textile | (e) hard shell, flock or coating | (f) tube or upholstered panel on open former |
|---|---|---|---|---|---|---|
| **Precedent** | Figure's patent permits stitched panels [2] | NEO [7] [8]; patent lists knitting [2] | None exact; Figure mixes housings and inserts [3] | NEO lattice under suit [6]; TPU skin lattice research [20] | Hard shells only [I]; no flocked humanoid found | Figure's panels over housings [2]; car A-pillar wrap when bonded [21] |
| **Joint stretch** | Good if tensioned; needs sliding length | Best: rib and stretch can be zoned | Good: shells clear, gaiter takes the fold | Per-joint design; solid TPU wrinkles [20] | None; relies on seams and hubs | Good: stops at hubs, gaiter where needed |
| **Heat** | Thin; mesh vents possible; loose fit traps air | Vents can be knit in [2] | Closed shell traps air; vents show | Open lattice should pass air [I]; solid skin poor | As closed shell | Best prospect [I]: fabric spans an open former |
| **Snagging** | Loose fabric can reach mechanisms; use rims or liner | Low if well fitted | Only gaiter exposed | Lattice can catch at folds | None | Low: former holds it off |
| **Removal, washing** | Hidden zips or hook strips; machine wash | As (a); one piece replaced whole | Gaiters wash, shells wipe | TPU wash life needs testing [22] | Wipe only | Tubes pull off and wash; bonded panels wipe only |
| **Fasteners** | Excellent once off | Depends on openings | Shells must hide screws | As shell | Show, or plug and touch up | Excellent: under each tube |
| **Abrasion** | Fair; add patches | Yarn decides | Shell strong; gaiter sacrificial | TPU can improve wear [22] [23] | Likely worst; bond untested | Fair; tube replaceable |
| **Fire near battery** | Nylon, polyester LOI about 20-21 and melt [24]; wool blend or FR on torso | Same; FR yarn can be specified | Shell plastic governs | TPU burns [I]; no data found | Thin; adhesive unknown | As (a), plus former plastic |
| **Added mass** | 0.3-0.8 kg, plus former | 0.3-0.8 kg | About 4.5 kg if solid 3 mm PA12 | Lattice 0.5-1.5 kg; solid 1 mm 1.8 kg | 0.1-0.3 kg over shells | 0.3-0.8 kg plus 1.5-3 kg ribbed former |
| **Cost** | $150-500 DIY; $500-1500 sewn by a maker | Unknown; maybe $2k-20k with iterations | $150-500 own printer; PA12 bureau much more | $100-300 | $100-400 | $200-600 own printer |
| **Lead time** | Days to 2 weeks | 2-4 months one-off; a vendor claims 1-3 weeks per sample [25] | 1-3 weeks printing | Weeks printing | Days after shells | 1-3 weeks |
| **Skills** | Zigzag or overlock | Mill's programmer; you write the spec | Printing, simple sewing | Print tuning; taut mounting on bed | Spray, masking, mess | Printing, one straight seam |

**Design tools**
- **Free:** Seams to Sewing Pattern for Blender, SVG in mm with a stretch overlay, is the first choice [26]. PatternLab (alpha) flattens STL regions with a strain map [27]. garment-flattening handles fabric that stretches unequally, as research code [28]. **Paid:** Marvelous Designer about $39/month [29]; CLO about $50/month [30]. Draping on an imported robot mesh is from memory; check it in a trial.
- **Unconfirmed in this search:** Rhino Squish and Boundary First Flattening. Cling-film-and-tape draping is a common cosplay method, also from memory.
- **Stretch fabric needs negative ease:** measure each fabric, start about 10-20 % undersize [E], per direction. Flattening does not do this for you.

## Designing around the fold

Fabric over a fold lengthens by about r x angle [E]. At 40 % working strain, free length is r x angle / 0.4: elbow (55 mm, 1.9 rad) needs about 105 mm extra over 260 mm; knee (70 mm, 2.15 rad) about 150 mm over 375 mm. So fabric cannot be glued across a fold, and continuous cloth over the knee is demanding. echo's rigid covers already pass the sweep and stop at hubs, which removes most of this: end each tube at the hub with a clamped ring or rim groove, and use gaiters only at the elbow's fold notch and behind the knee.

General cautions [I]: thin spandex shows print lines, so test double knit (ponte) too. Hook-and-loop snags knit. Retailers list magnetic encoders on RobStride units, so keep closure magnets away. Fabric is not a battery barrier: the pack needs its own enclosure and an uncovered vent path. Search excerpts give wool LOI about 25 without melting and cotton about 18, though I could not open that table; aramid is about 28-32 [24].

## Bench test (about $100-200, plus an actuator if you have one)

**Build.** A 1:1 thigh or calf section with a real aluminium bracket, and an elbow hinge reaching -1.9 rad. Fabrics: heather grey nylon/spandex, about $8-15/yd [31]; a wool-nylon-elastane blend, price unconfirmed; mesh or 3 mm spacer for vents, about $20-35/yd [32]. Day-zero shortcut: a shop-bought grey compression sleeve.

**Heat test.**
1. Source: an RS06, or a same-size aluminium dummy with a 25-40 W resistor inside (known power, risks nothing). Thermocouples on the housing, in the air gap, on the outer fabric, and in room air.
2. Hang in still air until change is under 1 K per 10 minutes, about 1-2 hours, at the same power each run. Configurations: bare; closed 3 mm shell; tight tube on open former; loose tube; tube with 5 mm foam; mesh window; tube with one small blower. Result: (T housing - T room) / W, as a ratio to bare.
3. Repeat the best two on the real actuator, at its stall rating in `spec/actuators.json` and at the sim's RMS torque, logging its temperature.
4. Soak each fabric at 80 C for 24 hours, then recheck stretch and recovery.

**What rules an option out** (thresholds are my judgement)

| result | consequence |
|---|---|
| Ratio above 1.5, and vents plus a fan cannot bring it under 1.25 | No cover over that motor; change the frame or expose a hub |
| Air under cover above 50 C scaled to a 30 C room (stated operating limit [16]); or any thermal fault at target duty | That configuration is out |
| Foam on the housing is much worse than an air path | Foam only at pinch zones, never on housings |
| Closed shell no better than open-former tube | (c) and (e) are not the thermally safe choice; they also need vents |
| Lost recovery or discolouring in the soak | Fabric out, or keep it off hot surfaces |
| Over 50 % strain at range end, over 5 % growth after 1000 cycles, or drag above about 1 % of rated torque (compare current) | Fabric or pattern out; lengthen free zone or stop at hub |
| Fabric pulled into the mechanism in 1000 cycles | Add rim or liner, else layout out |
| Panel takes over 5 minutes or tools to remove; over 5 % shrink or heavy pilling after 5 washes | Redesign closure; fabric out |
| Horizontal burn over 102 mm/min (FMVSS 302 limit [33]), or burning drips | Not on torso panels. Outdoors, small flame, extinguisher ready; an indicator, not certification |
| At 2 m beside Figure frames, reads as sportswear or shows print lines | Change fabric first, then consider a knit service |
| Flock rubs off in 1000 strokes, cracks, or does not read as knit at 1 m | (e) stays a static display finish |
| Knit quotes above about $5k or 12 weeks, or no mill takes a one-off | (b) deferred |

**If the tube passes:** make one arm, then a leg, re-run heat on the thigh, then torso panels with a hidden coil zip. Approach knit mills only after pattern and vents settle: 22 Factor [34] or other Shima Seiki contract knitters. A desktop Kniterate (expected $7,499 in 2017 [35]) is a costly detour.

## What I could not confirm

- Figure 03's fibre, thickness, supplier, closures, liner and limb cooling; any measured actuator temperature under a robot garment, from any company.
- RobStride's over-temperature threshold. Plate sizes for RS02, RS03 and RS06 come only from `spec/actuators.json`; I did not read the manuals.
- One-off knit prices, minimums and lead times (only one vendor's marketing [25]); Tesla and Unitree positions on fabric; flock adhesion on PA12; TPU on spandex for hobby filament; FR stretch knit prices.
- All resistance values, masses and most costs above are my estimates.

## Sources

1. https://www.figure.ai/news/introducing-figure-03
2. Figure AI, US20250242500A1: https://patents.google.com/patent/US20250242500A1/en
3. and 4. Figure AI: https://patents.google.com/patent/US20260061625A1/en and https://patents.google.com/patent/US12707556
5. https://robocouture.org/robots/figure-03
6. https://www.1x.tech/discover/neo-home-robot
7. https://blog.robozaps.com/b/1x-neo-review
8. https://www.designboom.com/technology/humanoid-robot-neo-gamma-house-chores-home-owners-autonomous-assistant-02-26-2025/
9. https://aiwiki.ai/wiki/1x_technologies_neo_gamma
10. THK, US12121082: https://patents.google.com/patent/US12121082
11. https://www.nippon.com/en/views/b00911/
12. https://www.eweek.com/news/sunday-memo-home-robot/
13. https://www.eweek.com/news/unitree-humanoid-robot-extreme-cold-walking/
14. https://nexuswild.com/news/knitreal-roboskin-humanoid-robot-fabric-covering/
15. https://mehrdadfarimani.substack.com/p/should-a-humanoid-wear-clothes
16. https://github.com/RobStride/Product_Information
17. https://openelab.io/blogs/learn/robstride-joint-torque-thermal-sizing-can-setup
18. Urs et al.: https://arxiv.org/html/2202.12395v1
19. https://www.engineeringtoolbox.com/clo-clothing-thermal-insulation-d_732.html
20. https://arxiv.org/pdf/2605.28352.pdf
21. https://www.crossfireforum.org/forum/interior/60835-cf-fabric-pillars-soon-headliner.html
22. and 23. https://polen.itu.edu.tr/entities/publication/0c972971-1ecc-44df-89fd-dafb2c68758a/full and Ehrmann and Steinmetz: https://cdatp.publia.org/cdatp/article/view/45
24. https://www.havep.com/en/node/2039
25. Vendor: https://ifactoryapp.com/industries/textile-manufacturing/3d-knitting-seamless-garment-manufacturing
26. to 28. https://thomaskole.nl/s2s/ then https://patternlab3d.com/ then https://github.com/CorentinDumery/garment-flattening
29. https://digitalproduction.com/2026/01/12/marvelous-designer-adds-indie-pricing-quietly/
30. https://www.xp-pen.com/blog/clothing-design-software.html
31. and 32. https://califabrics.com/spandex-solids and https://www.moodfabrics.com/taupe-spacer-mesh-110752
33. https://www.filkfreiberg.de/en/testing-certification/accredited-test-laboratory/standards-specifications/details/fmvss-302
34. https://wwd.com/sourcing-journal/industry-news/22-factor-shima-seiki-on-demand-3d-knitting-upcycled-yarn-1238741580/
35. https://www.core77.com/posts/64979/Affordable-Desktop-CNC-Machines-Expand-Into-Textiles
