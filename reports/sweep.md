# Physical checks

spec: `spec/robot.json`, 2026-10-31 12:04, 19 s

18 checks: 7 pass, 0 warn, **11 fail**

## sweep

| status | item | detail |
|---|---|---|
| FAIL | waist_yaw | lower -2.00 blocked at -0.42 rad by R_wrist_yaw_link / pelvis (2 cm3); upper +2.00 blocked at +0.42 rad by L_wrist_yaw_link / pelvis (2 cm3) |
| FAIL | waist_roll | lower -0.35 blocked at -0.07 rad by torso / pelvis (0 cm3); upper +0.35 blocked at +0.07 rad by torso / pelvis (0 cm3) |
| FAIL | waist_pitch | lower -0.35 blocked at -0.09 rad by torso / pelvis (1 cm3); upper +0.90 blocked at +0.15 rad by torso / waist_yaw_link (1 cm3) |
| PASS | neck_yaw | [-1.40, +1.40] rad clear |
| PASS | neck_pitch | [-0.45, +0.60] rad clear |
| PASS | L_shoulder_pitch | [-3.00, +1.00] rad clear |
| FAIL | L_shoulder_roll | lower -0.12 blocked at -0.04 rad by L_wrist_yaw_link / pelvis (2 cm3) |
| PASS | L_shoulder_yaw | [-2.60, +2.60] rad clear |
| FAIL | L_elbow | lower -2.50 blocked at -2.40 rad by L_elbow_link / L_shoulder_roll_link (1 cm3) |
| PASS | L_wrist_yaw | [-2.60, +2.60] rad clear |
| FAIL | L_wrist_pitch | lower -1.00 blocked at -0.46 rad by L_hand / pelvis (1 cm3) |
| PASS | L_wrist_roll | [-0.60, +0.60] rad clear |
| FAIL | L_hip_pitch | collides at 0: L_hip_roll_link / pelvis; lower -2.40 blocked at -0.40 rad by L_hip_roll_link / pelvis (1 cm3); upper +0.90 blocked at +0.04 rad by L_hip_roll_link / pelvis (1 cm3) |
| FAIL | L_hip_roll | collides at 0: L_hip_roll_link / pelvis; lower -0.35 blocked at -0.01 rad by L_hip_roll_link / pelvis (1 cm3); upper +1.40 blocked at +0.06 rad by L_hip_roll_link / pelvis (1 cm3) |
| FAIL | L_hip_yaw | lower -1.20 blocked at -0.55 rad by L_foot / R_foot (1 cm3) |
| PASS | L_knee | [+0.00, +2.40] rad clear |
| FAIL | L_ankle_pitch | lower -0.90 blocked at -0.67 rad by L_foot / L_knee_link (0 cm3); upper +0.55 blocked at +0.53 rad by L_foot / L_knee_link (0 cm3) |
| FAIL | L_ankle_roll | lower -0.30 blocked at -0.21 rad by L_foot / L_knee_link (0 cm3); upper +0.30 blocked at +0.21 rad by L_foot / L_knee_link (0 cm3) |
