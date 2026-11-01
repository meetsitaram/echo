"""MuJoCo test: can the replica hold itself up with the actuators we picked?

  python -I tools/sim_stand.py            stand 4 s in a slight crouch, then squat to 90 deg knees and back up

The model (urdf/echo.xml) limits every joint to its actuator's peak torque. Position servos hold the
targets; nothing balances actively, so the poses keep the centre of mass over the feet (ankles lean as in the
static check). PASS when the pelvis stays within 3 cm of its target height path, tilts less than 5 degrees, and
no joint sits at its torque limit for more than 0.2 s.
"""
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom as G  # noqa: E402

XML = os.path.join(G.ROOT, 'urdf', 'echo.xml')


def pose_vec(m, q):
    out = np.zeros(m.nu)
    for i in range(m.nu):
        out[i] = q.get(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i), 0.0)
    return out


def set_pose(m, d, q):
    for k, v in q.items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, k)
        d.qpos[m.jnt_qposadr[jid]] = v


def ground(m, d):
    """Drop the free base so the lowest sole corner touches the floor."""
    mujoco.mj_forward(m, d)
    low = min(d.geom_xpos[g][2] - m.geom_size[g][2] for g in range(m.ngeom) if m.geom_type[g] == 6)
    d.qpos[2] -= low - 0.0005
    mujoco.mj_forward(m, d)


def crouch(k):
    """k = 0 standing (slight crouch) .. 1 knees at 90 deg; ankles lean so the CoM stays over the feet."""
    knee = 0.3 + k * (1.57 - 0.3)
    hip = -0.15 - k * (0.80 - 0.15)
    ankle = -(knee + hip) + 0.0 * k
    q = {}
    for s in 'LR':
        q[f'{s}_hip_pitch'], q[f'{s}_knee'], q[f'{s}_ankle_pitch'] = hip, knee, ankle
    q['L_shoulder_pitch'] = q['R_shoulder_pitch'] = -0.4 * k            # arms forward as a counterweight
    return q


def main():
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    q0 = crouch(0)
    set_pose(m, d, q0)
    ground(m, d)
    limits = np.abs(m.actuator_forcerange[:, 1])
    T, dt = 8.0, m.opt.timestep
    worst_tilt, worst_dz, sat_time = 0.0, 0.0, np.zeros(m.nu)
    z_ref = None
    log = []
    for step in range(int(T / dt)):
        t = step * dt
        k = 0.0 if t < 3 else (min((t - 3) / 2, 1.0) if t < 5.5 else max(1 - (t - 5.5) / 2, 0.0))
        d.ctrl[:] = pose_vec(m, crouch(k))
        mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)):
            print('FAIL: simulation diverged at %.2f s' % t)
            return 1
        z = d.qpos[2]
        if t > 1.0 and k == 0 and z_ref is None:
            z_ref = z
        qw, qx, qy, qz = d.qpos[3:7]
        up = np.array([2 * (qx * qz + qw * qy), 2 * (qy * qz - qw * qx), 1 - 2 * (qx * qx + qy * qy)])
        tilt = np.degrees(np.arccos(np.clip(up[2], -1, 1)))
        if t > 0.5:
            worst_tilt = max(worst_tilt, tilt)
            sat = np.abs(d.actuator_force) >= 0.98 * limits
            sat_time = np.where(sat, sat_time + dt, 0.0)
        if step % int(0.5 / dt) == 0:
            log.append((t, z, tilt, k))
    for t, z, tilt, k in log:
        print(f'  t {t:4.1f} s  squat {k:4.2f}  pelvis z {z:.3f} m  tilt {tilt:4.1f} deg')
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    worst_sat = max(zip(sat_time, names))
    end_z = d.qpos[2]
    ok_tilt = worst_tilt < 5
    ok_height = z_ref is not None and abs(end_z - z_ref) < 0.03
    ok_sat = worst_sat[0] < 0.2
    print(f"tilt max {worst_tilt:.1f} deg ({'PASS' if ok_tilt else 'FAIL'}), back to standing height within "
          f"{abs(end_z - z_ref) * 100:.1f} cm ({'PASS' if ok_height else 'FAIL'}), longest torque saturation "
          f"{worst_sat[0]:.2f} s on {worst_sat[1]} ({'PASS' if ok_sat else 'FAIL'})")
    return 0 if ok_tilt and ok_height and ok_sat else 1


if __name__ == '__main__':
    sys.exit(main())
