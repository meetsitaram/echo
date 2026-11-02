"""NLF's per-frame SMPL body (tools/video_nlf.py, camera frame) -> a PICO-format tape for tools/retarget_pico.py.

1. Room frame from the fixed camera: the horizon row and the camera's height from the robot's own feet and head in
   its upright frames (horizon_fit), level (the back wall's base is level in the picture); the room's x axis toward
   the vanishing point of its depth lines (--vp-u on the horizon). (A floor fitted to the feet, 2026-11-01, came out
   tilted a degree or two: seen from 1.3 m up, almost level, that put the back wall 10.6 m away on the right and
   14 m on the left.)
2. Global joint rotations from SMPL's local axis-angles down the kinematic tree, into the room frame, then into the
   PICO convention: PICO's body joints differ from SMPL's by one fixed rotation (GMR's smplx_to_g1 and xrobot_to_g1
   tables give the same robot frames with offsets [0.5, -0.5, -0.5, -0.5] and [-0.5, 0.5, -0.5, -0.5]). The arm and
   head offsets are recalibrated per take by the retargeter, so only the body's convention matters.
3. Depth from the floor: NLF's depth jitters frame to frame (the whole body a little nearer or farther along the
   camera's rays), which played as the feet sliding (2026-11-01). Scaling a frame about the camera leaves its picture
   unchanged, so each frame is scaled until its lowest foot joint is at its flat height above the floor.
4. Frames NLF missed are filled (positions linearly, rotations by slerp; at the ends, held), then everything is
   smoothed (Gaussian, sigma 60 ms) and resampled to 50 Hz.

  python tools/nlf_to_tape.py reports/video/<clip>_nlf.npz [--out reports/video/<clip>_tape.npz]
  python tools/retarget_pico.py reports/video/<clip>_tape.npz --out reports/motion/<clip>.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as sRot, Slerp

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
FEET = (7, 8, 10, 11)                      # ankles and the balls of the feet
M_UNITY = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)     # Unity -> Z-up (retarget_pico._M_UNITY)
SMPL_TO_PICO = sRot.from_quat([0.5, -0.5, -0.5, -0.5], scalar_first=True) * \
    sRot.from_quat([-0.5, 0.5, -0.5, -0.5], scalar_first=True).inv()
FPS = 50
SIGMA_S = 0.06
FLAT_H = np.array([0.08, 0.08, 0.03, 0.03])   # ankle and ball-of-foot joints above the floor, foot flat (SMPL)
HEAD_JOINT_H = 0.925 * 1.727                  # SMPL's head joint on a body echo's height (standing)


def intrinsics(size, fov_deg):
    w, h = size
    f = 0.5 * w / np.tan(np.radians(fov_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])


def camera_frame(K, vp, height):
    """Room axes (rows: x, y, z in camera coordinates) and the floor point under the camera, from the depth lines'
    vanishing point (u, v) and a level horizon through it."""
    f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    up = np.array([0.0, -f, vp[1] - cy])              # normal to every ray on the horizon row (no roll)
    up /= np.linalg.norm(up)
    x = np.array([(vp[0] - cx) / f, (vp[1] - cy) / f, 1.0])
    x -= up * (x @ up)
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(up, x), up])
    return R, -height * up


def horizon_fit(J, K, ok):
    """Horizon row and camera height from the robot itself: in every upright frame, its lowest foot and its head
    joint are at known heights (FLAT_H, HEAD_JOINT_H), and (v_foot - v_hz) / (v_head - v_hz) = (h - z_foot) /
    (h - z_head) whatever the depth or focal length. (The rug's vanishing point put the horizon 9 px high: the robot
    came out 2.1 m tall 11 m away and 1.6 m at 4 m, 2026-11-01.)"""
    from scipy.optimize import least_squares
    with np.errstate(divide='ignore', invalid='ignore'):
        u = K[0, 0] * J[..., 0] / J[..., 2] + K[0, 2]
        v = K[1, 1] * J[..., 1] / J[..., 2] + K[1, 2]
    up = ok & (np.abs(u[:, 15] - u[:, 0]) < 0.12 * (v[:, 0] - v[:, 15]))       # head over pelvis, in the picture
    vf, vh = np.max(v[:, [10, 11]], axis=1)[up], v[up, 15]
    zf, zh = FLAT_H[2], HEAD_JOINT_H
    fit = least_squares(lambda p: (vf - p[0]) * (p[1] - zh) - (vh - p[0]) * (p[1] - zf), [K[1, 2], 1.2])
    return float(fit.x[0]), float(fit.x[1]), int(up.sum())


def floor_depth(J, R, origin, sigma_frames=1.5):
    """Per-frame scale about the camera that sets the lowest foot joint at its flat height: (n,)."""
    n, d0 = R[2], R[2] @ origin                       # floor: n . x = d0 (camera frame), heights n . x - d0
    a = J[:, FEET] @ n                                # (frames, 4), each < d0 (below the camera)
    s = np.min((FLAT_H + d0) / a, axis=1)             # the scale putting each joint at its height; the smallest wins
    return gauss(s[:, None], sigma_frames)[:, 0]


def global_rots(pose):
    """(n, 72) SMPL axis-angles -> (n, 24) global rotations (camera frame)."""
    loc = [sRot.from_rotvec(pose[:, 3 * j:3 * j + 3]) for j in range(24)]
    glob = [None] * 24
    for j, p in enumerate(PARENTS):
        glob[j] = loc[j] if p < 0 else glob[p] * loc[j]
    return glob


def fill_and_resample(t_src, ok, pos, rots, t_out, sigma_s=SIGMA_S):
    """Positions (n, 24, 3) and rotations (24 x Rotation(n)) at the good frames -> smoothed, at t_out."""
    good = np.flatnonzero(ok)
    tg = t_src[good]
    fps_src = 1.0 / np.median(np.diff(t_src))
    # positions: fill by linear interpolation on the source grid, smooth, then interpolate to t_out
    P = np.stack([np.stack([np.interp(t_src, tg, pos[good, j, k]) for k in range(3)], -1) for j in range(24)], 1)
    P = gauss(P.reshape(len(t_src), -1), sigma_s * fps_src).reshape(P.shape)
    P = np.stack([np.stack([np.interp(t_out, t_src, P[:, j, k]) for k in range(3)], -1) for j in range(24)], 1)
    Q = np.zeros((len(t_out), 24, 4))
    tc = np.clip(t_src, tg[0], tg[-1])
    for j in range(24):
        r = Slerp(tg, rots[j][good])(tc)                                       # fill, on the source grid
        q = r.as_quat(scalar_first=True)
        for i in range(1, len(q)):                                             # one hemisphere, frame to frame
            if q[i] @ q[i - 1] < 0:
                q[i] = -q[i]
        q = gauss(q, sigma_s * fps_src)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        Q[:, j] = Slerp(t_src, sRot.from_quat(q, scalar_first=True))(np.clip(t_out, t_src[0], t_src[-1])) \
            .as_quat(scalar_first=True)
    return P, Q


def gauss(x, sig):
    k = np.arange(-int(3 * sig), int(3 * sig) + 1)
    w = np.exp(-0.5 * (k / sig) ** 2)
    w /= w.sum()
    pad = np.pad(x, ((len(k) // 2, len(k) // 2), (0, 0)), mode='edge')
    return np.stack([np.convolve(pad[:, c], w, mode='valid') for c in range(x.shape[1])], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('nlf', type=Path)
    ap.add_argument('--vp-u', type=float, default=650.0,
                    help="column of the room's depth lines' vanishing point, pixels (the rug's edges and the ridge "
                         "beam in the living-room clip); the horizon row comes from the robot")
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    out = a.out or a.nlf.with_name(a.nlf.name.replace('_nlf.npz', '_tape.npz'))
    d = np.load(a.nlf)
    fps, ok = float(d['fps']), d['ok'].astype(bool)
    J, pose = d['joints'].astype(float), d['pose'].astype(float)
    t_src = np.arange(len(ok)) / fps

    K = intrinsics(d['size'], float(d['fov']))
    v_hz, height, n_up = horizon_fit(J, K, ok)
    R, origin = camera_frame(K, (a.vp_u, v_hz), height)
    s = np.ones(len(J))
    s[ok] = floor_depth(J[ok], R, origin)
    print(f'camera: {height:.2f} m up, horizon at row {v_hz:.0f} ({n_up} upright frames), '
          f'{-np.degrees(np.arcsin(R[2, 2])):.1f} deg down; depth scale {np.percentile(s[ok], 5):.2f}..'
          f'{np.percentile(s[ok], 95):.2f} (5th..95th percentile), frame to frame '
          f'{np.median(np.abs(np.diff(s[ok]))) * 100:.2f} % median')
    J = J * s[:, None, None]
    Jw = (J - origin) @ R.T
    rot_cw = sRot.from_matrix(R)
    glob = [rot_cw * g * SMPL_TO_PICO for g in global_rots(pose)]

    t_out = np.arange(0, t_src[-1], 1.0 / FPS)
    P, Q = fill_and_resample(t_src, ok, Jw, glob, t_out)
    body = np.zeros((len(t_out), 24, 7), np.float32)
    body[:, :, :3] = P @ M_UNITY                                              # Z-up -> Unity (M is a rotation)
    q_unity = sRot.from_matrix(M_UNITY).inv() * sRot.from_quat(Q.reshape(-1, 4), scalar_first=True)
    body[:, :, 3:] = q_unity.as_quat().reshape(len(t_out), 24, 4)             # x y z w, as PICO's
    filled = np.interp(t_out, t_src, (~ok).astype(float)) > 0.5
    np.savez_compressed(out, stamp_ns=(t_out * 1e9).astype(np.int64), body=body, body_ok=np.ones(len(t_out), bool),
                        filled=filled, source=str(a.nlf), cam_R=R, cam_origin=origin, K=K, size=d['size'])
    print(f'{out}: {len(t_out)} samples at {FPS} Hz ({filled.mean() * 100:.1f} % filled)')


if __name__ == '__main__':
    main()
