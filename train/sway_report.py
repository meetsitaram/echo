"""Is the pelvis's side-to-side motion in the source, or did retargeting add it? Numbers per motion, source BVH first.

  python train/sway_report.py <clip.bvh> <soma-retargeter dir> <take.csv> [<take.csv> ...]

Lateral: the pelvis's offset from its own smoothed path, across the direction of travel. Roll: its up axis leaning across that
direction. Gait lives near 1 and 2 Hz, so power above HF_HZ is what looks choppy: reported as a share and as RMS rate.
Run with the soma-retargeter environment (it reads the BVH with that repo's loader).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

HF_HZ = 5.0
FPS = 120.0
SKIP_S = 1.0     # the retargeter's lead-in and the clip's start


def bvh_pelvis(path, repo):
    """(positions (T, 3), rotations) of the BVH Hips in robot convention: Z up, pelvis frame X forward, Z up."""
    sys.path.insert(0, str(repo))
    import warp as wp
    import soma_retargeter.assets.bvh as bvh
    from soma_retargeter.utils.space_conversion_utils import SpaceConverter, get_facing_direction_type_from_str
    skel, anim = bvh.load_bvh(str(path))
    tx = np.array(SpaceConverter(get_facing_direction_type_from_str('Mujoco')).transform(wp.transform_identity()), dtype=float)
    names = list(skel.joint_names)
    hips, root = names.index('Hips'), names.index('Root')
    P, Q = [], []
    W = R.from_quat(tx[3:7])
    offset = R.from_quat([0.5, 0.5, 0.5, 0.5])      # configs: Hips joint frame -> pelvis frame
    for f in range(anim.num_frames):
        loc = np.array(anim.get_local_transforms(f), dtype=float)
        Rr = W * R.from_quat(loc[root, 3:7])
        pr = tx[:3] + W.apply(loc[root, :3])
        P.append(pr + Rr.apply(loc[hips, :3]))
        Q.append((Rr * R.from_quat(loc[hips, 3:7]) * offset).as_quat())
    fps = float(getattr(anim, 'sample_rate', FPS))
    return np.array(P), R.from_quat(np.array(Q)), fps


def csv_pelvis(path):
    d = np.loadtxt(path, delimiter=',', skiprows=1)
    return d[:, 1:4] / 100.0, R.from_euler('xyz', d[:, 4:7], degrees=True), FPS


def hf(x, fps):
    """(share of power above HF_HZ, RMS rate of that part) of a signal, its mean removed."""
    X = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(len(x), 1.0 / fps)
    p = np.abs(X) ** 2
    high = np.fft.irfft(np.where(f > HF_HZ, X, 0), len(x))
    return float(p[f > HF_HZ].sum() / max(p.sum(), 1e-12)), float(np.sqrt((np.gradient(high, 1.0 / fps) ** 2).mean()))


def metrics(p, rot, fps):
    n = int(SKIP_S * fps)
    p, rot = p[n:-n], rot[n:len(rot) - n]
    sm = gaussian_filter1d(p[:, :2], 0.5 * fps, axis=0, mode='nearest')
    v = np.gradient(sm, axis=0)
    fwd = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    left = np.column_stack((-fwd[:, 1], fwd[:, 0]))
    lat = ((p[:, :2] - sm) * left).sum(1)
    up = rot.apply([0.0, 0.0, 1.0])
    roll = np.degrees(np.arcsin(np.clip((up[:, :2] * left).sum(1), -1, 1)))
    pitch = np.degrees(np.arcsin(np.clip((up[:, :2] * fwd).sum(1), -1, 1)))
    lat_hf, lat_rate = hf(lat, fps)
    roll_hf, roll_rate = hf(roll, fps)
    return dict(fps=fps, lat_mm_rms=1e3 * lat.std(), lat_vel_mps_rms=float(np.gradient(lat, 1 / fps).std()), lat_hf_pct=100 * lat_hf, lat_hf_mm_s=1e3 * lat_rate,
                roll_deg_rms=roll.std(), roll_deg_peak=float(np.abs(roll - roll.mean()).max()), roll_rate_dps_rms=float(np.gradient(roll, 1 / fps).std()),
                roll_hf_pct=100 * roll_hf, roll_hf_dps=roll_rate, pitch_deg_rms=pitch.std(), z_mm_rms=1e3 * p[:, 2].std())


def main():
    bvh_path, repo, *csvs = sys.argv[1:]
    rows = [('source BVH', metrics(*bvh_pelvis(Path(bvh_path), Path(repo))))]
    rows += [(Path(c).stem, metrics(*csv_pelvis(c))) for c in csvs]
    keys = list(rows[0][1])
    print('motion'.ljust(30) + ''.join(k[:15].rjust(16) for k in keys))
    for name, m in rows:
        print(name[:29].ljust(30) + ''.join(f'{m[k]:16.2f}' for k in keys))


if __name__ == '__main__':
    main()
