"""What the two ankle motors would really carry. Training limits ankle pitch and roll separately, but one pair of motors
drives both: motor = pitch / 2 +- roll / (2 x 0.6). Estimated from takes at 50 Hz, torque = kp (target - q) - kd qd.

  python train/ankle_motors.py <dir of takes> [...] [--kp-p 72 --kd-p 4.32 --kp-r 43.2 --kd-r 2.6]
"""
import argparse
import glob
import os

import numpy as np

PEAK, RATED, RATIO = 36.0, 11.0, 0.6
EFF_P, EFF_R = 72.0, 43.2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--kp-p', type=float, default=72.0)
    ap.add_argument('--kd-p', type=float, default=4.32)
    ap.add_argument('--kp-r', type=float, default=43.2)
    ap.add_argument('--kd-r', type=float, default=2.6)
    a = ap.parse_args()
    for d in a.dirs:
        over, rms, rms_p, n, clips_over_rated, clips_over_peak = 0, [], [], 0, 0, 0
        files = glob.glob(os.path.join(d, 'motion', '*.npz'))
        for f in files:
            t = np.load(f, allow_pickle=True)
            names = [str(x) for x in t['joint_names']]
            q, c = t['q'], t['ctrl']
            qd = np.gradient(q, 0.02, axis=0)
            worst_rms, worst_over = 0.0, 0.0
            for s in 'LR':
                ip, ir = names.index(f'{s}_ankle_pitch'), names.index(f'{s}_ankle_roll')
                tp = np.clip(a.kp_p * (c[:, ip] - q[:, ip]) - a.kd_p * qd[:, ip], -EFF_P, EFF_P)
                tr = np.clip(a.kp_r * (c[:, ir] - q[:, ir]) - a.kd_r * qd[:, ir], -EFF_R, EFF_R)
                rms_p.append(np.sqrt((tp ** 2).mean()))
                for m in (tp / 2 + tr / (2 * RATIO), tp / 2 - tr / (2 * RATIO)):
                    r = np.sqrt((m ** 2).mean())
                    rms.append(r)
                    worst_rms = max(worst_rms, r)
                    o = (np.abs(m) > PEAK).sum()
                    over += o
                    n += len(m)
                    worst_over = max(worst_over, o)
            clips_over_rated += worst_rms > RATED
            clips_over_peak += worst_over > 0
        print(f'{d[-30:]}: {len(files)} clips; pitch torque RMS median {np.median(rms_p):.1f} N m (check against eval); per motor RMS median {np.median(rms):.1f}, max {np.max(rms):.1f} N m (rated {RATED}); clips with a motor over rated {clips_over_rated / len(files) * 100:.0f}%; motor demand past {PEAK} N m peak on {over / n * 100:.1f}% of motor-frames, in {clips_over_peak / len(files) * 100:.0f}% of clips')


if __name__ == '__main__':
    main()
