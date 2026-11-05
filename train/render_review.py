"""A retargeted take for review: the X2 source on the left, echo on the right, same clock, same camera angle, each framed
to its own height. Echo in its own colours (tools/render_motion.py's view model); kinematic playback, nothing simulated.

  python train/render_review.py train/review/<clip>.npz <source.pkl> [--view side|34|front] [--out x.mp4]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import joblib
import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'train'))
import render_motion as RM  # noqa: E402
from retarget_x2 import X2_XML  # noqa: E402

W, H, FPS = 800, 720, 25


def camera(view, track_xy, height):
    az, el, dist = RM.VIEWS[view]
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = az, el, dist * height / 1.68
    cam.lookat[:] = [track_xy[0], track_xy[1], height * 0.5]
    return cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('take', type=Path)
    ap.add_argument('source', type=Path)
    ap.add_argument('--view', default='side', choices=list(RM.VIEWS))
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    d = np.load(a.take)
    src = joblib.load(a.source)[str(d['clip'])]
    sp, sq, sdof = (np.asarray(src[k], dtype=float) for k in ('root_trans_offset', 'root_rot', 'dof'))

    me = RM.build_view_model()
    de = mujoco.MjData(me)
    adr = [me.jnt_qposadr[me.joint(str(n)).id] for n in d['joint_names']]
    mx = mujoco.MjModel.from_xml_path(str(X2_XML))
    dx = mujoco.MjData(mx)
    # the view model is a studio with its own sky and floor; the X2's file has neither, so give it the same off-screen size
    me.vis.global_.offwidth, me.vis.global_.offheight = max(W, me.vis.global_.offwidth), max(H, me.vis.global_.offheight)
    mx.vis.global_.offwidth, mx.vis.global_.offheight = max(W, mx.vis.global_.offwidth), max(H, mx.vis.global_.offheight)
    re, rx = mujoco.Renderer(me, H, W), mujoco.Renderer(mx, H, W)
    te = RM.smooth_track(d['root_pos'][:, :2], 30)
    tx = RM.smooth_track(sp[:, :2], 30)
    out = a.out or a.take.with_name(f'{a.take.stem}_{a.view}.mp4')
    writer = imageio.get_writer(out, fps=FPS, codec='libx264', quality=7, pixelformat='yuv420p', macro_block_size=8)
    for i in range(0, len(d['t']), 2):
        de.qpos[:3], de.qpos[3:7] = d['root_pos'][i], d['root_quat_wxyz'][i]
        de.qpos[adr] = d['q'][i]
        mujoco.mj_kinematics(me, de)
        mujoco.mj_camlight(me, de)
        re.update_scene(de, camera(a.view, te[i], 1.68))
        dx.qpos[:3], dx.qpos[3:7], dx.qpos[7:] = sp[i], [sq[i, 3], *sq[i, :3]], sdof[i]
        mujoco.mj_kinematics(mx, dx)
        mujoco.mj_camlight(mx, dx)
        rx.update_scene(dx, camera(a.view, tx[i], 1.3))
        left, right = Image.fromarray(rx.render()), Image.fromarray(re.render())
        RM.caption(left, (12, 10), f'X2 source  {d["clip"]}')
        RM.caption(right, (12, 10), 'echo, retargeted (kinematic playback, no physics)')
        RM.caption(right, (12, H - 24), f't {d["t"][i]:5.1f} s   1x   {a.view} view')
        writer.append_data(np.hstack((np.asarray(left), np.asarray(right))))
    writer.close()
    print(out)


if __name__ == '__main__':
    main()
