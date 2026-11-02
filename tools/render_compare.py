"""The source video beside echo replaying the motion captured from it, seen from the same camera: the video's camera,
calibrated by tools/nlf_to_tape.py on the room, moved into the take's frame (scaled by the retargeter's height ratio). echo stands in tools/scene_living_room.py's room, or in the studio.

  python tools/render_compare.py reports/motion/<take>.npz reports/video/<clip>.mp4 [--start S] [--dur D]
      [--studio] [--solo] [--height 1080] [--out reports/motion/<take>_compare.mp4]

For review only: the output holds the video's frames, so it stays out of the repo (reports/motion/ is ignored).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_motion as RM  # noqa: E402
import scene_living_room as SL  # noqa: E402


def video_camera(take):
    """(position, forward, horizontal fov, video size) of the video's camera in the take's frame: the camera
    tools/nlf_to_tape.py calibrated, scaled by the retargeter's height ratio (the take's floor is the room's)."""
    tape = np.load(str(take['tape']))
    R, origin, K = tape['cam_R'], tape['cam_origin'], tape['K']
    w, h = (int(v) for v in tape['size'])
    cam = (-origin @ R.T) * float(take['height_ratio'])
    fwd = R @ np.array([0.0, 0.0, 1.0])
    return cam, fwd, float(np.degrees(2 * np.arctan(w / 2 / K[0, 0]))), (w, h)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('take', type=Path)
    ap.add_argument('video', type=Path)
    ap.add_argument('--start', type=float, default=0.0)
    ap.add_argument('--dur', type=float, default=None)
    ap.add_argument('--studio', action='store_true', help='the grey studio instead of the living room')
    ap.add_argument('--fps', type=float, default=24.0)
    ap.add_argument('--height', type=int, default=540)
    ap.add_argument('--solo', action='store_true',
                    help="echo alone, no video and no captions (ours to publish: none of the footage is in it)")
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    out = a.out or a.take.with_name(a.take.stem + ('_solo.mp4' if a.solo else '_compare.mp4'))

    take = np.load(a.take)
    cam_pos, fwd, fov_h, (vw, vh) = video_camera(take)
    fovy = math.degrees(2 * math.atan(math.tan(math.radians(fov_h) / 2) * vh / vw))

    def room(ms):
        SL.add_living_room(ms)
        ms.visual.global_.fovy = fovy

    m = RM.build_view_model() if a.studio else RM.build_view_model(room=room)
    if a.studio:
        m.vis.global_.fovy = fovy
    data = mujoco.MjData(m)
    adr = [m.jnt_qposadr[m.joint(n).id] for n in take['joint_names']]
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    dist = 4.0
    cam.lookat[:] = cam_pos + fwd * dist
    cam.distance = dist
    cam.azimuth = math.degrees(math.atan2(fwd[1], fwd[0]))
    cam.elevation = math.degrees(math.asin(fwd[2]))
    h = a.height
    w = int(round(h * vw / vh))
    r = mujoco.Renderer(m, h, w)
    cap = cv2.VideoCapture(str(a.video))
    vfps = cap.get(cv2.CAP_PROP_FPS)
    t_take = take['t']
    t1 = t_take[-1] if a.dur is None else min(t_take[-1], a.start + a.dur)
    writer = imageio.get_writer(out, fps=a.fps, codec='libx264', quality=9 if a.solo else 7, pixelformat='yuv420p',
                                macro_block_size=8)
    for n, t in enumerate(np.arange(a.start, t1, 1.0 / a.fps)):
        i = int(np.clip(round(t * 50), 0, len(t_take) - 1))
        data.qpos[:3], data.qpos[3:7], data.qpos[adr] = take['root_pos'][i], take['root_quat_wxyz'][i], take['q'][i]
        mujoco.mj_kinematics(m, data)
        mujoco.mj_camlight(m, data)
        r.update_scene(data, cam)
        right = Image.fromarray(r.render())
        if a.solo:
            writer.append_data(np.asarray(right))
            if n % 240 == 0:
                print(f'  {t:5.1f} s', flush=True)
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round((t + float(take['start'])) * vfps)))
        okf, img = cap.read()
        left = Image.fromarray(cv2.cvtColor(cv2.resize(img, (w, h)), cv2.COLOR_BGR2RGB)) if okf else \
            Image.new('RGB', (w, h))
        frame = Image.new('RGB', (2 * w, h))
        frame.paste(left, (0, 0))
        frame.paste(right, (w, 0))
        RM.caption(frame, (8, 8), f'reference video  t {t:5.1f} s')
        RM.caption(frame, (w + 8, 8), 'echo: motion captured from the video (NLF -> GMR-based IK), kinematic replay')
        writer.append_data(np.asarray(frame))
        if n % 240 == 0:
            print(f'  {t:5.1f} s', flush=True)
    writer.close()
    r.close()
    print(out)


if __name__ == '__main__':
    main()
