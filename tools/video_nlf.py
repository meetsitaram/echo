"""Video -> per-frame SMPL body of the one robot in it, with NLF (Neural Localizer Fields, github.com/isarandi/nlf;
code MIT, models non-commercial research). The model file carries its own body model and person detector, so no SMPL
download is needed. Camera frame (x right, y down, z forward), metres.

Runs in NLF's own environment (PyTorch with CUDA 12.8 for the RTX 5090), not the repo's plain python:

  C:/Users/PC/projects/nlf/venv/Scripts/python tools/video_nlf.py reports/video/<clip>.mp4 [--fov 60]

Writes <clip>_nlf.npz: fps, size, fov, and per frame ok, box, joints (24 x 3), pose (72, axis-angle, SMPL order),
betas (10), trans (3). One person is kept per frame: the detection that best overlaps the moving region.

NLF's person detector misses the robot when it bends over (18-27 s of the living-room clip: every pick-up). The camera
is fixed, so the robot is also found by background difference (the median of 60 frames is the empty room) and that
box is passed to NLF as an extra detection; NLF then fits the bent pose.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision  # noqa: F401  (registers the ops, e.g. nms, that the TorchScript model calls)

MODEL = Path('C:/Users/PC/projects/nlf/models/nlf_l_multi_0.3.2.patch6.torchscript')


def intrinsics(w, h, fov_deg):
    f = 0.5 * w / math.tan(math.radians(fov_deg) / 2)      # horizontal field of view
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float32)


def background(cap, n_samples=60):
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, n - 1, n_samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        frames.append(cap.read()[1])
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return cv2.GaussianBlur(np.median(np.stack(frames), axis=0).astype(np.uint8), (5, 5), 0)


def moving_box(img, bg, thr=28, min_area=4000):
    """[x, y, w, h, score] of the largest region that differs from the empty room, or None."""
    d = cv2.absdiff(cv2.GaussianBlur(img, (5, 5), 0), bg).max(axis=2)
    mk = cv2.morphologyEx((d > thr).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mk = cv2.morphologyEx(mk, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    k, _, st, _ = cv2.connectedComponentsWithStats(mk)
    if k < 2:
        return None
    j = 1 + int(np.argmax(st[1:, 4]))
    if st[j, 4] < min_area:
        return None
    x, y, w, h = st[j, :4]
    return np.array([x - 10, y - 10, w + 20, h + 20, 1.0], np.float32)


def iou(a, b):
    x0, y0 = np.maximum(a[:, 0], b[0]), np.maximum(a[:, 1], b[1])
    x1, y1 = np.minimum(a[:, 0] + a[:, 2], b[0] + b[2]), np.minimum(a[:, 1] + a[:, 3], b[1] + b[3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    return inter / (a[:, 2] * a[:, 3] + b[2] * b[3] - inter)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('video', type=Path)
    ap.add_argument('--fov', type=float, default=60.0, help='horizontal field of view of the camera, degrees')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    out = a.out or a.video.with_name(a.video.stem + '_nlf.npz')

    cap = cv2.VideoCapture(str(a.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    K = torch.from_numpy(intrinsics(w, h, a.fov)).cuda()
    model = torch.jit.load(str(MODEL)).cuda().eval()

    bg = background(cap)
    rec = {k: [] for k in ('ok', 'box', 'joints', 'pose', 'betas', 'trans')}
    last = None

    def take(pred, n, moving):
        nonlocal last
        for i in range(n):
            boxes = pred['boxes'][i].cpu().numpy() if 'boxes' in pred else np.zeros((0, 5))   # none in the batch
            if len(boxes) == 0:
                rec['ok'].append(False)
                for k, s in (('box', 5), ('joints', (24, 3)), ('pose', 72), ('betas', 10), ('trans', 3)):
                    rec[k].append(np.zeros(s, np.float32))
                continue
            c = boxes[:, :2] + boxes[:, 2:4] / 2
            if moving[i] is not None:
                j = int(np.argmax(iou(boxes, moving[i])))
            else:
                j = int(np.argmax(boxes[:, 4])) if last is None else int(np.argmin(np.linalg.norm(c - last, axis=1)))
            last = c[j]
            rec['ok'].append(True)
            rec['box'].append(boxes[j])
            rec['joints'].append(pred['joints3d'][i][j].cpu().numpy() / 1000.0)   # mm -> m
            rec['pose'].append(pred['pose'][i][j].cpu().numpy())
            rec['betas'].append(pred['betas'][i][j].cpu().numpy())
            rec['trans'].append(pred['trans'][i][j].cpu().numpy() / 1000.0)

    frames, moving, n = [], [], 0
    with torch.inference_mode():
        while True:
            okf, img = cap.read()
            if okf:
                frames.append(torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1))
                moving.append(moving_box(img, bg))
            if frames and (len(frames) == a.batch or not okf):
                batch = torch.stack(frames).cuda()
                pred = model.detect_smpl_batched(batch, intrinsic_matrix=K[None].expand(len(frames), 3, 3),
                                                 model_name='smpl', detector_threshold=0.2,
                                                 extra_boxes=[torch.from_numpy(b[None] if b is not None
                                                                               else np.zeros((0, 5), np.float32)).cuda()
                                                              for b in moving])
                take(pred, len(frames), moving)
                n += len(frames)
                frames, moving = [], []
                if n % (a.batch * 20) == 0:
                    print(f'  {n} frames', flush=True)
            if not okf:
                break
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, fps=fps, size=np.array([w, h]), fov=a.fov,
                        **{k: np.array(v) for k, v in rec.items()})
    print(f'{out}: {n} frames, {int(np.sum(rec["ok"]))} with the robot found')


if __name__ == '__main__':
    main()
