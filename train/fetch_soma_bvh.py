"""Stream bones-studio/seed soma_uniform.tar.gz and keep only the listed clips' BVHs (no full archive on disk). The dataset
is gated: needs your Hugging Face login, and the data stays local, under its own licence, never in this repo.

  python train/fetch_soma_bvh.py <bones-seed dir> <clips .csv with move_name> [--also NAME ...] [--mbps 5]

Extracts under <bones-seed dir>/soma_uniform/bvh/<date>/ and stops as soon as every clip is found. Clips already on disk
are skipped, but a restart reads the archive again from the start (a gzip stream cannot seek).
"""
import argparse
import csv
import os
import tarfile
import time

import pandas as pd
import requests
from huggingface_hub import get_token, hf_hub_url


class Counted:
    """The download, capped at mbps: a sustained full-speed fetch has dropped this PC's Wi-Fi adapter."""

    def __init__(self, raw, mbps):
        self.raw, self.n, self.t, self.t0, self.rate = raw, 0, time.time(), time.time(), mbps * 1e6

    def read(self, k=-1):
        b = self.raw.read(k)
        self.n += len(b)
        ahead = self.n / self.rate - (time.time() - self.t0)
        if ahead > 0:
            time.sleep(ahead)
        if time.time() - self.t > 60:
            self.t = time.time()
            print(f'{self.n / 1e9:.1f} GB read', flush=True)
        return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('clips')
    ap.add_argument('--also', nargs='*', default=[])
    ap.add_argument('--mbps', type=float, default=5.0, help='megabytes per second')
    a = ap.parse_args()
    with open(a.clips, encoding='utf-8') as f:
        names = {r['move_name'] for r in csv.DictReader(f)} | set(a.also)
    meta = pd.read_parquet(os.path.join(a.root, 'metadata', 'seed_metadata_v004.parquet'))
    col = next(c for c in meta.columns if 'soma_uniform' in c and 'path' in c)
    by_lower = {str(n).lower(): str(p).lstrip('./') for n, p in zip(meta['move_name'], meta[col])}
    want = {by_lower[n.lower()] for n in names if n.lower() in by_lower}
    missing = sorted(n for n in names if n.lower() not in by_lower)
    print(f'want {len(want)} of {len(names)} clips, not in the metadata: {len(missing)} {missing[:5]}', flush=True)
    want = {w for w in want if not os.path.exists(os.path.join(a.root, w))}
    print(f'{len(want)} left to fetch', flush=True)
    if not want:
        return
    url = hf_hub_url('bones-studio/seed', 'soma_uniform.tar.gz', repo_type='dataset')
    r = requests.get(url, headers={'Authorization': 'Bearer ' + get_token()}, stream=True, timeout=60)
    r.raise_for_status()
    src = Counted(r.raw, a.mbps)
    got = 0
    with tarfile.open(fileobj=src, mode='r|gz') as tf:
        for m in tf:
            name = m.name.lstrip('./')
            if name in want and m.isfile():
                m.name = name
                tf.extract(m, a.root, filter='data')
                got += 1
                if got % 50 == 0 or got == len(want):
                    print(f'got {got} of {len(want)}: {name}', flush=True)
                if got == len(want):
                    break
    print(f'done {got} of {len(want)} after {src.n / 1e9:.1f} GB', flush=True)


if __name__ == '__main__':
    main()
