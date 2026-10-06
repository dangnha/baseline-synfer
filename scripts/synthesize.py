"""Synthesize new identities: sample -> neutralize -> apply target AU -> decode -> gate."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ferdiff.build import build_pipeline
from ferdiff.config import Config
from ferdiff.io import save_image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--W", required=True)
    p.add_argument("--target-au", required=True)
    p.add_argument("--emotion", required=True)
    p.add_argument("--num", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="outputs/synthesize")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config)
    W = np.load(args.W)
    pipeline = build_pipeline(config, W)  # decoder defaults to None; wire it before running

    au_names = list(W["au_names"])
    au_idx = au_names.index(args.target_au)
    a_target = np.zeros(len(au_names))
    a_target[au_idx] = 1.0

    os.makedirs(args.out_dir, exist_ok=True)
    zs = pipeline.sample_identity(args.num, seed=args.seed)
    for i, z in enumerate(zs):
        image, result = pipeline.synthesize(z, a_target, args.emotion)
        if image is not None:
            save_image(os.path.join(args.out_dir, f"{i:04d}.png"), image)
        print(i, "PASS" if result.passed else "FAIL", result.checks)


if __name__ == "__main__":
    main()
