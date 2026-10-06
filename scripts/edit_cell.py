"""Run A -> A+x editing for a manifest of source images (edit mode).

Needs the external models wired in ``ferdiff/interfaces.py`` and the SD decoder
callable passed to ``build_pipeline``. See README "Before you run".
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ferdiff.build import build_pipeline
from ferdiff.config import Config
from ferdiff.io import basename, load_image, save_image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--W", required=True, help="W.npz from learn_directions.py")
    p.add_argument("--sources", required=True, help="text file, one source image path per line")
    p.add_argument("--target-au", required=True, help="AU name to add, e.g. AU5")
    p.add_argument("--emotion", required=True)
    p.add_argument("--out-dir", default="outputs/edit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config)
    W = np.load(args.W)
    pipeline = build_pipeline(config, W)  # decoder defaults to None; wire it before running

    with open(args.sources, "r", encoding="utf-8") as fh:
        sources = [line.strip() for line in fh if line.strip()]

    au_names = list(W["au_names"])
    au_idx = au_names.index(args.target_au)
    a_target = np.zeros(len(au_names))
    a_target[au_idx] = 1.0

    os.makedirs(args.out_dir, exist_ok=True)
    for src in sources:
        image = load_image(src)
        out, result = pipeline.edit(image, a_target, args.emotion)
        if out is not None:
            save_image(os.path.join(args.out_dir, basename(src)), out)
        print(src, "PASS" if result.passed else "FAIL", result.checks)


if __name__ == "__main__":
    main()
