"""Fit the AU direction operator W from (z_sem, AU) pairs.

Usage:
    python scripts/learn_directions.py --Z z_sem.npy --A au_scores.npy \
        --au-names AU4 AU5 AU17 --out W.npz [--dependency-aware] \
        [--nuisance nuisance_basis.npy] [--coactivating coact.npy]

Outputs ``W.npz`` with W (d,K), bias (K,), au_names, and nuisance_basis.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ferdiff.directions import AUOperator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--Z", required=True, help="(n, d) semantic codes .npy")
    p.add_argument("--A", required=True, help="(n, K) AU scores .npy")
    p.add_argument("--au-names", nargs="+", required=True, help="AU names in column order")
    p.add_argument("--out", default="W.npz")
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--dependency-aware", action="store_true")
    p.add_argument("--coactivating", default=None, help="co-activation graph .npy (K lists)")
    p.add_argument("--nuisance", default=None, help="(d, m) nuisance basis .npy")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Z = np.load(args.Z)
    A = np.load(args.A)

    coactivating = None
    if args.coactivating:
        coactivating = [list(map(int, row)) for row in np.load(args.coactivating, allow_pickle=True)]

    op = AUOperator(Z.shape[1], args.au_names, ridge_alpha=args.ridge_alpha)
    op.fit(Z, A, dependency_aware=args.dependency_aware, coactivating=coactivating)
    if args.nuisance:
        op.set_nuisance_basis(np.load(args.nuisance))
        op.project_nuisance(inplace=True)

    np.savez(
        args.out,
        W=op.W,
        bias=op.bias,
        au_names=np.array(op.au_names),
        nuisance_basis=op.nuisance_basis if op.nuisance_basis is not None else np.zeros((op.latent_dim, 0)),
    )
    print(f"Saved {args.out}: W {op.W.shape}, bias {op.bias.shape}")


if __name__ == "__main__":
    main()
