"""Objective functions from docs/methodology.md §5.

All losses take torch tensors. The disentanglement loss accepts ``W`` as a
(d, K) tensor of edit directions and optionally the co-activation graph.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F


def diffusion_loss(noise_pred: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(noise_pred, noise)


def reconstruction_loss(x0: torch.Tensor, x0_hat: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x0, x0_hat)


def au_consistency_loss(au_pred: torch.Tensor, au_target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(au_pred, au_target)


def identity_loss(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
    """1 - cosine similarity between identity embeddings."""
    c = F.cosine_similarity(emb1, emb2, dim=-1)
    return (1.0 - c).mean()


def text_alignment_loss(img_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
    """Negative cosine similarity (CLIP-style alignment)."""
    c = F.cosine_similarity(img_emb, text_emb, dim=-1)
    return (-c).mean()


def orthogonality_term(W: torch.Tensor) -> torch.Tensor:
    """||W^T W - I||_F^2."""
    G = W.T @ W
    I = torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
    return torch.sum((G - I) ** 2)


def off_graph_term(W: torch.Tensor, graph: List[List[int]]) -> torch.Tensor:
    """Sum of squared inner products over AU pairs not in the co-activation graph."""
    allowed = set()
    for a, b in graph:
        allowed.add((min(a, b), max(a, b)))
    K = W.shape[1]
    terms = []
    for k in range(K):
        for l in range(k + 1, K):
            if (k, l) not in allowed:
                terms.append((W[:, k] * W[:, l]).sum() ** 2)
    if not terms:
        return torch.zeros((), dtype=W.dtype, device=W.device)
    return torch.stack(terms).sum()


def nuisance_term(W: torch.Tensor, nuisance_basis: torch.Tensor) -> torch.Tensor:
    """||P_nuisance W||_F^2 — leakage into the nuisance subspace."""
    P = nuisance_basis @ nuisance_basis.T  # assume orthonormal basis
    return torch.sum((P @ W) ** 2)


def disentanglement_loss(
    W: torch.Tensor,
    graph: Optional[List[List[int]]] = None,
    nuisance_basis: Optional[torch.Tensor] = None,
    lambda_ortho: float = 1.0,
    lambda_graph: float = 1.0,
    lambda_nuis: float = 1.0,
) -> torch.Tensor:
    loss = lambda_ortho * orthogonality_term(W)
    if graph:
        loss = loss + lambda_graph * off_graph_term(W, graph)
    if nuisance_basis is not None:
        loss = loss + lambda_nuis * nuisance_term(W, nuisance_basis)
    return loss
