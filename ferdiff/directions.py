"""The AU direction operator W and its disentanglement controls.

Everything here is pure NumPy so it can be unit-tested without PyTorch or the
pretrained models. The operator learns linear edit directions in DiffAE's semantic
space ``z``:

    ŷ_k = z^T w_k + w_{0k}        (AU predictor)
    z'  = z + s · w_k             (edit)

and removes leakage with dependency-aware conditioning + orthogonal projection.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np


def _as_float2d(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x.reshape(x.shape[0], -1)


def fit_ridge(Z: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Ridge regression ``min_w ||Z w - y||^2 + alpha ||w||^2`` (no bias).

    Returns the coefficient vector ``w`` (shape (d,)).
    """
    Z = _as_float2d(Z)
    y = np.asarray(y, dtype=np.float64)
    d = Z.shape[1]
    A = Z.T @ Z + alpha * np.eye(d)
    b = Z.T @ y
    return np.linalg.solve(A, b)


def fit_ridge_bias(Z: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> Tuple[np.ndarray, float]:
    """Ridge regression with an intercept term.

    Returns ``(w, w0)``.
    """
    Z = _as_float2d(Z)
    Zb = np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=1)
    wb = fit_ridge(Zb, y, alpha)
    return wb[:-1], float(wb[-1])


def orthogonal_complement(nuisance_basis: np.ndarray) -> np.ndarray:
    """Projection matrix onto the orthogonal complement of ``nuisance_basis`` columns.

    ``P_perp = I - U(U^T U)^-1 U^T``, computed via QR for numerical stability.
    """
    U = np.asarray(nuisance_basis, dtype=np.float64)
    if U.ndim == 1:
        U = U.reshape(-1, 1)
    U = U.reshape(U.shape[0], -1)
    Q, _ = np.linalg.qr(U)
    return np.eye(U.shape[0]) - Q @ Q.T


def project_out(w: np.ndarray, nuisance_basis: np.ndarray) -> np.ndarray:
    """Project the edit direction ``w`` away from the nuisance subspace."""
    return orthogonal_complement(nuisance_basis) @ np.asarray(w, dtype=np.float64)


def orthogonality_loss(W: np.ndarray) -> float:
    """``||W^T W - I||_F^2`` — soft orthogonality of the AU directions."""
    G = W.T @ W
    return float(np.sum((G - np.eye(G.shape[0])) ** 2))


def off_graph_loss(W: np.ndarray, graph: List[List[int]]) -> float:
    """Sum of squared inner products over AU pairs *not* in the co-activation graph."""
    K = W.shape[1]
    allowed = set()
    for a, b in graph:
        allowed.add((min(a, b), max(a, b)))
    loss = 0.0
    for k in range(K):
        for l in range(k + 1, K):
            if (k, l) not in allowed:
                loss += float((W[:, k] @ W[:, l]) ** 2)
    return loss


def neutralize_closed_form(
    z_sample: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    target: np.ndarray,
    lmbda: float = 0.1,
) -> np.ndarray:
    """Neutralize ``z`` toward ``target`` AUs using the linear predictor ``AU(z)=W^T z + b``.

    Solves ``min_z ||W^T z + b - target||^2 + lambda ||z - z_sample||^2`` in closed form:

        z = (W W^T + lambda I)^-1 (lambda·z_sample - W (b - target)).
    """
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    z_sample = np.asarray(z_sample, dtype=np.float64).reshape(-1)
    d = W.shape[0]
    M = W @ W.T + lmbda * np.eye(d)
    rhs = lmbda * z_sample - W @ (b - target)
    return np.linalg.solve(M, rhs)


def neutralize_gd(
    z_sample: np.ndarray,
    au_predictor: Callable[[np.ndarray], np.ndarray],
    target: np.ndarray,
    lmbda: float = 0.1,
    steps: int = 200,
    lr: float = 0.05,
) -> np.ndarray:
    """Gradient descent neutralization for an arbitrary (differentiable) AU predictor.

    ``au_predictor(z) -> a`` must be differentiable w.r.t. ``z`` (e.g. a linear/MLP
    probe). Keeps edits close to ``z_sample`` via the proximity regularizer.
    """
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    z = np.asarray(z_sample, dtype=np.float64).copy()

    def value(z: np.ndarray) -> float:
        return float(np.sum((au_predictor(z) - target) ** 2) + lmbda * np.sum((z - z_sample) ** 2))

    eps = 1e-5
    for _ in range(steps):
        grad = np.zeros_like(z)
        for i in range(z.shape[0]):
            zp = z.copy()
            zm = z.copy()
            zp[i] += eps
            zm[i] -= eps
            grad[i] = (value(zp) - value(zm)) / (2 * eps)
        z -= lr * grad
    return z


class AUOperator:
    """Learn and apply linear AU edit directions with disentanglement controls."""

    def __init__(self, latent_dim: int, au_names: List[str], ridge_alpha: float = 1.0):
        self.latent_dim = latent_dim
        self.au_names = list(au_names)
        self.ridge_alpha = ridge_alpha
        self.W: np.ndarray = np.zeros((latent_dim, len(au_names)))
        self.bias: np.ndarray = np.zeros(len(au_names))
        self.nuisance_basis: Optional[np.ndarray] = None

    # -- fitting -------------------------------------------------------------
    def fit(
        self,
        Z: np.ndarray,
        A: np.ndarray,
        dependency_aware: bool = False,
        coactivating: Optional[List[List[int]]] = None,
    ) -> "AUOperator":
        """Fit one ridge direction per AU.

        ``Z``: (n, d) semantic codes. ``A``: (n, K) AU scores.

        With ``dependency_aware``, AU ``k`` is regressed on ``[z; a_other]`` so the
        direction only captures the part of ``a_k`` not explained by co-activated AUs
        (``coactivating[k]`` lists the controlling AU indices). The direction is the
        first ``d`` coefficients.
        """
        Z = _as_float2d(Z)
        A = np.asarray(A, dtype=np.float64)
        K = A.shape[1]
        assert Z.shape[0] == A.shape[0], "Z and A must have the same number of rows"
        assert K == len(self.au_names), "A columns must match au_names"
        self.W = np.zeros((self.latent_dim, K))
        self.bias = np.zeros(K)
        for k in range(K):
            if dependency_aware and coactivating and coactivating[k]:
                controls = [c for c in coactivating[k] if c != k and 0 <= c < K]
                X = np.concatenate([Z, A[:, controls]], axis=1)
            else:
                X = Z
            w_all, b0 = fit_ridge_bias(X, A[:, k], self.ridge_alpha)
            self.W[:, k] = w_all[: self.latent_dim]
            self.bias[k] = b0
        return self

    # -- disentanglement -----------------------------------------------------
    def set_nuisance_basis(self, U: np.ndarray) -> "AUOperator":
        """Set the nuisance-attribute directions (columns of U) to project out."""
        self.nuisance_basis = np.asarray(U, dtype=np.float64).reshape(self.latent_dim, -1)
        return self

    def project_nuisance(self, inplace: bool = True) -> np.ndarray:
        """Project every AU direction onto the orthogonal complement of the nuisance basis."""
        if self.nuisance_basis is None or self.nuisance_basis.shape[1] == 0:
            return self.W.copy()
        P_perp = orthogonal_complement(self.nuisance_basis)
        W_proj = P_perp @ self.W
        if inplace:
            self.W = W_proj
        return W_proj

    def orthogonality_loss(self) -> float:
        return orthogonality_loss(self.W)

    def off_graph_loss(self, graph: List[List[int]]) -> float:
        return off_graph_loss(self.W, graph)

    # -- application ---------------------------------------------------------
    def predict(self, z: np.ndarray) -> np.ndarray:
        """AU score ``W^T z + b`` for a single code or a batch."""
        z = np.asarray(z, dtype=np.float64)
        single = z.ndim == 1
        z = z.reshape(-1, self.latent_dim) if single else z.reshape(-1, self.latent_dim)
        a = z @ self.W + self.bias
        return a[0] if single else a

    def edit(self, z: np.ndarray, a_res: np.ndarray) -> np.ndarray:
        """Apply ``z + W · a_res`` (a_res is the target residual AU vector)."""
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        a_res = np.asarray(a_res, dtype=np.float64).reshape(-1)
        return z + self.W @ a_res

    def direction(self, au_index: int) -> np.ndarray:
        return self.W[:, au_index].copy()
