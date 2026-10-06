"""Unit tests for the AU direction operator (pure NumPy, no external models)."""

import numpy as np

from ferdiff.directions import (
    AUOperator,
    fit_ridge,
    neutralize_closed_form,
    off_graph_loss,
    orthogonal_complement,
    orthogonality_loss,
    project_out,
)


def test_ridge_recovers_linear_direction():
    rng = np.random.default_rng(0)
    d = 20
    w_true = rng.standard_normal(d)
    Z = rng.standard_normal((200, d))
    y = Z @ w_true
    w = fit_ridge(Z, y, alpha=1e-6)
    assert np.allclose(w, w_true, atol=1e-3)


def test_orthogonal_complement_projection():
    d = 10
    U = np.random.default_rng(1).standard_normal((d, 3))
    P = orthogonal_complement(U)
    assert np.allclose(P @ U, 0.0, atol=1e-8)
    assert np.allclose(P, P.T, atol=1e-8)
    assert np.allclose(P @ P, P, atol=1e-8)


def test_project_out_is_orthogonal_to_nuisance():
    d = 10
    w = np.arange(d, dtype=np.float64)
    U = np.random.default_rng(2).standard_normal((d, 2))
    w_perp = project_out(w, U)
    assert np.allclose(U.T @ w_perp, 0.0, atol=1e-8)


def test_orthogonality_loss_zero_for_orthonormal():
    W = np.eye(5, 3)  # W.T W = I_3
    assert orthogonality_loss(W) < 1e-10
    assert orthogonality_loss(np.ones((5, 3))) > 0.0


def test_off_graph_loss_ignores_allowed_edges():
    W = np.random.default_rng(4).standard_normal((6, 4))
    # All pairs allowed -> zero loss.
    graph = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    assert off_graph_loss(W, graph) < 1e-10


def test_neutralize_closed_form_reduces_au():
    d, K = 4, 2
    W = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    b = np.zeros(K)
    z = np.array([0.5, -0.5, 1.0, 1.0])
    z_n = neutralize_closed_form(z, W, b, np.zeros(K), lmbda=1e-4)
    assert np.linalg.norm(W.T @ z_n) < 0.1 * np.linalg.norm(W.T @ z)


def test_au_operator_fit_and_edit():
    rng = np.random.default_rng(5)
    d, K = 6, 2
    W_true = rng.standard_normal((d, K))
    Z = rng.standard_normal((100, d))
    A = Z @ W_true  # no bias
    op = AUOperator(d, ["AU4", "AU5"], ridge_alpha=1e-6)
    op.fit(Z, A)
    assert np.allclose(op.W, W_true, atol=1e-3)
    z0 = rng.standard_normal(d)
    a_res = np.array([1.0, 0.0])
    assert np.allclose(op.edit(z0, a_res), z0 + W_true[:, 0], atol=1e-3)
