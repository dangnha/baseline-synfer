"""Unit tests for the decoupled cross-attention adapters (torch only)."""

import pytest

torch = pytest.importorskip("torch")

from ferdiff.adapter import (  # noqa: E402
    DecoupledCrossAttention,
    Projector,
    SemanticAUAdapter,
    _attention,
    _reshape_heads,
)


def test_projector_shape():
    p = Projector(in_dim=10, token_dim=8, num_tokens=4)
    x = torch.randn(2, 10)
    out = p(x)
    assert out.shape == (2, 4, 8)


def test_decoupled_cross_attention_shape():
    dim, heads = 16, 4
    m = DecoupledCrossAttention(dim, heads)
    z = torch.randn(2, 5, dim)
    c_text = torch.randn(2, 3, dim)
    c_sem = torch.randn(2, 4, dim)
    c_au = torch.randn(2, 4, dim)
    assert m(z, c_text, c_sem, c_au).shape == z.shape


def test_decoupled_attention_reduces_to_text_only():
    dim, heads = 16, 4
    m = DecoupledCrossAttention(dim, heads)
    m.lambda_sem = 0.0
    m.lambda_au = 0.0
    z = torch.randn(2, 5, dim)
    c_text = torch.randn(2, 3, dim)
    c_sem = torch.randn(2, 4, dim)
    c_au = torch.randn(2, 4, dim)
    out = m(z, c_text, c_sem, c_au)

    q = _reshape_heads(m.to_q(z), m.num_heads)
    ref = _attention(
        q, _reshape_heads(m.to_k(c_text), m.num_heads), _reshape_heads(m.to_v(c_text), m.num_heads)
    )
    b, n, _ = z.shape
    ref = ref.permute(0, 2, 1, 3).reshape(b, n, dim)
    ref = m.to_out(ref)
    assert torch.allclose(out, ref, atol=1e-5)


def test_semantic_au_adapter_shapes():
    adapter = SemanticAUAdapter(semantic_dim=512, au_dim=18, token_dim=768, num_sem_tokens=4, num_au_tokens=4)
    c_sem, c_au = adapter(torch.randn(2, 512), torch.randn(2, 18))
    assert c_sem.shape == (2, 4, 768)
    assert c_au.shape == (2, 4, 768)
