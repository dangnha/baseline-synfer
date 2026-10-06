"""Decoupled cross-attention adapters for text / semantic / AU conditioning.

Implements the IP-Adapter-style "same query, separate key/value" attention from
docs/methodology.md §4.5:

    Z_new = Attn(Q, K_t, V_t) + lambda_sem·Attn(Q, K_sem, V_sem) + lambda_au·Attn(Q, K_au, V_au)

The core ``DecoupledCrossAttention`` is self-contained and unit-testable; the
``SemanticAUAdapter`` turns the semantic code ``z_edit`` and the AU vector ``a`` into
the extra tokens. ``attach_to_unet`` is a best-effort hook into diffusers.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _reshape_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    b, n, c = x.shape
    return x.reshape(b, n, num_heads, c // num_heads).permute(0, 2, 1, 3)


def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scale = q.shape[-1] ** -0.5
    weights = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
    return weights @ v


class Projector(nn.Module):
    """Linear -> LayerNorm -> N tokens (IP-Adapter image projection, repurposed)."""

    def __init__(self, in_dim: int, token_dim: int, num_tokens: int):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.linear = nn.Linear(in_dim, token_dim * num_tokens)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        h = h.reshape(-1, self.num_tokens, self.token_dim)
        return self.norm(h)


class DecoupledCrossAttention(nn.Module):
    """Cross-attention with decoupled semantic and AU branches.

    The semantic/AU K/V projections are initialized from the text K/V weights so the
    branches start from a "text-like" behavior (docs/methodology.md §3).
    """

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)
        self.to_k_sem = nn.Linear(dim, dim)
        self.to_v_sem = nn.Linear(dim, dim)
        self.to_k_au = nn.Linear(dim, dim)
        self.to_v_au = nn.Linear(dim, dim)
        self._init_decoupled()
        self.lambda_sem = 1.0
        self.lambda_au = 1.0

    def _init_decoupled(self) -> None:
        with torch.no_grad():
            for src, dst in (
                (self.to_k, self.to_k_sem),
                (self.to_v, self.to_v_sem),
                (self.to_k, self.to_k_au),
                (self.to_v, self.to_v_au),
            ):
                dst.weight.copy_(src.weight)
                dst.bias.copy_(src.bias)

    def _project(self, layer: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        return _reshape_heads(layer(x), self.num_heads)

    def forward(
        self,
        z: torch.Tensor,
        c_text: torch.Tensor,
        c_sem: Optional[torch.Tensor] = None,
        c_au: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, nz, _ = z.shape
        q = self._project(self.to_q, z)
        out = _attention(q, self._project(self.to_k, c_text), self._project(self.to_v, c_text))
        if c_sem is not None and self.lambda_sem != 0.0:
            out = out + self.lambda_sem * _attention(
                q, self._project(self.to_k_sem, c_sem), self._project(self.to_v_sem, c_sem)
            )
        if c_au is not None and self.lambda_au != 0.0:
            out = out + self.lambda_au * _attention(
                q, self._project(self.to_k_au, c_au), self._project(self.to_v_au, c_au)
            )
        out = out.permute(0, 2, 1, 3).reshape(b, nz, self.dim)
        return self.to_out(out)


class SemanticAUAdapter(nn.Module):
    """Projects the edited semantic code and the AU vector into decoupled tokens."""

    def __init__(
        self,
        semantic_dim: int = 512,
        au_dim: int = 18,
        token_dim: int = 768,
        num_sem_tokens: int = 4,
        num_au_tokens: int = 4,
    ):
        super().__init__()
        self.semantic_projector = Projector(semantic_dim, token_dim, num_sem_tokens)
        self.au_projector = Projector(au_dim, token_dim, num_au_tokens)

    def forward(self, z_edit: torch.Tensor, a: torch.Tensor):
        return self.semantic_projector(z_edit), self.au_projector(a)


def attach_to_unet(unet, adapter: SemanticAUAdapter, lambda_sem: float = 1.0, lambda_au: float = 1.0):
    """Best-effort hook to install decoupled semantic/AU branches into a diffusers U-Net.

    NOTE: the exact ``AttnProcessor`` API varies across diffusers versions, so this is a
    documented starting point rather than a guaranteed drop-in. See README "Extending".
    """
    try:
        from diffusers.models.attention_processor import AttnProcessor  # noqa: F401
    except ImportError as exc:
        raise ImportError("diffusers is required to attach adapters to the U-Net.") from exc

    # The team should wrap DecoupledCrossAttention into an AttnProcessor subclass here,
    # mirroring the IP-Adapter reference implementation. This hook only records intent.
    raise NotImplementedError(
        "attach_to_unet is a scaffold: implement a DecoupledAttnProcessor for your "
        "installed diffusers version (see README). The core logic is in "
        "DecoupledCrossAttention / SemanticAUAdapter."
    )
