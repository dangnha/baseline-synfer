"""Frozen DiffAE semantic encoder wrapper.

DiffAE turns an image into a two-part code ``(z_sem, x_T)``. Only the semantic code
``z_sem`` is used by the pipeline (the stochastic channel is produced by DDIM inversion
in SD latent space). The encoder is frozen and treated as a black box: wire your
``phizaz/diffae`` checkpoint (or an existing encoder callable) into ``interfaces.py``.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


class DiffAESemanticEncoder:
    def __init__(
        self,
        checkpoint: str = "",
        latent_dim: int = 512,
        device: str = "cuda",
        encoder_fn: Optional[Callable] = None,
    ):
        self.checkpoint = checkpoint
        self.latent_dim = latent_dim
        self.device = device
        self._encoder_fn = encoder_fn
        self._model = None

    def load(self) -> "DiffAESemanticEncoder":
        if self._encoder_fn is not None:
            return self
        if not self.checkpoint:
            raise FileNotFoundError(
                "Set diffae.checkpoint in configs/default.yaml to load the DiffAE encoder."
            )
        # The DiffAE model definition lives in the team's fork (phizaz/diffae).
        # Wire it here: build the model, load the checkpoint, set eval() and freeze.
        raise NotImplementedError(
            "Load your DiffAE checkpoint in interfaces.load_diffae_encoder() and pass "
            "its encode() as encoder_fn."
        )

    @torch_no_grad_guard
    def encode(self, images) -> np.ndarray:
        """Return semantic codes ``z_sem`` of shape (n, latent_dim)."""
        if self._encoder_fn is None:
            self.load()
        z = self._encoder_fn(images)
        z = np.asarray(z, dtype=np.float64).reshape(-1, self.latent_dim)
        return z


def torch_no_grad_guard(fn: Callable) -> Callable:
    """Wrap an encoder call in torch.no_grad() when torch is available."""
    try:
        import torch

        def wrapper(*args, **kwargs):
            with torch.no_grad():
                return fn(*args, **kwargs)

        return wrapper
    except ImportError:  # pragma: no cover
        return fn
