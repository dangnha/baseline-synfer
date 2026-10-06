"""Wire the frozen external models into a ``FerDiffusion`` pipeline.

This is the single place that connects ``configs/default.yaml`` to the loader stubs in
``interfaces.py``. Edit ``interfaces.py`` (not this file) to plug in your checkpoints.
"""

from __future__ import annotations

from typing import Optional

from . import interfaces
from .config import Config
from .directions import AUOperator
from .pipeline import FerDiffusion


def build_operator(config: Config, W_dict) -> AUOperator:
    op = AUOperator(config.diffae.latent_dim, list(W_dict["au_names"]))
    op.W = W_dict["W"]
    op.bias = W_dict["bias"]
    nb = W_dict.get("nuisance_basis")
    if nb is not None and nb.shape[1]:
        op.set_nuisance_basis(nb)
    return op


def build_pipeline(config: Config, W_dict, decoder=None) -> FerDiffusion:
    """Build the pipeline. ``decoder(x_T, z_edit, c_text, a) -> image`` is the SD decode fn."""
    return FerDiffusion(
        config,
        encoder=interfaces.load_diffae_encoder(
            config.diffae.checkpoint, config.diffae.latent_dim, config.diffae.device
        ),
        operator=build_operator(config, W_dict),
        au_estimator=interfaces.load_au_estimator(config.paths.opengraphau, config.diffae.device),
        emotion_classifier=interfaces.load_emotion_classifier(config.paths.aucanet, config.diffae.device),
        identity=interfaces.load_identity_descriptor(),
        decoder=decoder,
        clip_text=interfaces.load_clip_text_encoder(config.paths.clip_model_id),
        arcface=interfaces.load_arcface(config.paths.arcface, config.diffae.device),
    )
