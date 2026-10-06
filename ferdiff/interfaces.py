"""Pluggable loaders for the external frozen models the pipeline depends on.

Each loader returns a callable/predictor. A missing path or optional dependency raises
a clear error, so the team fills in exactly what their environment already has. These
are the only places that touch external repositories / checkpoints.
"""

from __future__ import annotations

from typing import Callable, Optional


def load_diffae_encoder(checkpoint: str, latent_dim: int = 512, device: str = "cuda"):
    """Return ``encode(images) -> z_sem`` for the frozen DiffAE model.

    The team's fork (phizaz/diffae) already does this; adapt the import/load here.
    """
    if not checkpoint:
        raise FileNotFoundError("diffae.checkpoint is empty; set it in configs/default.yaml.")
    # Wire your diffae fork: build the model, load checkpoint, return its encode().
    raise NotImplementedError("Wire your DiffAE fork here and return encode(images) -> z_sem.")


def load_au_estimator(checkpoint: str, device: str = "cuda") -> Callable:
    """Return ``au_estimator(images) -> (n, K)`` AU scores (e.g. OpenGraphAU)."""
    if not checkpoint:
        raise FileNotFoundError("OpenGraphAU checkpoint path is empty.")
    raise NotImplementedError("Wire your OpenGraphAU model here and return a scoring callable.")


def load_emotion_classifier(checkpoint: str, device: str = "cuda") -> Callable:
    """Return ``classifier(images) -> predictions`` (e.g. AUCANet)."""
    if not checkpoint:
        raise FileNotFoundError("AUCANet checkpoint path is empty.")
    raise NotImplementedError("Wire your AUCANet model here and return a predict callable.")


def load_identity_descriptor() -> Callable:
    """Return ``descriptor(images) -> embeddings`` plus a ``distance(a, b)`` helper (dlib)."""
    try:
        import dlib  # noqa: F401
    except ImportError as exc:
        raise ImportError("dlib is required for the identity gate.") from exc
    raise NotImplementedError("Wire your dlib face-recognition model (68-point + recognizer) here.")


def load_arcface(checkpoint: str, device: str = "cuda") -> Callable:
    """Return ``arcface(images) -> embeddings`` for the identity loss."""
    if not checkpoint:
        raise FileNotFoundError("ArcFace checkpoint path is empty.")
    raise NotImplementedError("Wire your ArcFace / face-recognition backbone here.")


def load_clip_text_encoder(model_id: str = "openai/clip-vit-large-patch14") -> Callable:
    """Return ``encode_text(texts) -> embeddings`` for the global description."""
    try:
        from transformers import CLIPTextModel, CLIPTokenizer  # noqa: F401
    except ImportError as exc:
        raise ImportError("transformers is required for the CLIP text encoder.") from exc
    raise NotImplementedError("Wire your CLIP text encoder here and return encode_text(texts).")
