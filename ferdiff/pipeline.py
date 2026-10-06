"""Orchestrates the full pipeline: encode -> AU edit -> decode -> gate.

The class is deliberately component-based: the team injects their existing frozen
models (DiffAE encoder, SD decoder, OpenGraphAU, AUCANet, dlib) through the loader
functions in ``interfaces.py``. The edit/synthesize orchestration, dose sweep, and
neutralization live here.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from .config import Config
from .directions import AUOperator, neutralize_closed_form
from .gates import GateConfig, GateResult, demographic_acceptance, run_gates


class FerDiffusion:
    def __init__(
        self,
        config: Config,
        *,
        encoder: Callable,
        operator: AUOperator,
        au_estimator: Callable,
        emotion_classifier: Callable,
        identity: object,
        decoder: Callable,
        clip_text: Optional[Callable] = None,
        arcface: Optional[Callable] = None,
    ):
        self.config = config
        self.encoder = encoder
        self.operator = operator
        self.au_estimator = au_estimator
        self.emotion_classifier = emotion_classifier
        self.identity = identity
        self.decoder = decoder
        self.clip_text = clip_text
        self.arcface = arcface

    # -- building blocks -----------------------------------------------------
    def encode(self, image: np.ndarray) -> np.ndarray:
        return self.encoder(image)

    def _au_scores(self, image: np.ndarray) -> np.ndarray:
        scores = np.asarray(self.au_estimator(image), dtype=np.float64).reshape(-1)
        return scores

    def _decode(self, x_T, z_edit: np.ndarray, emotion: str, a_target: np.ndarray):
        c_text = self.clip_text(f"a photo of a {emotion} face") if self.clip_text else None
        return self.decoder(x_T, z_edit, c_text, a_target)

    def _sweep(
        self,
        z_sem: np.ndarray,
        a_res: np.ndarray,
        x_T,
        emotion: str,
        a_target: np.ndarray,
        a_source: np.ndarray,
        source_label: Optional[str],
    ) -> Tuple[Optional[np.ndarray], GateResult]:
        cfg = self.config.gates
        for dose in np.arange(self.config.train.dose_min, self.config.train.dose_max + 1e-9, self.config.train.dose_step):
            z_edit = z_sem + dose * self.operator.W @ a_res
            image = self._decode(x_T, z_edit, emotion, a_target)
            result = self._evaluate(image, a_target, a_source, source_label, cfg)
            if result.passed:
                return image, result
        return None, GateResult(passed=False)

    def _evaluate(
        self,
        image: np.ndarray,
        a_target: np.ndarray,
        a_source: np.ndarray,
        source_label: Optional[str],
        cfg: GateConfig,
    ) -> GateResult:
        scores = self._au_scores(image)
        # Identify the target AU as the index of the max residual.
        target_idx = int(np.argmax(np.abs(a_target - a_source)))
        target_au = self.config.au.au_names[target_idx]
        threshold = self.config.au.thresholds.get(target_au, 0.5)

        a_au_edit = {au: float(scores[i]) for i, au in enumerate(self.config.au.au_names) if a_source[i] > 0.0}
        a_au_recon = {au: float(a_source[i]) for i, au in enumerate(self.config.au.au_names) if a_source[i] > 0.0}

        emotion_pred_edit = str(self.emotion_classifier(image))
        emotion_pred_recon = source_label if source_label is not None else emotion_pred_edit
        identity_distance = self.identity.distance(image)

        return run_gates(
            target_au_score=float(scores[target_idx]),
            target_au_recon=float(a_source[target_idx]),
            target_au_threshold=threshold,
            a_au_edit=a_au_edit,
            a_au_recon=a_au_recon,
            emotion_pred_edit=emotion_pred_edit,
            emotion_pred_recon=emotion_pred_recon,
            identity_distance=identity_distance,
            source_label=source_label,
            cfg=cfg,
        )

    # -- public API ----------------------------------------------------------
    def edit(
        self,
        source: np.ndarray,
        a_target: np.ndarray,
        emotion: str,
        source_label: Optional[str] = None,
        x_T=None,
    ) -> Tuple[Optional[np.ndarray], GateResult]:
        """Add the target AU configuration to a source image (A -> A+x)."""
        z_sem = self.encode(source)
        a_source = self._au_scores(source)
        a_res = np.asarray(a_target, dtype=np.float64) - a_source
        return self._sweep(z_sem, a_res, x_T, emotion, a_target, a_source, source_label)

    def synthesize(
        self,
        z_sample: np.ndarray,
        a_target: np.ndarray,
        emotion: str,
        x_T=None,
    ) -> Tuple[Optional[np.ndarray], GateResult]:
        """Neutralize a sampled identity, then apply the target AU and decode."""
        a_target = np.asarray(a_target, dtype=np.float64)
        z_neutral = neutralize_closed_form(
            z_sample,
            self.operator.W,
            self.operator.bias,
            np.zeros_like(a_target),
            lmbda=self.config.train.neutralization_lambda,
        )
        z_edit = z_neutral + self.operator.W @ a_target
        image = self._decode(x_T, z_edit, emotion, a_target)
        result = self._evaluate(image, a_target, np.zeros_like(a_target), emotion, self.config.gates)
        return image, result

    def sample_identity(self, num: int, seed: Optional[int] = None) -> np.ndarray:
        """Sample semantic codes from a standard Gaussian prior (synthesis mode)."""
        rng = np.random.default_rng(seed)
        return rng.standard_normal((num, self.operator.latent_dim))

    def filter_demographics(self, predictions: dict, desired_bins: dict) -> bool:
        return demographic_acceptance(predictions, desired_bins)
