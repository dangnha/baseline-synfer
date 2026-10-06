"""Acceptance--rejection gates (docs/methodology.md §4.7, already.md "gates").

The four frozen gates decide whether a generated image is accepted. Each is a pure
predicate on scalar scores; ``GateResult`` records which gates passed for reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class GateConfig:
    target_au_delta: float = 0.10
    keep_a_delta: float = 0.10
    identity_threshold: float = 0.6
    demographic_bins: Optional[Dict[str, List[str]]] = None


@dataclass
class GateResult:
    passed: bool = False
    checks: Dict[str, bool] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.passed


def gate_target_au(score: float, recon_score: float, threshold: float, delta: float) -> bool:
    """Target AU must be active and rise by at least ``delta`` vs reconstruction."""
    return score >= threshold and (score - recon_score) >= delta


def gate_keep_a(edit_scores: Dict[str, float], recon_scores: Dict[str, float], delta: float) -> bool:
    """Every AU of the source configuration must not drop by more than ``delta``."""
    for au, s in edit_scores.items():
        if recon_scores.get(au, 0.0) - s > delta:
            return False
    return True


def gate_emotion(pred_edit: str, pred_recon: str, source_label: Optional[str] = None) -> bool:
    """Emotion must not get worse than the reconstruction (or equal the source label)."""
    if source_label is not None and pred_edit == source_label:
        return True
    return pred_edit == pred_recon


def gate_identity(distance: float, threshold: float) -> bool:
    return distance <= threshold


def demographic_acceptance(predictions: Dict[str, str], desired_bins: Dict[str, List[str]]) -> bool:
    """Accept only if every demographic attribute falls into a desired bin."""
    return all(predictions.get(attr) in bins for attr, bins in desired_bins.items())


def run_gates(
    *,
    target_au_score: float,
    target_au_recon: float,
    target_au_threshold: float,
    a_au_edit: Dict[str, float],
    a_au_recon: Dict[str, float],
    emotion_pred_edit: str,
    emotion_pred_recon: str,
    identity_distance: float,
    source_label: Optional[str] = None,
    cfg: GateConfig = GateConfig(),
) -> GateResult:
    checks = {
        "target_au": gate_target_au(target_au_score, target_au_recon, target_au_threshold, cfg.target_au_delta),
        "keep_a": gate_keep_a(a_au_edit, a_au_recon, cfg.keep_a_delta),
        "emotion": gate_emotion(emotion_pred_edit, emotion_pred_recon, source_label),
        "identity": gate_identity(identity_distance, cfg.identity_threshold),
    }
    return GateResult(passed=all(checks.values()), checks=checks)
