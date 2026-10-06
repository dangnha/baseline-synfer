"""Configuration dataclasses for the FER-Diffusion pipeline.

All checkpoint/data paths live here so the team only edits ``configs/default.yaml``
(or constructs a ``Config`` programmatically). Missing paths raise a clear error at
load time (see ``interfaces.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

try:  # PyYAML is optional; fall back to a dict loader if absent.
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class DiffAEConfig:
    """Frozen DiffAE semantic encoder."""

    checkpoint: str = ""
    latent_dim: int = 512
    device: str = "cuda"
    # Nuisance-attribute directions (glasses/beard/gender/age) come from the
    # CelebA-HQ classifiers bundled with the DiffAE fork.
    nuisance_labels: List[str] = field(default_factory=lambda: ["glasses", "beard", "gender", "age"])


@dataclass
class SDConfig:
    """Frozen Stable Diffusion decoder (diffusers)."""

    model_id: str = "stabilityai/stable-diffusion-2-1-base"  # or a local path
    device: str = "cuda"
    dtype: str = "float16"
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    # Decoupled cross-attention strength dials.
    lambda_sem: float = 1.0
    lambda_au: float = 1.0


@dataclass
class AUConfig:
    """AU vocabulary, activation thresholds and the allowed co-activation graph."""

    au_names: List[str] = field(default_factory=list)
    thresholds: Dict[str, float] = field(default_factory=dict)
    # Allowed co-activation edges as pairs of AU *indices* into au_names.
    # Pairs not listed are penalised by the off-graph disentanglement loss.
    coactivation_graph: List[List[int]] = field(default_factory=list)


@dataclass
class GateConfig:
    target_au_delta: float = 0.10   # target AU must rise >= this vs recon
    keep_a_delta: float = 0.10      # each A-AU may drop <= this vs recon
    identity_threshold: float = 0.6  # dlib distance upper bound
    # synthesis-mode demographic acceptance--rejection
    demographic_bins: Optional[Dict[str, List[str]]] = None


@dataclass
class LossConfig:
    alpha: float = 1.0   # reconstruction
    beta: float = 1.0    # AU consistency
    gamma: float = 1.0   # identity
    delta: float = 1.0   # disentanglement
    eta: float = 1.0     # text alignment


@dataclass
class TrainConfig:
    ridge_alpha: float = 1.0
    neutralization_lambda: float = 0.1
    neutralization_steps: int = 200
    neutralization_lr: float = 0.05
    dose_min: float = 1.0
    dose_max: float = 30.0
    dose_step: float = 1.0


@dataclass
class ModelPaths:
    """Checkpoint paths for the frozen detectors / auxiliary encoders."""

    opengraphau: str = ""   # OpenGraphAU checkpoint (AU estimator)
    aucanet: str = ""       # AUCANet checkpoint (emotion classifier), e.g. rafdb_best.pth
    arcface: str = ""       # ArcFace / face-recognition backbone
    clip_model_id: str = "openai/clip-vit-large-patch14"


@dataclass
class Config:
    diffae: DiffAEConfig = field(default_factory=DiffAEConfig)
    sd: SDConfig = field(default_factory=SDConfig)
    au: AUConfig = field(default_factory=AUConfig)
    gates: GateConfig = field(default_factory=GateConfig)
    losses: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    paths: ModelPaths = field(default_factory=ModelPaths)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        if yaml is None:
            raise ImportError("PyYAML is required to load a config file (pip install pyyaml).")
        with open(path, "r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Config":
        cfg = cls()
        for section, dataclass_type in (
            ("diffae", DiffAEConfig),
            ("sd", SDConfig),
            ("au", AUConfig),
            ("gates", GateConfig),
            ("losses", LossConfig),
            ("train", TrainConfig),
            ("paths", ModelPaths),
        ):
            if section in raw:
                setattr(cfg, section, dataclass_type(**raw[section]))
        return cfg

    def with_overrides(self, **kwargs: Any) -> "Config":
        return replace(self, **kwargs)
