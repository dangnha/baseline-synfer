"""FER-Diffusion: AU-conditioned face synthesis pipeline.

Hybrid DiffAE (semantic encoder) + Stable Diffusion (decoder) with a disentangled
AU direction operator W and decoupled text/AU cross-attention adapters.

See docs/methodology.md for the full method.
"""

from .config import Config
from .directions import AUOperator
from .pipeline import FerDiffusion

__all__ = ["Config", "AUOperator", "FerDiffusion"]
__version__ = "0.1.0"
