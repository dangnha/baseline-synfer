"""DDIM deterministic sampling and inversion (docs/methodology.md §2.2).

``alphas_cumprod`` is the noise schedule (a 1-D tensor of length T+1). ``model`` is a
callable ``model(x, t, cond) -> eps`` predicting the noise at step ``t``.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import torch


def _step(x_t: torch.Tensor, model: Callable, t: int, cond, alphas_cumprod: torch.Tensor):
    eps = model(x_t, t, cond)
    alpha_t = alphas_cumprod[t]
    x0 = (x_t - torch.sqrt(1.0 - alpha_t) * eps) / torch.sqrt(alpha_t)
    return x0, eps


def ddim_step(
    model: Callable,
    x_t: torch.Tensor,
    t: int,
    t_prev: int,
    alphas_cumprod: torch.Tensor,
    cond=None,
    eta: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """One deterministic (eta=0) DDIM reverse step."""
    x0, eps = _step(x_t, model, t, cond, alphas_cumprod)
    alpha_t = alphas_cumprod[t]
    alpha_prev = alphas_cumprod[t_prev]
    sigma = eta * torch.sqrt((1.0 - alpha_prev) / (1.0 - alpha_t)) * torch.sqrt(
        1.0 - alpha_t / alpha_prev
    )
    noise = torch.randn_like(x_t, generator=generator) if eta > 0 else torch.zeros_like(x_t)
    return torch.sqrt(alpha_prev) * x0 + torch.sqrt(1.0 - alpha_prev - sigma**2) * eps + sigma * noise


def ddim_invert_step(
    model: Callable,
    x_t: torch.Tensor,
    t: int,
    t_next: int,
    alphas_cumprod: torch.Tensor,
    cond=None,
) -> torch.Tensor:
    """Approximate DDIM inversion (forward direction): x_t -> x_{t+1}."""
    x0, eps = _step(x_t, model, t, cond, alphas_cumprod)
    alpha_next = alphas_cumprod[t_next]
    return torch.sqrt(alpha_next) * x0 + torch.sqrt(1.0 - alpha_next) * eps


def ddim_sample(
    model: Callable,
    x_T: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    steps: List[int],
    cond=None,
    eta: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Reverse from x_T over the given (descending) timesteps."""
    x = x_T
    for i in range(len(steps) - 1):
        t, t_prev = steps[i], steps[i + 1]
        x = ddim_step(model, x, t, t_prev, alphas_cumprod, cond, eta, generator)
    return x


def ddim_invert(
    model: Callable,
    x_0: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    steps: List[int],
    cond=None,
) -> torch.Tensor:
    """Forward (inversion) from x_0 over ascending timesteps -> x_T."""
    x = x_0
    for i in range(len(steps) - 1):
        t, t_next = steps[i], steps[i + 1]
        x = ddim_invert_step(model, x, t, t_next, alphas_cumprod, cond)
    return x
