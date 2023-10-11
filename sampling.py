"""This module implements the low temperature sampling that is used in Chroma. The paper introduces a
new low temperature diffusion sampling procedure that increase sampling of high-likelihood states at the
expense of reduced sample diversity."""

import jax
from model.polymer import get_stable_1malpha, SNR


def stable_scaling_factor(t):
    """Returns the stable scaling factor used during sampling."""
    return get_stable_1malpha(t) * jax.grad(SNR)(t)
