import jax
from polymer import get_stable_1malpha, SNR


def stable_scaling_factor(t):
    """Returns the stable scaling factor used during sampling."""
    return get_stable_1malpha(t) * jax.grad(SNR)(t)
