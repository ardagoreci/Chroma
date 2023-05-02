"""Random graph generation and featurization for protein graphs. The Chroma architecture uses random graph neural
networks to constrain computational costs to O(N) for a system of N atoms. This module defines the random graph
generation and graph featurization processes."""
import jax.numpy as jnp
import jax
import numpy as np


@jax.jit
def get_internode_distances(coordinates) -> jnp.array:
    """Computes internode distances for all atoms.
    Args:
        coordinates: atom coordinates of shape [N_at, 3]
    Returns:
        a distance matrix of size N_at x N_at where the matrix[i,j] gives the internode distances between nodes i and j.
        This is a symmetric matrix.
    Important: if this method were to be generalized to distance for two vectors a and b, the resulting distance matrix
    should be transposed.
    (unit-tested)
    """
    n_atoms, _ = coordinates.shape
    x = coordinates
    broad_x = jnp.broadcast_to(x, shape=(n_atoms, n_atoms, 3))
    broad_x_T = jnp.transpose(broad_x, axes=(1, 0, 2))
    dists = jnp.sqrt(jnp.sum((broad_x - broad_x_T) ** 2, axis=2))
    return dists


def sample_random_graph():
    pass

# Further methods for featurization
