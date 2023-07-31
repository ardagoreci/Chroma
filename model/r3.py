"""Transformations for 3D coordinates.

This module has been adapted from the AlphaFold codebase:
https://github.com/deepmind/alphafold/blob/main/alphafold/model/r3.py

This Module contains objects for representing Vectors (Vecs), Rotation Matrices
(Rots) and proper Rigid transformation (Rigids). These are represented as
named tuples with arrays for each entry, for example a set of
[N, M] points would be represented as a Vecs object with arrays of shape [N, M]
for x, y and z.

This is being done to improve readability by making it very clear what objects
are geometric objects rather than relying on comments and array shapes.
Another reason for this is to avoid using matrix multiplication primitives
like matmul or einsum, on modern accelerator hardware these can end up on
specialized cores such as tensor cores on GPU or the MXU on
cloud TPUs, this often involves lower computational precision which can be
problematic for coordinate geometry. Also, these cores are typically optimized
for larger matrices than 3 dimensional, this code is written to avoid any
unintended use of these cores on both GPUs and TPUs.
"""
import collections
from typing import List
import jax.numpy as jnp
import tree

# Array of 3-component vectors, stored as individual array for
# each component.
Vecs = collections.namedtuple('Vecs', ['x', 'y', 'z'])

# Array of 3x3 rotation matrices, stored as individual array for
# each component.
Rots = collections.namedtuple('Rots', ['xx', 'xy', 'xz',
                                       'yx', 'yy', 'yz',
                                       'zx', 'zy', 'zz'])
# Array of rigid 3D transformations, stored as array of rotations and
# array of translations.
Rigids = collections.namedtuple('Rigids', ['rot', 'trans'])


def squared_difference(x, y):
    return jnp.square(x - y)


def invert_rigids(r: Rigids) -> Rigids:
    """Computes group inverse of rigid transformations 'r'."""
    inv_rots = invert_rots(r.rot)
    t = rots_mul_vecs(inv_rots, r.trans)
    inv_trans = Vecs(-t.x, -t.y, -t.z)
    return Rigids(inv_rots, inv_trans)


def invert_rots(m: Rots) -> Rots:
    """Computes inverse of rotations 'm'."""
    return Rots(m.xx, m.yx, m.zx,
                m.xy, m.yy, m.zy,
                m.xz, m.yz, m.zz)


def rigids_from_3_points(
        point_on_neg_x_axis: Vecs,  # shape (...)
        origin: Vecs,  # shape (...)
        point_on_xy_plane: Vecs,  # shape (...)
) -> Rigids:  # shape (...)
    """Create Rigids from 3 points.

    Jumper et al. (2021) Suppl. Alg. 21 "rigidFrom3Points"
    This creates a set of rigid transformations from 3 points by Gram Schmidt
    orthogonalization.

    Args:
        point_on_neg_x_axis: Vecs corresponding to points on the negative x axis
        origin: Origin of resulting rigid transformations
        point_on_xy_plane: Vecs corresponding to points in the xy plane
    Returns:
        Rigid transformations from global frame to local frames derived from
        the input points.
    """
    m = rots_from_two_vecs(
        e0_unnormalized=vecs_sub(origin, point_on_neg_x_axis),
        e1_unnormalized=vecs_sub(point_on_xy_plane, origin))

    return Rigids(rot=m, trans=origin)


def rots_from_two_vecs(e0_unnormalized: Vecs, e1_unnormalized: Vecs) -> Rots:
    """Create rotation matrices from unnormalized vectors for the x and y-axes.

    This creates a rotation matrix from two vectors using Gram-Schmidt
    orthogonalization.
    Args:
        e0_unnormalized: vectors lying along x-axis of resulting rotation
        e1_unnormalized: vectors lying in xy-plane of resulting rotation
    Returns:
        Rotations resulting from Gram-Schmidt procedure.
    """
    # Normalize the unit vector for the x-axis, e0.
    e0 = vecs_robust_normalize(e0_unnormalized)

    # make e1 perpendicular to e0.
    c = vecs_dot_vecs(e1_unnormalized, e0)
    e1 = Vecs(e1_unnormalized.x - c * e0.x,
              e1_unnormalized.y - c * e0.y,
              e1_unnormalized.z - c * e0.z)
    e1 = vecs_robust_normalize(e1)

    # Compute e2 as cross product of e0 and e1.
    e2 = vecs_cross_vecs(e0, e1)

    return Rots(e0.x, e1.x, e2.x, e0.y, e1.y, e2.y, e0.z, e1.z, e2.z)


def vecs_robust_normalize(v: Vecs, epsilon: float = 1e-8) -> Vecs:
    """Normalizes vectors 'v'.

    Args:
        v: vectors to be normalized.
        epsilon: small regularizer added to squared norm before taking square root.
    Returns:
        normalized vectors
    """
    norms = vecs_robust_norm(v, epsilon)
    return Vecs(v.x / norms, v.y / norms, v.z / norms)


def vecs_robust_norm(v: Vecs, epsilon: float = 1e-8) -> jnp.ndarray:
    """Computes norm of vectors 'v'.
    Args:
        v: vectors to be normalized.
        epsilon: small regularizer added to squared norm before taking square root.
    Returns:
        norm of 'v'
    """
    return jnp.sqrt(jnp.square(v.x) + jnp.square(v.y) + jnp.square(v.z) + epsilon)


def rots_mul_vecs(m: Rots, v: Vecs) -> Vecs:
    """Apply rotations 'm' to vectors 'v'."""
    return Vecs(m.xx * v.x + m.xy * v.y + m.xz * v.z,
                m.yx * v.x + m.yy * v.y + m.yz * v.z,
                m.zx * v.x + m.zy * v.y + m.zz * v.z)


def vecs_add(v1: Vecs, v2: Vecs) -> Vecs:
    """Add two vectors 'v1' and 'v2'."""
    return Vecs(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z)


def vecs_dot_vecs(v1: Vecs, v2: Vecs) -> jnp.ndarray:
    """Dot product of vectors 'v1' and 'v2'."""
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z


def vecs_cross_vecs(v1: Vecs, v2: Vecs) -> Vecs:
    """Cross product of vectors 'v1' and 'v2'."""
    return Vecs(v1.y * v2.z - v1.z * v2.y,
                v1.z * v2.x - v1.x * v2.z,
                v1.x * v2.y - v1.y * v2.x)


def vecs_from_tensor(x: jnp.ndarray  # shape (..., 3)
                     ) -> Vecs:  # shape (...)
    """Converts from tensor of shape (3,) to Vecs."""
    num_components = x.shape[-1]
    assert num_components == 3
    return Vecs(x[..., 0], x[..., 1], x[..., 2])


def vecs_sub(v1: Vecs, v2: Vecs) -> Vecs:
    """Computes v1 - v2."""
    return Vecs(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z)


def vecs_squared_distance(v1: Vecs, v2: Vecs) -> jnp.ndarray:
    """Computes squared euclidean difference between 'v1' and 'v2'."""
    return (squared_difference(v1.x, v2.x) +
            squared_difference(v1.y, v2.y) +
            squared_difference(v1.z, v2.z))
