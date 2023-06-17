"""This file implements the methods and classes that are useful when working with protein geometry."""
import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple
from protein_graph import gather_edges, gather_nodes


class Transforms(NamedTuple):
    """A residue wise transform that stores a [..., 3, 3] orientation matrix and a [..., 3] translation vector."""
    translations: jnp.array
    orientations: jnp.array


def invert_transform(t, O) -> Tuple[jnp.array, jnp.array]:
    """Computes the inverse of a given transform as T(-1) = (dot(-O(-1), t), O(-1)) where O(-1) is the inverse of
    the orientation. This method can also be used to compute the inverses of relative transformations as T_ba =
    T_ab(-1).
    Args:
        t: the translation vector of shape [3,]
        O: the rotation matrix of shape [3, 3]
    Returns: inverted translation and rotation arrays
    (unit-tested)
    """
    new_t = - jnp.dot(O.T, t)
    new_O = O.T
    return new_t, new_O


def compose_transforms(transform_a, transform_b) -> Transforms:
    """Composes two transforms. This method needs to be jax.vmap transformed when the Transforms object passed
    are of higher rank.
    Args:
        transform_a: a Transforms object with arrays of shape [3,] and [3, 3]
        transform_b: a Transforms object with arrays of shape [3,] and [3, 3]

    (unit-tested)"""
    t_a = transform_a.translations
    O_a = transform_a.orientations
    t_b = transform_b.translations
    O_b = transform_b.orientations

    translation = t_a + jnp.dot(O_a, t_b)
    orientation = jnp.dot(O_a, O_b)
    return Transforms(translation, orientation)


def transforms_to_structure(transforms) -> jnp.array:
    """Computes the position of backbone atoms given transforms. The coordinates are computed using the ideal bond
    lengths of Gly with the following values (taken from alphafold.common.residue_constants.py):
    'GLY': ['N': (-0.572, 1.337, 0.000)],
           ['CA': (0.000, 0.000, 0.000)],
           ['C': (1.517, -0.000, -0.000)],
           ['O': (0.626, 1.062, -0.000)]
    It is important to note that N, CA, and C are in the backbone rigid group. However, O is in the psi group, so its
    coordinates vary depending on the torsion angle psi. The Ramachandran plots show that the value of psi is changes
    depending on the secondary structures alpha helices and beta sheets. For now, I will assume that the O is part of
    the backbone rigid group. This can then be fixed using residual updates to the final coordinates predicted from
    node embeddings - the network can easily learn the correct relative position of the O atom depending on the
    secondary structure context.
    Empirically, this has a root-mean-square deviation of 1.0 Angstrom from the true coordinates when given perfect
    transforms. Importantly, the carbon alpha positions are exactly the same.
    Args:
        transforms: a Transforms object containing translation of shape [N, 3] and orientations (rotations) of
        shape [N, 3, 3]
    Returns:
        an array of shape [N, 4, 3] encoding the coordinates
    (unit-tested)
    TODO: test if gradients are actually flowing through this method
    """

    def _frame_to_coor(t, O) -> jnp.array:
        """
        Args:
            t: translation vector of shape [3,]
            O: orientation vector of shape [3, 3]
        Returns:
            a jnp.array of shape [4, 3] encoding the transformed coordinates of N, CA, C, O in that order.
        """
        N_at = jnp.array([-0.572, 1.337, 0.000])
        Ca_at = jnp.array([0.000, 0.000, 0.000])
        C_at = jnp.array([1.517, -0.000, -0.000])
        O_at = jnp.array([0.626, 1.062, -0.000])

        coor = jnp.stack([N_at, Ca_at, C_at, O_at], axis=0)

        # Apply orientation transform to each atom
        def _apply_transform(x):
            """Applies the rotation 'O' matrix to a [3,] array"""
            return jnp.matmul(O, x) + t
        apply_transform_fn = jax.vmap(_apply_transform)

        return apply_transform_fn(coor)

    frames_to_coor_fn = jax.vmap(_frame_to_coor)  # accepts [N,3] translations and [N,3,3] rotations, returns [N,4,3]
    coordinates = frames_to_coor_fn(transforms.translations, transforms.orientations)
    return coordinates


def structure_to_transforms(coordinates) -> Transforms:
    """Constructs frames using the position of three atoms from the ground truth PDB structures using a Gram-Schmidt
    process. Note: the translation vector is assigned to the centre atom. Following AlphaFold, it uses N as x1, Ca as x2
    and C as x3 for backbone frames, so the frame has Ca at the centre. This function is written without the batch dim,
    it should be jax.vmap transformed if used with batch dimension.
    For details, see AlphaFold Supplementary Information, Alg. 21.
    Args:
        coordinates: structure coordinates of shape [N, 4, 3]
    Returns: a Transforms object storing the residue transforms, each with a translation vector and a rotation matrix.
    (unit-tested)"""

    def _single_rigid_from_3_points(x1, x2, x3) -> Transforms:
        """Single example implementation fo rigid_from_3_points. The implementation is easier and less bug-prone if
        written for a single example and then jax.vmap transformed.
        Args:
            x1: coordinate array of shape [3,]
            x2: coordinate array of shape [3,]
            x3: coordinate array of shape [3,]
        Returns: a Transforms object storing the residue transforms, each with a translation vector and a rotation
                 matrix.
        """
        v1 = x3 - x2
        v2 = x1 - x2
        e1 = v1 / jnp.linalg.norm(v1)
        u2 = v2 - (e1 * jnp.dot(e1, v2))
        e2 = u2 / jnp.linalg.norm(u2)
        e3 = jnp.cross(e1, e2)
        R = jnp.stack([e1, e2, e3], axis=0)
        t = x2  # translation atom assigned to center atom x2
        return Transforms(t, R)

    N = coordinates[:, 0, :]
    Ca = coordinates[:, 1, :]
    C = coordinates[:, 2, :]
    rigid_from_3_points_fn = jax.vmap(_single_rigid_from_3_points)
    return rigid_from_3_points_fn(N, Ca, C)


class PairwiseGeometries(NamedTuple):
    """A pairwise geometry that stores Transform objects and associated confidence values."""
    transforms: Transforms
    confidences: jnp.array  # this will be changed when switching to Backbone Network B


def compute_pairwise_geometries(transforms, topology):
    """Given current transforms of a protein, computes inter-residue transforms.
    Args:
        transforms: [N, ...]
        topology: graph topology of shape [N,]
    """
    t_i = transforms.translations  # [N, 3]
    O_i = transforms.orientations  # [N, 3, 3]

    # Gather current transforms of edge frames
    t_j = gather_nodes(transforms.translations, topology)  # [N, K, 3]
    O_j = gather_nodes(transforms.orientations, topology)  # [N, K, 3, 3]

    # Invert T_i
    inverse_t_i, inverse_O_i = jax.vmap(invert_transform)(t_i, O_i)

    inverse_t_i = jnp.broadcast_to(jnp.expand_dims(inverse_t_i, axis=1), shape=t_j.shape)
    inverse_O_i = jnp.broadcast_to(jnp.expand_dims(inverse_O_i, axis=1), shape=O_j.shape)

    # Compute T_ij from inverse_T_i and T_j
    composer = jax.vmap(jax.vmap(compose_transforms))
    T_ij = composer(Transforms(inverse_t_i, inverse_O_i),
                    Transforms(t_j, O_j))
    # "This is how you would go from the real position of i to my current position" - said by edge j
    return T_ij
