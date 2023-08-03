"""Ops for all atom representations.

 - Converting coordinates to backbone frames
 - Converting backbone frames back to coordinates
 - Methods to measure the structural violations between residues such as
   violation of the chain constraint and steric clashes between residues.
 - Frame Aligned Point Error (FAPE) loss
"""
from typing import Union, Tuple

import r3
import jax.numpy as jnp
from common import residue_constants


def structure_to_transforms(
        backbone_coordinates: jnp.ndarray,  # (..., 4, 3)
) -> r3.Rigids:
    """Computes the backbone frames given the coordinates of the 4 backbone atoms: N, CA, C, O."""
    N = r3.vecs_from_tensor(backbone_coordinates[..., 0, :])  # N atom
    CA = r3.vecs_from_tensor(backbone_coordinates[..., 1, :])  # Ca
    C = r3.vecs_from_tensor(backbone_coordinates[..., 2, :])  # C
    return r3.rigids_from_3_points(x1=N, x2=CA, x3=C)


def transforms_to_structure(
        backbone_frames: r3.Rigids,
) -> r3.Vecs:  # (N, 4)
    """Computes the backbone coordinates given backbone frames.
    The coordinates are computed using the ideal bond lengths of Gly with the following values:
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
    transforms. Importantly, the carbon alpha positions are exactly the same."""

    # Gather the literature position of GLY
    # r3.Vecs with shape (..., 4)
    lit_positions = gather_literature_position("GLY", backbone_frames.trans.x.shape)  # same shape as backbone frames

    # Transform each atom from its local frame to the global frame.
    # r3.Vecs with shape (..., 4)
    pred_positions = r3.rigids_mul_vecs(backbone_frames, lit_positions)
    return pred_positions


def gather_literature_position(
        residue_name: str,
        shape: tuple
) -> r3.Vecs:
    """Gathers the literature position of a given residue and returns it as r3.Vecs object
    whose elements are in the given range."""
    positions = residue_constants.rigid_group_atom_positions[residue_name]
    N, CA, C, O = positions[0][2], positions[1][2], positions[2][2], positions[3][2]  # extract coordinates
    local_coordinates = jnp.stack([N, CA, C, O], axis=0)  # local_coordinates.shape == (4, 3)

    # Broadcast the coordinates to required shape and create a Vecs object
    broadcast_shape = shape + (3,)  # add the dims for the 3 coordinates
    x = jnp.broadcast_to(local_coordinates[:, 0], shape=broadcast_shape)
    y = jnp.broadcast_to(local_coordinates[:, 1], shape=broadcast_shape)
    z = jnp.broadcast_to(local_coordinates[:, 2], shape=broadcast_shape)
    return r3.Vecs(x, y, z)




