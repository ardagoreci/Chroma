"""Ops for all atom representations.

 - Converting coordinates to backbone frames
 - Converting backbone frames back to coordinates
 - Methods to measure the structural violations between residues such as
   violation of the chain constraint and steric clashes between residues.
 - Frame Aligned Point Error (FAPE) loss
"""
import jax
import jax.numpy as jnp
from model import r3, protein_graph
from common import residue_constants
from typing import NamedTuple, Tuple, Optional


def coordinates_to_backbone_frames(
        backbone_coordinates: jnp.ndarray,  # (..., 4, 3)
) -> r3.Rigids:
    """Computes the backbone frames given the coordinates of the 4 backbone atoms: N, CA, C, O."""
    N = r3.vecs_from_tensor(backbone_coordinates[..., 0, :])  # N atom
    CA = r3.vecs_from_tensor(backbone_coordinates[..., 1, :])  # Ca
    C = r3.vecs_from_tensor(backbone_coordinates[..., 2, :])  # C
    return r3.rigids_from_3_points(x1=N, x2=CA, x3=C)


def backbone_frames_to_coordinates(
        backbone_frames: r3.Rigids,  # (N)
) -> jnp.ndarray:  # (N, 4, 3)
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

    # Expand dims, r3.Rigids with shape (N) to (N, 1)
    map_atoms_to_global = jax.tree_map(lambda x: x[:, None], backbone_frames)

    # Gather the literature position of GLY
    # r3.Vecs with shape (..., 4)
    lit_positions = gather_literature_position("GLY", (backbone_frames.trans.x.shape[0], 4))

    # Transform each atom from its local frame to the global frame.
    # r3.Vecs with shape (..., 4)
    pred_positions = r3.rigids_mul_vecs(map_atoms_to_global, lit_positions)
    return r3.vecs_to_tensor(pred_positions)


def gather_literature_position(
        residue_name: str,
        shape: Tuple
) -> r3.Vecs:
    """Gathers the literature position of a given residue and returns it as r3.Vecs object
    whose elements are in the given range."""
    positions = residue_constants.rigid_group_atom_positions[residue_name]
    N, CA, C, O = positions[0][2], positions[1][2], positions[2][2], positions[3][2]  # extract coordinates
    local_coordinates = jnp.stack([jnp.array(N),
                                   jnp.array(CA),
                                   jnp.array(C),
                                   jnp.array(O)], axis=0)  # local_coordinates.shape == (4, 3)

    # Broadcast local coordinates and create Vecs object
    x = jnp.broadcast_to(jnp.expand_dims(local_coordinates[:, 0], axis=0), shape)
    y = jnp.broadcast_to(jnp.expand_dims(local_coordinates[:, 1], axis=0), shape)
    z = jnp.broadcast_to(jnp.expand_dims(local_coordinates[:, 2], axis=0), shape)
    return r3.Vecs(x, y, z)


def compute_pairwise_frames(
        backbone_frames: r3.Rigids,  # (N)
        topology: jnp.ndarray  # (N, K)
) -> r3.Rigids:
    """Computes pairwise geometries given backbone frames and graph topology."""
    # 1. Gather current transforms of edge frames
    T_j = jax.tree_map(lambda x: protein_graph.gather_nodes(x, topology), backbone_frames)

    # 2. Invert T_i (broadcast to right shape)
    inverse_T_i = r3.invert_rigids(backbone_frames)
    inverse_T_i = jax.tree_map(lambda x: jnp.broadcast_to(x[:, None], shape=topology.shape),
                               inverse_T_i)

    # 3. Compute T_ij from inverse_T_i and T_j
    T_ij = r3.rigids_mul_rigids(inverse_T_i, T_j)

    # T_ij: "This is how you would go from the real position of i to my current position"
    # - said by edge j
    return T_ij


def frame_aligned_point_error(
        pred_frames: r3.Rigids,  # shape (num_frames)
        target_frames: r3.Rigids,  # shape (num_frames)
        pred_positions: r3.Vecs,  # shape (num_positions)
        target_positions: r3.Vecs,  # shape (num_positions)
        # length_scale: float = 10.0,  # 10 Angstroms
        l1_clamp_distance: Optional[float] = None,
        epsilon=1e-4
) -> jnp.ndarray:  # shape ()
    """Measure point error under different alignments.

    Jumper et al. (2021) Suppl. Alg. 28 "computeFAPE"
    Computes error between two structures with B points under A alignments derived
    from the given pairs of frames.
    Args:
        pred_frames: num_frames reference frames for 'pred_positions'.
        target_frames: num_frames reference frames for 'target_positions'.
        pred_positions: num_positions predicted positions of the structure.
        target_positions: num_positions target positions of the structure.
        length_scale: length scale to divide loss by.
        l1_clamp_distance: Distance cutoff on error beyond which gradients will
          be zero.
        epsilon: small value used to regularize denominator for masked average.
    Returns:
        Masked Frame Aligned Point Error.
    """

    # Compute array of predicted positions in the predicted frames.
    # r3.Vecs (num_frames, num_positions)
    local_pred_pos = r3.rigids_mul_vecs(
        jax.tree_map(lambda r: r[:, None], r3.invert_rigids(pred_frames)),
        jax.tree_map(lambda x: x[None, :], pred_positions))

    # Compute array of target positions in the target frames.
    # r3.Vecs (num_frames, num_positions)
    local_target_pos = r3.rigids_mul_vecs(
        jax.tree_map(lambda r: r[:, None], r3.invert_rigids(target_frames)),
        jax.tree_map(lambda x: x[None, :], target_positions))

    # Compute errors between the structures
    # jnp.ndarray (num_frames, num_positions)
    error_dist = jnp.sqrt(
        r3.vecs_squared_distance(local_pred_pos, local_target_pos)
        + epsilon)

    # Normalized mean
    if l1_clamp_distance:
        error_dist = jnp.clip(error_dist, a_min=0, a_max=l1_clamp_distance)
    fape = jnp.mean(error_dist) / 10.0  # scale by the length scale
    return fape

