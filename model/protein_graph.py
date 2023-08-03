"""Random graph generation for protein graphs. The Chroma architecture uses random graph neural networks to constrain
computational costs to O(N) for a system of N atoms. This module defines the random graph generation process. """
import jax.numpy as jnp
import jax
import numpy as np


def gather_edges(features, topology) -> jnp.array:
    """Utility function that extracts relevant edge features from "features" given graph topology. This function is
    written for a single example. If used with the batch dimension, it should be jax.vmap transformed.
    Features [N,N,C] at Neighbor indices [N,K] => Neighbor features [N,K,C]
    Args:
        features: an array of shape [N, N, C] where N is the number of nodes and C is the number of channels
        topology: an array of shape [N, K] where K indicates the number of edges and the row at the ith index gives a
        list of K edges where the elements encode the indices of the jth node
    Returns: an array of shape [N, K, C] where the elements are gathered from features
            [N,N,C] at topology [N,K] => edge features [N,K,C]
    (unit-tested)"""
    N, N, C = features.shape
    _, K = topology.shape
    neighbours = jnp.broadcast_to(jnp.expand_dims(topology, axis=-1), shape=(N, K, C))  # [N, K]=> [N, K, 1]=> [N, K, C]
    edge_features = jnp.take_along_axis(features, indices=neighbours, axis=1)
    return edge_features


def gather_nodes(features, topology) -> jnp.array:
    """Utility function that extracts relevant node features from "features" given graph topology. This function is
    written for a single example. If used with the batch dimension, it should be jax.vmap transformed.
    Features [N,C] at Neighbor indices [N,K] => [N,K,C]
    Args:
        features: an array of shape [N, C] where N is the number of nodes and C is the number of channels
        topology: an array of shape [N, K] where K indicates the number of edges and the row at the ith index gives a
        list of K edges where the elements encode the indices of the jth node
    Returns: an array of shape [N, K, C] where the elements are gathered from features
             [N,C] at topology [N,K] => node features [N,K,C]
    (unit-tested)"""
    return jnp.take(features, topology, axis=0)


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
    broad_x = jnp.broadcast_to(x, shape=(n_atoms, n_atoms, 3))  # cheap algorithmic broadcasting, no memory overhead
    broad_x_T = jnp.transpose(broad_x, axes=(1, 0, 2))  # transposing the first two axes but leaving xyz
    dists = jnp.sqrt(jnp.sum((broad_x - broad_x_T) ** 2, axis=2))  # symmetric distance matrix
    return dists


def extract_carbon_alpha_coordinates(backbone_coordinates) -> jnp.array:
    """Extracts CA coordinates from all atom backbone coordinates.
    Args:
        backbone_coordinates: backbone coordinates for the protein of shape [N_res, 4, 3]. The backbone atoms should be
        arranged in order (N, CA, C, O)
    Returns:
        an array of shape [N_res, 3] containing the coordinates of only CA atoms
    (unit-tested)
    """
    return jnp.squeeze(backbone_coordinates[:, 1:2, :], axis=1)


def sample_random_graph(key, backbone_coordinates, temperature=1.0) -> jnp.array:
    """This method samples a stochastic graph topology given backbone coordinates of a protein. The sampling follows
    the conventions in the Chroma Paper: the top 20 nearest neighbours + 40 inverse cubic distance attachment
    propensity. For details, see Algorithm 1 and Supplementary Figure 3 in the Chroma Paper. Note: this method is
    written for a single example.
    Args:
        key: random PRNG key
        backbone_coordinates: backbone coordinates for the protein of shape [N_res, 4, 3]. The backbone atoms should be
        arranged in order (N, CA, C, O)
        temperature: temperature for sampling (used as softmax temperature)
    Returns:
        a graph topology represented by an array of shape [N_res, K] where K = 60, corresponding to the 60 edges.
        The elements of the array encode residue indices. For instance, topology[i] = [0, 5, ..., 231] suggests that
        the (i + 1)th residue in the protein is connected to the (0+1)st, (5+1)th, and (231 + 1)nd residues. The actual
        residues in the protein are indexed starting from 0 when represented in arrays.
    TODO: measure exact probabilities for cubic sampling
    """
    ca_coordinates = extract_carbon_alpha_coordinates(backbone_coordinates)
    distances = get_internode_distances(ca_coordinates)  # distances.shape == (N_res, N_res)
    # First get top 20 nearest neighbours (unit-tested)
    argsorted_distances = jnp.argsort(distances, axis=1)
    k_nearest = 20
    k_cubic = 40
    top_k_nearest_neighbours = argsorted_distances[:, :k_nearest]  # top_k_nearest_neighbours.shape == (N_res, 20)

    # Sample rest with inverse cubic distance
    distances = custom_put_along_axis(distances, top_k_nearest_neighbours, jnp.inf, axis=1)  # mask with inf distances
    remaining_distances = distances[:, k_nearest:]
    uniform_noise = jax.random.uniform(key, shape=remaining_distances.shape, minval=0.0, maxval=1.0)
    gumbel_perturbed = (1.0 / temperature) * (-3 * jnp.log(remaining_distances)) - jnp.log(
        -jnp.log(uniform_noise))  # TopK Gumbel-Max Trick (See Kool et al. 2019 for details)
    gumbel_perturbed_argsorted = jnp.argsort(gumbel_perturbed, axis=1)
    top_k_cubic_samples = gumbel_perturbed_argsorted[:, -k_cubic:] + 20  # top_k_cubic indices

    return jnp.concatenate((top_k_nearest_neighbours, top_k_cubic_samples), axis=1)


def custom_put_along_axis(arr, indices, values, axis):
    """This method implements np.put_along_axis function in Jax. It would be much easier if Jax simply
    implemented this function.
    Essentially copied the code from:
    https://github.com/numpy/numpy/blob/4adc87dff15a247e417d50f10cc4def8e1c17a03/numpy/lib/shape_base.py#L29
    Parameters
    ----------
    arr : ndarray (Ni..., M, Nk...)
        Destination array.
    indices : ndarray (Ni..., J, Nk...)
        Indices to change along each 1d slice of `arr`. This must match the
        dimension of arr, but dimensions in Ni and Nj may be 1 to broadcast
        against `arr`.
    values : array_like (Ni..., J, Nk...)
        values to insert at those indices. Its shape and dimension are
        broadcast to match that of `indices`.
    axis : int
        The axis to take 1d slices along. If axis is None, the destination
        array is treated as if a flattened 1d view had been created of it.

    """

    def _make_along_axis_idx(in_arr_shape, in_indices, in_axis):
        # compute dimensions to iterate over
        if not np.core.numeric.issubdtype(in_indices.dtype, np.core.numeric.integer):
            raise IndexError("`indices` must be an integer array")
        if len(in_arr_shape) != in_indices.ndim:
            raise ValueError("`indices` and `arr` must have the same number of dimensions")
        shape_ones = (1,) * in_indices.ndim
        dest_dims = list(range(in_axis)) + [None] + list(range(in_axis + 1, in_indices.ndim))

        # build a fancy index, consisting of orthogonal aranges, with the
        # requested index inserted at the right location
        fancy_index = []
        for dim, n in zip(dest_dims, in_arr_shape):
            if dim is None:
                fancy_index.append(in_indices)
            else:
                ind_shape = shape_ones[:dim] + (-1,) + shape_ones[dim + 1:]
                fancy_index.append(np.core.numeric.arange(n).reshape(ind_shape))

        return tuple(fancy_index)

    # normalize inputs
    if axis is None:
        arr = arr.flat
        axis = 0
        arr_shape = (len(arr),)  # flatiter has no .shape
    else:
        # axis = normalize_axis_index(axis, arr.ndim)
        arr_shape = arr.shape

    # use the fancy index
    arr = arr.at[tuple(_make_along_axis_idx(arr_shape, indices, axis))].set(values)
    return arr


# -----------------------------------------------------------------------------
# Visualization of the Protein Graph
# -----------------------------------------------------------------------------


def visualize_connections(pdb_path, residue_pairs, cylinder_color='red', sphere_color='red', radius=0.2,
                          sphere_radius=0.8):
    """A method that visualizes connections between residue pairs given residue pairs and a pdb file.
    Args:
        pdb_path: the path for the pdb file
        residue_pairs: a list of tuples (res1, res2) that indicates which residues are connected to each other
        cylinder_color: the connections are added as cylinders in py3Dmol, the color of cylinders
        sphere_color: at the end of connections, there is a sphere to indicate the residue, the color of spheres
        radius: the radius of connecting cylinder
        sphere_radius: the radius of spheres indicating the residues
    Credit: GPT-4"""

    # Private method to add connections
    def _add_connections(in_viewer, in_residue_pairs):
        """# Function to add connections between residues in py3Dmol."""
        for pair in in_residue_pairs:
            # Add cylinder
            in_viewer.addCylinder({'start': {'resi': pair[0], 'chain': 'A', 'atom': 'CA'},
                                   'end': {'resi': pair[1], 'chain': 'A', 'atom': 'CA'},
                                   'color': cylinder_color, 'radius': radius})
            # Add spheres at the ends of the cylinder
            in_viewer.addSphere({'center': {'resi': pair[0], 'chain': 'A', 'atom': 'CA'},
                                 'color': sphere_color, 'radius': sphere_radius})
            in_viewer.addSphere({'center': {'resi': pair[1], 'chain': 'A', 'atom': 'CA'},
                                 'color': sphere_color, 'radius': sphere_radius})

    # Read the PDB file content
    with open(pdb_path, 'r') as f:
        pdb_data = f.read()

    # Create the py3Dmol viewer and add the PDB data
    viewer = py3Dmol.view()
    viewer.addModel(pdb_data, 'pdb')

    # Set protein color to gray
    viewer.setStyle({'cartoon': {'color': 'gray'}})

    # Add connections between the residues
    _add_connections(viewer, residue_pairs)

    # Render the viewer
    viewer.zoomTo()
    viewer.show()
