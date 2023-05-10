"""Definition of the Chroma Model. The model takes in Noisy Coordinates [B, N, 4, 3] of the backbone atoms and
outputs Denoised Coordinates [B, N, 4, 3]. The entire model is end-to-end trainable.
The main components include:
    1. Graph Sampling to create a random graph (handled by protein_graph.py module, not trainable)
    2. Graph Featurization
    3. Graph Neural Network Layer
    4. Inter-residue Geometry Prediction
    5. Backbone Solver
This module includes all components that are trainable.
"""
# Dependencies
import flax
import jax.numpy as jnp
import jax
from flax import linen as nn
from typing import Tuple, NamedTuple
from protein_graph import sample_random_graph


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
    N, C = features.shape
    _, K = topology.shape
    # Flatten and broadcast [N,K] => [NK, 1] => [NK,C]
    neighbours_flat = jnp.reshape(topology, (N * K, 1))
    neighbours_flat = jnp.broadcast_to(neighbours_flat, shape=(N * K, C))
    # Take and repack
    neighbour_features = jnp.take_along_axis(features, indices=neighbours_flat, axis=0)
    neighbour_features = jnp.reshape(neighbour_features, (N, K, C))
    return neighbour_features


def cat_neighbours_nodes(h_nodes, h_edges, topology) -> jnp.array:
    """Utility function that concatenates node embeddings with that of its neighbours given graph topology. Concatenate ij with j.
    This function is written for a single example. If used with the batch dimension, it should be jax.vmap transformed.
    Args:
        h_nodes: node embeddings of shape [N, C]
        h_edges: edge embeddings of shape [N, K, C]
        topology: topology array of shape [N, K]
    Returns: concatenated features of shape [N, K, 2*C]
    (unit-tested)"""
    h_nodes = gather_nodes(h_nodes, topology)
    h_nn = jnp.concatenate([h_nodes, h_edges], axis=-1)
    return h_nn


def compute_distances(a, b) -> jnp.array:
    """A function that computes an inter-atomic distance matrix given two arrays a and b encoding the distances.
    Args:
        a: an array of shape [N, 3]
        b: another array of shape [N, 3]
    """
    n_atoms, _ = a.shape
    broad_a = jnp.broadcast_to(a, shape=(n_atoms, n_atoms, 3))  # cheap algorithmic broadcasting, no memory overhead
    broad_b = jnp.broadcast_to(b, shape=(n_atoms, n_atoms, 3))

    broad_b_T = jnp.transpose(broad_b, axes=(1, 0, 2))  # transposing the first two axes but leaving xyz
    dists = jnp.sqrt(jnp.sum((broad_a - broad_b_T) ** 2, axis=2)).T  # symmetric distance matrix
    return dists


class PositionalEncodings(nn.Module):
    """A module that implements relative positional encodings as edge features. Given an edge between two residues i
    and j, the relative positional encoding encodes the distance between i and j in the primary amino acid sequence.
    In the ProteinMPNN work, the relative positional encoding was capped at ±32 residues.

    num_embeddings: the number of positional embeddings
    max_relative_features: the cap for distance in primary sequence, defaults to 32
    """
    num_embeddings: int
    max_relative_feature: int = 32

    @nn.compact
    def __call__(self, offset, mask):
        """
        Args:
            offset: array of shape [B, N_res, K] encoding the primary sequence offset
            mask: the binary feature array of shape [B, N_res, K] encoding whether the i and j residues forming the edge
            are in the same chain or in different chains. An entry is equal to 1.0 if they are from the same chain, 0.0
            if not. An additional token is added if residues are in different chains (hence the +1).
        Returns:
            positional encodings of shape [B, N_res, K, num_embeddings]
        (unit-tested)"""
        d = jax.lax.clamp(x=offset + self.max_relative_feature,
                          min=0, max=2 * self.max_relative_feature) * mask + (1 - mask) * (
                    2 * self.max_relative_feature + 1)
        d_onehot = jax.nn.one_hot(d, 2 * self.max_relative_feature + 1 + 1)  # d_onehot.shape == (B, N, K, 32*2+1+1)
        # input features size: 2*max_relative_feature+1+1
        # output features size: num_embeddings
        E = nn.Dense(self.num_embeddings)(d_onehot)  # Embed with trainable weights
        return E


# noinspection PyAttributeOutsideInit
class ProteinFeatures(nn.Module):
    """A module that creates node and edge features given protein backbone coordinates and the graph topology.

    edge_features_dim: the dimensionality of edge features
    node_features_dim: the dimensionality of node features (will match edge features)
    """
    edge_features_dim: int
    node_features_dim: int
    num_positional_embeddings: int = 16
    num_rbf: int = 16
    num_chain_embeddings: int = 16

    def setup(self):
        self.pos_embeddings = PositionalEncodings(self.num_positional_embeddings)
        self.edge_embeddings = nn.Dense(self.edge_features_dim)
        self.layer_norm = nn.LayerNorm()

    def __call__(self, coordinates, topologies):  # X, mask, residue_idx, chain_labels
        return jax.vmap(self.single_example_forward)(coordinates, topologies)

    def _rbf(self, D) -> jnp.array:
        """This function computes a number of Gaussian radial basis functions between 2 and 22 Angstroms to
        effectively encode the distance between residues. Note: this function is written for a single example.
        Args:
            D: an array encoding the distances of shape [N, K] where N is the number of residues, K is the
            number of edges
        Returns: an array of shape [N, K, self.num_rbf] encoding distances in the form of radial basis functions

        TODO: this function was written for a nearest neighbour approach. 2 and 22 Angstroms are likely completely
         fine for that purpose, as it is unlikely that your nearest neighbours are further away than 22 Angstroms.
         However, when I am doing the random graph neural networks with connections to atoms that can be really far
         away, this might not cut it. I might have to expand the range and add more radial basis functions - should be
         a simple ratio calculation given the desired range.
        """
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = jnp.linspace(D_min, D_max, D_count)
        D_mu = D_mu.reshape((1, 1, -1))  # (1, 1, 1, -1) for batched version of function
        D_sigma = (D_max - D_min) / D_count
        D_expand = jnp.expand_dims(D, -1)
        RBF = jnp.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        return RBF

    def _get_rbf(self, A, B, topology) -> jnp.array:
        """Computes all to all distances between A and B, then gathers edge distances given topology. Encodes those
        distances with Gaussian radial basis functions. Note: this function is written for a single example.
        Args:
            A: coordinate array of shape [N, 3]
            B: another coordinate array of shape [N, 3]
            topology: graph topology of shape [N, K]
        Returns: array of shape [N, K, self.num_rbf] encoding edge distances in Gaussian RBFs.
        """
        D_A_B = compute_distances(A, B)  # [N, N]
        D_A_B_neighbours = gather_edges(D_A_B[:, :, None], topology)[:, :, 0]  # [N, K]
        RBF_A_B = self._rbf(D_A_B_neighbours)
        return RBF_A_B

    def single_example_forward(self, coordinates, topology) -> jnp.array:  # X, mask, residue_idx, chain_labels
        """The call function written for a single example. It will be jax.vmap transformed in __call__ function.
        Args:
            coordinates: [N, B, 3]
            topology: [N, K]
        Returns: a tuple of edge features with shape [N, K, self.edge_features_dim] and node features
                 [N, self.node_features_dim]. The node features are zeros.
        """
        N = coordinates[:, 0, :]
        Ca = coordinates[:, 1, :]
        C = coordinates[:, 2, :]
        O = coordinates[:, 3, :]

        # Compute virtual Cß coordinates
        b = Ca - N
        c = C - Ca
        a = jnp.cross(b, c)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca

        # Add N, Ca, C, O, Cb distances
        RBF_all = [self._get_rbf(Ca, Ca, topology), self._get_rbf(N, N, topology), self._get_rbf(C, C, topology),
                   self._get_rbf(O, O, topology), self._get_rbf(Cb, Cb, topology), self._get_rbf(Ca, N, topology),
                   self._get_rbf(Ca, C, topology), self._get_rbf(Ca, O, topology), self._get_rbf(Ca, Cb, topology),
                   self._get_rbf(N, C, topology), self._get_rbf(N, O, topology), self._get_rbf(N, Cb, topology),
                   self._get_rbf(Cb, C, topology), self._get_rbf(Cb, O, topology), self._get_rbf(O, C, topology),
                   self._get_rbf(N, Ca, topology), self._get_rbf(C, Ca, topology), self._get_rbf(O, Ca, topology),
                   self._get_rbf(Cb, Ca, topology), self._get_rbf(C, N, topology), self._get_rbf(O, N, topology),
                   self._get_rbf(Cb, N, topology), self._get_rbf(C, Cb, topology), self._get_rbf(O, Cb, topology),
                   self._get_rbf(C, O, topology)]
        RBF_all = jnp.concatenate(RBF_all, axis=-1)  # concatenate along last axis, shape: [N, K, 25*self.num_rbf]
        # Compute distances in primary amino acid sequence and positional embeddings
        offset = topology - jnp.arange(coordinates.shape[0])[:, None]  # [N, K]
        positional_embedding = self.pos_embeddings(offset, mask=1.0)  # mask hardcoded for now
        E = jnp.concatenate([positional_embedding, RBF_all], axis=-1)  # concatenate along last axis
        E = self.edge_embeddings(E)  # embed with linear layer
        node_zeros = jnp.zeros((coordinates.shape[0], self.node_features_dim))  # zeros as node features
        return node_zeros, E


# Graph Neural Network Modules
class PositionWiseFeedForward(nn.Module):
    """A module that applies position-wise feedforward operation as in the Transformer Paper. (unit-tested)"""
    num_hidden: int
    num_ff: int

    @nn.compact
    def __call__(self, h_V):
        h = jax.nn.gelu(nn.Dense(self.num_ff)(h_V))
        h = nn.Dense(self.num_hidden)(h)
        return h


# noinspection PyAttributeOutsideInit
class MPNNLayer(nn.Module):
    """Implements a message passing neural network layer with node and edge updates."""
    node_embedding_dim: int
    edge_embedding_dim: int
    edge_mlp_hidden_dim: int
    node_mlp_hidden_dim: int
    dropout: float = 0.1
    scale: int = 60

    def setup(self):
        # node message MLP
        self.node_mlp = nn.Sequential([
            nn.Dense(self.node_mlp_hidden_dim),
            jax.nn.gelu,
            nn.Dense(self.node_mlp_hidden_dim),
            jax.nn.gelu,
            nn.Dense(self.node_embedding_dim)
        ])
        # edge update MLP
        self.edge_mlp = nn.Sequential([
            nn.Dense(self.edge_mlp_hidden_dim),
            jax.nn.gelu,
            nn.Dense(self.edge_mlp_hidden_dim),
            jax.nn.gelu,
            nn.Dense(self.edge_embedding_dim)
        ])
        # Normalization Layers
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.norm3 = nn.LayerNorm()

        # Dropout
        self.dropout_layer = nn.Dropout(rate=self.dropout, deterministic=False)

        # Position-wise feedforward (acts as node update MLP)
        self.dense = PositionWiseFeedForward(num_hidden=self.node_embedding_dim,
                                             num_ff=self.node_mlp_hidden_dim)

    def __call__(self, h_V, h_E, topology):
        """Parallel computation of full MPNN layer
        Args:
            h_V: node activations of shape [B, N, V]
            h_E: edge activations of shape [B, N, K, E]
        """
        B, N, K, _ = h_E.shape
        # Concat i, ij, j
        h_EV = jax.vmap(cat_neighbours_nodes)(h_V, h_E, topology)
        h_V_broad = jnp.broadcast_to(jnp.expand_dims(h_V, axis=-2),
                                     shape=(B, N, K, self.node_embedding_dim))  # expand and broadcast
        h_EV = jnp.concatenate([h_V_broad, h_EV], axis=-1)

        # Compute node Message with MLP
        h_message = self.node_mlp(h_EV)
        # Sum to i
        dh = jnp.sum(h_message, axis=-2) / self.scale
        h_V = self.norm1(h_V + self.dropout_layer(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout_layer(dh))

        # Edge Updates
        h_EV = jax.vmap(cat_neighbours_nodes)(h_V, h_E, topology)
        h_V_broad = jnp.broadcast_to(jnp.expand_dims(h_V, axis=-2),
                                     shape=(B, N, K, self.node_embedding_dim))  # expand and broadcast
        h_EV = jnp.concatenate([h_V_broad, h_EV], axis=-1)

        # Compute edge update with MLP
        h_update = self.edge_mlp(h_EV)
        h_E = self.norm3(h_E + self.dropout_layer(h_update))
        return h_V, h_E


class BackboneGNN(nn.Module):
    """Implements the Chroma backbone GNN that takes in backbone coordinates and outputs a graph topology,
    node embeddings and edge embeddings."""
    node_embedding_dim: int = 512  # the default values are those of Backbone Network A in Chroma
    edge_embedding_dim: int = 256
    node_mlp_hidden_dim: int = 512
    edge_mlp_hidden_dim: int = 128
    num_gnn_layers: int = 12
    dropout: float = 0.1  # dropout rate

    @nn.compact
    def __call__(self, key, noisy_coordinates):
        """
        h_V.shape == (B, N, node_embedding_dim) and h_E.shape == (B, N, K, edge_embedding_dim)
        Args:
            key: random PRNGKey
            noisy_coordinates: noisy coordinates of shape (B, N, 4, 3)
        Returns:
        """
        B, N, _, _ = noisy_coordinates.shape
        keys = jax.random.split(jax.random.PRNGKey(1), num=B)[:, 0]  # keys.shape == (B,2)

        # Graph sampling
        topologies = jax.vmap(sample_random_graph)(keys, noisy_coordinates)

        # Graph featurization
        h_V, h_E = ProteinFeatures(edge_features_dim=self.edge_embedding_dim,
                                   node_features_dim=self.node_embedding_dim)(noisy_coordinates, topologies)

        # MPNN layers
        for _ in range(self.num_gnn_layers):
            h_V, h_E = MPNNLayer(node_embedding_dim=self.node_embedding_dim,
                                 edge_embedding_dim=self.edge_embedding_dim,
                                 edge_mlp_hidden_dim=self.edge_mlp_hidden_dim,
                                 node_mlp_hidden_dim=self.node_mlp_hidden_dim,
                                 dropout=self.dropout)(h_V, h_E, topologies)

        return h_V, h_E, topologies


# Inter-residue Geometry Prediction
class AnisotropicConfidence(NamedTuple):
    """A more sophisticated anisotropic confidence that is parameterized by separating the precision term w of
    isotropic confidence into three components: rotational precision and two components for position: radial distance
    precision and lateral precision. The radial and lateral precision are eigenvalues of the full 3x3 precision
    matrix P_ij for translation errors. For details, see section E.2 in Chroma Paper.
    rotational_precision: [..., 1]
    radial_distance_precision: [..., 1]
    lateral_precision: [..., 1]
    """
    rotational_precision: jnp.array
    radial_distance_precision: jnp.array
    lateral_precision: jnp.array


class Transforms(NamedTuple):
    """A residue wise transform that stores a [..., 3, 3] orientation matrix and a [..., 3] translation vector."""
    translations: jnp.array
    orientations: jnp.array


class PairwiseGeometries(NamedTuple):
    """A pairwise geometry that stores Transform objects and associated confidence values."""
    transforms: Transforms
    confidences: jnp.array  # this will be changed when switching to Backbone Network B


# noinspection PyAttributeOutsideInit
class PairwiseGeometryPrediction(nn.Module):
    """Implements the inter-residue geometry prediction that predicts pairwise transforms and confidences given node
    embeddings and edge embeddings. """
    num_confidence_values: int = 1  # variable depending on whether isotropic or anisotropic confidences are used

    def setup(self):
        self.linear = nn.Dense(3+3+self.num_confidence_values)

    def __call__(self, h_V, h_E):
        batch_pairwise_geometries = jax.vmap(self.backbone_update_with_confidence)(h_E)
        return batch_pairwise_geometries

    def backbone_update_with_confidence(self, pair_embeddings) -> PairwiseGeometries:
        """Given an embedding, computes a quaternion for the rotation and a vector for the translation. Given an
        embedding, a linear layer predicts a vector for the translation and three additional components that define the
        Euler axis. See AlphaFold Supplementary information section 1.8.3, Algorithm 23 "Backbone update".
        This function is written for a single example.
        Args:
            pair_embeddings: [N, K, C]
        Returns:
            a tuple of (translations, rotations, confidences) where:
            *translations: an array of shape [N, K, 3] encoding a translation in 3D space (units of nanometers)
            *rotations: an array of shape [N, K, 3, 3] encoding rotation matrices along the last two axes of shape
                        [3,3] that are derived from quaternion predictions from the neural network.
            *confidences: an array of shape [N, K, self.num_confidence_values] encoding the confidence values
        """
        output = self.linear(pair_embeddings)  # [N,K, self.num_confidence_values+3+3]
        confidences = output[:, :, :self.num_confidence_values]
        translations = output[:, :, self.num_confidence_values:self.num_confidence_values+3]
        # Compute Rotations
        get_rotation_matrix_fn = jax.vmap(jax.vmap(PairwiseGeometryPrediction._get_rotation_matrix))  # pretend N and
        # K dimensions are batch dim to vectorize function with jax.vmap
        rotations = get_rotation_matrix_fn(output[:, :, self.num_confidence_values+3:])
        # Return pairwise geometries
        transforms = Transforms(translations=translations, orientations=rotations)
        pairwise_geometries = PairwiseGeometries(transforms=transforms, confidences=confidences)
        return pairwise_geometries

    @staticmethod
    def _get_rotation_matrix(array):
        """Converts (non-unit) quaternion to rotation matrix.
        For details, see Wikipedia https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation.
        Args:
            array: encodes three components and has shape [3,] that define the Euler axis. These components are
            predicted by the neural network.
        Returns: a rotation matrix of shape [3, 3] computed from the quaternion. This procedure guarantees a valid
                 normalized quaternion and favours small rotations over large rotations.

        (unit-tested, the angles make sense and definitely have a relationship with SciPy quaternion matrix conversion,
        for a learning system, it should not be a problem.)"""
        b, c, d = array[0], array[1], array[2]
        quaternion = jnp.array([1, b, c, d])  # the first component of non-unit quaternion is fixed to 1.
        (a, b, c, d) = quaternion / jnp.sqrt(jnp.sum(quaternion**2))
        rot = jnp.array([[(a**2 + b**2 - c**2 - d**2), (2*b*c - 2*a*d), (2*b*d + 2*a*c)],
                         [(2*b*c + 2*a*d), (a**2 - b**2 + c**2 - d**2), (2*c*d - 2*a*b)],
                         [(2*b*d - 2*a*c), (2*c*d + 2*a*b), (a**2 - b**2 - c**2 + d**2)]])
        return rot

# Backbone solver
class BackboneSolver(nn.Module):
    pass
