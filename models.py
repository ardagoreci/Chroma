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
from protein_graph import sample_random_graph


# from typing import i

# Methods for featurization
# class PositionalEncodings
# class ProteinFeatures


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
    neighbours_flat = jnp.reshape(topology, (N*K, 1))
    neighbours_flat = jnp.broadcast_to(neighbours_flat, shape=(N*K, C))
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


def compute_distances(a, b):
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

    def __call__(self, coordinates, topology):  # X, mask, residue_idx, chain_labels
        return jax.vmap(self.single_example_forward)(coordinates, topology)

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
        Returns: an array of edge features with shape [N, K, self.edge_features_dim]
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
        return E


# Graph Neural Network
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
    num_hidden: int
    num_in: int
    dropout: float = 0.1
    scale: int = 60

    def setup(self):
        # node message MLP
        self.node_mlp = nn.Sequential([
            nn.Dense(self.num_hidden),
            jax.nn.gelu,
            nn.Dense(self.num_hidden),
            jax.nn.gelu,
            nn.Dense(self.num_hidden)
        ])
        # edge update MLP
        self.edge_mlp = nn.Sequential([
            nn.Dense(self.num_hidden),
            jax.nn.gelu,
            nn.Dense(self.num_hidden),
            jax.nn.gelu,
            nn.Dense(self.num_hidden)
        ])
        # Normalization Layers
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.norm3 = nn.LayerNorm()

        # Dropout
        self.dropout = nn.Dropout(rate=self.dropout)

        # Position-wise feedforward (acts as node update MLP)
        self.dense = PositionWiseFeedForward(num_hidden=self.num_hidden,
                                             num_ff=self.num_hidden*4)  # following ProteinMPNN

    def __call__(self, h_V, h_E, topology):
        """Parallel computation of full MPNN layer
        Args:
            h_V: node activations of shape [B, N, C]
            h_E: edge activations of shape [B, N, K, C]
        """
        B, N, K, C = h_E.shape
        # Concat i, ij, j
        h_EV = jax.vmap(cat_neighbours_nodes)(h_V, h_E, topology)
        h_V_broad = jnp.broadcast_to(jnp.expand_dims(h_V, axis=-2), shape=(B, N, K, C))  # expand and broadcast
        h_EV = jnp.concatenate([h_V_broad, h_EV], axis=-1)

        # Compute node Message with MLP
        h_message = self.node_mlp(h_EV)
        # Sum to i
        dh = jnp.sum(h_message, axis=-2) / self.scale
        h_V = self.norm1(h_V + self.dropout(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout(dh))

        # Edge Updates
        h_EV = jax.vmap(cat_neighbours_nodes)(h_V, h_E, topology)
        h_V_broad = jnp.broadcast_to(jnp.expand_dims(h_V, axis=-2), shape=(B, N, K, C))  # expand and broadcast
        h_EV = jnp.concatenate([h_V_broad, h_EV], axis=-1)

        # Compute edge update with MLP
        h_update = self.edge_mlp(h_EV)
        h_E = self.norm3(h_E + self.dropout(h_update))
        return h_V, h_E

# Inter-residue Geometry Prediction
# Backbone solver
