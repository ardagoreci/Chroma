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
import jax.random
from flax import linen as nn
from model.protein_graph import sample_random_graph, gather_edges
from model.geometry import *
from model import r3, all_atom


def cat_neighbours_nodes(h_nodes, h_edges, topology) -> jnp.array:
    """Utility function that concatenates node embeddings with that of its neighbours given graph topology. Concatenate
    ij with j. This function is written for a single example. If used with the batch dimension, it should be
    jax.vmap transformed.
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


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    Args:
        timesteps: a 1-D Tensor of B indices, one per batch element.
        dim: the dimension of the output
        max_period: controls the minimum frequency of the embeddings.
    Returns:
        a Tensor of shape (B, dim) of positional embeddings
    """
    timesteps = timesteps * 1000  # Convert [0,1] scale to [0,1000]
    half = dim // 2
    freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(0, half, dtype=jnp.float32) / half)
    args = timesteps[:, None] * freqs[None]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
        embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
    return embedding


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
    num_chain_embeddings: int = 32

    def setup(self):
        self.pos_embeddings = PositionalEncodings(self.num_positional_embeddings)
        self.edge_embeddings = nn.Dense(self.edge_features_dim)
        self.norm_edges = nn.LayerNorm()

    def __call__(
            self,
            coordinates: jnp.ndarray,
            transforms: r3.Rigids,
            topologies: jnp.ndarray
            # X, mask, residue_idx, chain_labels
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return jax.vmap(self.single_example_forward)(coordinates, transforms, topologies)

    def _rbf(self, D) -> jnp.ndarray:
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
        D_min, D_max, D_count = 2., 42., self.num_rbf
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

    def _get_pairwise_frame_features(
            self,
            pairwise_frames: r3.Rigids  # shape (N, K)
    ) -> jnp.ndarray:  # shape (N, K, 9 + 3 * self.num_rbf)
        """Computes pairwise frame features (interresidue transforms). The rotation features are the flattened
        rotation matrix, and the translation features are the x, y, z axes of the translation vector encoded
        with Gaussian RBFs."""
        N, K = pairwise_frames.trans.x.shape
        rot_features = r3.rots_to_tensor_flat9(pairwise_frames.rot)  # (N, K, 9)

        # Translation features
        translations = r3.vecs_to_tensor(pairwise_frames.trans)  # (N, K, 3)
        trans_rbfs = jax.vmap(self._rbf)(translations)  # (N, K, 3, self.num_rbf)
        trans_features = trans_rbfs.reshape(N, K, -1)  # (N, K, 3 * self.num_rbf)

        # Concatenate features along last axis
        return jnp.concatenate([rot_features, trans_features], axis=-1)

    def single_example_forward(
            self,
            coordinates: jnp.ndarray,  # (N, 4, 3)
            transforms: r3.Rigids,  # (N)
            topology: jnp.ndarray  # (N, K)
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """The call function written for a single example. It will be jax.vmap transformed in __call__ function.
        Args:
            coordinates: [N, B, 3]
            transforms: r3.Rigids object describing current transforms of the protein
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

        # Compute inter-residue geometry  TODO: change required for r3
        pairwise_frames = all_atom.compute_pairwise_frames(transforms, topology)  # T_ij (N, K)
        interresidue_transforms = self._get_pairwise_frame_features(pairwise_frames)

        # Compute features and embed
        E = jnp.concatenate([positional_embedding, RBF_all, interresidue_transforms], axis=-1)
        E = self.edge_embeddings(E)  # embed with linear layer
        E = self.norm_edges(E)  # normalize edges
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
    residual_scale: float = 0.7071  # scale skip connections with 1/sqrt(2)

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

        # Dropout layers
        self.dropout1 = nn.Dropout(rate=self.dropout, deterministic=False)
        self.dropout2 = nn.Dropout(rate=self.dropout, deterministic=False)
        self.dropout3 = nn.Dropout(rate=self.dropout, deterministic=False)

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
        h_V = self.norm1(self.residual_scale * h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(self.residual_scale * h_V + self.dropout2(dh))

        # Edge Updates
        h_EV = jax.vmap(cat_neighbours_nodes)(h_V, h_E, topology)
        h_V_broad = jnp.broadcast_to(jnp.expand_dims(h_V, axis=-2),
                                     shape=(B, N, K, self.node_embedding_dim))  # expand and broadcast
        h_EV = jnp.concatenate([h_V_broad, h_EV], axis=-1)

        # Compute edge update with MLP
        h_update = self.edge_mlp(h_EV)
        h_E = self.norm3(self.residual_scale * h_E + self.dropout3(h_update))
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
    use_timestep_embedding: bool = True
    residual_scale: float = 0.7071

    @nn.compact
    def __call__(
            self,
            key: jax.random.PRNGKey,
            noisy_coordinates: jnp.ndarray,
            transforms: r3.Rigids,
            timesteps: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        h_V.shape == (B, N, node_embedding_dim) and h_E.shape == (B, N, K, edge_embedding_dim)
        Args:
            key: random PRNGKey
            noisy_coordinates: noisy coordinates of shape (B, N, 4, 3)
            transforms: r3.Rigids object describing the current transforms for protein
            timesteps: timesteps of shape [B,]
        Returns:
            node embeddings (h_V), edge embeddings (h_E), topologies
        """
        B, N, _, _ = noisy_coordinates.shape
        keys = jax.random.split(key, num=B)  # keys.shape == (B,2)

        # Graph sampling
        topologies = jax.vmap(sample_random_graph)(keys, noisy_coordinates)

        # Graph featurization
        h_V, h_E = ProteinFeatures(edge_features_dim=self.edge_embedding_dim,
                                   node_features_dim=self.node_embedding_dim)(noisy_coordinates, transforms, topologies)

        # Add timestep embedding to node embeddings
        if self.use_timestep_embedding:
            timestep_embeddings = timestep_embedding(timesteps, dim=h_V.shape[-1])  # [B, node_embedding_dim]
            h_V = h_V + jnp.broadcast_to(jnp.expand_dims(timestep_embeddings, axis=1), shape=h_V.shape)
            # broadcast and add

        # MPNN layers
        for _ in range(self.num_gnn_layers):
            h_V, h_E = MPNNLayer(node_embedding_dim=self.node_embedding_dim,
                                 edge_embedding_dim=self.edge_embedding_dim,
                                 edge_mlp_hidden_dim=self.edge_mlp_hidden_dim,
                                 node_mlp_hidden_dim=self.node_mlp_hidden_dim,
                                 dropout=self.dropout,
                                 residual_scale=self.residual_scale)(h_V, h_E, topologies)

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


class PairwiseGeometries(NamedTuple):
    """A pairwise geometry that stores Transform objects and associated confidence values."""
    pairwise_frames: r3.Rigids  # as T_ji
    # T_ji, 'This is how you would go from my position to the real position of i' - said by edge j

    # Edge wise confidence scalars that indicate how confident the network is from its prediction
    confidences: jnp.array  # this will be changed when switching to Backbone Network B


# noinspection PyAttributeOutsideInit
class PairwiseGeometryPrediction(nn.Module):
    """Implements the inter-residue geometry prediction that predicts pairwise transforms and confidences given node
    embeddings and edge embeddings.
    TODO: the get_rotation_matrix can be integrated into r3.py so that I can just call that
     instead of having it here as a static method. DeepMind made it into a quat_affine.py module.
    """
    num_confidence_values: int = 1  # variable depending on whether isotropic or anisotropic confidences are used

    def setup(self):
        self.linear = nn.Dense(3 + 3 + self.num_confidence_values)
        self.node_mlp = nn.Dense(4 * 3)  # 4 atom coordinates for each node

    def __call__(self, h_V, h_E):
        batch_pairwise_geometries = jax.vmap(self.backbone_update_with_confidence)(h_E)
        batch_node_updates = jax.vmap(self.node_coordinate_update)(h_V)
        return batch_node_updates, batch_pairwise_geometries

    def node_coordinate_update(self, node_embeddings):
        """Given node embeddings, computes a residual update for all backbone atom coordinates.
        This allows Chroma to predict all backbone atom coordinates simultaneously, and the denoising
        objective acts as a version of Noisy Nodes (Godwin et al. 2022) that regularizes deep MPNNs.
        Args:
            node_embeddings: node embeddings of shape [N, C]
        Returns:
            an array of shape [N, 4, 3] that are residual updates to backbone atom coordinates
        """
        output = self.node_mlp(node_embeddings)
        residual_updates = output.reshape((node_embeddings.shape[0], 4, 3))
        return residual_updates

    def backbone_update_with_confidence(self, pair_embeddings) -> PairwiseGeometries:
        """Given an embedding, computes a quaternion for the rotation and a vector for the translation. Given an
        embedding, a linear layer predicts a vector for the translation, three additional components that define the
        Euler axis, and a confidence value. See AlphaFold Supplementary information section 1.8.3, Algorithm 23
        "Backbone update". This function is written for a single example.
        Args:
            pair_embeddings: [N, K, C]
        Returns:
            a tuple of (translations, rotations, confidences) where:
            *translations: an array of shape [N, K, 3] encoding a translation in 3D space (units of nanometers)
            *rotations: an array of shape [N, K, 3, 3] encoding rotation matrices along the last two axes of shape
                        [3,3] that are derived from quaternion predictions from the neural network.
            *confidences: an array of shape [N, K, self.num_confidence_values] encoding the confidence values
        """
        output = self.linear(pair_embeddings)  # [N,K, 1+3+3]
        confidences = output[:, :, :self.num_confidence_values]
        translations = output[:, :, self.num_confidence_values:self.num_confidence_values + 3] * 10.0  # predict
        # translations in nanometers, convert to Angstroms

        # Compute Rotations
        get_rotation_matrix_fn = jax.vmap(jax.vmap(PairwiseGeometryPrediction._get_rotation_matrix))  # pretend N and
        # K dimensions are batch dim to vectorize function with jax.vmap
        rotations = get_rotation_matrix_fn(output[:, :, self.num_confidence_values + 3:])

        # Return pairwise geometries
        pairwise_frames = r3.Rigids(r3.rots_from_tensor3x3(rotations),
                                    r3.vecs_from_tensor(translations))
        pairwise_geometries = PairwiseGeometries(pairwise_frames, confidences)
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

        (unit-tested, the angles make sense and definitely have a relationship with SciPy quaternion matrix conversion.
        For a learning system, it should not be a problem.)"""
        b, c, d = array[0], array[1], array[2]
        quaternion = jnp.array([1, b, c, d])  # the first component of non-unit quaternion is fixed to 1.
        (a, b, c, d) = quaternion / jnp.sqrt(jnp.sum(quaternion ** 2))
        rot = jnp.array([[(a ** 2 + b ** 2 - c ** 2 - d ** 2), (2 * b * c - 2 * a * d), (2 * b * d + 2 * a * c)],
                         [(2 * b * c + 2 * a * d), (a ** 2 - b ** 2 + c ** 2 - d ** 2), (2 * c * d - 2 * a * b)],
                         [(2 * b * d - 2 * a * c), (2 * c * d + 2 * a * b), (a ** 2 - b ** 2 - c ** 2 + d ** 2)]])
        return rot


# Backbone solver
class BackboneSolver(nn.Module):
    """Implements the Backbone solver that predicts equivariant consensus structure from weighted inter-residue
    geometries. The BackboneGNN predicts a set of inter-residue geometries T_ij together with confidences w_ij. This
    module solves either fully or approximately for the consensus structure that best satisfies this set of pairwise
    predictions.
    """
    num_iterations: int = 1  # for backbone network B, num_iterations = 10

    @nn.compact
    def __call__(
            self,
            transforms: r3.Rigids,  # (B, N)
            pairwise_geometries: Tuple[r3.Rigids, jnp.ndarray],  # (B, N, K)
            topology  # (B, N, K)
    ) -> r3.Rigids:
        """
        Args:
            transforms: a r3.Rigids describing the current pose
            pairwise_geometries: a pairwise geometries object (batched arrays within)
        Returns: updated transforms
        """
        batch_update_frames = jax.vmap(BackboneSolver.update_frames)
        return batch_update_frames(transforms, pairwise_geometries, topology)

    @staticmethod
    def update_frames(
            backbone_frames: r3.Rigids,  # (N)
            pairwise_geometries: PairwiseGeometries,  # (N, K)
            topology  # (N, K)
    ) -> r3.Rigids:
        """Updates the frame of a single residue i given the pairwise geometries of its neighbours.
        The method notation is written to closely follow the notation in the Chroma Paper, section E.2.
        Note: this method will be jax.vmap transformed for the batch dimension.
        Args:
            backbone_frames: r3.Rigids object for the current transforms describing the pose (N)
            pairwise_geometries: a PairwiseGeometries object predicted by the network containing confidences
                                 and pairwise frames as a r3.Rigids object of shape (N, K)
            topology: graph topology of shape [N, K]
        Returns: updated frames
        """

        # Gather current transforms of edge frames
        T_j = jax.tree_map(lambda x: gather_nodes(x, topology), backbone_frames)  # (N, K)

        # Normalize confidences
        w_ij = jnp.squeeze(pairwise_geometries.confidences, axis=-1)  # [N, K], isotropic confidence
        p_ij = w_ij / jnp.sum(w_ij, axis=1, keepdims=True)  # sum over j, p_ij.shape == [N, K]

        # T_ji as network prediction (I hypothesize it will lead to faster learning, as inversion of T_ij is nontrivial)
        T_ji = pairwise_geometries.pairwise_frames  # r3.invert_rigids(pairwise_geometries.pairwise_frames)  # (N, K)

        #  Compose transforms: T_j and T_ji
        pred_T_i_per_edge = r3.rigids_mul_rigids(T_j, T_ji)

        #  Confidence weighted averaging over edges
        pred_T_i_averages = jax.tree_map(lambda x: jnp.sum(p_ij * x, axis=1), pred_T_i_per_edge)

        #  Project rots with SVD so they remain valid rotations
        projector = jax.vmap(BackboneSolver.proj_with_svd)  # acts on [N, 3, 3]
        rot_array = projector(r3.rots_to_tensor(pred_T_i_averages.rot))
        # TODO: the projection somehow inverts my rotations (I hadn't seen this in the previous version of the solver),
        #  I have to correct it:
        pred_T_i = r3.Rigids(rot=r3.invert_rots(r3.rots_from_tensor3x3(rot_array)),
                             trans=pred_T_i_averages.trans)

        # Return updated transforms
        return pred_T_i

    @staticmethod
    def proj_with_svd(matrix):
        """Implements a projection operator with singular value decomposition (SVD) as in the Kabsch algorithm for
        optimal RMSD decomposition. For details, see: https://en.wikipedia.org/wiki/Kabsch_algorithm
        This function is written for a single example.
        Args:
            matrix: a matrix of size [3, 3]
        Returns: a rotation matrix of shape [3, 3] derived from 'matrix' via SVD
        (unit-tested)"""
        u, s, vh = jnp.linalg.svd(matrix, full_matrices=False)  # JAX grad creates problems if full_matrices = True
        V = vh.T
        # Decide whether we need to correct our rotation matrix to ensure a right-handed coordinate system
        d = jnp.sign(jnp.linalg.det(jnp.matmul(V, u.T)))

        # Compute optimal rotation matrix
        intermediate = jnp.array([[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, d]])
        intermediate = jnp.matmul(intermediate, u.T)
        rot = jnp.matmul(V, intermediate)
        return rot


class Chroma(nn.Module):
    """The full Chroma network"""
    node_embedding_dim: int = 512  # the default values are those of Backbone Network A in Chroma
    edge_embedding_dim: int = 256
    node_mlp_hidden_dim: int = 512
    edge_mlp_hidden_dim: int = 128
    num_gnn_layers: int = 12
    dropout: float = 0.1  # dropout rate
    backbone_solver_iterations: int = 1  # this is not implemented for more than 1 yet.
    use_timestep_embedding: bool = True
    residual_scale: float = 1.0

    @nn.compact
    def __call__(self, key, noisy_coordinates, timesteps):
        """
        Args:
            key: random PRNGKey
            noisy_coordinates: noisy coordinates of shape [B, N, 4, 3]
            timesteps: timesteps of shape [B,]
        Returns:
            denoised coordinates
        """
        # Current transforms
        transforms = all_atom.coordinates_to_backbone_frames(noisy_coordinates)  # (B, N)

        # BackboneGNN
        h_V, h_E, topologies = BackboneGNN(node_embedding_dim=self.node_embedding_dim,
                                           edge_embedding_dim=self.edge_embedding_dim,
                                           node_mlp_hidden_dim=self.node_mlp_hidden_dim,
                                           edge_mlp_hidden_dim=self.edge_mlp_hidden_dim,
                                           num_gnn_layers=self.num_gnn_layers,
                                           use_timestep_embedding=self.use_timestep_embedding,
                                           dropout=self.dropout,
                                           residual_scale=self.residual_scale)(key, noisy_coordinates,
                                                                               transforms, timesteps)

        # Interresidue Geometry Prediction
        node_updates, pairwise_geometries = PairwiseGeometryPrediction()(h_V, h_E)

        # Backbone Solver
        updated_transforms = BackboneSolver()(transforms, pairwise_geometries, topologies)

        # TransformsToStructure (going back to 3D coordinates)
        denoised_coordinates = jax.vmap(all_atom.backbone_frames_to_coordinates)(updated_transforms)

        # Residual updates for all-atom prediction
        denoised_coordinates = denoised_coordinates + node_updates
        return denoised_coordinates
