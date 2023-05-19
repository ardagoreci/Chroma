""""""

import flax
import jax
import ml_collections
import optax
from jax import lax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax.training import checkpoints
from flax.training import common_utils
from absl import logging
from clu import metric_writers, periodic_actions
import input_pipeline
from models import Chroma
import polymer


def create_model(config):
    """Creates the Chroma model."""
    model = Chroma(node_embedding_dim=config.node_embedding_dim,
                   edge_embedding_dim=config.edge_embedding_dim,
                   node_mlp_hidden_dim=config.node_mlp_hidden_dim,
                   edge_mlp_hidden_dim=config.edge_mlp_hidden_dim,
                   num_gnn_layers=config.num_gnn_layers,
                   dropout=config.dropout)
    return model


def initialize(key, model, config, local_batch_size):
    """Utility function to initialize the model."""
    key1, key2 = jax.random.split(key, num=2)
    B, N = local_batch_size, config.crop_size
    dummy_coordinates = jnp.ones((B, N, 4, 3))
    dummy_timesteps = jnp.zeros((local_batch_size,))
    params = model.init(rngs={'params': key1, 'dropout': key2},
                        key=key,
                        noisy_coordinates=dummy_coordinates,
                        timesteps=dummy_timesteps)
    return params


def mean_squared_error(x, y):
    """Computes the element-wise mean squared error between x and y."""
    return jnp.mean(jnp.square(x - y))


def compute_metrics(x_pred, x0):
    """Returns a dictionary of metrics for the given logits and labels."""
    mse = mean_squared_error(x_pred, x0)
    return {'loss': mse}


def create_learning_rate_fn(config: ml_collections.ConfigDict):
    # TODO: implement the schedule that the authors have used.
    def _base_fn(step):
        return config.learning_rate
    return _base_fn


def prepare_tf_data(xs):
    """Convert a input batch from tf Tensors to numpy arrays."""
    local_device_count = jax.local_device_count()

    def _prepare(x):
        # Use _numpy() for zero-copy conversion between TF and NumPy.
        x = x._numpy()  # pylint: disable=protected-access
        # reshape (host_batch_size, height, width, 3) to
        # (local_devices, device_batch_size, height, width, 3)
        return x.reshape((local_device_count, -1) + x.shape[1:])

    return jax.tree_util.tree_map(_prepare, xs)


def create_input_iter(config):
    """Creates an iterator for the given dataset and split.
    Args:
        config: the config dictionary
    Returns:
        an iterator of containing batched training examples
    """
    dataset = input_pipeline.create_protein_dataset(config.crop_size,
                                                    config.batch_size)
    iterator = map(prepare_tf_data, dataset)
    return iterator


def train_step(key: jax.random.PRNGKey,
               state: TrainState,
               batch) -> TrainState:
    """
    Perform a single training step for diffusion.
    Args:
        key: random key for sampling timesteps
        state: train state
        batch: batch of protein xyz coordinates of shape [B, N, 4, 3] cropped to size N
    1. Sample timesteps
    2. Get noised protein batch
    3. Apply model to noised protein batch
    4. Compute loss as
    5. All-reduce gradients
    6. Update train state
    """
    xyz = batch[0]  # TODO: named accession instead of index
    R = batch[1]
    R_inverse = batch[2]
    batch_size, N_res, B, _ = xyz.shape[0]
    n_atoms = N_res * B
    timesteps = jax.random.uniform(key, minval=0, maxval=1.0, shape=(batch_size,))
    x0 = jnp.reshape(xyz, newshape=(batch_size, n_atoms, 3))
    # Noise protein
    epsilons = jax.random.normal(key, shape=x0.shape)  # epsilons drawn from normal distribution
    x_t = jax.vmap(polymer.diffuse)(epsilons, R, x0, timesteps)
    key, dropout_key = jax.random.split(key, num=2)  # for model apply_fn

    def loss_fn(params):
        """Computes the diffusion loss given the parameters of the network.
        The loss is computed according to the formula in section A.2 of Chroma Paper.
        This loss is highly analogous to the diffusion loss used in Kingma et al. 2021."""
        x_theta = state.apply_fn(params, key, x_t, timesteps, rngs={'dropout': dropout_key})
        derivative_snr = jax.vmap(jax.grad(polymer.SNR))(timesteps)
        tau_t = (-0.5) * derivative_snr  # used to scale the loss in a time-dependent manner
        regularized_inverse = R_inverse + jnp.expand_dims(0.1 * jnp.identity(n=n_atoms), axis=0)  # regularized for
        # absolute errors in x space, expanded for broadcasting across batch dimension
        offset = jax.vmap(jnp.matmul)(regularized_inverse, (x_theta - x0))  # element-wise offset from truth
        reconstruction_error = mean_squared_error(offset, jnp.zeros_like(offset))
        loss = tau_t * reconstruction_error
        return loss

    # Compute gradient
    grad_fn = jax.value_and_grad(loss_fn)
    mse, grads = grad_fn(state.params)
    # Update parameters (all-reduce gradients)
    grads = jax.lax.pmean(grads, axis_name='batch')
    metrics = {'loss': mse}

    # Update train state
    new_state = state.apply_gradients(grads=grads)
    return new_state, metrics


def diffusion_eval_step(key, state, batch):
    """Perform a single evaluation step for diffusion."""
    images = batch[0]
    epsilons = batch[1]
    batch_size, *_ = images.shape
    return None  # compute_metrics(epsilon_theta, epsilons)
