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


def create_model(config):
    """Creates and initializes the UNet model."""
    return None


def initialize(key, image_size, model, local_batch_size):
    """Utility function to initialize the model."""
    dummy_x = jnp.zeros((local_batch_size, image_size, image_size, model.in_channels))
    dummy_timesteps = jnp.zeros((local_batch_size,))
    params = model.init(key, dummy_x, dummy_timesteps)
    return params


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


def diffusion_train_step(key: jax.random.PRNGKey,
                         state: TrainState,
                         batch) -> TrainState:
    """
    Perform a single training step for diffusion.
    Args:
        key: random key for sampling timesteps
        state: train state
        batch: batch of images containing images and epsilons
    1. Sample timesteps
    2. Get noised protein batch
    3. Apply model to noised protein batch
    4. Compute loss as
    5. All-reduce gradients
    6. Update train state
    """
    images = batch[0]  # xyz coordinates
    epsilons = jax.random.normal(key, shape=())  # epsilons drawn from normal distribution
    batch_size, *_ = images.shape
    timesteps = jax.random.randint(key, minval=1, maxval=1000, shape=(batch_size,))  # Diffusion for 1000 steps
    # Noise protein

    def loss_fn(params):
        loss = 0  # (timestep scaling) * (R^-1 + omega*I)(x_theta(x_t, t) - x)
        return loss

    # Compute gradient
    grad_fn = jax.value_and_grad(loss_fn)
    mse, grads = grad_fn(state.params)  # TODO: something may be wrong here with the order of aux
    # Update parameters (all-reduce gradients)
    grads = jax.lax.pmean(grads, axis_name='batch')
    metrics = {'loss': mse}

    # Update train state
    new_state = state.apply_gradients(grads=grads)
    return new_state, metrics
