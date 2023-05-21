""""""

import flax
import jax
import ml_collections
import optax
import time
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


def squared_distance(x, y):
    """Computes the squared distance between x and y."""
    return jnp.sum(jnp.square(x - y))


def compute_metrics(x_pred, x0):
    """Returns a dictionary of metrics for the given logits and labels."""
    mse = mean_squared_error(x_pred, x0)
    return {'error': mse}


def create_learning_rate_fn(config):
    """Creates learning rate schedule."""
    def _base_fn(step):
        return config.base_learning_rate

    warmup_fn = optax.linear_schedule(
        init_value=0.0001, end_value=config.base_learning_rate,
        transition_steps=config.warmup_epochs * config.steps_per_epoch)
    cosine_epochs = max(config.num_epochs - config.warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=config.base_learning_rate,
        decay_steps=cosine_epochs * config.steps_per_epoch)
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[config.warmup_epochs * config.steps_per_epoch])

    if config.use_constant_lr:
        return _base_fn
    else:
        return schedule_fn


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


def create_denoising_input_iters(config):
    train_ds, test_ds = input_pipeline.create_denoising_datasets(config.crop_size,
                                                                 config.batch_size)
    train_iter = map(prepare_tf_data, train_ds)
    test_iter = map(prepare_tf_data, test_ds)
    return train_iter, test_iter


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
    batch_size, N_res, B, _ = xyz.shape
    n_atoms = N_res * B
    timesteps = jax.random.uniform(key, minval=0, maxval=1.0, shape=(batch_size,))
    x0 = jnp.reshape(xyz, newshape=(batch_size, n_atoms, 3))
    # Noise protein
    epsilons = jax.random.normal(key, shape=x0.shape)  # epsilons drawn from normal distribution
    x_t = jax.vmap(polymer.diffuse)(epsilons, R, x0, timesteps).reshape(xyz.shape)  # [B, N, 4, 3]
    key, dropout_key = jax.random.split(key, num=2)  # for model apply_fn

    def loss_fn(params):
        """Computes the diffusion loss given the parameters of the network.
        The loss is computed according to the formula in section A.2 of Chroma Paper.
        This loss is highly analogous to the diffusion loss used in Kingma et al. 2021."""
        x_theta = state.apply_fn(params, key, x_t, timesteps, rngs={'dropout': dropout_key})
        x_theta = x_theta.reshape(x0.shape)
        regularized_inverse = R_inverse + jnp.expand_dims(0.1 * jnp.identity(n=n_atoms), axis=0)  # regularized for
        # absolute errors in x space, expanded for broadcasting across batch dimension
        offset = jax.vmap(jnp.matmul)(regularized_inverse, (x_theta - x0))  # element-wise offset from truth
        distances = jax.vmap(squared_distance)(offset, jnp.zeros_like(offset))  # [B,]
        derivative_snr = jax.vmap(jax.grad(polymer.SNR))(timesteps)
        tau_t = ((-0.5) * derivative_snr).reshape(batch_size, 1, 1)  # used to scale the loss in a time-dependent manner
        loss = jnp.mean(tau_t * distances)  # average over batch dim
        return loss

    # Compute gradient
    grad_fn = jax.value_and_grad(loss_fn)
    mse, grads = grad_fn(state.params)
    # All-reduce gradients across devices
    grads = jax.lax.pmean(grads, axis_name='batch')
    metrics = {'loss': mse}

    # Update train state
    new_state = state.apply_gradients(grads=grads)
    return new_state, metrics


def eval_step(key, state, batch):
    """Perform a single evaluation step for diffusion."""
    xyz = batch[0]  # TODO: named accession instead of index
    R = batch[1]
    R_inverse = batch[2]
    batch_size, N_res, B, _ = xyz.shape
    n_atoms = N_res * B
    timesteps = jax.random.uniform(key, minval=0, maxval=1.0, shape=(batch_size,))
    x0 = jnp.reshape(xyz, newshape=(batch_size, n_atoms, 3))
    # Noise protein
    epsilons = jax.random.normal(key, shape=x0.shape)  # epsilons drawn from normal distribution
    x_t = jax.vmap(polymer.diffuse)(epsilons, R, x0, timesteps).reshape(xyz.shape)
    key, dropout_key = jax.random.split(key, num=2)  # for model apply_fn
    # Predict x0
    x_theta = state.apply_fn(state.params, key, x_t, timesteps, rngs={'dropout': dropout_key}).reshape(x0.shape)
    # Compute metrics as mean squared error
    return compute_metrics(x_theta, x0)


def denoising_train_step(key: jax.random.PRNGKey,
                         state: TrainState,
                         batch) -> TrainState:
    noised_xyz = batch[0]
    xyz = batch[1]
    dummy_timesteps = jnp.zeros((xyz.shape[0],))
    key, dropout_key = jax.random.split(key, num=2)  # for model apply_fn

    def loss_fn(params):
        x_theta = state.apply_fn(params, key, noised_xyz, dummy_timesteps, rngs={'dropout': dropout_key})
        return mean_squared_error(xyz, x_theta)

    # Compute gradient
    grad_fn = jax.value_and_grad(loss_fn)
    mse, grads = grad_fn(state.params)
    # All-reduce gradients across devices
    grads = jax.lax.pmean(grads, axis_name='batch')
    metrics = {'loss': mse}

    # Update train state
    new_state = state.apply_gradients(grads=grads)
    return new_state, metrics


def denoising_eval_step(key, state, batch):
    noised_xyz = batch[0]
    xyz = batch[1]
    dummy_timesteps = jnp.zeros((xyz.shape[0],))
    key, dropout_key = jax.random.split(key, num=2)  # for model apply_fn

    x_theta = state.apply_fn(state.params,
                             key,
                             noised_xyz,
                             dummy_timesteps,
                             rngs={'dropout': dropout_key})
    return compute_metrics(x_theta, xyz)


def save_checkpoint(workdir, state):
    state = jax.device_get(state)
    step = int(state.step)
    checkpoints.save_checkpoint(workdir, target=state, step=step, keep=3)


def create_train_state(rng,
                       config: ml_collections.ConfigDict,
                       model,
                       learning_rate_fn):
    """
    Creates the initial train state object.
    Args:
        rng: random number generator
        config: hyperparameter configuration
        model: Flax model
        image_size: integer specifying the height and width of input images
        learning_rate_fn: function that returns the learning rate for a given step.
    Returns:
        the initial train state object.
    """
    params = initialize(rng, model, config,
                        local_batch_size=config.batch_size // jax.device_count())
    optimizer = optax.adam(learning_rate_fn)
    opt_state = optimizer.init(params)
    state = TrainState(apply_fn=model.apply,
                       params=params,
                       tx=optimizer,
                       step=0,
                       opt_state=opt_state)
    return state


def summarize_metrics(metrics):
    """Summarizes the metrics."""
    summary = {}
    for metric in metrics:
        for key, value in metric.items():
            if summary.get(key) is None:
                summary[key] = value
            else:
                summary[key] += value
    # Average metrics
    for key, value in summary.items():
        summary[key] = value / len(metrics)
    return summary


def train_and_evaluate(config: ml_collections.ConfigDict,
                       workdir: str):
    """
    Executes model training and evaluation loop.
    Args:
        config: Hyperparameter configuration for training and evaluation.
        workdir: Directory where the Tensorboard summaries are written to.

    Returns:
        final train state.
    """
    # Initialize writer
    writer = metric_writers.create_default_writer(logdir=workdir,
                                                  just_logging=jax.process_index() != 0)
    rng = jax.random.PRNGKey(config.seed)
    # compute local_batch_size (with the appropriate divisibility assertion)
    if config.batch_size % jax.device_count() > 0:
        raise ValueError("Global batch size should be divisible by the number of devices.")
    local_batch_size = config.batch_size // jax.device_count()
    print(f"Local batch size: {local_batch_size}")
    platform = jax.local_devices()[0].platform

    # Create input iterator
    # train_iter = create_input_iter(config)
    # test_iter = create_input_iter(config)  # training dataset is used for testing for now
    train_iter, test_iter = create_denoising_input_iters(config)
    # Compute num_train_steps
    steps_per_epoch = config.steps_per_epoch
    if config.num_train_steps == -1:
        num_steps = int(steps_per_epoch * config.num_epochs)
    else:
        num_steps = config.num_train_stepsc
    steps_per_checkpoint = config.steps_per_checkpoint

    if config.steps_per_eval == -1:
        steps_per_eval = 300  # this is just a number, can be changed with config
    else:
        steps_per_eval = config.steps_per_eval

    # Create model
    model = create_model(config)
    # Create learning rate function
    learning_rate_fn = create_learning_rate_fn(config)
    # Create train_state
    state = create_train_state(rng, config, model, learning_rate_fn)
    # restore checkpoint
    state = checkpoints.restore_checkpoint(workdir, state)
    # step_offset > 0 if we are resuming training
    step_offset = int(state.step)  # 0 usually
    state = flax.jax_utils.replicate(state)

    # pmap transform train_step and eval_step
    p_train_step = jax.pmap(denoising_train_step, axis_name='batch')  # replace denoising_train_step with train_step
    p_eval_step = jax.pmap(denoising_eval_step, axis_name='batch')  # replace denoising_eval_step with eval_step

    # Create train loop
    train_metrics = []
    hooks = []
    if jax.process_index() == 0:
        hooks += [periodic_actions.Profile(num_profile_steps=5, logdir=workdir)]
    train_metrics_last_t = time.time()
    logging.info("Initial compilation, this might take some minutes...")
    for step, batch in zip(range(step_offset, num_steps), train_iter):
        keys = jax.random.split(jax.random.PRNGKey(config.seed + step), jax.device_count())
        state, metrics = p_train_step(keys, state, batch)
        if step == step_offset:
            logging.info("Initial compilation done.")
        if config.log_every_n_steps:
            train_metrics.append(metrics)
            if (step + 1) % config.log_every_n_steps == 0:
                train_metrics = common_utils.get_metrics(train_metrics)
                summary = {
                    f'train_{k}': v
                    for k, v in jax.tree_util.tree_map(lambda x: x.mean(), train_metrics).items()
                }
                summary['steps_per_second'] = config.log_every_n_steps / (
                        time.time() - train_metrics_last_t)
                # Write scalars to Tensorboard
                writer.write_scalars(step + 1, summary)
                train_metrics = []
                train_metrics_last_t = time.time()

            if (step + 1) % steps_per_epoch == 0:
                epoch = step // steps_per_epoch
                eval_metrics = []
                for _ in range(steps_per_eval):
                    eval_batch = next(test_iter)
                    keys = jax.random.split(jax.random.PRNGKey(config.seed + step), jax.device_count())
                    metrics = p_eval_step(keys, state, eval_batch)
                    eval_metrics.append(metrics)
                eval_metrics = common_utils.get_metrics(eval_metrics)
                summary = jax.tree_util.tree_map(lambda x: x.mean(), eval_metrics)
                # summary = summarize_metrics(eval_metrics)
                logging.info('eval epoch: %d, error: %.4f',
                             epoch, summary['error'])
                writer.write_scalars(
                    step + 1, {f'eval_{key}': val for key, val in summary.items()})
                # TODO: write sampled proteins to Tensorboard
                writer.flush()
                # if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_steps:
                save_checkpoint(workdir, flax.jax_utils.unreplicate(state))  # TODO: save checkpoint with eval loss

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
    return state
