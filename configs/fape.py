import ml_collections
from configs import tpu


def get_config() -> ml_collections.ConfigDict:
    """Get the hyperparameter configuration for backbone network A."""
    config = tpu.get_config()

    # Model hyperparameters
    config.use_timestep_embedding = True
    config.num_gnn_layers = 3

    # Loss Hyperparameters
    config.fape_clamp_rate = 1.0  # FAPE is clamped at 10.0 Angstroms for all training examples

    # Optimization details
    config.batch_size = 64
    config.base_learning_rate = 1e-3  # this will need to be tuned
    config.use_constant_lr = False
    config.steps_per_epoch = 500
    config.warmup_epochs = 20  # 10,000 warmup steps
    config.num_epochs = 2000  # 1 million training steps
    config.weight_decay = 1e-2
    config.global_clipping_value = 0.1
    config.exponential_moving_avg_decay = 0.9999
    return config
