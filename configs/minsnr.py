import ml_collections
from configs import tpu


def get_config() -> ml_collections.ConfigDict:
    """Get the hyperparameter configuration for backbone network A."""
    config = tpu.get_config()

    # Model hyperparameters
    config.use_timestep_embedding = True
    config.num_gnn_layers = 3

    # Optimization details
    config.batch_size = 28
    config.base_learning_rate = 1e-5  # this will need to be tuned
    config.use_constant_lr = True
    config.warmup_epochs = 20
    config.num_epochs = 5_000  # 1 million training steps
    config.weight_decay = 1e-3
    config.global_clipping_value = 0.1
    return config
