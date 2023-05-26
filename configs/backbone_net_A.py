import ml_collections
from configs import tpu


def get_config() -> ml_collections.ConfigDict:
    """Get the hyperparameter configuration for backbone network A."""
    config = tpu.get_config()

    config.use_timestep_embedding = True
    config.num_gnn_layers = 12

    config.batch_size = 128
    config.base_learning_rate = 0.5 * 1e-4  # this will need to be tuned
    config.use_constant_lr = False
    config.warmup_epochs = 1

    return config
