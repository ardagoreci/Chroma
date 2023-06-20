import ml_collections
from configs import tpu


def get_config() -> ml_collections.ConfigDict:
    """Get the hyperparameter configuration for backbone network A."""
    config = tpu.get_config()

    config.use_timestep_embedding = True
    config.num_gnn_layers = 12

    config.batch_size = 64
    config.base_learning_rate = 1e-5  # this will need to be tuned
    config.use_constant_lr = True
    config.warmup_epochs = 20
    config.num_epochs = 10_000  # 2 million training steps

    config.steps_per_eval = 100
    config.steps_per_epoch = 1000

    return config
