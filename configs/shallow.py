import ml_collections
from configs import tpu


def get_config() -> ml_collections.ConfigDict:
    """Get the hyperparameter configuration for backbone network A."""
    config = tpu.get_config()

    config.use_timestep_embedding = True
    config.num_gnn_layers = 3

    config.batch_size = 64  # with 8 workers local batch size becomes 4
    config.base_learning_rate = 1e-5  # this will need to be tuned
    config.use_constant_lr = True
    config.warmup_epochs = 20
    config.num_epochs = 5_000  # 1 million training steps

    return config
