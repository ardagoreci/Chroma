import ml_collections
from configs import tpu


def get_config() -> ml_collections.ConfigDict:
    """Get the hyperparameter configuration for experiment 2: timestep embeddings."""
    config = tpu.get_config()
    config.use_timestep_embedding = True
    config.base_learning_rate = 0.5 * 1e-6  # this will need to be tuned
    return config
