import ml_collections
from configs import tpu


def get_config() -> ml_collections.ConfigDict:
    """Get the hyperparameter configuration for experiment 2: timestep embeddings."""
    config = tpu.get_config()
    # Do not use timestep embedding
    config.use_timestep_embedding = False
    return config
