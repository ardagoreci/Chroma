import ml_collections


def get_config() -> ml_collections.ConfigDict:
    """Get the hyperparameter configuration for training on TPUs."""
    config = ml_collections.ConfigDict()

    config.model = 'Chroma'
    config.dataset = 'thermophiles'
    config.crop_size = 256

    # Backbone Network A in Chroma
    config.node_embedding_dim = 512
    config.edge_embedding_dim = 256
    config.node_mlp_hidden_dim = 512
    config.edge_mlp_hidden_dim = 128
    config.num_gnn_layers = 12
    config.dropout = 0.1  # dropout rate
    config.backbone_solver_iterations = 1  # this is not implemented for more than 1 yet.

    # Optimization parameters
    config.learning_rate = 0.1
    config.warmup_epochs = 5.0
    config.momentum = 0.9

    config.num_epochs = 100
    config.log_every_n_steps = 100

    config.num_train_steps = -1
    config.steps_per_eval = -1

    config.batch_size = 256
    config.cache = True
    config.half_precision = True

    return config
