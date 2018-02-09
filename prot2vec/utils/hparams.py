"""Hparams"""

import tensorflow as tf

__all__ = ["get_default_hparams",]

VALID_HPARAMS = {
        "optimizer": ["adam", "sgd", "momentum"],
        "unit_type": ["lstm", "nlstm"],
        "train_helper": ["teacher", "sched"],
        "sched_decay": ["linear", "expon", "inv_sig"],
        "initializer": ["glorot_normal", "glorot_uniform", "orthogonal"],
        "decoder": ["greedy", "beam"],
        }
HPARAMS = ["logdir", "num_features", "num_labels", "batch_size", "num_epochs",
           "optimizer", "learning_rate", "momentum", "unit_type", "num_units",
           "num_layers", "depth", "num_residual_layers", "forget_bias",
           "dropout", "max_gradient_norm", "colocate_gradients_with_ops",
           "num_keep_ckpts", "dense_input", "train_helper", "sched_decay",
           "initializer", "decoder", "beam_width"]

def get_hparams(setting):
    """Return the hyperparameter settings given by name."""
    if setting == "default":
        hparams = tf.contrib.training.HParams(
            logdir="/home/dillon/thesis/models/prot2vec/default",
            num_features=43,
            num_labels=9,
            batch_size=32,
            num_epochs=4,
            optimizer="adam",
            learning_rate=0.005,
            unit_type="lstm",
            num_units=128,
            num_layers=1,
            depth=0,
            num_residual_layers=0,
            forget_bias=1,
            dropout=0.0,
            max_gradient_norm=5.0,
            colocate_gradients_with_ops=False,
            num_keep_ckpts=2,
            dense_input=False,
            train_helper="sched",
            sched_decay="inv_sig",
            initializer="glorot_uniform"
        )

    return hparams
