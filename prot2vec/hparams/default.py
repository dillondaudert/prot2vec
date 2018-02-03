"""Default hparams"""

import tensorflow as tf

__all__ = ["get_default_hparams",]

def get_default_hparams():
    hparams = tf.contrib.training.HParams(
        num_features=43,
        num_labels=9,
        batch_size=32,
        num_epochs=1,
        learning_rate=0.007,
        unit_type="lstm",
        num_units=128,
        num_layers=1,
        num_residual_layers=0,
        forget_bias=1,
        dropout=0.5,
        max_gradient_norm=5.0
    )

    return hparams
