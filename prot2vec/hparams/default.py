"""Hparams"""

import tensorflow as tf

__all__ = ["get_default_hparams",]

def get_default_hparams():
    hparams = tf.contrib.training.HParams(
        dir="/home/dillon/thesis/models/prot2vec/tuning",
        num_features=43,
        num_labels=9,
        batch_size=32,
        num_epochs=2,
        optimizer="adam",
        learning_rate=0.005,
        unit_type="lstm",
        num_units=128,
        num_layers=1,
        num_residual_layers=0,
        forget_bias=1,
        dropout=0.0,
        max_gradient_norm=5.0,
        colocate_gradients_with_ops=False,
        num_keep_ckpts=4,
        dense_input=False,
        train_helper="sched",
        sched_rate=0.4
    )

    return hparams
