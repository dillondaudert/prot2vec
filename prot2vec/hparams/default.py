"""Hparams"""

import tensorflow as tf

__all__ = ["get_default_hparams",]

def get_default_hparams():
    hparams = tf.contrib.training.HParams(
        dir="/home/dillon/thesis/models/prot2vec/tuning2",
        num_features=43,
        num_labels=9,
        batch_size=32,
        num_epochs=4,
        optimizer="adam",
        learning_rate=0.05,
        unit_type="nlstm",
        num_units=128,
        num_layers=2,
        depth=3,
        num_residual_layers=1,
        forget_bias=1,
        dropout=0.4,
        max_gradient_norm=2.0,
        colocate_gradients_with_ops=False,
        num_keep_ckpts=4,
        dense_input=True,
        train_helper="sched",
        sched_decay="inv_sig"
    )

    return hparams
