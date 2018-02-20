"""Hparams"""

import tensorflow as tf
from pathlib import Path

HOME = str(Path.home())

__all__ = ["get_hparams",]

VALID_HPARAMS = {
        "optimizer": ["adam", "sgd"],
        "unit_type": ["lstm", "nlstm", "gru"],
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
            model="cpdb",
            logdir=HOME+"/thesis/models/prot2vec/default",
            train_file=HOME+"/data/cpdb/cv_5/cpdb_6133_filter_train_1.tfrecords",
            valid_file=HOME+"/data/cpdb/cv_5/cpdb_6133_filter_valid_1.tfrecords",
            num_features=43,
            num_labels=9,
            unit_type="lstm",
            initializer="glorot_uniform",
            dense_input=False,
            num_units=128,
            num_layers=1,
            num_residual_layers=0,
            depth=0,
            forget_bias=1,
            dropout=0.0,
            batch_size=50,
            num_epochs=10,
            optimizer="sgd",
            learning_rate=0.05,
            momentum=0.0,
            max_gradient_norm=5.0,
            colocate_gradients_with_ops=False,
            train_helper="sched",
            sched_decay="inv_sig",
            num_keep_ckpts=1,
            #tag="2"
        )

    elif setting == "copy":
        hparams = tf.contrib.training.HParams(
            model="copy",
            logdir=HOME+"/thesis/models/prot2vec/copy",
            train_file = HOME+"/data/synthetic/copy/train_10-50L_12V_10k.tfrecords",
            valid_file = HOME+"/data/synthetic/copy/valid_10-50L_12V_1k.tfrecords",
            num_features=12,
            num_labels=12,
            unit_type="lstm",
            initializer="glorot_uniform",
            dense_input=False,
            num_units=128,
            num_layers=1,
            num_residual_layers=0,
            depth=0,
            forget_bias=1,
            dropout=0.0,
            batch_size=128,
            num_epochs=100,
            optimizer="sgd",
            learning_rate=0.5,
            momentum=0.0,
            max_gradient_norm=5.0,
            colocate_gradients_with_ops=False,
            train_helper="sched",
            sched_decay="linear",
            num_keep_ckpts=1,
            tag="sched_10-50L_10k"
        )

    return hparams
