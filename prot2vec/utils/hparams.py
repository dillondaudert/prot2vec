"""Hparams"""
import argparse as ap
import tensorflow as tf
from pathlib import Path

HOME = str(Path.home())

HPARAM_CHOICES= {
        "optimizer": ["adam", "sgd"],
        "unit_type": ["lstm", "nlstm", "gru"],
        "train_helper": ["teacher", "sched"],
        "sched_decay": ["linear", "expon", "inv_sig"],
        "initializer": ["glorot_normal", "glorot_uniform", "orthogonal"],
        "decoder": ["greedy", "beam"],
        }

HPARAMS = ["num_features", "num_labels", "initializer", "dense_input",
           "unit_type", "num_layers", "depth", "num_residual_layers",
           "forget_bias", "dropout", "decoder", "beam_width", "batch_size",
           "num_epochs", "train_helper", "sched_decay", "optimizer",
           "learning_rate", "momentum", "max_gradient_norm",
           "colocate_gradients_with_ops", "num_keep_ckpts",
           "model", "train_data", "valid_data", "infer_data", "modeldir"]

def hparams_to_str(hparams):
    print("Hyperparameters")
    for hp in HPARAMS:
        if hp in vars(hparams):
            print("\t"+hp+": ", vars(hparams)[hp])

def get_hparam_parser():
    parser = ap.ArgumentParser(description="Hyperparameters", add_help=False,
                               argument_default=ap.SUPPRESS)
    arch_group = parser.add_argument_group("architecture")
    arch_group.add_argument("--num_features", type=int)
    arch_group.add_argument("--num_labels", type=int)
    arch_group.add_argument("--initializer", type=str,
                        choices=HPARAM_CHOICES["initializer"])
    arch_group.add_argument("--dense_input", type=bool)
    arch_group.add_argument("--unit_type", type=str,
                        choices=HPARAM_CHOICES["unit_type"])
    arch_group.add_argument("--num_layers", type=int)
    arch_group.add_argument("--depth", type=int)
    arch_group.add_argument("--num_residual_layers", type=int)
    arch_group.add_argument("--forget_bias", type=float)
    arch_group.add_argument("--dropout", type=float)
    arch_group.add_argument("--decoder", type=str)
    arch_group.add_argument("--beam_width", type=int)

    tr_group = parser.add_argument_group("training")
    tr_group.add_argument("--batch_size", type=int)
    tr_group.add_argument("--num_epochs", type=int)
    tr_group.add_argument("--train_helper", type=str,
                        choices=HPARAM_CHOICES["train_helper"])
    tr_group.add_argument("--sched_decay", type=str,
                        choices=HPARAM_CHOICES["sched_decay"])
    tr_group.add_argument("--optimizer", type=str,
                         choices=HPARAM_CHOICES["optimizer"])
    tr_group.add_argument("--learning_rate", type=float)
    tr_group.add_argument("--momentum", type=float)
    tr_group.add_argument("--max_gradient_norm", type=float)
    tr_group.add_argument("--colocate_gradients_with_ops", type=bool)
    tr_group.add_argument("--num_keep_ckpts", type=int)

    return parser


def get_hparams(setting):
    """Return the hyperparameter settings given by name."""
    hparams = tf.contrib.training.HParams()
    if setting == "cpdb":
        hparams = tf.contrib.training.HParams(
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
            num_epochs=1,
            optimizer="sgd",
            learning_rate=0.05,
            momentum=0.0,
            max_gradient_norm=5.0,
            colocate_gradients_with_ops=False,
            train_helper="sched",
            sched_decay="inv_sig",
            num_keep_ckpts=1,
        )

    elif setting == "copy":
        hparams = tf.contrib.training.HParams(
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
            num_epochs=2,
            optimizer="sgd",
            learning_rate=0.5,
            momentum=0.0,
            max_gradient_norm=5.0,
            colocate_gradients_with_ops=False,
            train_helper="sched",
            sched_decay="linear",
            num_keep_ckpts=1,
        )

    return hparams
