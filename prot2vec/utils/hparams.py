"""Hparams"""
import argparse as ap
import tensorflow as tf
from pathlib import Path

HOME = str(Path.home())

HPARAM_CHOICES= {
        "model": ["cpdb", "copy", "bdrnn", "cpdb2", "cpdb2_prot"],
        "optimizer": ["adam", "sgd", "adadelta"],
        "unit_type": ["lstm", "nlstm", "gru"],
        "train_helper": ["teacher", "sched"],
        "sched_decay": ["linear", "expon", "inv_sig"],
        "initializer": ["glorot_normal", "glorot_uniform", "orthogonal"],
        "decoder": ["greedy", "beam"],
        }

HPARAMS = ["num_features", "num_labels", "initializer", "dense_input",
           "unit_type", "num_units", "num_layers", "depth", "num_residual_layers",
           "forget_bias", "dropout", "decoder", "beam_width", "batch_size",
           "num_epochs", "train_helper", "sched_decay", "optimizer",
           "learning_rate", "momentum", "max_gradient_norm",
           "colocate_gradients_with_ops", "num_keep_ckpts",
           "model", "train_file", "valid_file", "infer_file", "modeldir",
           "train_source_file", "train_target_file", "valid_source_file",
           "valid_target_file", "infer_source_file", "infer_target_file"]

def hparams_to_str(hparams):
    print("Hyperparameters")
    for hp in HPARAMS:
        if hp in vars(hparams):
            print("\t"+hp+": ", vars(hparams)[hp])

def get_hparam_parser():
    parser = ap.ArgumentParser(description="Hyperparameters", add_help=False,
                               argument_default=ap.SUPPRESS)
    gen_group = parser.add_argument_group("general")
    gen_group.add_argument("-m", "--model", type=str,
                           choices=HPARAM_CHOICES["model"])
    gen_group.add_argument("--train_file", type=str)
    gen_group.add_argument("--valid_file", type=str)
    gen_group.add_argument("--infer_file", type=str)
    gen_group.add_argument("--train_source_file", type=str)
    gen_group.add_argument("--train_target_file", type=str)
    gen_group.add_argument("--valid_source_file", type=str)
    gen_group.add_argument("--valid_target_file", type=str)
    gen_group.add_argument("--infer_source_file", type=str)
    gen_group.add_argument("--infer_target_file", type=str)


    arch_group = parser.add_argument_group("architecture")
    arch_group.add_argument("--num_features", type=int)
    arch_group.add_argument("--num_labels", type=int)
    arch_group.add_argument("--initializer", type=str,
                        choices=HPARAM_CHOICES["initializer"])
    arch_group.add_argument("--dense_input", type=bool)
    arch_group.add_argument("--unit_type", type=str,
                        choices=HPARAM_CHOICES["unit_type"])
    arch_group.add_argument("--num_units", type=int)
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
            model="cpdb",
            num_features=43,
            num_labels=9,
            unit_type="lstmblock",
            initializer="glorot_uniform",
            dense_input=False,
            num_units=256,
            num_layers=3,
            num_residual_layers=2,
            depth=0,
            forget_bias=1,
            dropout=0.0,
            batch_size=64,
            num_epochs=400,
            optimizer="adadelta",
            learning_rate=0.05,
            momentum=0.0,
            max_gradient_norm=5.,
            colocate_gradients_with_ops=False,
            train_helper="sched",
            sched_decay="none",
            num_keep_ckpts=2,
            train_file="/home/dillon/data/cpdb/cv_5/cpdb_6133_filter_train_1.tfrecords",
            valid_file="/home/dillon/data/cpdb/cv_5/cpdb_6133_filter_valid_1.tfrecords",
        )
    elif setting == "cpdb2_prot":
        hparams = tf.contrib.training.HParams(
            model="cpdb2_prot",
            num_features=30,
            num_labels=10,
            unit_type="lstmblock",
            initializer="glorot_uniform",
            dense_input=False,
            num_units=128,
            num_layers=3,
            num_residual_layers=2,
            depth=0,
            forget_bias=1,
            dropout=0.0,
            batch_size=32,
            num_epochs=1,
            optimizer="adadelta",
            learning_rate=0.05,
            momentum=0.0,
            max_gradient_norm=5.,
            colocate_gradients_with_ops=False,
            train_helper="sched",
            sched_decay="none",
            num_keep_ckpts=2,
            train_source_file="/home/dillon/data/cpdb2/cpdb2_train_source.txt",
            train_target_file="/home/dillon/data/cpdb2/cpdb2_train_target.txt",
            valid_source_file="/home/dillon/data/cpdb2/cpdb2_valid_source.txt",
            valid_target_file="/home/dillon/data/cpdb2/cpdb2_valid_target.txt",
        )
    elif setting == "copy":
        hparams = tf.contrib.training.HParams(
            model="copy",
            num_features=12,
            num_labels=12,
            unit_type="nlstm",
            initializer="glorot_uniform",
            dense_input=False,
            num_units=128,
            num_layers=1,
            num_residual_layers=0,
            depth=3,
            forget_bias=1,
            dropout=0.0,
            batch_size=100,
            num_epochs=500,
            optimizer="sgd",
            learning_rate=0.5,
            momentum=0.,
            max_gradient_norm=1.0,
            colocate_gradients_with_ops=False,
            train_helper="sched",
            sched_decay="linear",
            num_keep_ckpts=1,
            train_file="/home/dillon/data/synthetic/copy/train_100L_10k.tfrecords",
            valid_file="/home/dillon/data/synthetic/copy/valid_100L_1k.tfrecords",
        )

    elif setting == "bdrnn":
        hparams = tf.contrib.training.HParams(
            model="bdrnn",
            num_features=43,
            num_labels=9,
            unit_type="lstmblock",
            initializer="glorot_uniform",
            num_units=300,
            num_layers=3,
            forget_bias=1,
            num_dense_units=200,
            dropout=0.5,
            batch_size=128,
            num_epochs=100,
            optimizer="adadelta",
            learning_rate=1.,
            max_gradient_norm=0.5,
            colocate_gradients_with_ops=False,
            num_keep_ckpts=4,
            train_helper="teacher",
            train_file="/home/dillon/data/cpdb/cv_5/cpdb_6133_filter_train_1.tfrecords",
            valid_file="/home/dillon/data/cpdb/cv_5/cpdb_6133_filter_valid_1.tfrecords",
        )

    return hparams
