'''Functions and utilities for handling and preprocessing data.'''
import numpy as np
import tensorflow as tf
import pandas as pd
from pathlib import Path

HOME = str(Path.home())

__all__ = [
    "cpdb_parser", "cpdb2_parser", "autoenc_parser", "copytask_parser",
]

def cpdb_parser(record, hparams):
    """
    Parse a CPDB tfrecord Record into a tuple of tensors.
    """

    keys_to_features = {
        "seq_len": tf.FixedLenFeature([], tf.int64),
        "seq_data": tf.VarLenFeature(tf.float32),
        "label_data": tf.VarLenFeature(tf.float32),
        }

    parsed = tf.parse_single_example(record, keys_to_features)

    seq_len = parsed["seq_len"]
    seq_len = tf.cast(seq_len, tf.int32)
    seq = tf.sparse_tensor_to_dense(parsed["seq_data"])
    label = tf.sparse_tensor_to_dense(parsed["label_data"])
    src = tf.reshape(seq, [-1, hparams.num_features])
    tgt = tf.reshape(label, [-1, hparams.num_labels])

    # prepend and append 'NoSeq' to create dec_input / dec_target
    noseq = tf.constant([[0., 0., 0., 0., 0., 0., 0., 0., 1.]])
    tgt_input = tf.concat([noseq, tgt], 0)
    tgt_output = tf.concat([tgt, noseq], 0)

    return src, tgt_input, tgt_output, seq_len

def cpdb2_parser(record, hparams):
    """
    Parse a CPDB tfrecord Record into a tuple of tensors.
    """

    ss_feats = pd.read_csv(HOME+"/thesis/prot2vec/prot2vec/datasets/cpdb2_ss_features.csv", index_col=0)

    keys_to_features = {
        "dssp_id": tf.FixedLenFeature([], tf.string),
        "seq_len": tf.FixedLenFeature([], tf.int64),
        "seq_data": tf.VarLenFeature(tf.float32),
        "label_data": tf.VarLenFeature(tf.float32),
        }

    parsed = tf.parse_single_example(record, keys_to_features)

    seq_len = parsed["seq_len"]
    seq_len = tf.cast(seq_len, tf.int32)
    seq = tf.sparse_tensor_to_dense(parsed["seq_data"])
    label = tf.sparse_tensor_to_dense(parsed["label_data"])
    src = tf.reshape(seq, [-1, hparams.num_features])
    tgt = tf.reshape(label, [-1, hparams.num_labels])

    # prepend and append 'NoSeq' to create dec_input / dec_target
    sos = tf.constant(ss_feats.loc["SOS"].values.reshape(1, -1), dtype=tf.float32)
    eos = tf.constant(ss_feats.loc["EOS"].values.reshape(1, -1), dtype=tf.float32)

    tgt_input = tf.concat([sos, tgt], 0)
    tgt_output = tf.concat([tgt, eos], 0)

    return src, tgt_input, tgt_output, seq_len


def autoenc_parser(record, hparams):
    keys_to_features = {
        "seq_len": tf.FixedLenFeature([], tf.int64),
        "seq_data": tf.VarLenFeature(tf.float32),
        "label_data": tf.VarLenFeature(tf.float32),
        }

    parsed = tf.parse_single_example(record, keys_to_features)

    seq_len = parsed["seq_len"]
    seq_len = tf.cast(seq_len, tf.int32)
    seq = tf.sparse_tensor_to_dense(parsed["seq_data"])
    label = tf.sparse_tensor_to_dense(parsed["label_data"])
    src = tf.reshape(seq, [-1, hparams.num_features])
    tgt = tf.reshape(label, [-1, hparams.num_labels])

    # prepend and append 'NoSeq' to create dec_input / dec_target
    noseq = tf.constant([[0., 0., 0., 0., 0., 0., 0., 0.,
                          0., 0., 0., 0., 0., 0., 0., 0.,
                          0., 0., 0., 0., 0., 1.]])
    tgt_input = tf.concat([noseq, tgt], 0)
    tgt_output = tf.concat([tgt, noseq], 0)

    return src, tgt_input, tgt_output, seq_len

def copytask_parser(record, hparams):
    keys_to_features = {
        "seq_len": tf.FixedLenFeature([], tf.int64),
        "seq_data": tf.VarLenFeature(tf.float32),
        "tgt_in": tf.VarLenFeature(tf.float32),
        "tgt_out": tf.VarLenFeature(tf.float32),
        }

    parsed = tf.parse_single_example(record, keys_to_features)

    seq_len = parsed["seq_len"]
    seq_len = tf.cast(seq_len, tf.int32)
    seq = tf.sparse_tensor_to_dense(parsed["seq_data"])
    tgt_in = tf.sparse_tensor_to_dense(parsed["tgt_in"])
    tgt_out = tf.sparse_tensor_to_dense(parsed["tgt_out"])
    src = tf.reshape(seq, [-1, hparams.num_features])
    tgt_in = tf.reshape(tgt_in, [-1, hparams.num_labels])
    tgt_out = tf.reshape(tgt_out, [-1, hparams.num_labels])

    return src, tgt_in, tgt_out, seq_len


def bdrnn_parser(record, hparams):
    """
    Parse a CPDB tfrecord Record into a tuple of tensors.
    """

    keys_to_features = {
        "seq_len": tf.FixedLenFeature([], tf.int64),
        "seq_data": tf.VarLenFeature(tf.float32),
        "label_data": tf.VarLenFeature(tf.float32),
        }

    parsed = tf.parse_single_example(record, keys_to_features)

    seq_len = parsed["seq_len"]
    seq_len = tf.cast(seq_len, tf.int32)
    seq = tf.sparse_tensor_to_dense(parsed["seq_data"])
    label = tf.sparse_tensor_to_dense(parsed["label_data"])
    seq = tf.reshape(seq, [-1, hparams.num_features])
    tgt_outputs = tf.reshape(label, [-1, hparams.num_labels])

    return seq, tgt_outputs, seq_len
