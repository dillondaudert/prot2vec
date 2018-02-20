'''Functions and utilities for handling and preprocessing data.'''
from pathlib import Path
import numpy as np
import tensorflow as tf

__all__ = [
    "pssp_dataset", "autoenc_dataset",
]

HOME = str(Path.home())

def pssp_dataset(filename, shuffle, num_features, num_labels, batch_size=32, num_epochs=None):
    """
    Read the cpdb dataset and return as a TensorFlow Dataset.
    Args:
        filename    - a Tensor containing a list of strings of input files
        shuffle     - a bool
        batch_size  - the integer size of each minibatch (Default: 32)
        num_epochs  - how many epochs to repeat the dataset (Default: None)
    """

    # 1) define a source to construct a dataset
    dataset = tf.data.TFRecordDataset(filename)

    # use tf.parse_single_example() to extract data from a tf.Example proto buffer
    def parser(record):
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
        src = tf.reshape(seq, [-1, num_features])
        tgt = tf.reshape(label, [-1, num_labels])

        # prepend and append 'NoSeq' to create dec_input / dec_target
        noseq = tf.constant([[0., 0., 0., 0., 0., 0., 0., 0., 1.]])
        tgt_input = tf.concat([noseq, tgt], 0)
        tgt_output = tf.concat([tgt, noseq], 0)

        return src, tgt_input, tgt_output, seq_len


    # shuffle logic
    if shuffle:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=6000, count=num_epochs))
    else:
        dataset = dataset.repeat(num_epochs)

    # apply parser transformation to parse out individual samples
    dataset = dataset.map(parser, num_parallel_calls=4)

#    dataset = dataset.cache()

    dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=(tf.TensorShape([None, num_features]),
                           tf.TensorShape([None, num_labels]),
                           tf.TensorShape([None, num_labels]),
                           tf.TensorShape([]))
            )

    dataset = dataset.prefetch(1)

    return dataset

def autoenc_dataset(filename, shuffle, num_features, num_labels, batch_size=32, num_epochs=None):
    """
    Read the cpdb autoencode dataset and return as a TensorFlow Dataset.
    Args:
        filename    - a Tensor containing a list of strings of input files
        shuffle     - a boolean Tensor
        batch_size  - the integer size of each minibatch (Default: 32)
        num_epochs  - how many epochs to repeat the dataset (Default: None)
    """

    # 1) define a source to construct a dataset
    dataset = tf.data.TFRecordDataset(filename)

    # use tf.parse_single_example() to extract data from a tf.Example proto buffer
    def parser(record):
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
        src = tf.reshape(seq, [-1, num_features])
        tgt = tf.reshape(label, [-1, num_labels])

        # prepend and append 'NoSeq' to create dec_input / dec_target
        noseq = tf.constant([[0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 1.]])
        tgt_input = tf.concat([noseq, tgt], 0)
        tgt_output = tf.concat([tgt, noseq], 0)

        return src, tgt_input, tgt_output, seq_len


    # shuffle logic
    if shuffle:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=6000, count=num_epochs))
    else:
        dataset = dataset.repeat(num_epochs)

    # apply parser transformation to parse out individual samples
    dataset = dataset.map(parser, num_parallel_calls=4)

#    dataset = dataset.cache()

    dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=(tf.TensorShape([None, num_features]),
                           tf.TensorShape([None, num_labels]),
                           tf.TensorShape([None, num_labels]),
                           tf.TensorShape([]))
            )

    dataset = dataset.prefetch(1)

    return dataset