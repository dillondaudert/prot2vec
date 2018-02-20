'''Functions and utilities for handling and preprocessing data.'''
from pathlib import Path
import numpy as np
import tensorflow as tf

__all__ = [
    "copytask_dataset",
]

HOME = str(Path.home())

def copytask_dataset(filename, shuffle, batch_size=32, num_epochs=None):
    """
    Read the copy task dataset and return as a TensorFlow Dataset.
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
            "tgt_in": tf.VarLenFeature(tf.float32),
            "tgt_out": tf.VarLenFeature(tf.float32),
            }

        parsed = tf.parse_single_example(record, keys_to_features)

        seq_len = parsed["seq_len"]
        seq_len = tf.cast(seq_len, tf.int32)
        seq = tf.sparse_tensor_to_dense(parsed["seq_data"])
        tgt_in = tf.sparse_tensor_to_dense(parsed["tgt_in"])
        tgt_out = tf.sparse_tensor_to_dense(parsed["tgt_out"])
        src = tf.reshape(seq, [-1, 10])
        tgt_in = tf.reshape(tgt_in, [-1, 10])
        tgt_out = tf.reshape(tgt_out, [-1, 10])

        return src, tgt_in, tgt_out, seq_len


    # shuffle logic
    def shfl(dataset):
#        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=6000, count=num_epochs))
        dataset = dataset.shuffle(buffer_size=6000)
        dataset = dataset.repeat(num_epochs)
        return tf.no_op()
    tf.cond(shuffle, lambda: shfl(dataset), lambda: tf.no_op())

    # apply parser transformation to parse out individual samples
    dataset = dataset.map(parser, num_parallel_calls=4)

#    dataset = dataset.cache()

    dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=(tf.TensorShape([None, 10]),
                           tf.TensorShape([None, 10]),
                           tf.TensorShape([None, 10]),
                           tf.TensorShape([]))
            )

    dataset = dataset.prefetch(10)

    return dataset
