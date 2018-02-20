'''Functions and utilities for handling and preprocessing data.'''
import numpy as np
import tensorflow as tf

__all__ = [
    "copytask_dataset",
]

def copytask_dataset(dataset, shuffle, num_features, num_labels, batch_size=32, num_epochs=None):
    """
    Read the copy task dataset and return as a TensorFlow Dataset.
    Args:
        dataset     - a Dataset object source
        shuffle     - a boolean Tensor
        batch_size  - the integer size of each minibatch (Default: 32)
        num_epochs  - how many epochs to repeat the dataset (Default: None)
    """

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
        src = tf.reshape(seq, [-1, num_features])
        tgt_in = tf.reshape(tgt_in, [-1, num_labels])
        tgt_out = tf.reshape(tgt_out, [-1, num_labels])

        return src, tgt_in, tgt_out, seq_len


    # shuffle logic
    if shuffle:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=6000, count=num_epochs))
    else:
        dataset = dataset.repeat(num_epochs)

    # if shuffling, then call the shuffle_and_repeat op. Else just repeat
#    dataset = tf.cond(shuffle, true_fn=lambda: shfl(dataset), false_fn=lambda: dataset.repeat(num_epochs))

    # apply parser transformation to parse out individual samples
    dataset = dataset.map(parser, num_parallel_calls=4)

    # cache the dataset
    dataset = dataset.cache()

    dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=(tf.TensorShape([None, num_features]),
                           tf.TensorShape([None, num_labels]),
                           tf.TensorShape([None, num_labels]),
                           tf.TensorShape([]))
            )

    dataset = dataset.prefetch(1)

    return dataset
