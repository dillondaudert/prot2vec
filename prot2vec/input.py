'''Functions and utilities for handling and preprocessing data.'''

from pathlib import Path
import numpy as np
import tensorflow as tf

HOME = str(Path.home())

def cpdb_6133_input_fn(batch_size, shuffle, num_epochs, mode):
    """
    An Estimator input function using the tf.data API
    """

    if mode == "TRAIN":
        filename = HOME+"/data/cpdb/cpdb_6133_train.tfrecords"
    elif mode == "VALID":
        filename = HOME+"/data/cpdb/cpdb_6133_valid.tfrecords"
    elif mode == "TEST":
        filename = HOME+"/data/cpdb/cpdb_6133_test.tfrecords"
    else:
        print("Invalid mode passed to cpdb_dataset_input_fn!\n")
        quit()

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
        seq = tf.sparse_tensor_to_dense(parsed["seq_data"])
        label = tf.sparse_tensor_to_dense(parsed["label_data"])
        src = tf.reshape(seq, [-1, 43])
        tgt = tf.reshape(label, [-1, 9])

        # prepend and append 'NoSeq' to create dec_input / dec_target
        noseq = tf.constant([[0., 0., 0., 0., 0., 0., 0., 0., 1.]])
        tgt_input = tf.concat([noseq, tgt], 0)
        tgt_output = tf.concat([tgt, noseq], 0)

        return src, tgt_input, tgt_output

    # apply parser transformation to parse out individual samples
    dataset = dataset.map(parser)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=3000)
    dataset = dataset.repeat(num_epochs)

    dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=(tf.TensorShape([None, 43]),
                           tf.TensorShape([None, 9]),
                           tf.TensorShape([None, 9]))
            )


    # 3) make an iterator object
    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()
