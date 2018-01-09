# utility functions
from pathlib import Path
import numpy as np
import tensorflow as tf

HOME = str(Path.home())

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def cpdb_to_tfrecord():
    """
    Convert the numpy array format for cpdb files to TFRecord format
    Saves training, validation, and test files
    """

    data = np.load(HOME+"/data/cpdb/cpdb_6133.npy.gz").reshape(6133, 700, 57)

    seqs = np.concatenate([data[:, :, 0:22].copy(), data[:, :, 35:56].copy()], axis=2).reshape(6133, -1)
    labels = data[:, :, 22:31].copy().reshape(6133, 700, -1)

    num_features = 43
    num_labels = 9

    # Count the protein sequence lengths for all samples
    def get_length(seq_labels):
        assert seq_labels.shape == (700, 9)
        noseq = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 1.]])
        return np.logical_not(np.all(np.equal(seq_labels, noseq), axis=1)).sum()

    seq_lengths = [get_length(labels[l, :, :]) for l in range(6133)]

    # Flatten labels
    labels = labels.reshape(6133, -1)

    train_examples = range(5600)
    valid_examples = range(5600,5877)
    test_examples = range(5877,6133)

    trainfile = HOME+"/data/cpdb/cpdb_6133_train.tfrecords"
    validfile = HOME+"/data/cpdb/cpdb_6133_valid.tfrecords"
    testfile = HOME+"/data/cpdb/cpdb_6133_test.tfrecords"
    print("Writing ", trainfile)
    trainwriter = tf.python_io.TFRecordWriter(trainfile)

    for index in train_examples:
        example = tf.train.Example(features=tf.train.Features(feature={
            'seq_len': _int64_feature(seq_lengths[index]),
            'seq_data': _floats_feature(seqs[index, 0:num_features*seq_lengths[index]]),
            'label_data': _floats_feature(labels[index, 0:num_labels*seq_lengths[index]])}))
        trainwriter.write(example.SerializeToString())
    trainwriter.close()

    print("Writing ", validfile)
    validwriter = tf.python_io.TFRecordWriter(validfile)
    for index in valid_examples:
        example = tf.train.Example(features=tf.train.Features(feature={
            'seq_len': _int64_feature(seq_lengths[index]),
            'seq_data': _floats_feature(seqs[index, 0:num_features*seq_lengths[index]]),
            'label_data': _floats_feature(labels[index, 0:num_labels*seq_lengths[index]])}))
        validwriter.write(example.SerializeToString())
    validwriter.close()


    print("Writing ", testfile)
    testwriter = tf.python_io.TFRecordWriter(testfile)
    for index in test_examples:
        example = tf.train.Example(features=tf.train.Features(feature={
            'seq_len': _int64_feature(seq_lengths[index]),
            'seq_data': _floats_feature(seqs[index, 0:num_features*seq_lengths[index]]),
            'label_data': _floats_feature(labels[index, 0:num_labels*seq_lengths[index]])}))
        testwriter.write(example.SerializeToString())
    testwriter.close()

if __name__ == "__main__":
    cpdb_to_tfrecord()

