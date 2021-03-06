# utility functions
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold

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


def cpdb_filt_to_tfrecord():
    """
    Convert the numpy array format for cpdb files to TFRecord format
    Saves training and validation splits using KFold
    """

    #TODO: Consolidate various aspects of these functions (such as get_length) for DNRY
    #      and to make everything simpler

    data = np.load(HOME+"/data/cpdb/cpdb_6133_filtered.npy.gz").reshape(-1, 700, 57)
    num_samples = data.shape[0]

    # shuffle data
    np.random.seed(248317)
    data = np.random.permutation(data)

    # calculate mean/stdev for pssm features
    mu = np.mean(data[:, :, 35:56], axis=1, keepdims=True)
    std = np.std(data[:, :, 35:56], axis=1, keepdims=True)
    # scale the data
    data[:, :, 35:56] = (data[:, :, 35:56] - mu)/std

    seqs = np.concatenate([data[:, :, 0:22].copy(), data[:, :, 35:56].copy()], axis=2).reshape(num_samples, -1)
    labels = data[:, :, 22:31].copy().reshape(num_samples, 700, -1)

    num_features = 43
    num_labels = 9

    # Count the protein sequence lengths for all samples
    def get_length(seq_labels):
        assert seq_labels.shape == (700, 9)
        noseq = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 1.]])
        return np.logical_not(np.all(np.equal(seq_labels, noseq), axis=1)).sum()

    seq_lengths = [get_length(labels[l, :, :]) for l in range(num_samples)]

    # Flatten labels
    labels = labels.reshape(num_samples, -1)

    # Calculate the indices of the 5 folds:
    kf = KFold(n_splits=5)
    fold = 0
    for train_inds, valid_inds in kf.split(np.arange(num_samples)):
        fold += 1
        print("Creating fold %d: Train %d, Valid %d\n" % (fold, train_inds.size, valid_inds.size))

        trainfile = HOME+"/data/cpdb/cpdb_6133_filter_train_"+str(fold)+".tfrecords"
        validfile = HOME+"/data/cpdb/cpdb_6133_filter_valid_"+str(fold)+".tfrecords"
        print("Writing ", trainfile)
        trainwriter = tf.python_io.TFRecordWriter(trainfile)

        for index in train_inds:
            example = tf.train.Example(features=tf.train.Features(feature={
                'seq_len': _int64_feature(seq_lengths[index]),
                'seq_data': _floats_feature(seqs[index, 0:num_features*seq_lengths[index]]),
                'label_data': _floats_feature(labels[index, 0:num_labels*seq_lengths[index]])}))
            trainwriter.write(example.SerializeToString())
        trainwriter.close()

        print("Writing ", validfile)
        validwriter = tf.python_io.TFRecordWriter(validfile)
        for index in valid_inds:
            example = tf.train.Example(features=tf.train.Features(feature={
                'seq_len': _int64_feature(seq_lengths[index]),
                'seq_data': _floats_feature(seqs[index, 0:num_features*seq_lengths[index]]),
                'label_data': _floats_feature(labels[index, 0:num_labels*seq_lengths[index]])}))
            validwriter.write(example.SerializeToString())
        validwriter.close()

def cpdb_513_to_tfrecord():
    """
    Convert the numpy array format for cpdb_513 to a TFRecord file.
    """

    data = np.load(HOME+"/data/cpdb/cpdb_513.npy.gz").reshape(-1, 700, 57)
    # get indices for train/valid sets
    num_samples = data.shape[0]

    # calculate mean/stdev for pssm features
    mu = np.mean(data[:, :, 35:56], axis=1, keepdims=True)
    std = np.std(data[:, :, 35:56], axis=1, keepdims=True)
    # scale the data
    data[:, :, 35:56] = (data[:, :, 35:56] - mu)/std

    seqs = np.concatenate([data[:, :, 0:22].copy(), data[:, :, 35:56].copy()], axis=2).reshape(num_samples, -1)
    labels = data[:, :, 22:31].copy().reshape(num_samples, 700, -1)

    num_features = 43
    num_labels = 9

    # Count the protein sequence lengths for all samples
    def get_length(seq_labels):
        assert seq_labels.shape == (700, 9)
        noseq = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 1.]])
        return np.logical_not(np.all(np.equal(seq_labels, noseq), axis=1)).sum()

    seq_lengths = [get_length(labels[l, :, :]) for l in range(num_samples)]

    # Flatten labels
    labels = labels.reshape(num_samples, -1)

    filename = HOME+"/data/cpdb/cpdb_513.tfrecords"
    print("Writing ", filename)
    writer = tf.python_io.TFRecordWriter(filename)

    for index in range(num_samples):
        example = tf.train.Example(features=tf.train.Features(feature={
            'seq_len': _int64_feature(seq_lengths[index]),
            'seq_data': _floats_feature(seqs[index, 0:num_features*seq_lengths[index]]),
            'label_data': _floats_feature(labels[index, 0:num_labels*seq_lengths[index]])}))
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == "__main__":
    cpdb_to_tfrecord()

