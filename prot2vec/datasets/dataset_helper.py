"""Utilities for input pipelines."""

from pathlib import Path
import tensorflow as tf, numpy as np
import .parsers as prs

def create_dataset(hparams, mode):
    """
    Create a tf.Dataset from a file.
    Args:
        hparams - Hyperparameters for the dataset
        mode    - the mode, one of tf.contrib.learn.ModeKeys.{TRAIN, EVAL, INFER}
    Returns:
        dataset - A tf.data.Dataset object
    """

    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        input_file = Path(hparams.train_file)
        shuffle = True
        batch_size = hparams.batch_size
        num_epochs = hparams.num_epochs
    elif mode == tf.contrib.learn.ModeKeys.EVAL:
        input_file = Path(hparams.valid_file)
        shuffle = False
        batch_size = hparams.batch_size
        num_epochs = 1
    else:
        input_file = Path(hparams.infer_file)
        shuffle = False
        num_epochs = 1
        batch_size = 1

    if input_file is None or input_file == "":
        print("Input file must be specified in create_dataset()!")
        exit()

    if hparams.model == "cpdb":
        parser = prs.cpdb_parser
    elif hparams.model == "copy":
        parser = prs.copytask_parser
    elif hparams.model == "autoenc":
        parser = prs.autoenc_parser

    # create the initial dataset from the file
    if input_file.suffix == ".tfrecords":
        dataset = tf.data.TFRecordDataset(str(input_file.absolute()))

    elif input_file.suffix == ".npz":
        dataset = load_dataset_from_npz(str(input_file.absolute()))
        parser = None

    # perform the appropriate transformations and return

    if shuffle:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=batch_size*100, count=num_epochs))
    else:
        dataset = dataset.repeat(num_epochs)

    if parser is not None:
        dataset = dataset.map(parser, num_parallel_calls=4)

    dataset = dataset.cache()

    dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=(tf.TensorShape([None, hparams.num_features]),
                           tf.TensorShape([None, hparams.num_labels]),
                           tf.TensorShape([None, hparams.num_labels]),
                           tf.TensorShape([]))

    dataset = dataset.prefetch(1)

    return dataset

def load_dataset_from_npz(filename):
    """
    Create a tf.data.Dataset source from a .npz file.
    Returns:
        dataset - a tf.data.Dataset object
    """

    npzfile = np.load(filename)
    sample_list = npzfile["arr_0"]
    num_values = sample_list[0].shape[1]
    def sample_gen():
        for s in sample_list:
            yield (s, s.shape[0])

    dataset = tf.data.Dataset.from_generator(sample_gen,
                                             (tf.float32, tf.int32),
                                             (tf.TensorShape([None, num_values]), tf.TensorShape([])))

    return dataset
