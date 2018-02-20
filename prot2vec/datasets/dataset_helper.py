"""Utilities for input pipelines."""

from pathlib import Path
import tensorflow as tf
from .cpdb_dataset import pssp_dataset
from .synth_dataset import copytask_dataset

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

    # create the initial dataset from the file
    if input_file.suffix == ".tfrecords":
        dataset = tf.data.TFRecordDataset(str(input_file.absolute()))

    elif input_file.suffix == ".csv":
        # read in the data and create dataset
        dataset = load_dataset_from_csv(str(input_file.absolute()))

    # perform the appropriate transformations and return
    if hparams.model == "cpdb":
        dataset_creator = pssp_dataset
    elif hparams.model == "copy":
        dataset_creator = copytask_dataset

    dataset = dataset_creator(dataset=dataset,
                              shuffle=shuffle,
                              num_features=hparams.num_features,
                              num_labels=hparams.num_labels,
                              batch_size=batch_size,
                              num_epochs=num_epochs)

    return dataset

def load_dataset_from_csv(filename):
    return
