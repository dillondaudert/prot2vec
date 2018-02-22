#!/usr/bin/env python3
import tensorflow as tf
from utils.hparams import get_hparams
from datasets.dataset_helper import create_dataset
from model_helper import create_model
# functions for using trained models to generate predictions


def inference(ckpt,
              input_file,
              output_file,
              hparams):
    """
    Generate predictions.
    Args:
        ckpt        - the model checkpoint file to use
        input_file  - the input file with the inference data
        output_file - the name of the file to save the predictions to
        hparams     - hyperparameters for this model
    """

    infer_tuple = create_model(hparams, tf.contrib.learn.ModeKeys.INFER)
    infer_tuple.model.saver.restore(infer_tuple.session, ckpt)
    infer_tuple.session.run([infer_tuple.iterator.initializer])

    while True:
        try:
            sample, output = infer_tuple.model.infer(infer_tuple.session)
            print("Sample: ", sample)
            print("Output: ", output)
        except tf.errors.OutOfRangeError:
            print(" - Done -")
            break

if __name__ == "__main__":
    ckpt = "/home/dillon/thesis/models/prot2vec/copy/sched50k/ckpts-39100"
    input_file = "/home/dillon/data/synthetic/copy/infer_30L_10V.npz"
    output_file = ""
    hparams = get_hparams("copy")
    hparams.model = "copy"
    hparams.infer_file = input_file
    hparams.batch_size = 1
    hparams.num_epochs = 1
    inference(ckpt, input_file, output_file, hparams)
