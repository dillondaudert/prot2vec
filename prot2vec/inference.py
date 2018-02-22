#!/usr/bin/env python3
import tensorflow as tf
import numpy as np, pandas as pd
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

    outputs = []
    while True:
        try:
            sample, output = infer_tuple.model.infer(infer_tuple.session)
            outputs.append((sample, output))
        except tf.errors.OutOfRangeError:
            print(" - Done -")
            break

    return outputs

if __name__ == "__main__":
    ckpt = "/home/dillon/thesis/models/prot2vec/copy/sched50k/ckpts-39100"
    input_file = "/home/dillon/data/synthetic/copy/infer_30L_10V.npz"
    output_file = ""
    hparams = get_hparams("copy")
    hparams.model = "copy"
    hparams.infer_file = input_file
#    hparams.num_features = 12
#    hparams.num_labels = 12
    hparams.batch_size = 1
    hparams.num_epochs = 1
    outputs = inference(ckpt, input_file, output_file, hparams)

    columns = ["sample_len", "output_len", "step_accuracy", "overall_accuracy"]
    rows = []
    for i, t in enumerate(outputs):
        s = t[0][0]
        p = t[1][0]
        diff = s - p[:s.shape[0], :s.shape[1]]
        # print lengths and accuracy of each sample
        sample_len = s.shape[0]
        output_len = p.shape[0]
        step_accuracy = np.mean(np.all(diff == 0., axis=1))
        overall_accuracy = np.mean(diff == 0.)
        rows.append((sample_len, output_len, step_accuracy, overall_accuracy))

    df = pd.DataFrame.from_records(rows, columns=columns)
    df.to_csv("sched50k_30L_infer_outputs.csv")

