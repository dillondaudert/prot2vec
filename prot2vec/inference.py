#!/usr/bin/env python3
from utils.hparams import get_hparams
from datasets.dataset_helper import create_dataset
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




    return


if __name__ == "__main__":
    hparams = get_hparams("copy")
    input_file = "/home/dillon/data/synthetic/copy/infer_30L_10V.npz"
    ckpt = "/home/dillon/thesis/models/prot2vec/copy/sched50k/ckpts-39100.index"
    output_file = "sched50k_30L_10V_infer.csv"
    # ... other setup
    inference()
