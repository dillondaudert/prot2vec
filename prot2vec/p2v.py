#!/usr/bin/env python3
# prot2vec driver and command line
import argparse as ap
from pathlib import Path
from train import train
from utils.hparams import *

def main():


    # Define the main argument parser
    parser = ap.ArgumentParser(prog="p2v", description="prot2vec",
                               argument_default=ap.SUPPRESS)

    subparsers = parser.add_subparsers(title='subcommands')

    # get the hyperparameter parser parent
    hp_parser = get_hparam_parser()

    # -- training subparser --
    tr_parser = subparsers.add_parser("train", help="Begin or continue training a model",
                                      parents=[hp_parser])

    tr_parser.add_argument("-H", "--hparams", required=True,
                           help="file specifying the hyperparameters for the model")
    tr_parser.add_argument("--logdir", required=True,
                           help="the directory where model checkpoints and logs will\
                                 be saved")
    tr_parser.add_argument("-n", "--name", required=True,
                           help="Name of model directory (logdir/name)")

    tr_parser.add_argument("-s", "--saved", type=str, default=None,
                           help="A checkpoint to a saved model to resume training.")

    tr_parser.set_defaults(entry="train")

    args = parser.parse_args()

    if args.entry == "train":
        # run training
        hparams = get_hparams(args.hparams)

        # replace any hparams specified as cl args
        for argkey in vars(args):
            if argkey in HPARAMS:
                vars(hparams)[argkey] = vars(args)[argkey]

        logpath = Path(args.logdir)
        hparams.modeldir = str(Path(logpath, args.name).absolute())
        if "train_file" in vars(hparams):
            trainpath = Path(hparams.train_file)
            hparams.train_file = str(trainpath.absolute())
        if "valid_file" in vars(hparams):
            validpath = Path(hparams.valid_file)
            hparams.valid_file = str(validpath.absolute())
        if args.saved is not None:
            hparams.saved = str(Path(args.saved).absolute())
        else:
            hparams.saved = None

        hparams_to_str(hparams)

        cont = input("Continue? [y]/n: ")
        if cont == "" or cont == "y":
            print("Continuing")
            train(hparams)
        else:
            print("Quitting.")


if __name__ == "__main__":
    main()
