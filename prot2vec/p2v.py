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

    tr_parser.add_argument('-m', '--model', choices=['cpdb', 'copy'],
                           type=str, help='specify the Model class')
    tr_parser.add_argument('--train_data', required=True,
                           help="training data file")
    tr_parser.add_argument('--valid_data', required=True,
                           help="validation data file")
    tr_parser.add_argument("--logdir", required=True,
                           help="the directory where model checkpoints and logs will\
                                 be saved")
    tr_parser.add_argument("-H", "--hparams", required=True,
                           help="file specifying the hyperparameters for the model")
    # TODO: the --name flag default should be hparams_to_name
    tr_parser.add_argument("--name", default="default", help="name of this trained model")

    tr_parser.set_defaults(entry="train")

    args = parser.parse_args()

    if args.entry == "train":
        # run training
        logpath = Path(args.logdir)
        trainpath = Path(args.train_data)
        validpath = Path(args.valid_data)
        hparams = get_hparams(args.model)
        hparams.model = args.model
        hparams.modeldir = str(Path(logpath, args.name).absolute())
        hparams.train_file = str(trainpath.absolute())
        hparams.valid_file = str(validpath.absolute())

        # replace any hparams specified as cl args
        for argkey in vars(args):
            if argkey in HPARAMS:
                vars(hparams)[argkey] = vars(args)[argkey]

        hparams_to_str(hparams)

        cont = input("Continue? [y]/n: ")
        if cont == "" or cont == "y":
            print("Continuing")
            train(hparams)
        else:
            print("Quitting.")


if __name__ == "__main__":
    main()
