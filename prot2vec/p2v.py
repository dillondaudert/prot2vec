#!/usr/bin/env python3
# prot2vec driver and command line
import argparse as ap
from pathlib import Path
from train import train
from utils.hparams import get_hparams

def main():

    # Define the main argument parser
    parser = ap.ArgumentParser(prog="p2v", description="prot2vec")

    subparsers = parser.add_subparsers(title='subcommands')

    # -- training subparser --
    parser_tr = subparsers.add_parser("train", help="Begin or continue training a model")

    parser_tr.add_argument('-m', '--model', nargs=1, choices=['cpdb', 'copy'],
                           help='specify the Model class')
    parser_tr.add_argument('--train_data', required=True,
                           help="training data file")
    parser_tr.add_argument('--valid_data', required=True,
                           help="validation data file")
    parser_tr.add_argument("--logdir", required=True,
                           help="the directory where model checkpoints and logs will\
                                 be saved")
    parser_tr.add_argument("-H", "--hparams", required=True,
                           help="file specifying the hyperparameters for the model")
    # TODO: Add optional flags for all hyperparameters
    # TODO: the --name flag default should be hparams_to_name
    parser_tr.add_argument("--name", default="default", help="name of this trained model")

    parser_tr.set_defaults(entry="train")

    args = parser.parse_args()

    if args.entry == "train":
        # run training
        logpath = Path(args.logdir).absolute()
        trainpath = Path(args.train_data)
        validpath = Path(args.valid_data)
        hparams = get_hparams(args.model)
        # TODO: hparams parser
        hparams.modeldir = str(Path(logpath, args.name))
        hparams.train_data = str(trainpath.absolute())
        hparams.valid_data = str(validpath.absolute())

        # TODO: print training info, and continue Y/n

        # TODO: call training driver
        print(args)


if __name__ == "__main__":
    main()


#basedir = hparams.logdir+"/LR%.3f_MG%1.1f_U%d_D%s_NL%d_NR%d_H%s_SD%s_DR%.2f_DP%d_OPT%s_%s" % \
#                                                                    (hparams.learning_rate,
#                                                                    hparams.max_gradient_norm,
#                                                                    hparams.num_units,
#                                                                    hparams.dense_input,
#                                                                    hparams.num_layers,
#                                                                    hparams.num_residual_layers,
#                                                                    hparams.train_helper,
#                                                                    hparams.sched_decay,
#                                                                    hparams.dropout,
#                                                                    hparams.depth,
#                                                                    hparams.optimizer,
#                                                                    hparams.tag)

