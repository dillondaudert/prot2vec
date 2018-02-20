#!/usr/bin/env python3
# prot2vec driver and command line
from train import train
from utils.hparams import get_hparams

# TODO: Add command line arguments for hparams

hparams = get_hparams("copy")

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

train(hparams)
