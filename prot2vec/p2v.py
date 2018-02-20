#!/usr/bin/env python3
# prot2vec driver and command line
from train import train
from utils.hparams import get_hparams

# TODO: Add command line arguments for hparams

hparams = get_hparams("default")
hparams.tag = "test"

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

train_files = ["/home/dillon/data/cpdb/cv_5/cpdb_6133_filter_train_%d.tfrecords" % (i) for i in range(1, 6)]
valid_files = ["/home/dillon/data/cpdb/cv_5/cpdb_6133_filter_valid_%d.tfrecords" % (i) for i in range(1, 6)]
test_files = ["/home/dillon/data/cpdb/cpdb_513.tf_records"]

#train_files = ["/home/dillon/data/synthetic/copy/train_25L_10V.tfrecords"]
#valid_files = ["/home/dillon/data/synthetic/copy/valid_25L_10V.tfrecords"]

hparams.train_file = train_files[0]
hparams.valid_file = valid_files[0]
hparams.test_file = test_files[0]

train(hparams)
