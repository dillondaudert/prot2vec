#!/usr/bin/env python3
# basic example of training a network end-to-end
from time import process_time
import tensorflow as tf, numpy as np
from dataset import pssp_dataset
from synth_dataset import copytask_dataset
from model_helper import *
import model
from utils.hparams import get_hparams
from utils.vocab_utils import create_table

hparams = get_hparams("copy")
hparams.tag = ""

basedir = hparams.logdir+"/LR%.3f_MG%1.1f_U%d_D%s_NL%d_NR%d_H%s_SD%s_DR%.2f_DP%d_OPT%s_%s" % \
                                                                    (hparams.learning_rate,
                                                                    hparams.max_gradient_norm,
                                                                    hparams.num_units,
                                                                    hparams.dense_input,
                                                                    hparams.num_layers,
                                                                    hparams.num_residual_layers,
                                                                    hparams.train_helper,
                                                                    hparams.sched_decay,
                                                                    hparams.dropout,
                                                                    hparams.depth,
                                                                    hparams.optimizer,
                                                                    hparams.tag)

#train_files = ["/home/dillon/data/cpdb/cv_5/cpdb_6133_filter_train_%d.tfrecords" % (i) for i in range(1, 6)]
#valid_files = ["/home/dillon/data/cpdb/cv_5/cpdb_6133_filter_valid_%d.tfrecords" % (i) for i in range(1, 6)]
#test_files = ["/home/dillon/data/cpdb/cpdb_513.tf_records"]

train_files = ["/home/dillon/data/synthetic/copy/train_25L_10V.tfrecords"]
valid_files = ["/home/dillon/data/synthetic/copy/valid_25L_10V.tfrecords"]


def train_cv(fold):
    """Perform training for a fold in the cross validation"""

    modeldir = basedir + "/" + str(fold)
    ckptsdir = modeldir+"/ckpts"
    logdir = modeldir

    train_graph = tf.Graph()
    eval_graph = tf.Graph()

    # build training graph
    with train_graph.as_default():
        train_dataset = pssp_dataset(tf.constant(train_files[fold-1], tf.string),
                                     tf.constant(True, tf.bool),
                                     batch_size=hparams.batch_size,
                                     num_epochs=hparams.num_epochs)
        train_iterator = train_dataset.make_initializable_iterator()
        ss_table = create_table("ss")

        train_model = model.Model(hparams=hparams,
                                  iterator=train_iterator,
                                  mode=tf.contrib.learn.ModeKeys.TRAIN,
                                  target_lookup_table=ss_table)

        initializer = tf.global_variables_initializer()
        tables_initializer = tf.tables_initializer()

    with eval_graph.as_default():
        eval_dataset = pssp_dataset(tf.constant(valid_files[fold-1], tf.string),
                                    tf.constant(True, tf.bool),
                                    batch_size=hparams.batch_size,
                                    num_epochs=1)
        eval_iterator = eval_dataset.make_initializable_iterator()
        ss_table = create_table("ss")

        eval_model = model.Model(hparams=hparams,
                                 iterator=eval_iterator,
                                 mode=tf.contrib.learn.ModeKeys.EVAL,
                                 target_lookup_table=ss_table)
        local_initializer = tf.local_variables_initializer()

    # Summary writers
    summary_writer = tf.summary.FileWriter(logdir, train_graph)


    train_sess = tf.Session(graph=train_graph)
    eval_sess = tf.Session(graph=eval_graph)

    train_sess.run([initializer, tables_initializer])

    # Train for num_epochs
    for i in range(hparams.num_epochs):
        train_sess.run([train_iterator.initializer])
        start_time = process_time()
        step_time = []
        while True:
            try:
                curr_time = process_time()
                _, train_loss, global_step, _, summary = train_model.train(train_sess)
                step_time.append(process_time() - curr_time)

                # write train summaries
                if global_step == 1:
                    summary_writer.add_summary(summary, global_step)
                if global_step % 5 == 0:
                    summary_writer.add_summary(summary, global_step)
                    print("Step: %d, Training Loss: %f, Avg Step/Sec: %2.2f" % (global_step, train_loss, np.mean(step_time)))

                if global_step % 20 == 0:
                    checkpoint_path = train_model.saver.save(train_sess,
                                                             ckptsdir,
                                                             global_step=global_step)
                    eval_model.saver.restore(eval_sess, checkpoint_path)
                    eval_sess.run([eval_iterator.initializer, local_initializer])
                    while True:
                        try:
                            eval_loss, eval_acc, eval_summary, _ = eval_model.eval(eval_sess)
                            # summary_writer.add_summary(summary, global_step)
                        except tf.errors.OutOfRangeError:
                            print("Step: %d, Eval Loss: %f, Eval Accuracy: %f" % (global_step,
                                                                                  eval_loss,
                                                                                  eval_acc))
                            summary_writer.add_summary(eval_summary, global_step)
                            break

            except tf.errors.OutOfRangeError:
                print("- End of Epoch %d -" % (i+1))
                break
        summary_writer.flush()

    # End of training
    summary_writer.close()
    print("Total Training Time: %4.2f" % (process_time() - start_time))

print("Saving to %s" % (basedir))

for i in range(1, 2):
    print("TRAINING FOLD %d" % i)
    train_cv(i)
