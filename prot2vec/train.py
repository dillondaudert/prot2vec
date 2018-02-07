#!/usr/bin/env python3
# basic example of training a network end-to-end
import tensorflow as tf
from datasets import pssp_dataset
from model_helper import *
import model
from hparams.default import get_default_hparams

modeldir = "/home/dillon/thesis/models/prot2vec/basic_3"
ckptsdir = modeldir+"/ckpts"
logdir = modeldir+"/log"

train_files = ["/home/dillon/data/cpdb/cpdb_6133_filter_train_%d.tfrecords" % (i) for i in range(1, 11)]
valid_files = ["/home/dillon/data/cpdb/cpdb_6133_filter_valid_%d.tfrecords" % (i) for i in range(1, 11)]
test_files = ["/home/dillon/data/cpdb/cpdb_513.tf_records"]

hparams = get_default_hparams()

train_graph = tf.Graph()
eval_graph = tf.Graph()

# build training graph
with train_graph.as_default():
    train_dataset = pssp_dataset(tf.constant(train_files[2], tf.string),
                                 tf.constant(True, tf.bool),
                                 batch_size=hparams.batch_size,
                                 num_epochs=hparams.num_epochs)
    train_iterator = train_dataset.make_initializable_iterator()

    train_model = model.Model(hparams=hparams,
                              iterator=train_iterator,
                              mode=tf.contrib.learn.ModeKeys.TRAIN)

    initializer = tf.global_variables_initializer()

with eval_graph.as_default():
    eval_dataset = pssp_dataset(tf.constant(valid_files[2], tf.string),
                                tf.constant(True, tf.bool),
                                batch_size=hparams.batch_size,
                                num_epochs=1)
    eval_iterator = eval_dataset.make_initializable_iterator()

    eval_model = model.Model(hparams=hparams,
                             iterator=eval_iterator,
                             mode=tf.contrib.learn.ModeKeys.EVAL)
    local_initializer = tf.local_variables_initializer()

# Summary writers
train_writer = tf.summary.FileWriter(logdir+"/train", train_graph)
eval_writer = tf.summary.FileWriter(logdir+"/eval")


train_sess = tf.Session(graph=train_graph)
eval_sess = tf.Session(graph=eval_graph)

train_sess.run([initializer])

# Train for num_epochs
for i in range(hparams.num_epochs):
    train_sess.run([train_iterator.initializer])
    while True:
        try:
            _, train_loss, global_step, summary = train_model.train(train_sess)
            # write train summaries
            train_writer.add_summary(summary, global_step)
            print("Step: %d, Training Loss: %f" % (global_step, train_loss))

            if global_step % 20 == 0:
                # TODO: Need to write eval summary not on eval minibatch, but entire eval
                # evaluate progress
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
                        eval_writer.add_summary(eval_summary, global_step)
                        break

        except tf.errors.OutOfRangeError:
            print("- End of Epoch %d -" % (i))
            break
