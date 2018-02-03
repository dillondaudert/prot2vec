# basic example of training a network end-to-end
import tensorflow as tf
from datasets import pssp_dataset
from model_helper import *
import model
from hparams.default import get_default_hparams

train_files = ["/home/dillon/data/cpdb/cpdb_6133_filter_train_%d.tfrecords" % (i) for i in range(1, 11)]
valid_files = ["/home/dillon/data/cpdb/cpdb_6133_filter_valid_%d.tfrecords" % (i) for i in range(1, 11)]
test_files = ["/home/dillon/data/cpdb/cpdb_513.tf_records"]

hparams = get_default_hparams()

train_graph = tf.Graph()
eval_graph = tf.Graph()

# build training graph
with train_graph.as_default():
    train_dataset = pssp_dataset(tf.constant(train_files[0], tf.string),
                                 tf.constant(True, tf.bool),
                                 batch_size=hparams.batch_size,
                                 num_epochs=hparams.num_epochs)
    train_iterator = train_dataset.make_initializable_iterator()

    train_model = model.Model(hparams=hparams,
                              iterator=train_iterator,
                              mode=tf.contrib.learn.ModeKeys.TRAIN)

    initializer = tf.global_variables_initializer()

with eval_graph.as_default():
    eval_dataset = pssp_dataset(tf.constant(valid_files[0], tf.string),
                                tf.constant(True, tf.bool),
                                batch_size=hparams.batch_size,
                                num_epochs=1)
    eval_iterator = eval_dataset.make_initializable_iterator()

    eval_model = model.Model(hparams=hparams,
                             iterator=eval_iterator,
                             mode=tf.contrib.learn.ModeKeys.EVAL)

checkpoints_path = "/home/dillon/thesis/models/prot2vec/model1"

train_sess = tf.Session(graph=train_graph)
eval_sess = tf.Session(graph=eval_graph)

train_sess.run([initializer])

# Train for num_epochs
for i in range(hparams.num_epochs):
    train_sess.run([train_iterator.initializer])
    while True:
        try:
            _, train_loss, global_step = train_model.train(train_sess)
            print("Step: %d, Training Loss: %f" % (global_step, train_loss))

            if global_step % 20 == 0:
                # evaluate progress
                checkpoint_path = train_model.saver.save(train_sess, checkpoints_path, global_step=global_step)
                eval_model.saver.restore(eval_sess, checkpoint_path)
                eval_sess.run(eval_iterator.initializer)
                eval_step = 1
                while True:
                    try:
                        eval_loss, eval_acc = eval_model.eval(eval_sess)
                        print("Eval Step: %d, Eval Loss: %f, Eval Accuracy: %f" % (eval_step,
                                                                                   eval_loss,
                                                                                   eval_acc))
                        eval_step += 1
                    except tf.errors.OutOfRangeError:
                        break

        except tf.errors.OutOfRangeError:
            print("- End of Epoch %d -" % (i))
            break

    # TODO: evaluate
