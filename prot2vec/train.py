# basic example of training a network end-to-end
from time import process_time
import tensorflow as tf, numpy as np
from dataset import pssp_dataset
from synth_dataset import copytask_dataset
from model_helper import create_model
from utils.hparams import get_hparams
from utils.vocab_utils import create_table

def train(hparams):
    """Build and train the model as specified in hparams"""

    basedir = hparams.logdir
    modeldir = basedir + "/" + hparams.tag
    ckptsdir = modeldir+"/ckpts"
    logdir = modeldir

    # build training and eval graphs
    train_tuple = create_model(hparams, tf.contrib.learn.ModeKeys.TRAIN)
    eval_tuple = create_model(hparams, tf.contrib.learn.ModeKeys.EVAL)

    with train_tuple.graph.as_default():
        initializer = tf.global_variables_initializer()

    with eval_tuple.graph.as_default():
        local_initializer = tf.local_variables_initializer()

    # Summary writers
    summary_writer = tf.summary.FileWriter(logdir, train_tuple.graph)

    train_tuple.session.run([initializer])

    # Train for num_epochs
    for i in range(hparams.num_epochs):
        train_tuple.session.run([train_tuple.iterator.initializer])
        start_time = process_time()
        step_time = []
        while True:
            try:
                curr_time = process_time()
                _, train_loss, global_step, _, summary = train_tuple.model.train(train_tuple.session)
                step_time.append(process_time() - curr_time)

                # write train summaries
                if global_step == 1:
                    summary_writer.add_summary(summary, global_step)
                if global_step % 5 == 0:
                    summary_writer.add_summary(summary, global_step)
                    print("Step: %d, Training Loss: %f, Avg Step/Sec: %2.2f" % (global_step, train_loss, np.mean(step_time)))

                if global_step % 20 == 0:
                    checkpoint_path = train_tuple.model.saver.save(train_tuple.session,
                                                                   ckptsdir,
                                                                   global_step=global_step)
                    eval_tuple.model.saver.restore(eval_tuple.session, checkpoint_path)
                    eval_tuple.session.run([eval_tuple.iterator.initializer, local_initializer])
                    while True:
                        try:
                            eval_loss, eval_acc, eval_summary, _ = eval_tuple.model.eval(eval_tuple.session)
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
