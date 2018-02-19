"""BaseModel class from which all models inherit."""

import abc
import tensorflow as tf
import base_model
from model_helper import *
from metrics import streaming_confusion_matrix, cm_summary


class BaseModel(object):

    def __init__(self, hparams, iterator, mode, target_lookup_table, scope=None):
        self.hparams = hparams
        self.iterator = iterator
        self.mode = mode
        self.target_lookup_table = target_lookup_table
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.sample_probability = self._get_sample_probability(hparams)

        # set initializer
        if hparams.initializer == "glorot_normal":
            initializer = tf.glorot_normal_initializer()
        elif hparams.initializer == "glorot_uniform":
            initializer = tf.glorot_uniform_initializer()
        elif hparams.initializer == "orthogonal":
            initializer = tf.orthogonal_initializer()

        tf.get_variable_scope().set_initializer(initializer)

        res = self._build_graph(hparams, scope=scope)

        # Graph losses
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.train_loss = res[1]

        elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
            self.eval_loss = res[1]
            self.accuracy = res[2][0]
            self.confusion = res[2][1]
            self.update_metrics = res[3]

        elif self.mode == tf.contrib.learn.ModeKeys.INFER:
            # TODO: Implement Inference
            raise NotImplementedError("Inference not implemented yet!")

        params = tf.trainable_variables()

        # training update ops
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.learning_rate = hparams.learning_rate
            self.momentum = hparams.momentum

            # optimizer
            if hparams.optimizer == "adam":
                opt = tf.train.AdamOptimizer(self.learning_rate)
            elif hparams.optimizer == "sgd":
                if self.momentum == 0.:
                    opt = tf.train.GradientDescentOptimizer(self.learning_rate)
                else:
                    opt = tf.train.MomentumOptimizer(self.learning_rate,
                                                     self.momentum,
                                                     use_nesterov=True)
            else:
                raise ValueError("Optimizer %s not recognized!" % (hparams.optimizer))

            # gradients
            gradients = tf.gradients(self.train_loss,
                                     params,
                                     colocate_gradients_with_ops=hparams.colocate_gradients_with_ops)

            clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, hparams.max_gradient_norm)


            self.update = opt.apply_gradients(zip(clipped_gradients, params),
                                              global_step=self.global_step)

            # Summaries
            tf.summary.scalar("grad_norm", gradient_norm, collections=["train"])
            tf.summary.scalar("train_loss", self.train_loss, collections=["train"])
            tf.summary.scalar("sample_probability", self.sample_probability, collections=["train"])
            self.train_summary = tf.summary.merge_all("train")

        elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
            # Evaluation summaries
            tf.summary.scalar("eval_loss", self.eval_loss, collections=["eval"])
            tf.summary.scalar("accuracy", self.accuracy, collections=["eval"])
            tf.add_to_collection("eval", cm_summary(self.confusion, hparams.num_labels))
            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var, collections=["eval"])
            self.eval_summary = tf.summary.merge_all("eval")

            # TODO: Add inference summaries

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=hparams.num_keep_ckpts)

    @abc.abstractmethod
    def _build_graph(self, hparams, scope):
        """Subclasses must implement this.
        Args:
            hparams: The hyperparameters for configuration
            scope: The variable scope name for this subgraph, default "dynamic_seq2seq"
        Returns:
            A tuple with (logits, loss, metrics, update_ops)
        """
        pass

    def _get_sample_probability(self, hparams):
        """Get the probability for sampling from the outputs instead of
        ground truth."""

        eps = tf.constant(0.99)
        if hparams.sched_decay == "expon":
            # exponential decay
            eps = tf.pow(eps, tf.cast(self.global_step, tf.float32))
        elif hparams.sched_decay == "linear":
            min_eps = tf.constant(0.035)
            eps = tf.maximum(min_eps, (eps - tf.divide(tf.cast(self.global_step, tf.float32),
                                                       tf.constant(4600, dtype=tf.float32))))
        elif hparams.sched_decay == "inv_sig":
            k = tf.constant(90.)
            start_offset = tf.constant(1.4)
            eps = (k / (k + tf.exp(tf.cast(self.global_step, tf.float32)/k)))/start_offset
        sample_probability = tf.constant(1.) - eps

        return sample_probability



    def train(self, sess):
        """Do a single training step."""
        assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
        return sess.run([self.update,
                         self.train_loss,
                         self.global_step,
                         self.sample_probability,
                         self.train_summary])


    def eval(self, sess):
        """Evaluate the model."""
        assert self.mode == tf.contrib.learn.ModeKeys.EVAL
        return sess.run([self.eval_loss,
                         self.accuracy,
                         self.eval_summary,
                         self.update_metrics])
