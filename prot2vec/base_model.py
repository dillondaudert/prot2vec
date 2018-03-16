"""BaseModel class from which all models inherit."""

import abc
import tensorflow as tf
import base_model
from model_helper import *


class BaseModel(object):

    def __init__(self, hparams, iterator, mode, target_lookup_table=None, scope=None):
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
            self.update_metrics = res[3]

        elif self.mode == tf.contrib.learn.ModeKeys.INFER:
            self.inputs = res[0]
            self.sample_ids = res[1]

        params = tf.trainable_variables()

        # training update ops
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.learning_rate = hparams.learning_rate

            # optimizer
            if hparams.optimizer == "adam":
                opt = tf.train.AdamOptimizer(self.learning_rate)
            elif hparams.optimizer == "sgd":
                if "momentum" in vars(hparams):
                    self.momentum = hparams.momentum
                else:
                    self.momentum = 0.

                if self.momentum == 0.:
                    opt = tf.train.GradientDescentOptimizer(self.learning_rate)
                else:
                    opt = tf.train.MomentumOptimizer(self.learning_rate,
                                                     self.momentum,
                                                     use_nesterov=True)
            elif hparams.optimizer == "adadelta":
                opt = tf.train.AdadeltaOptimizer(self.learning_rate,
                                                 epsilon=1e-06)
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
            if hparams.train_helper == "sched":
                tf.summary.scalar("sample_probability", self.sample_probability, collections=["train"])
            self.train_summary = tf.summary.merge_all("train")

        elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
            # Evaluation summaries
            tf.summary.scalar("eval_loss", self.eval_loss, collections=["eval"])
            tf.summary.scalar("accuracy", self.accuracy, collections=["eval"])
            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var, collections=["eval"])
            self.eval_summary = tf.summary.merge_all("eval")


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
                                                       tf.constant(30000, dtype=tf.float32))))
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


    def train_with_profile(self, sess, writer):
        """Do a single training step, with profiling"""
        assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        retvals = sess.run([self.update,
                            self.train_loss,
                            self.global_step,
                            self.sample_probability,
                            self.train_summary], options=run_options,
                                              run_metadata=run_metadata)
        writer.add_run_metadata(run_metadata, "step "+str(retvals[2]), retvals[2])
        return retvals


    def eval(self, sess):
        """Evaluate the model."""
        assert self.mode == tf.contrib.learn.ModeKeys.EVAL
        return sess.run([self.eval_loss,
                         self.accuracy,
                         self.eval_summary,
                         self.update_metrics])

    def infer(self, sess):
        """Inference."""
        assert self.mode == tf.contrib.learn.ModeKeys.INFER
        return sess.run([self.inputs, self.sample_ids])
