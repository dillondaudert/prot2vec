"""Sequence-to-sequence model using a dynamic RNN."""
import tensorflow as tf
import base_model
from model_helper import *


class Model(base_model.BaseModel):

    def __init__(self, hparams, iterator, mode, scope=None):
        self.hparams = hparams
        self.iterator = iterator
        self.mode = mode

        res = self._build_graph(hparams, scope=scope)

        # Graph losses
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.train_loss = res[1]

        elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
            self.eval_loss = res[1]
            self.accuracy = res[2]
            self.conf_matrix = res[3]
        elif self.mode == tf.contrib.learn.ModeKeys.INFER:
            # TODO: Implement Inference
            raise NotImplementedError("Inference not implemented yet!")

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        params = tf.trainable_variables()

        # training update ops
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.learning_rate = hparams.learning_rate

            # optimizer
            if hparams.optimizer == "adam":
                opt = tf.train.AdamOptimizer(self.learning_rate)
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
            with tf.variable_scope("train_summaries"):
                grad_summary = [tf.summary.scalar("grad_norm", gradient_norm),]
                self.train_summary = tf.summary.merge([
    #                tf.summary.scalar("lr", self.learning_rate),
                    tf.summary.scalar("train_loss", self.train_loss),
                    ] + grad_summary)

        elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
            # Evaluation summaries
            with tf.variable_scope("eval_summaries"):
                self.eval_summary = tf.summary.merge([
                    tf.summary.scalar("eval_loss", self.eval_loss),
                    tf.summary.scalar("accuracy", self.accuracy),])
    #                tf.summary.image("conf_matrix", self.conf_matrix)])


            # TODO: Add inference summaries

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=hparams.num_keep_ckpts)


    def _build_graph(self, hparams, scope=None):
        """Construct a sequence-to-sequence model.
        Args:
            hparams: The hyperparameters for configuration
            scope: The variable scope name for this subgraph, default "dynamic_seq2seq"
        Returns:
            A tuple with (logits, loss, accuracy)
        """

        enc_inputs, dec_inputs, dec_outputs, seq_len = self.iterator.get_next()

        with tf.variable_scope(scope or "dynamic_seq2seq", dtype=tf.float32):
            # create encoder
            enc_cells = create_rnn_cell(unit_type=hparams.unit_type,
                                        num_units=hparams.num_units,
                                        num_layers=hparams.num_layers,
                                        num_residual_layers=hparams.num_residual_layers,
                                        forget_bias=hparams.forget_bias,
                                        dropout=hparams.dropout,
                                        mode=self.mode)

            # run encoder
            enc_outputs, enc_state = tf.nn.dynamic_rnn(cell=enc_cells,
                                                       inputs=enc_inputs,
                                                       sequence_length=seq_len,
                                                       dtype=tf.float32,
                                                       scope="encoder")

            tgt_seq_len = tf.add(seq_len, tf.constant(1, tf.int32))

            # TODO: Add Inference decoder

            # create decoder
            dec_cells = create_rnn_cell(unit_type=hparams.unit_type,
                                        num_units=hparams.num_units,
                                        num_layers=hparams.num_layers,
                                        num_residual_layers=hparams.num_residual_layers,
                                        forget_bias=hparams.forget_bias,
                                        dropout=hparams.dropout,
                                        mode=self.mode)

            # output project layer
            projection_layer = tf.layers.Dense(hparams.num_labels, use_bias=False)

            # teacher forcing
            helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_inputs,
                                                       sequence_length=tgt_seq_len)

            decoder = tf.contrib.seq2seq.BasicDecoder(cell=dec_cells,
                                                      helper=helper,
                                                      initial_state=enc_state,
                                                      output_layer=projection_layer)

            # run decoder
            final_outputs, final_states, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder=decoder,
                    impute_finished=True,
                    scope="decoder")

            logits = final_outputs.rnn_output

            # mask out entries longer than target sequence length
            mask = tf.sequence_mask(tgt_seq_len, dtype=tf.float32)

            crossent = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                                  labels=dec_outputs,
                                                                  name="crossent")

            loss = (tf.reduce_sum(crossent*mask)/hparams.batch_size)

            #
            accuracy = None
            conf_matrix = None
            if self.mode == tf.contrib.learn.ModeKeys.EVAL:
                predictions = tf.argmax(input=logits, axis=-1)
                targets = tf.argmax(input=dec_outputs, axis=-1)
                accuracy = tf.contrib.metrics.accuracy(predictions=predictions,
                                                       labels=targets)
#                conf_matrix = tf.confusion_matrix(labels=targets,
#                                                  predictions=predictions,
#                                                  name="conf_matrix")

            return logits, loss, accuracy, conf_matrix


    def train(self, sess):
        """Do a single training step."""
        assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
        return sess.run([self.update,
                         self.train_loss,
                         self.global_step,
                         self.train_summary])


    def eval(self, sess):
        """Evaluate the model."""
        assert self.mode == tf.contrib.learn.ModeKeys.EVAL
        return sess.run([self.eval_loss,
                         self.accuracy,
                         self.eval_summary])
