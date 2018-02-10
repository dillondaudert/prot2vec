"""Sequence-to-sequence model using a dynamic RNN."""
import tensorflow as tf
import base_model
from model_helper import *
from metrics import streaming_confusion_matrix, cm_summary


class Model(base_model.BaseModel):

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
            self.eval_summary = tf.summary.merge_all("eval")

            # TODO: Add inference summaries

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=hparams.num_keep_ckpts)


    def _build_graph(self, hparams, scope=None):
        """Construct a sequence-to-sequence model.
        Args:
            hparams: The hyperparameters for configuration
            scope: The variable scope name for this subgraph, default "dynamic_seq2seq"
        Returns:
            A tuple with (logits, loss, metrics, update_ops)
        """

        enc_inputs, dec_inputs, dec_outputs, seq_len = self.iterator.get_next()

        with tf.variable_scope(scope or "dynamic_seq2seq", dtype=tf.float32):
            # create encoder
            dense_input_layer = tf.layers.Dense(hparams.num_units)

            if hparams.dense_input:
                enc_inputs = dense_input_layer(enc_inputs)

            enc_cells = create_rnn_cell(unit_type=hparams.unit_type,
                                        num_units=hparams.num_units,
                                        num_layers=hparams.num_layers,
                                        depth=hparams.depth,
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
                                        depth=hparams.depth,
                                        num_residual_layers=hparams.num_residual_layers,
                                        forget_bias=hparams.forget_bias,
                                        dropout=hparams.dropout,
                                        mode=self.mode)

            # output project layer
            projection_layer = tf.layers.Dense(hparams.num_labels, use_bias=False)

            if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
                if hparams.train_helper == "teacher":
                    # teacher forcing
                    helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_inputs,
                                                               sequence_length=tgt_seq_len)
                elif hparams.train_helper == "sched":
                    embedding = tf.eye(hparams.num_labels)
                    # scheduled sampling
                    helper = tf.contrib.seq2seq.\
                             ScheduledEmbeddingTrainingHelper(inputs=dec_inputs,
                                                              sequence_length=tgt_seq_len,
                                                              embedding=embedding,
                                                              sampling_probability=self.sample_probability,
                                                              )
            elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
                embedding = tf.eye(hparams.num_labels)
                helper = tf.contrib.seq2seq.\
                         ScheduledEmbeddingTrainingHelper(inputs=dec_inputs,
                                                          sequence_length=tgt_seq_len,
                                                          embedding=embedding,
                                                          sampling_probability=tf.constant(1.0))

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

            loss = (tf.reduce_sum(crossent*mask)/(hparams.batch_size*tf.reduce_mean(tf.cast(tgt_seq_len,
                                                                                            tf.float32))))

            metrics = []
            update_ops = []
            if self.mode == tf.contrib.learn.ModeKeys.EVAL:
                predictions = tf.argmax(input=logits, axis=-1)
                targets = tf.argmax(input=dec_outputs, axis=-1)
                acc, acc_update = tf.metrics.accuracy(predictions=predictions,
                                                      labels=targets,
                                                      weights=mask)
                # flatten for confusion matrix
                # TODO: remove the zero padding so that zeros aren't counted
                targets_flat = tf.reshape(targets, [-1])
                predictions_flat = tf.reshape(predictions, [-1])
                mask_flat = tf.reshape(mask, [-1])
                cm, cm_update = streaming_confusion_matrix(labels=targets_flat,
                                                           predictions=predictions_flat,
                                                           num_classes=hparams.num_labels,
                                                           weights=mask_flat)
                metrics = [acc, cm]
                update_ops = [acc_update, cm_update]

            return logits, loss, metrics, update_ops

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
                                                       tf.constant(260, dtype=tf.float32))))
        elif hparams.sched_decay == "inv_sig":
            k = tf.constant(60.)
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
