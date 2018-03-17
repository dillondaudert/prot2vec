"""Sequence-to-sequence model using a dynamic RNN."""
import tensorflow as tf
import base_model
import model_helper as mdl_help
from metrics import streaming_confusion_matrix, cm_summary

__all__ = [
    "CopyModel",
]

class CopyModel(base_model.BaseModel):
    """A sequence-to-sequence model for the copy task on synthetic data.
    """

    def _build_graph(self, hparams, scope=None):
        """Construct the train, evaluation, and inference graphs.
        Args:
            hparams: The hyperparameters for configuration
            scope: The variable scope name for this subgraph, default "dynamic_seq2seq"
        Returns:
            A tuple with (logits, loss, metrics, update_ops)
        """

        sample = self.iterator.get_next()
        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            enc_inputs, dec_inputs, dec_outputs, seq_len = sample
        else:
            # At inference, only two inputs are given
            enc_inputs, seq_len, dec_start = sample
            #indices = (hparams.num_labels-1)*tf.ones([enc_inputs.shape[0]], tf.int32)
            #depth = hparams.num_labels
            #dec_start = tf.one_hot(indices, depth, axis=-1)

        with tf.variable_scope(scope or "dynamic_seq2seq", dtype=tf.float32):
            # create encoder
            dense_input_layer = tf.layers.Dense(hparams.num_units)

            if hparams.dense_input:
                enc_inputs = dense_input_layer(enc_inputs)

            enc_cells = mdl_help.create_rnn_cell(unit_type=hparams.unit_type,
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

            # create decoder
            dec_cells = mdl_help.create_rnn_cell(unit_type=hparams.unit_type,
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
                    # scheduled sampling
                    helper = tf.contrib.seq2seq.\
                             ScheduledOutputTrainingHelper(inputs=dec_inputs,
                                                           sequence_length=tgt_seq_len,
                                                           sampling_probability=self.sample_probability,
                                                           next_inputs_fn=lambda x: mdl_help.multiclass_sample(x),
                                                           )
            elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
                helper = tf.contrib.seq2seq.\
                         ScheduledOutputTrainingHelper(inputs=dec_inputs,
                                                       sequence_length=tgt_seq_len,
                                                       sampling_probability=tf.constant(1.0),
                                                       next_inputs_fn=lambda x: mdl_help.multiclass_sample(x))

            else: # running inference
                def end_fn(sample_ids):
                    are_eq = tf.equal(dec_start, sample_ids)
                    reduce_eq = tf.reduce_all(are_eq, axis=-1)
                    return reduce_eq
                helper = tf.contrib.seq2seq.\
                         InferenceHelper(sample_fn=lambda x: mdl_help.multiclass_sample(x),
                                         sample_shape=[hparams.num_labels],
                                         sample_dtype=tf.float32,
                                         start_inputs=dec_start,
                                         end_fn=lambda x: end_fn(x))

            max_len = tf.reduce_max(tgt_seq_len)

            decoder = tf.contrib.seq2seq.BasicDecoder(cell=dec_cells,
                                                      helper=helper,
                                                      initial_state=enc_state,
                                                      output_layer=projection_layer)

            # run decoder
            final_outputs, final_states, final_lengths = tf.contrib.seq2seq.dynamic_decode(
                    decoder=decoder,
                    impute_finished=True,
                    maximum_iterations=tf.constant(2)*max_len,
                    scope="decoder")

            logits = final_outputs.rnn_output
            sample_ids = final_outputs.sample_id

            if self.mode == tf.contrib.learn.ModeKeys.INFER:
                return enc_inputs, sample_ids

            # mask out entries longer than target sequence length
            mask = tf.expand_dims(tf.sequence_mask(tgt_seq_len, dtype=tf.float32), axis=-1)

            #stop gradient thru labels by crossent op
            labels = tf.stop_gradient(dec_outputs)

            crossent = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                               labels=labels,
                                                               name="crossent")


            loss = tf.reduce_sum((crossent * mask) / tf.expand_dims(
                tf.expand_dims(tf.cast(tgt_seq_len, tf.float32), -1), -1))/hparams.batch_size

#            loss = tf.reduce_sum(crossent*mask)/(hparams.batch_size*tf.reduce_mean(tf.cast(tgt_seq_len,
#                                                                                           tf.float32)))

            metrics = []
            update_ops = []
            if self.mode == tf.contrib.learn.ModeKeys.EVAL:
                # for predictions, we will scale the logits and then count each class as
                # active if it is over .5
                predictions = mdl_help.multiclass_prediction(logits)
                targets = dec_outputs
                acc, acc_update = tf.metrics.accuracy(predictions=predictions,
                                                      labels=targets,
                                                      weights=mask)
                metrics = [acc]
                update_ops = [acc_update]

            return logits, loss, metrics, update_ops
