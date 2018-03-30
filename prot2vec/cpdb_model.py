"""Sequence-to-sequence model using a dynamic RNN."""
import tensorflow as tf
import base_model
import model_helper as mdl_help
from metrics import streaming_confusion_matrix, cm_summary

__all__ = [
    "CPDBModel",
]

class CPDBModel(base_model.BaseModel):
    """A sequence-to-sequence model for the CPDB data.
    """

    def _build_graph(self, hparams, scope=None):
        """Construct the train, evaluation, and inference graphs.
        Args:
            hparams: The hyperparameters for configuration
            scope: The variable scope name for this subgraph, default "dynamic_seq2seq"
        Returns:
            A tuple with (logits, loss, metrics, update_ops)
        """

        enc_inputs, dec_inputs, dec_outputs, seq_len = self.iterator.get_next()

        # get the size of the batch
        batch_size = tf.shape(enc_inputs)[0]

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
                                                       swap_memory=True,
                                                       dtype=tf.float32,
                                                       scope="encoder")

            tgt_seq_len = tf.add(seq_len, tf.constant(1, tf.int32))

            # TODO: Add Inference decoder
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
                    swap_memory=True,
                    scope="decoder")

            logits = final_outputs.rnn_output

            # mask out entries longer than target sequence length
            mask = tf.sequence_mask(tgt_seq_len, dtype=tf.float32)

            #stop gradient thru labels by crossent op
            labels = tf.stop_gradient(dec_outputs)

            crossent = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                                  labels=labels,
                                                                  name="crossent")

#            loss = (tf.reduce_sum(crossent*mask)/(hparams.batch_size*tf.reduce_mean(tf.cast(tgt_seq_len,
#                                                                                            tf.float32))))


            loss = tf.reduce_sum((crossent * mask) / tf.expand_dims(
                tf.expand_dims(tf.cast(tgt_seq_len, tf.float32), -1), -1)) / tf.cast(batch_size, tf.float32)

            metrics = []
            update_ops = []
            if self.mode == tf.contrib.learn.ModeKeys.EVAL:
                predictions = tf.argmax(input=logits, axis=-1)
                targets = tf.argmax(input=dec_outputs, axis=-1)
                acc, acc_update = tf.metrics.accuracy(predictions=predictions,
                                                      labels=targets,
                                                      weights=mask)
                # flatten for confusion matrix
                targets_flat = tf.reshape(targets, [-1])
                predictions_flat = tf.reshape(predictions, [-1])
                mask_flat = tf.reshape(mask, [-1])
                cm, cm_update = streaming_confusion_matrix(labels=targets_flat,
                                                           predictions=predictions_flat,
                                                           num_classes=hparams.num_labels,
                                                           weights=mask_flat)
                tf.add_to_collection("eval", cm_summary(cm, hparams.num_labels))
                metrics = [acc, cm]
                update_ops = [acc_update, cm_update]

            return logits, loss, metrics, update_ops
