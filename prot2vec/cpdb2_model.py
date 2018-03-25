"""Sequence-to-sequence model using a dynamic RNN."""
import tensorflow as tf
import base_model
import model_helper as mdl_help
from metrics import streaming_confusion_matrix, cm_summary

__all__ = [
    "CPDB2Model",
]

class CPDB2Model(base_model.BaseModel):
    """A sequence-to-sequence model for the CPDB2 data.
    """

    def _build_graph(self, hparams, scope=None):
        """Construct the train, evaluation, and inference graphs.
        Args:
            hparams: The hyperparameters for configuration
            scope: The variable scope name for this subgraph, default "dynamic_seq2seq"
        Returns:
            A tuple with (logits, loss, metrics, update_ops)
        """

        src_in_ids, tgt_in_ids, tgt_out_ids, src_len, tgt_len = self.iterator.get_next()

        # embeddings
        self.init_embeddings()

        src_in = tf.nn.embedding_lookup(self.enc_embedding, src_in_ids)
        tgt_in = tf.nn.embedding_lookup(self.dec_embedding, tgt_in_ids)
        tgt_out = tf.nn.embedding_lookup(self.dec_embedding, tgt_out_ids)

        with tf.variable_scope(scope or "dynamic_seq2seq", dtype=tf.float32):
            # create encoder
            dense_input_layer = tf.layers.Dense(hparams.num_units)

            if hparams.dense_input:
                src_in = dense_input_layer(src_in)

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
                                                       inputs=src_in,
                                                       sequence_length=src_len,
                                                       swap_memory=True,
                                                       dtype=tf.float32,
                                                       scope="encoder")

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
                    helper = tf.contrib.seq2seq.TrainingHelper(inputs=tgt_in,
                                                               sequence_length=tgt_len)
                elif hparams.train_helper == "sched":
                    embedding = tf.eye(hparams.num_labels)
                    # scheduled sampling
                    helper = tf.contrib.seq2seq.\
                             ScheduledEmbeddingTrainingHelper(inputs=tgt_in,
                                                              sequence_length=tgt_len,
                                                              embedding=embedding,
                                                              sampling_probability=self.sample_probability,
                                                              )
            elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
                embedding = tf.eye(hparams.num_labels)
                helper = tf.contrib.seq2seq.\
                         ScheduledEmbeddingTrainingHelper(inputs=tgt_in,
                                                          sequence_length=tgt_len,
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
            mask = tf.sequence_mask(tgt_len, dtype=tf.float32)

            #stop gradient thru labels by crossent op
            labels = tf.stop_gradient(tgt_out)

            crossent = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                                  labels=labels,
                                                                  name="crossent")

#            loss = (tf.reduce_sum(crossent*mask)/(hparams.batch_size*tf.reduce_mean(tf.cast(tgt_len,
#                                                                                            tf.float32))))

            loss = tf.reduce_sum((crossent * mask) / tf.expand_dims(
                tf.expand_dims(tf.cast(tgt_len, tf.float32), -1), -1)) / hparams.batch_size

            metrics = []
            update_ops = []
            if self.mode == tf.contrib.learn.ModeKeys.EVAL:
                predictions = tf.argmax(input=logits, axis=-1)
                targets = tf.argmax(input=tgt_out, axis=-1)
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

    def init_embeddings(self):
        """
        Initialize the embedding variables.
        """
        self.enc_embedding, self.dec_embedding = mdl_help.create_embeddings("cpdb2")
