"""Sequence-to-sequence model using a dynamic RNN."""
import tensorflow as tf
import base_model
import model_helper as mdl_help
from metrics import streaming_confusion_matrix, cm_summary

__all__ = [
    "BDRNNModel",
]

class BDRNNModel(base_model.BaseModel):

    def _build_graph(self, hparams, scope=None):
        """Construct the train, evaluation, and inference graphs.
        Args:
            hparams: The hyperparameters for configuration
            scope: The variable scope name for this subgraph
        Returns:
            A tuple with (logits, loss, metrics, update_ops)
        """

        sample = self.iterator.get_next()

        inputs, tgt_outputs, seq_len = sample

        with tf.variable_scope(scope or "dynamic_bdrnn", dtype=tf.float32):
            # TODO: hidden activations are passed thru FC net
            # TODO: hidden-to-hidden network has skip connections (residual)
            # TODO: initial hidden and cell states are learned


            # create bdrnn
            fw_cells = mdl_help.create_rnn_cell(unit_type=hparams.unit_type,
                                                num_units=hparams.num_units,
                                                num_layers=hparams.num_layers,
                                                depth=0,
                                                num_residual_layers=0,
                                                forget_bias=hparams.forget_bias,
                                                dropout=0.,
                                                mode=self.mode,
                                                num_gpus=1,
                                                base_gpu=0)

            bw_cells = mdl_help.create_rnn_cell(unit_type=hparams.unit_type,
                                                num_units=hparams.num_units,
                                                num_layers=hparams.num_layers,
                                                depth=0,
                                                num_residual_layers=0,
                                                forget_bias=hparams.forget_bias,
                                                dropout=0.,
                                                mode=self.mode,
                                                num_gpus=1,
                                                base_gpu=0)

#            print(fw_cells.zero_state(1, dtype=tf.float32))
#            initial_fw_state = tf.get_variable("initial_fw_state", shape=fw_cells.state_size)
#            initial_bw_state = tf.get_variable("initial_bw_state", shape=bw_cells.state_size)
#            initial_fw_state_tiled = tf.tile(initial_fw_state, [hparams.batch_size, 1])
#            initial_bw_state_tiled = tf.tile(initial_bw_state, [hparams.batch_size, 1])

            # run bdrnn
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cells,
                                                                     cell_bw=bw_cells,
                                                                     inputs=inputs,
                                                                     sequence_length=seq_len,
                                                                     initial_state_fw=None,
                                                                     initial_state_bw=None,
                                                                     dtype=tf.float32)
            # outputs is a tuple (output_fw, output_bw)
            # output_fw/output_bw are tensors [batch_size, max_time, cell.output_size]
            # outputs_states is a tuple (output_state_fw, output_state_bw) containing final states for
            # forward and backward rnn

            # concatenate the outputs of each direction
            combined_outputs = tf.concat([outputs[0], outputs[1]], axis=-1)

            # dense output layers
            dense1 = tf.layers.dense(inputs=combined_outputs,
                                     units=hparams.num_dense_units,
                                     activation=tf.nn.relu,
                                     use_bias=True)
            drop1 = tf.layers.dropout(inputs=dense1,
                                      rate=hparams.dropout,
                                      training=self.mode==tf.contrib.learn.ModeKeys.TRAIN)
            dense2 = tf.layers.dense(inputs=drop1,
                                     units=hparams.num_dense_units,
                                     activation=tf.nn.relu,
                                     use_bias=True)
            drop2 = tf.layers.dropout(inputs=dense2,
                                      rate=hparams.dropout,
                                      training=self.mode==tf.contrib.learn.ModeKeys.TRAIN)

            logits = tf.layers.dense(inputs=drop2,
                                     units=hparams.num_labels,
                                     use_bias=False)

            # mask out entries longer than target sequence length
            mask = tf.sequence_mask(seq_len, dtype=tf.float32)

            #stop gradient thru labels by crossent op
            tgt_outputs = tf.stop_gradient(tgt_outputs)

            crossent = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                                  labels=tgt_outputs,
                                                                  name="crossent")

            # divide loss by batch_size * mean(seq_len)
            loss = (tf.reduce_sum(crossent*mask)/(hparams.batch_size*tf.reduce_mean(tf.cast(seq_len,
                                                                                            tf.float32))))

            metrics = []
            update_ops = []
            if self.mode == tf.contrib.learn.ModeKeys.EVAL:
                predictions = tf.argmax(input=logits, axis=-1)
                tgt_labels = tf.argmax(input=tgt_outputs, axis=-1)
                acc, acc_update = tf.metrics.accuracy(predictions=predictions,
                                                      labels=tgt_labels,
                                                      weights=mask)
                # confusion matrix
                targets_flat = tf.reshape(tgt_labels, [-1])
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
