# basic example of training a network end-to-end
import tensorflow as tf
from datasets import pssp_dataset
from model_helper import *

train_files = ["/home/dillon/data/cpdb/cpdb_6133_filter_train_%d.tfrecords" % (i) for i in range(1, 11)]
valid_files = ["/home/dillon/data/cpdb/cpdb_6133_filter_valid_%d.tfrecords" % (i) for i in range(1, 11)]
test_files = ["/home/dillon/data/cpdb/cpdb_513.tf_records"]

hparams = tf.contrib.training.HParams(
    num_features=43,
    num_labels=9,
    batch_size=32,
    num_epochs=1,
    learning_rate=0.007,
    unit_type="lstm",
    num_units=128,
    num_layers=1,
    num_residual_layers=0,
    forget_bias=1,
    dropout=0.5,
    max_gradient_norm=5.0
)



train_graph = tf.Graph()

# build training graph
with train_graph.as_default():
    train_dataset = pssp_dataset(tf.constant(train_files[0], tf.string),
                                 tf.constant(True, tf.bool),
                                 batch_size=hparams.batch_size,
                                 num_epochs=hparams.num_epochs)
    train_iterator = train_dataset.make_one_shot_iterator()

    # build model
    enc_inputs, dec_inputs, dec_outputs, seq_len = train_iterator.get_next()

    # create encoder
    enc_cells = create_rnn_cell(unit_type=hparams.unit_type,
                                  num_units=hparams.num_units,
                                  num_layers=hparams.num_layers,
                                  num_residual_layers=hparams.num_residual_layers,
                                  forget_bias=hparams.forget_bias,
                                  dropout=hparams.dropout,
                                  mode=tf.contrib.learn.ModeKeys.TRAIN)

    # Run encoder
    # Input shape:
    #   enc_inputs: [batch_size, max_len, num_features]
    # Output shapes:
    #   enc_outputs: [batch_size, max_len, cell.output_size]
    #   enc_state: ?
    enc_outputs, enc_state = tf.nn.dynamic_rnn(
            cell=enc_cells,
            inputs=enc_inputs,
            sequence_length=seq_len,
            dtype=tf.float32,
            scope="encoder")


    projection_layer = tf.layers.Dense(hparams.num_labels, use_bias=False)

    tgt_seq_len = tf.add(seq_len, tf.constant(1, tf.int32))

    # create decoder
    dec_cells = create_rnn_cell(unit_type=hparams.unit_type,
                                  num_units=hparams.num_units,
                                  num_layers=hparams.num_layers,
                                  num_residual_layers=hparams.num_residual_layers,
                                  forget_bias=hparams.forget_bias,
                                  dropout=hparams.dropout,
                                  mode=tf.contrib.learn.ModeKeys.TRAIN)

    # feed the ground truth at each time step
    helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_inputs,
                                               sequence_length=tgt_seq_len)

    decoder = tf.contrib.seq2seq.BasicDecoder(cell=dec_cells,
                                              helper=helper,
                                              initial_state=enc_state,
                                              output_layer=projection_layer)

    final_outputs, final_states, final_sequence_len = tf.contrib.seq2seq.dynamic_decode(
            decoder,
            impute_finished=True,
           # maximum_iterations=tf.reduce_max(seq_len),
            scope="decoder")

    logits = final_outputs.rnn_output

    # mask out entries longer than seq_len
    mask = tf.sequence_mask(tgt_seq_len, dtype=tf.float32)

    crossent = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                          labels=dec_outputs,
                                                          name="crossent")

    # NOTE: this part will likely go into the Eval graph
    predictions = tf.argmax(input=logits, axis=-1)
    targets = tf.argmax(input=dec_outputs, axis=-1)
#    _, acc = tf.metrics.accuracy(labels=targets,
#                                 predictions=predictions)
    acc = tf.contrib.metrics.accuracy(predictions=predictions,
                                      labels=targets)



    train_loss = (tf.reduce_sum(crossent*mask)/hparams.batch_size)

    global_step = tf.Variable(0, name="global_step", trainable=False)

    # calculate and clip gradient
    params = tf.trainable_variables()
    gradients = tf.gradients(train_loss, params)
    clipped_gradients, _ = tf.clip_by_global_norm(
                gradients, hparams.max_gradient_norm)

    # Optimization
    optimizer = tf.train.AdamOptimizer(hparams.learning_rate)
    update_step = optimizer.apply_gradients(
                zip(clipped_gradients, params), global_step=global_step)

    initializer = tf.global_variables_initializer()
    loc_initializer = tf.local_variables_initializer()


train_sess = tf.Session(graph=train_graph)

train_sess.run([initializer, loc_initializer])

# print out the input sequence length, output sequence length:
#preds, tgts = train_sess.run([predictions, targets])
#print("predictions: ", preds)
#print("targets: ", tgts)
#print("cross entropy: (shape): ", cross_ent.shape)
#print(cross_ent)
#quit()

for i in range(100):
    _, loss_val, accuracy = train_sess.run([update_step, train_loss, acc])
    print("Step: %d, Loss %f, Acc: %f" % (i, loss_val, accuracy))
