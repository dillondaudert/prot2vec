# Contains the model definition

from pathlib import Path
import tensorflow as tf
from keras import backend as K
from keras.callbacks import Callback, LearningRateScheduler
from keras.utils import print_summary
from keras.models import Model
from keras.layers import Input, LSTMCell, RNN, Dense, Masking
from keras.optimizers import SGD
from input import cpdb_dataset

HOME = str(Path.home())
# data files
train_files = [HOME+'/data/cpdb/cpdb_6133_train.tfrecords']
valid_files = [HOME+'/data/cpdb/cpdb_6133_valid.tfrecords']
test_files = [HOME+'/data/cpdb/cpdb_6133_test.tfrecords']

num_features = 43
num_targets = 9
epochs = 5

# define placeholders for the dataset
filenames = tf.placeholder(tf.string, shape=[None])
shuffle = tf.placeholder(tf.bool)

dataset = cpdb_dataset(filenames, shuffle, 32, epochs)

iterator = dataset.make_initializable_iterator()

src_input, tgt_input, tgt_output = iterator.get_next()


# MODEL DEFINITION ---

# Define input layer
enc_inputs = Input(tensor=src_input)

# Mask out empty time steps
enc_masks = Masking(mask_value=0.)(enc_inputs)

# Encoder architecture
enc_cells = [
    LSTMCell(units=512),# dropout=0.2, recurrent_dropout=0.2),
    LSTMCell(units=512),# dropout=0.2, recurrent_dropout=0.2),
#    LSTMCell(units=512),# dropout=0.2, recurrent_dropout=0.2),
]

enc = RNN(enc_cells, return_state=True, go_backwards=True)
enc_out = enc(enc_masks)
enc_states = enc_out[1:]

# Decoder architecture
dec_inputs = Input(tensor=tgt_input)
dec_masks = Masking(mask_value=0.)(dec_inputs)

dec_cells = [
    LSTMCell(units=512),# dropout=0.2, recurrent_dropout=0.2),
    LSTMCell(units=512),# dropout=0.2, recurrent_dropout=0.2),
#    LSTMCell(units=512),# dropout=0.2, recurrent_dropout=0.2),
]
dec_lstm = RNN(dec_cells, return_sequences=True, return_state=True)

dec_out = dec_lstm(dec_masks, initial_state=enc_states)

dec_dense = Dense(num_targets, activation='softmax')
dec_outputs = dec_dense(dec_out[0])

# Define model to turn 'enc_input_data' & 'dec_input_data' into 'dec_target_data'
model = Model([enc_inputs, dec_inputs], dec_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'], target_tensors=[tgt_output])
#print_summary(model, line_length=100)

# training callback class
class TrainHelper(Callback):
    def on_train_begin(self, logs={}):
        self.eval_metrics = []
        self.sess = K.get_session()
        self.sess.run(iterator.initializer, feed_dict={filenames: train_files, shuffle: True})

    def on_epoch_end(self, epoch, logs):
        # check which metrics are available
        #print("Available metrics at on_epoch_end: %s\n" % ','.join(list(logs.keys())))
        # evaluate the model
        self.sess.run(iterator.initializer, feed_dict={filenames: valid_files, shuffle: False})
        self.eval_metrics.append(self.model.evaluate(steps=7, verbose=0))
        self.sess.run(iterator.initializer, feed_dict={filenames: train_files, shuffle: True})

train_helper = TrainHelper()
lr_rate = {0: 0.001, 1: 0.001, 2: 0.0005, 3: 0.00025, 4: 0.0001}
lr_scheduler = LearningRateScheduler(lambda e: lr_rate[e])

model.fit(steps_per_epoch=175,
          epochs=epochs,
          callbacks=[train_helper, lr_scheduler],
          verbose=1)

for i, metrics in enumerate(train_helper.eval_metrics):
    print("Epoch %2d:\t%2.4f\t%.3f\n" % (i, metrics[0], metrics[1]))
