# Contains the model definition

import tensorflow as tf
from keras.utils import print_summary
from keras.models import Model
from keras.layers import Input, LSTMCell, RNN, Dense, Masking
from keras.optimizers import SGD
from input import cpdb_6133_input_fn

batch_size = 32
epochs = 5

num_features = 43
num_targets = 9

# training data
enc_train_x, dec_train_x, dec_train_y = cpdb_6133_input_fn(batch_size, True, epochs, "TRAIN")
# validation data
enc_valid_x, dec_valid_x, dec_valid_y = cpdb_6133_input_fn(batch_size, False, 1, "VALID")

# Define input layer
enc_inputs = Input(tensor=enc_train_x)

# Mask out empty time steps
enc_masks = Masking(mask_value=0.)(enc_inputs)

# Encoder architecture
enc_cells = [
    LSTMCell(units=512),# dropout=0.2, recurrent_dropout=0.2),
    LSTMCell(units=512),# dropout=0.2, recurrent_dropout=0.2),
    LSTMCell(units=512),# dropout=0.2, recurrent_dropout=0.2),
]

enc = RNN(enc_cells, return_state=True)
enc_out = enc(enc_masks)
enc_states = enc_out[1:]

# Decoder architecture
dec_inputs = Input(tensor=dec_train_x)
dec_masks = Masking(mask_value=0.)(dec_inputs)

dec_cells = [
    LSTMCell(units=512),# dropout=0.2, recurrent_dropout=0.2),
    LSTMCell(units=512),# dropout=0.2, recurrent_dropout=0.2),
    LSTMCell(units=512),# dropout=0.2, recurrent_dropout=0.2),
]
dec_lstm = RNN(dec_cells, return_sequences=True, return_state=True)

dec_out = dec_lstm(dec_masks, initial_state=enc_states)

dec_dense = Dense(num_targets, activation='softmax')
dec_outputs = dec_dense(dec_out[0])

# Define model to turn 'enc_input_data' & 'dec_input_data' into 'dec_target_data'
model = Model([enc_inputs, dec_inputs], dec_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'], target_tensors=[dec_train_y])
#print_summary(model, line_length=100)

model.fit(steps_per_epoch=150,
          epochs=epochs,
          verbose=1)
