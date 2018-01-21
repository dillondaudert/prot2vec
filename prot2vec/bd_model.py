import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, LSTMCell, RNN, Dense, Masking, Bidirectional

class BDEncDecModel():
    '''LSTM bidirectional encoder-decoder network as multiple Keras models.'''

    models = {}

    def __init__(self,
                 num_features,
                 num_targets,
                 src_in,
                 tgt_in,
                 tgt_out,
                 filename=None):
        self.num_features = num_features
        self.num_targets = num_targets
        self.src_in = src_in
        self.tgt_in = tgt_in
        self.tgt_out = tgt_out
        self._create_models()
        if filename != None:
            # load the weights into the models
            for model in self.models:
                model.load_weights(filename, by_name=True)

    def _create_models(self):
        '''Create Keras Model objects for training, encoding, and decoding.'''

        # -- ENCODER --
        encoder_in = Input(tensor=self.src_in)
        # Mask out empty time steps
        encoder_mask = Masking(mask_value=0.)(encoder_in)

        encoder_bd = Bidirectional(
                         LSTM(units=128, unit_forget_bias=True, dropout=0.2, recurrent_dropout=0.2,
                              return_sequences=True),
                         name='enc1_bd'
                         )(encoder_mask)

        # encoder_bd has shape (batch_size, time_steps, units)

        encoder_cells = [
            LSTMCell(units=128, dropout=0.2, recurrent_dropout=0.2, name='enc2', unit_forget_bias=True),
            LSTMCell(units=128, dropout=0.2, recurrent_dropout=0.2, name='enc3', unit_forget_bias=True),
            ]

        encoder = RNN(encoder_cells, return_state=True, name='enc_stack')
        encoder_out = encoder(encoder_bd)

        # Add encoder to class.models dictionary
        self.models['encoder'] = Model(encoder_in, encoder_out)

        # -- DECODER --
        encoder_states = encoder_out[1:]

        decoder_in = Input(tensor=self.tgt_in)
        decoder_mask = Masking(mask_value=0.)(decoder_in)

        decoder_cells = [
            LSTMCell(units=128, dropout=0.2, recurrent_dropout=0.2, name='dec1', unit_forget_bias=True),
            LSTMCell(units=128, dropout=0.2, recurrent_dropout=0.2, name='dec2', unit_forget_bias=True),
            ]
        decoder_lstm = RNN(decoder_cells, return_sequences=True, return_state=True, name='dec_stack')

        decoder_outputs = decoder_lstm(decoder_mask, initial_state=encoder_states)

        decoder_dense = Dense(self.num_targets, activation='softmax', name='dec_dense')
        decoder_out = decoder_dense(decoder_outputs[0])

        # Define model to turn 'enc_input_data' & 'dec_input_data' into 'dec_target_data'
        self.models['decoder'] = Model([encoder_in, decoder_in], decoder_out)