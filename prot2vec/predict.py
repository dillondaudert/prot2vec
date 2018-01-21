from pathlib import Path
import tensorflow as tf, numpy as np
from keras import backend as K
from keras.optimizers import SGD, Adam
# local imports
from input import pssp_dataset, encode_dataset
from model import EncDecModel
from bd_model import BDEncDecModel

HOME = str(Path.home())

if __name__ == '__main__':
    num_features = 43
    num_targets = 9

    test_files = [HOME+'/data/cpdb/cpdb_513.tfrecords']

    # define placeholders for the dataset
    filenames = tf.placeholder(tf.string, shape=[None])
    shuffle = tf.placeholder(tf.bool)

    dataset = pssp_dataset(filenames, shuffle, 51, 1)

    iterator = dataset.make_initializable_iterator()

    src_input, tgt_input, tgt_output = iterator.get_next()

    encdec = EncDecModel(num_features, num_targets, src_input, tgt_input, tgt_output)

    model = encdec.models['decoder']

    adam = Adam(clipnorm=5.0)
    model.compile(optimizer=adam,
            loss='categorical_crossentropy',
            metrics=['accuracy'],
            target_tensors=[tgt_output])

    print("Loading model weights...\n")
    model.load_weights('test_weights.h5')

    sess = K.get_session()
    sess.run(iterator.initializer, feed_dict={filenames: test_files, shuffle: False})

    print(model.evaluate(steps=10, verbose=1))

