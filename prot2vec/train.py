from pathlib import Path
import tensorflow as tf, numpy as np
from keras import backend as K
from keras.callbacks import Callback, LearningRateScheduler
from keras.utils import print_summary
from keras.optimizers import SGD, Adam
# local imports
from input import pssp_dataset, encode_dataset
from model import EncDecModel
from bd_model import BDEncDecModel

HOME = str(Path.home())

# training callback class
class ValidationMonitor(Callback):
    '''Swap validation Dataset, track validation metrics, (optional) early stopping.'''
    def __init__(self,
            train_files,
            valid_files,
            stop_early=True,
            min_delta=0,
            patience=0,
            verbose=0):
        self.train_files = train_files
        self.valid_files = valid_files
        self.stop_early = stop_early
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose

    def on_train_begin(self, logs={}):
        self.eval_metrics = []
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf
        # initialize dataset pipeline
        self.sess = K.get_session()
        self.sess.run(iterator.initializer, feed_dict={filenames: train_files, shuffle: True})

    def on_epoch_end(self, epoch, logs):
        # check which metrics are available
        #print("Available metrics at on_epoch_end: %s\n" % ','.join(list(logs.keys())))
        # evaluate the model
        self.sess.run(iterator.initializer, feed_dict={filenames: valid_files, shuffle: False})
        eval_metrics = self.model.evaluate(steps=4, verbose=0)
        self.eval_metrics.append(eval_metrics)
        self.sess.run(iterator.initializer, feed_dict={filenames: train_files, shuffle: True})

        if self.verbose > 1:
            print('\nEvaluation Metrics: ')
            for metric, val in zip(self.model.metrics_names, eval_metrics):
                print('%s: %2.4f\t' % (metric, val))


        if self.stop_early:
            current = eval_metrics[0]

            # if the validation loss has decreased by at least min_delta
            if current + self.min_delta < self.best:
                self.best = current
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %2d: early stopping' % (self.stopped_epoch + 1))

if __name__ == '__main__':
    num_features = 43
    num_targets = 9
    epochs = 10
    savedir = HOME+"/thesis/models/encdec3"

    for i in range(0, 5):
        print("Cross Validation: Fold %d\nTraining on dataset %d" % (i+1, i+1))
        # data files
        train_files = [HOME+'/data/cpdb/cpdb_6133_filter_train_'+str(i+1)+'.tfrecords']
        valid_files = [HOME+'/data/cpdb/cpdb_6133_filter_valid_'+str(i+1)+'.tfrecords']

        # define placeholders for the dataset
        filenames = tf.placeholder(tf.string, shape=[None])
        shuffle = tf.placeholder(tf.bool)

        dataset = pssp_dataset(filenames, shuffle, 64, epochs)

        iterator = dataset.make_initializable_iterator()

        src_input, tgt_input, tgt_output = iterator.get_next()

        encdec = EncDecModel(num_features, num_targets, src_input, tgt_input, tgt_output)

        model = encdec.models['decoder']

        opt = Adam(clipnorm=5.0, lr=0.7)
        model.compile(optimizer=opt,
                loss='categorical_crossentropy',
                metrics=['accuracy'],
                target_tensors=[tgt_output])


        val_monitor = ValidationMonitor(train_files,
                valid_files,
                stop_early=True,
                min_delta=1e-2,
                patience=3,
                verbose=2)
#        lr_rate = [1e-3, 1e-3, 1e-3, 5e-4, 2.5e-4, 1e-4, 5e-5, 2.5e-5, 1e-5, 1e-5]
        lr_rate = [1e-3, 1e-3, 1e-3, 5e-4, 2.5e-4, 1e-4, 5e-5, 2.5e-5, 1e-5, 1e-5]
        lr_scheduler = LearningRateScheduler(lambda e: lr_rate[e])

        model.fit(steps_per_epoch=82,
                  epochs=epochs,
                  callbacks=[val_monitor, lr_scheduler],
                  verbose=1)

        model.save_weights(savedir+"/cv_"+str(i+1)+"_weights.h5")
        print("Clearing session...")
        K.clear_session()
