import tensorflow as tf
from synth_dataset import copytask_dataset

filename = tf.constant("/home/dillon/data/synthetic/copy/train_25L_10V.tfrecords")
shuffle = tf.constant(False)
dataset = copytask_dataset(filename, shuffle, 2)
iterator = dataset.make_initializable_iterator()
vals = iterator.get_next()

initializer = tf.global_variables_initializer()

sess = tf.Session()
sess.run([initializer])
sess.run([iterator.initializer])
ret = sess.run([vals])
print(ret)
