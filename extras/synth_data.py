# Generate various synthetic datasets
from pathlib import Path
import numpy as np
import tensorflow as tf

HOME = str(Path.home())

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def copy_task(filename, num_seqs, len_min, len_max, num_vals):
    """
    Generate sequences of random binary vectors for the copy task
    and save as .tfrecords file
    Args:
        filename - the name of the file to save
        num_seqs - the number of sequences to generate
        len_min/max - the minimum and maximum length of the sequences
        num_vals    - the number of values per step
    Each sequence will therefore be of size [length, num_vals] flattened
    to shape [length * (num_vals)] when written to the file.
    """

    print("Writing to file %s" % filename)
    with tf.python_io.TFRecordWriter(filename) as writer:
        # Generate and write the sequences
        for i in range(num_seqs):
            length = np.random.randint(len_min, len_max+1)
            seq = np.random.randint(0, 2, size=(length, (num_vals)))
            # the last value of the last location is only used for the delimiter
            seq[:,-1] = 0.
            seq[-1,:] = 0.
            seq = seq.astype(np.float32)

            target_seq = np.copy(seq)
            # prepend and append the delimiter for the target input and output
            delim = np.zeros(shape=(1, num_vals), dtype=np.float32)
            delim[0,-1] = 1.
            target_seq_in = np.concatenate([delim, target_seq], 0)
            target_seq_out = np.concatenate([target_seq, delim], 0)

            seq = seq.reshape(-1)
            target_seq_in = target_seq_in.reshape(-1)
            target_seq_out = target_seq_out.reshape(-1)

            example = tf.train.Example(features=tf.train.Features(feature={
                'seq_len': _int64_feature(length),
                'seq_data': _floats_feature(seq),
                'tgt_in': _floats_feature(target_seq_in),
                'tgt_out': _floats_feature(target_seq_out)}))
            writer.write(example.SerializeToString())
