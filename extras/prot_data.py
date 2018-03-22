# Generate various synthetic datasets
from pathlib import Path
import numpy as np, pandas as pd
import tensorflow as tf
from features import prot_to_vector

HOME = str(Path.home())

aminos_df = pd.read_csv("./aminos_table.csv", index_col=0)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def make_prot_tfrecords(filename, num_train, num_valid, num_test):
    """
    Save protein strings as float vectors in tfrecords file format.
    Args:
        filename - the name of the file to save
        num_train - the number of training files to save
        num_valid - the number of validation fiels to save
        num_test - the number of test files to save
    The total number of files generated will be equal to num_train + num_valid
    + num_test. Each file will contain the total number of sequences divided
    by the total number of files.
    """

    # open the file
    seqs = pd.read_csv(filename, names=["seq"])
    num_seqs = len(seqs)
    num_files = num_train + num_valid + num_test
    seqs_per_file = num_seqs//num_files

    print("Writing %d sequences to %d train, %d validation, and %d test files" % (num_seqs, num_train, num_valid, num_test))
    print("Each file will contain %d sequences" % seqs_per_file)

    for i in range(num_files):
        start_index = i*seqs_per_file

        if num_train - i > 0:
            print("Writing train file %d" % i)
            filename = "prot_train_%d.tfrecords" % i
        elif (num_train+num_valid) - i > 0:
            print("Writing valid file %d" % i-num_train)
            filename = "prot_valid_%d.tfrecords" % (i-num_train)
        else:
            print("Writing test file %d" % i-(num_train+num_valid))
            filename = "prot_test_%d.tfrecords" % (i-(num_train+num_valid))

        with tf.python_io.TFRecordWriter(filename) as writer:
            # convert the strings to feature vectors and write to file
            for i in range(start_index, start_index+seqs_per_file):
                length = len(seqs.iloc[i].seq)
                seq = prot_to_vector(seqs.iloc[i].seq)

                target_seq = np.copy(seq[:, 0:21])
                # prepend and append the delimiter for the target input and output
                delim = aminos_df.loc["EOS"].values[0:21].reshape(1, -1)
                target_seq_in = np.concatenate([delim, target_seq], 0)
                target_seq_out = np.concatenate([target_seq, delim], 0)
                print(seq)
                print(seq.shape)
                print(target_seq_in)
                print(target_seq_in.shape)
                print(target_seq_out)
                print(target_seq_out.shape)

                seq = seq.reshape(-1)
                target_seq_in = target_seq_in.reshape(-1)
                target_seq_out = target_seq_out.reshape(-1)

                quit()

                example = tf.train.Example(features=tf.train.Features(feature={
                    'seq_len': _int64_feature(length),
                    'seq_data': _floats_feature(seq),
                    'tgt_in': _floats_feature(target_seq_in),
                    'tgt_out': _floats_feature(target_seq_out)}))
                writer.write(example.SerializeToString())
