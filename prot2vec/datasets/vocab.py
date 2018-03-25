"""Utilities for protein and structure strings."""

import tensorflow as tf
import pandas as pd

# NOTE: the ordering of these alphabets, and the inclusion of the NoSeq token,
#       was taken from the CullPDB datasets provided by Zhou & Troyanskaya

aminos_df = pd.read_csv("~/thesis/prot2vec/prot2vec/datasets/aminos_vocab_features.csv", index_col=0)

CPDB2_SOURCE_ALPHABET = list(aminos_df.columns)

CPDB2_TARGET_ALPHABET = ["H", "B", "E", "G", "I",
                         "T", "S", "U", "SOS", "EOS"]

def create_lookup_table(vocab, reverse=False):
    """Create a lookup table that turns amino acid or secondary structure
    characters into an integer id.
    Args:
        vocab: One of "aa" or "ss" for amino acids or secondary structures, respectively.
        reverse: Whether or not this table will convert strings to ids (default) or ids to strings.
    Returns:
        A lookup table that maps strings to ids (or ids to strings if reverse=True)
    """

    if vocab == "aa":
        alphabet = CPDB2_SOURCE_ALPHABET
    elif vocab == "ss":
        alphabet = CPDB2_TARGET_ALPHABET
    else:
        raise ValueError("Unrecognized value for vocab: %s" % (vocab))

    if not reverse:
        table = tf.contrib.lookup.index_table_from_tensor(tf.constant(alphabet))
    else:
        table = tf.contrib.lookup.index_to_string_table_from_tensor(tf.constant(alphabet))

    return table

def create_cpdb2_embedding(vocab):
    """
    Create embedding matrices for the CPDB2 dataset.
    Args:
        vocab: One of "aa" or "ss" for amino acids or secondary structures, respectively.
    Returns:
        An embedding Tensor
    """

    if vocab == "aa":
        emb = tf.constant(aminos_df.values, dtype=tf.float32)
    elif vocab == "ss":
        emb = tf.eye(len(CPDB2_TARGET_ALPHABET))
    else:
        raise ValueError("Unrecognized value for vocab: %s" % (vocab))

    return emb
