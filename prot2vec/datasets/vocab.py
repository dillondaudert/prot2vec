"""Utilities for protein and structure strings."""

import tensorflow as tf

# NOTE: the ordering of these alphabets, and the inclusion of the NoSeq token,
#       was taken from the CullPDB datasets provided by Zhou & Troyanskaya

AMINO_ACIDS = ["A", "C", "E", "D", "G",
               "F", "I", "H", "K", "M",
               "L", "N", "Q", "P", "S",
               "R", "T", "W", "V", "Y",
               "X", "NoSeq"]

SEC_STRUCTS = ["L", "B", "E", "G", "I",
               "H", "S", "T", "NoSeq"]

def create_table(vocab):
    """Create a lookup table that turns amino acid or secondary structure
    characters into an integer id.
    Args:
        vocab: One of "aa" or "ss" for amino acids or secondary structures, respectively.
    Returns:
        A lookup table that maps strings to ids
    """

    if vocab == "aa":
        alphabet = AMINO_ACIDS
    elif vocab == "ss":
        alphabet = SEC_STRUCTS
    else:
        raise ValueError("Unrecognized value for vocab: %s" % (vocab))

    table = tf.contrib.lookup.index_table_from_tensor(tf.constant(alphabet))

    return table
