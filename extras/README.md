# extras
...
## Dataset Preparation
See *MakeTFRecords.ipynb* notebook for example.

The original dataset was prepared for Zhou & Troyanskaya (see references), 
described in the *cpdb_readme.txt* file. This dataset consisted of proteins
taken from the PDB up to 700 residues in length, where proteins shorter than
this were padded with 'NoSeq' tokens. 

Input to seq2seq models allows for variable-length sequences, so the original
dataset was modified and saved as a TFRecord format. This format makes it easy
to read into TensorFlow using the tf.data API.
