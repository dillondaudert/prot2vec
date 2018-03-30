"""Utility functions for building models."""
import tensorflow as tf
from collections import namedtuple
from nlstm.rnn_cell import NLSTMCell
from cpdb_model import CPDBModel
from cpdb2_model import CPDB2ProtModel
from synth_model import CopyModel
from bdrnn_model import BDRNNModel
from datasets.dataset_helper import create_dataset
from datasets.vocab import create_cpdb2_embedding

__all__ = [
    "create_rnn_cell", "create_model", "multiclass_sample", "multiclass_prediction",
    "create_embeddings",
]

def create_embeddings(embedding):
    """
    Create or load embeddings for encoder and decoder.
    """

    if embedding == "cpdb2":
        src_embed = create_cpdb2_embedding("aa")
        tgt_embed = create_cpdb2_embedding("ss")
        return src_embed, tgt_embed
    else:
        raise ValueError()


def multiclass_prediction(logits):
    """
    Calculate the predictions for a multiclass output where the classes are not
    mutually exclusive.
    The logits are scaled via a sigmoid, and each class is counted as active if
    the value is greater than .5
    """

    scaled_logits = tf.sigmoid(logits)
    preds = tf.greater_equal(scaled_logits, tf.constant(0.5, tf.float32))
    return tf.cast(preds, tf.float32)

def multiclass_sample(logits):
    """
    Sample from a multiclass distribution where the classes are not mutually
    exclusive.
    Takes a uniform sample from each output class, returns a float Tensor of 0s
    and 1s.
    """
    scaled_logits = tf.sigmoid(logits)
    probs = tf.random_uniform(shape=tf.shape(logits), maxval=1.)
    leq = tf.less_equal(probs, scaled_logits)
#    tf.summary.histogram("scaled_logits", scaled_logits, collections=["eval"])
    return tf.cast(leq, tf.float32)

ModelTuple = namedtuple('ModelTuple', ['graph', 'iterator', 'model', 'session'])

def create_model(hparams, mode):
    """
    Return a tuple of a tf Graph, Iterator, Model, and Session for training.
    Args:
        hparams - Hyperparameter named tuple
        mode    - the tf.contrib.learn mode (TRAIN, EVAL, INFER)
    Returns a ModelTuple(graph, iterator, model, session)
    """

    if hparams.model == "cpdb" or hparams.model == "cpdb2":
        model_creator = CPDBModel
    elif hparams.model == "copy":
        model_creator = CopyModel
    elif hparams.model == "bdrnn":
        model_creator = BDRNNModel
    elif hparams.model == "cpdb2_prot":
        model_creator = CPDB2ProtModel
    else:
        print("Error! Model %s unrecognized" % (hparams.model))
        exit()

    graph = tf.Graph()

    with graph.as_default():
        dataset = create_dataset(hparams, mode)
        iterator = dataset.make_initializable_iterator()
        model = model_creator(hparams=hparams,
                              iterator=iterator,
                              mode=mode)

    config = tf.ConfigProto()
    #jit_level = tf.OptimizerOptions.ON_1
    #config.graph_options.optimizer_options.global_jit_level = jit_level
    sess = tf.Session(graph=graph,)
    #                  config=config)

    modeltuple = ModelTuple(graph=graph, iterator=iterator,
                            model=model, session=sess)

    return modeltuple



def _single_cell(unit_type, num_units, depth, forget_bias, dropout, mode,
                 residual_connection=False, residual_fn=None, device_str=None):
    """Define a single recurrent cell."""

    # Set dropout to 0 if not training
    dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0

    if unit_type == "lstm":
        single_cell = tf.nn.rnn_cell.LSTMCell(name="lstm",
                                              num_units=num_units,
                                              forget_bias=forget_bias)
    elif unit_type == "lstmblock":
        single_cell = tf.contrib.rnn.LSTMBlockCell(name="lstm",
                                                   num_units=num_units,
                                                   forget_bias=forget_bias)
    elif unit_type == "nlstm":
        single_cell = NLSTMCell(name="nlstm",
                                num_units=num_units,
                                depth=depth)
    elif unit_type == "gru":
        single_cell = tf.nn.rnn_cell.GRUCell(name="gru",
                                             num_units=num_units)
    else:
        raise ValueError("Unknown unit type %s!" % unit_type)

    if dropout > 0.0:
        single_cell = tf.nn.rnn_cell.DropoutWrapper(
            cell=single_cell, input_keep_prob=(1.0 - dropout))

    if residual_connection:
        single_cell = tf.nn.rnn_cell.ResidualWrapper(
            cell=single_cell, residual_fn=residual_fn)

    if device_str:
        single_cell = tf.nn.rnn_cell.DeviceWrapper(single_cell, device_str)

    return single_cell

def _cell_list(unit_type, num_units, num_layers, num_residual_layers, depth,
               forget_bias, dropout, mode, num_gpus, base_gpu, residual_fn=None):
    """Create a list of RNN cells."""

    cell_list = []
    for i in range(num_layers):
        if num_gpus == 1:
            device_str = "/device:GPU:%d" % (base_gpu)
        else:
            device_str = "/device:GPU:%d" % ((i+base_gpu) % num_gpus)
        single_cell = _single_cell(
            unit_type=unit_type,
            num_units=num_units,
            depth=depth,
            forget_bias=forget_bias,
            dropout=dropout,
            mode=mode,
            residual_connection=(i >= num_layers - num_residual_layers),
            residual_fn=residual_fn,
            device_str=device_str
        )
        cell_list.append(single_cell)
    return cell_list

def create_rnn_cell(unit_type, num_units, num_layers, num_residual_layers, depth,
                    forget_bias, dropout, mode, num_gpus=1, base_gpu=0):
    """Create single- or multi-layer RNN cell.

    Args:
        unit_type: string representing the unit type, e.g. "lstm"
        num_units: the depth of each unit
        num_layers: the number of cells
        num_residual_layers: the number of residual layers. if
          num_residual_layers < num_layers, then the last num_residual_layers
          will have residual connections (ResidualWrapper)
        depth: only used for NLSTM; the depth of the nesting
        forget_bias: the initial forget bias of the RNNCell(s)
        dropout: floating point between 0.0 and 1.0, the probability of dropout
        device_str: the device this rnn will be placed on
        mode: either tf.contrib.learn.TRAIN/EVAL/INFER

    Returns:
        An 'RNNCell' instance
    """

    cell_list = _cell_list(unit_type=unit_type,
                           num_units=num_units,
                           num_layers=num_layers,
                           num_residual_layers=num_residual_layers,
                           depth=depth,
                           forget_bias=forget_bias,
                           dropout=dropout,
                           mode=mode,
                           num_gpus=num_gpus,
                           base_gpu=base_gpu
                           )

    if len(cell_list) == 1:
        return cell_list[0]
    else:
        return tf.nn.rnn_cell.MultiRNNCell(cell_list)
