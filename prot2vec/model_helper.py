"""Utility functions for building models."""
import tensorflow as tf

__all__ = [
    "create_rnn_cell",
]

def _single_cell(unit_type, num_units, forget_bias, dropout, mode,
                 residual_connection=False, residual_fn=None):
    """Define a single recurrent cell."""

    # Set dropout to 0 if not training
    dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0

    if unit_type == "lstm":
        single_cell = tf.nn.rnn_cell.BasicLSTMCell(
            num_units,
            forget_bias=forget_bias)
    else:
        raise ValueError("Unknown unit type %s!" % unit_type)

    if dropout > 0.0:
        single_cell = tf.nn.rnn_cell.DropoutWrapper(
            cell=single_cell, input_keep_prob=(1.0 - dropout))

    if residual_connection:
        single_cell = tf.nn.rnn_cell.ResidualWrapper(
            cell=single_cell, residual_fn=residual_fn)

    return single_cell

def _cell_list(unit_type, num_units, num_layers, num_residual_layers,
               forget_bias, dropout, mode, residual_fn=None):
    """Create a list of RNN cells."""

    cell_list = []
    for i in range(num_layers):
        single_cell = _single_cell(
            unit_type=unit_type,
            num_units=num_units,
            forget_bias=forget_bias,
            dropout=dropout,
            mode=mode,
            residual_connection=(i >= num_layers - num_residual_layers),
            residual_fn=residual_fn
        )
        cell_list.append(single_cell)
    return cell_list

def create_rnn_cell(unit_type, num_units, num_layers, num_residual_layers,
                    forget_bias, dropout, mode):
    """Create single- or multi-layer RNN cell.

    Args:
        unit_type: string representing the unit type, e.g. "lstm"
        num_units: the depth of each unit
        num_layers: the number of cells
        num_residual_layers: the number of residual layers. if
          num_residual_layers < num_layers, then the last num_residual_layers
          will have residual connections (ResidualWrapper)
        forget_bias: the initial forget bias of the RNNCell(s)
        dropout: floating point between 0.0 and 1.0, the probability of dropout
        mode: either tf.contrib.learn.TRAIN/EVAL/INFER

    Returns:
        An 'RNNCell' instance
    """

    cell_list = _cell_list(unit_type=unit_type,
                           num_units=num_units,
                           num_layers=num_layers,
                           num_residual_layers=num_residual_layers,
                           forget_bias=forget_bias,
                           dropout=dropout,
                           mode=mode)

    if len(cell_list) == 1:
        return cell_list[0]
    else:
        return tf.nn.rnn_cell.MultiRNNCell(cell_list)
