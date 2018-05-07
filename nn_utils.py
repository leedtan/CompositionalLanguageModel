import tensorflow as tf
import numpy as np


def dense_scaled(prev_layer, layer_size, name=None, reuse=False, scale=1.0):
  output = tf.layers.dense(prev_layer, layer_size, reuse=reuse) * scale
  return output


def apply_clipped_optimizer(opt_fcn,
                            loss,
                            clip_norm=.1,
                            clip_single=.03,
                            clip_global_norm=False):
  gvs = opt_fcn.compute_gradients(loss)

  if clip_global_norm:
    gs, vs = zip(*[(g, v) for g, v in gvs if g is not None])
    capped_gs, grad_norm_total = tf.clip_by_global_norm([g for g in gs],
                                                        clip_norm)
    capped_gvs = list(zip(capped_gs, vs))
  else:
    grad_norm_total = tf.sqrt(
        tf.reduce_sum([
            tf.reduce_sum(tf.square(grad)) for grad, var in gvs
            if grad is not None
        ]))
    capped_gvs = [(tf.clip_by_value(grad, -1 * clip_single, clip_single), var)
                  for grad, var in gvs if grad is not None]
    capped_gvs = [(tf.clip_by_norm(grad, clip_norm), var)
                  for grad, var in capped_gvs if grad is not None]

  optimizer = opt_fcn.apply_gradients(capped_gvs)

  return optimizer, grad_norm_total


def mlp(x, hidden_sizes, output_size=None, name='', reuse=False):
  prev_layer = x

  for idx, l in enumerate(hidden_sizes):
    dense = dense_scaled(prev_layer, l, name='mlp' + name + '_' + str(idx))
    prev_layer = tf.nn.leaky_relu(dense)

  output = prev_layer

  if output_size is not None:
    output = dense_scaled(prev_layer, output_size, name='mlp' + name + 'final')

  return output


def encode(
    x,
    num_layers,
    cells,
    initial_states,
    lengths,
    tf_bs,
    name='',
):
  prev_layer = x
  shortcut = x
  hiddenlayers = []
  returncells = []
  cell_fw, cell_bw = cells
  bs = tf_bs
  for idx in range(num_layers):
    prev_layer, c = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=cell_fw[idx],
        cell_bw=cell_bw[idx],
        inputs=prev_layer,
        sequence_length=lengths,
        initial_state_fw=None,
        initial_state_bw=None,
        dtype=tf.float32,
        scope='encoder' + str(idx))
    if idx == num_layers - 1:
      fw = prev_layer[0]
      bw = prev_layer[1]
      stacked = tf.stack([tf.range(bs), lengths - 1], 1)
      fw_final = tf.gather_nd(fw, stacked, name=None)
      bw_final = bw[:, 0, :]
      output = tf.concat((fw_final, bw_final), 1)
    prev_layer = tf.concat(prev_layer, 2)
    prev_layer = tf.nn.leaky_relu(prev_layer)
    returncells.append(c)
    hiddenlayers.append(prev_layer)
    if idx == num_layers - 1:
      #pdb.set_trace()
      #stacked = tf.stack([tf.range(bs), lengths - 1], 1)
      #output = tf.gather_nd(prev_layer,stacked,name=None)
      return prev_layer, returncells, hiddenlayers, output, fw, stacked
    prev_layer = tf.concat((prev_layer, shortcut), 2)


def subprogram(
    x,
    num_layers,
    cells,
    initial_states,
    lengths,
    hidden_filters,
    hidden_filters_subprogram,
    reuse,
    name='',
):
  prev_layer = x
  shortcut = x
  hiddenlayers = []
  returncells = []
  bs = tf.shape(x)[0]
  for idx in range(num_layers):
    print(idx)
    if idx == 0:
      num_past_units = hidden_filters
    else:
      num_past_units = hidden_filters_subprogram
    with tf.variable_scope(name + 'subprogram' + str(idx), reuse=reuse):
      #             self_attention_mechanism = tf.contrib.seq2seq.LuongAttention(
      #                 num_units=num_past_units, memory=prev_layer,
      #                 memory_sequence_length=tf.expand_dims(tf.range(10), 0))
      #             cell_with_selfattention = tf.contrib.seq2seq.AttentionWrapper(
      #                     cells[idx], self_attention_mechanism, attention_layer_size = num_past_units)

      prev_layer, c = tf.nn.dynamic_rnn(
          cell=cells[idx],
          inputs=prev_layer,
          sequence_length=lengths,
          initial_state=None,
          dtype=tf.float32,
      )
      prev_layer = tf.concat(prev_layer, 2)
      prev_layer = tf.nn.leaky_relu(prev_layer)
      returncells.append(c)
      hiddenlayers.append(prev_layer)
      if idx == num_layers - 1:
        output = tf.gather_nd(
            prev_layer, tf.stack([tf.range(bs), lengths], 1), name=None)
        return prev_layer, returncells, hiddenlayers, output
      prev_layer = tf.concat((prev_layer, shortcut), 2)
