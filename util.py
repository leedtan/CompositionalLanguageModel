
"""
Util functions shared across scripts
"""

import tensorflow as tf

#================ Data Processing =================

#================ NN functions ====================
def dense_scaled(prev_layer, layer_size, name=None, reuse=False, scale=1.0):
    output = tf.layers.dense(prev_layer, layer_size, reuse=reuse) * scale
    return output


def dense_relu(dense_input, layer_size, scale=1.0):
    dense = dense_scaled(dense_input, layer_size, scale=scale)
    output = tf.nn.leaky_relu(dense)

    return output


def get_grad_norm(opt_fcn, loss):
    gvs = opt_fcn.compute_gradients(loss)
    grad_norm = tf.sqrt(tf.reduce_sum(
        [tf.reduce_sum(tf.square(grad)) for grad, var in gvs if grad is not None]))
    return grad_norm


def apply_clipped_optimizer(opt_fcn, loss, clip_norm=.1, clip_single=.03, clip_global_norm=False):
    gvs = opt_fcn.compute_gradients(loss)

    if clip_global_norm:
        gs, vs = zip(*[(g, v) for g, v in gvs if g is not None])
        capped_gs, grad_norm_total = tf.clip_by_global_norm([g for g in gs], clip_norm)
        capped_gvs = list(zip(capped_gs, vs))
    else:
        grad_norm_total = tf.sqrt(
            tf.reduce_sum([tf.reduce_sum(tf.square(grad)) for grad, var in gvs if grad is not None]))
        capped_gvs = [(tf.clip_by_value(grad, -1 * clip_single, clip_single), var)
                      for grad, var in gvs if grad is not None]
        capped_gvs = [(tf.clip_by_norm(grad, clip_norm), var) for grad, var in capped_gvs if grad is not None]

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


def mlp_with_adversaries(x, hidden_sizes, output_size=None, name='', reuse=False):
    prev_layer = x
    adv_phs = []
    for idx, l in enumerate(hidden_sizes):
        adversary = tf.placeholder_with_default(tf.zeros_like(prev_layer), prev_layer.shape)
        prev_layer = prev_layer + adversary
        adv_phs.append(adversary)

        dense = dense_scaled(prev_layer, l, name='mlp' + name + '_' + str(idx))
        prev_layer = tf.nn.leaky_relu(dense)

    output = prev_layer

    if output_size is not None:
        output = dense_scaled(prev_layer, output_size, name='mlp' + name + 'final')

    return output, adv_phs
