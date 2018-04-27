
# coding: utf-8

# In[1]:



# Import common dependencies
import pandas as pd  # noqa
import numpy as np
import matplotlib  # noqa
import matplotlib.pyplot as plt
import datetime  # noqa
import PIL  # noqa
import glob  # noqa
import pickle  # noqa
from pathlib import Path  # noqa
from scipy import misc  # noqa
import sys
import tensorflow as tf
import pdb
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
TRADE_COST_FRAC = .003
EPSILON = 1e-10
ADV_MULT = 1e-3


# In[2]:


uni_tokens = set()
uni_commands = set()
uni_actions = set()
fname = 'tasks_with_length_tags.txt'
with open(fname) as f:
    content = f.readlines()
content2 = [c.split(' ') for c in content]
# you may also want to remove whitespace characters like `\n` at the end of each line
commands = []
actions = []
content = [l.replace('\n', '') for l in content]
commands = [x.split(':::')[1].split(' ')[1:-1] for x in content]
actions = [x.split(':::')[2].split(' ')[1:-2] for x in content]
structures = [x.split(':::')[3].split(' ')[2:] for x in content]

structures = [[int(l) for l in program] for program in structures]
#actions = [[wd.replace('\n', '') for wd in res] for res in actions]


# In[3]:


max_actions_per_subprogram = max([max([s for s in struct]) for struct in structures]) + 1
max_num_subprograms = max([len(s) for s in structures]) + 1
max_cmd_len = max([len(s) for s in commands]) + 1
max_act_len = max([len(a) for a in actions]) + 1
cmd_lengths_list = [len(s)+1 for s in commands]
cmd_lengths_np = np.array(cmd_lengths_list)
max_num_subprograms, max_cmd_len, max_act_len, max_actions_per_subprogram


# In[4]:


def build_fmap_invmap(unique, num_unique):
    fmap = dict(zip(unique, range(num_unique)))
    invmap = dict(zip(range(num_unique), unique))
    return fmap, invmap


# In[5]:


for li in content2:
    for wd in li:
        uni_tokens.add(wd)
for li in commands:
    for wd in li:
        uni_commands.add(wd)
for li in actions:
    for wd in li:
        uni_actions.add(wd)
uni_commands.add('end_command')
uni_actions.add('end_subprogram')
uni_actions.add('end_action')
num_cmd = len(uni_commands)
num_act = len(uni_actions)
command_map, command_invmap = build_fmap_invmap(uni_commands, num_cmd)
action_map, action_invmap = build_fmap_invmap(uni_actions, num_act)


# In[6]:




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



# In[7]:



commands_ind = [[command_map[c] for c in cmd] + [0] * (max_cmd_len - len(cmd)) for cmd in commands]
actions_ind = [[action_map[a] for a in act] + [0] * (max_act_len - len(act)) for act in actions]
cmd_np = np.array(commands_ind)
actions_structured = []
mask_structured = []
for row in range(len(structures)):
    mask_row = []
    action_row = []
    act = actions_ind[row]
    struct = structures[row]
    start = 0
    for step in struct:
        end = start + step
        a = act[start:end]
        padding = max_actions_per_subprogram - step - 1
        action_row.append(a + [action_map['end_action']] + [0] * padding)
        start = end
    actions_structured.append(
        action_row + [[action_map['end_subprogram']] + [0] * (max_actions_per_subprogram - 1)] +
        [[0] * max_actions_per_subprogram] * (max_num_subprograms - len(struct) - 1)
    )
act_np = np.array(actions_structured)
struct_padded = [[sa + 1 for sa in s] + [1] + [0] * (max_num_subprograms - len(s) - 1) for s in structures]
struct_np = np.array(struct_padded)

mask_list = [[np.concatenate((np.ones(st), np.zeros(max_actions_per_subprogram - st)), 0) 
              for st in s] for s in struct_np]
mask_np = np.array(mask_list)


# In[23]:


tf.reset_default_graph()
default_sizes = 128
size_emb = 64
num_layers_encoder = 4
hidden_filters = 256
num_layers_subprogram = 2
hidden_filters_subprogram = 256
init_mag = 1e-3
cmd_mat = tf.Variable(init_mag*tf.random_normal([num_cmd, size_emb]))
act_mat = tf.Variable(init_mag*tf.random_normal([num_act, size_emb]))
act_st_emb = tf.Variable(init_mag*tf.random_normal([size_emb]))
global_bs = None
global_time_len = 7
action_lengths = None
max_num_actions= None
# global_bs = 8
global_time_len = 7
max_num_actions = 9
output_keep_prob = tf.placeholder_with_default(1.0, ())
state_keep_prob = tf.placeholder_with_default(1.0, ())
cmd_ind = tf.placeholder(tf.int32, shape=(global_bs, 10,))
act_ind = tf.placeholder(tf.int32, shape=(global_bs, global_time_len, 9))
mask_ph = tf.placeholder(tf.float32, shape=(global_bs, global_time_len, 9))
cmd_lengths = tf.placeholder(tf.int32, shape=(global_bs,))
act_lengths = tf.placeholder(tf.int32, shape=(global_bs, 7))
learning_rate = tf.placeholder(tf.float32, shape = (None))

cmd_emb = tf.nn.embedding_lookup(cmd_mat, cmd_ind)
act_emb = tf.nn.embedding_lookup(act_mat, act_ind)
tf_bs = tf.shape(act_ind)[0]
act_st_emb_expanded = tf.tile(tf.reshape(
    act_st_emb, [1, 1, 1, size_emb]), [tf_bs, global_time_len, 1, 1])
act_emb_with_st = tf.concat((act_st_emb_expanded, act_emb), 2)

first_cell_encoder = [tf.nn.rnn_cell.LSTMCell(
    hidden_filters, forget_bias=1., name = 'layer1_'+d) for d in ['f', 'b']]
hidden_cells_encoder = [[tf.nn.rnn_cell.LSTMCell(
    hidden_filters,forget_bias=1., name = 'layer' + str(lidx) + '_' + d)  for d in ['f', 'b']]
                        for lidx in range(num_layers_encoder - 1)]
hidden_cells_encoder = [[tf.nn.rnn_cell.DropoutWrapper(cell,
    output_keep_prob=output_keep_prob, state_keep_prob=state_keep_prob,
    variational_recurrent=True, dtype=tf.float32) for cell in cells] 
                        for cells in hidden_cells_encoder[:-1]] + [hidden_cells_encoder[-1]]
cells_encoder = [first_cell_encoder] + hidden_cells_encoder
c1, c2 = zip(*cells_encoder)
cells_encoder = [c1, c2]
def encode(x, num_layers, cells, initial_states, lengths, name='',):
    prev_layer = x
    shortcut = x
    hiddenlayers = []
    returncells = []
    cell_fw, cell_bw = cells
    bs = tf_bs
    for idx in range(num_layers):
        prev_layer, c = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = cell_fw[idx],
                cell_bw = cell_bw[idx],
                inputs = prev_layer,
                sequence_length=lengths,
                initial_state_fw=None,
                initial_state_bw=None,
                dtype=tf.float32,
                scope='encoder'+str(idx)
            )
        if idx == num_layers - 1:
            fw = prev_layer[0]
            bw = prev_layer[1]
            stacked = tf.stack([tf.range(bs), lengths - 1], 1)
            fw_final = tf.gather_nd(fw,stacked,name=None)
            bw_final = bw[:,0,:]
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
encoding_last_layer, encoding_final_cells, encoding_hidden_layers, encoding_last_timestep, dbg1, dbg2 = encode(
    cmd_emb, num_layers_encoder, cells_encoder,None, lengths = cmd_lengths, name = 'encoder')
# encoding_last_timestep = encoding_last_layer[:,cmd_lengths, :]
hidden_filters_encoder = encoding_last_timestep.shape[-1].value
first_cell_subprogram = tf.nn.rnn_cell.LSTMCell(
    hidden_filters_subprogram, forget_bias=1., name = 'subpogramlayer1_')
hidden_cells_subprogram = [tf.nn.rnn_cell.LSTMCell(
    hidden_filters_subprogram,forget_bias=1., name = 'subpogramlayer' + str(lidx))
                        for lidx in range(num_layers_subprogram - 1)]

cells_subprogram_rnn = [first_cell_subprogram] + hidden_cells_subprogram

attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
    num_units=hidden_filters_encoder, memory=encoding_last_layer,
    memory_sequence_length=cmd_lengths)
attention_mechanism = tf.contrib.seq2seq.LuongAttention(
    num_units=hidden_filters_encoder//2, memory=encoding_last_layer,
    memory_sequence_length=cmd_lengths)
cells_subprogram = [
    tf.contrib.seq2seq.AttentionWrapper(
        cell, attention_mechanism, attention_layer_size = hidden_filters_subprogram) 
    for cell in cells_subprogram_rnn]

def subprogram(x, num_layers, cells, initial_states, lengths, reuse, name='',):
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
                    cell = cells[idx],
                    inputs = prev_layer,
                    sequence_length=lengths,
                    initial_state = None,
                    dtype=tf.float32,
                )
            prev_layer = tf.concat(prev_layer, 2)
            prev_layer = tf.nn.leaky_relu(prev_layer)
            returncells.append(c)
            hiddenlayers.append(prev_layer)
            if idx == num_layers - 1:
                stacked = tf.stack([tf.range(bs), lengths], 1)
                output = tf.gather_nd(
                            prev_layer,
                            stacked,
                            name=None
                        )
                return prev_layer, returncells, hiddenlayers, output
            prev_layer = tf.concat((prev_layer, shortcut), 2)
encodings = [encoding_last_timestep]
last_encoding = encoding_last_timestep
initial_cmb_encoding = last_encoding
loss = 0
action_probabilities_presoftmax = []
for sub_idx in range(max_num_subprograms): 
    from_last_layer = tf.tile(tf.expand_dims(tf.concat((
        last_encoding, initial_cmb_encoding), 1), 1), [1, max_num_actions + 1, 1])
    autoregressive = act_emb_with_st[:,sub_idx, :, :]
    x_input = tf.concat((from_last_layer, autoregressive), -1)
    subprogram_last_layer, _, subprogram_hidden_layers, subprogram_output = subprogram(
        x_input, num_layers_subprogram, cells_subprogram,None, 
        lengths = act_lengths[:, sub_idx], reuse = (sub_idx > 0), name = 'subprogram')
    action_prob_flat = mlp(
        tf.reshape(subprogram_last_layer, [-1, hidden_filters_subprogram]),
        [], output_size = num_act, name = 'action_choice_mlp', reuse = (sub_idx > 0))
    action_prob_expanded = tf.reshape(action_prob_flat, [-1, max_num_actions + 1, num_act])
    action_probabilities_layer = tf.nn.softmax(action_prob_expanded, axis=-1)
    action_probabilities_presoftmax.append(action_prob_expanded)
    delta1, delta2 = [mlp(
        subprogram_output, [256,], output_size = hidden_filters_encoder, name = 'global_transform' + str(idx),
        reuse = (sub_idx > 0)
    ) for idx in range(2)]
    remember = tf.sigmoid(delta1)
    last_encoding = last_encoding * remember + delta2
    encodings.append(last_encoding)
act_presoftmax = tf.stack(action_probabilities_presoftmax, 1)[:, :, :-1, :]
#batch, subprogram, timestep, action_selection
logprobabilities = tf.nn.log_softmax(act_presoftmax, -1)
act_presoftmax_flat = tf.reshape(act_presoftmax, [-1, 9, num_act])
mask_ph_flat = tf.reshape(mask_ph, [-1, max_actions_per_subprogram])
act_ind_flat = tf.reshape(act_ind, [-1, max_actions_per_subprogram])
ppl_loss = tf.contrib.seq2seq.sequence_loss(
    logits = act_presoftmax_flat,
    targets = act_ind_flat,
    weights = mask_ph_flat,
    average_across_timesteps=False,
    average_across_batch=False,
    softmax_loss_function=None,
    name=None
)
ppl_loss_avg = tf.reduce_mean(tf.pow(ppl_loss, 2.0)) * 10000 # + tf.reduce_mean(tf.pow(ppl_loss, 1.0)) * 100

tfvars = tf.trainable_variables()
weight_norm = tf.reduce_mean([tf.reduce_sum(tf.square(var)) for var in tfvars])*1e-3

action_taken = tf.argmax(act_presoftmax, -1, output_type=tf.int32)
correct_mat = tf.cast(tf.equal(action_taken, act_ind), tf.float32) * mask_ph
correct_percent = tf.reduce_sum(correct_mat, [1, 2])/tf.reduce_sum(mask_ph, [1, 2])
percent_correct = tf.reduce_mean(correct_percent)
percent_fully_correct = tf.reduce_mean(tf.cast(tf.equal(correct_percent, 1), tf.float32))

loss = ppl_loss_avg + weight_norm

opt_fcn = tf.train.AdamOptimizer(learning_rate=learning_rate)
#opt_fcn = tf.train.MomentumOptimizer(learning_rate=learning_rate, use_nesterov=True, momentum=.8)
optimizer, grad_norm_total = apply_clipped_optimizer(opt_fcn, loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[24]:


hidden_filters


# In[25]:


encoding_last_layer


# In[ ]:


np.random.seed(0)
trn_percent = .1
num_samples = mask_np.shape[0]
ordered_samples = np.arange(num_samples)
np.random.shuffle(ordered_samples)
trn_samples = ordered_samples[:int(np.ceil(num_samples*trn_percent))]
val_samples_all = ordered_samples[int(np.ceil(num_samples*trn_percent)):]
val_samples = val_samples_all
trn_samples.shape, val_samples.shape


# In[ ]:


total_eval_itr = -1
bs = 256 # trn_samples.shape[0] // 2
bs_eval = 4096 * 2
num_eval_samples = float(val_samples_all.shape[0])
num_eval_batches = int(np.ceil(num_eval_samples / bs_eval))
for itr in range(100000):
    if itr == 0:
        samples = np.random.choice(trn_samples, size=bs, replace=False)
        trn_feed_dict = {
            cmd_ind: cmd_np[samples],
            act_ind: act_np[samples],
            mask_ph: mask_np[samples],
            act_lengths: np.clip(struct_np[samples], a_min=1, a_max=None),
            cmd_lengths: cmd_lengths_np[samples],
        }

    trn_feed_dict[learning_rate] = .02 / (np.power(itr + 10, .6))
    #pdb.set_trace()
    _, trn_loss, acc_trn_single, acc_trn = sess.run(
        [optimizer, loss, percent_correct, percent_fully_correct], trn_feed_dict)
    if itr == 0:
        trn_loss_avg = trn_loss
        acc_trn_avg = acc_trn
        acc_trn_single_avg = acc_trn_single
    else:
        trn_loss_avg = trn_loss_avg * .9 + trn_loss * .1
        acc_trn_avg = acc_trn_avg * .9 + acc_trn * .1
        acc_trn_single_avg = acc_trn_single_avg * .9 + acc_trn_single * .1
    if itr % 10 == 0 and itr > 0:
        total_eval_itr += 1.
        val_loss = 0.
        acc_val = 0.
        for eval_itr in range(num_eval_batches):
            val_samples = val_samples_all[eval_itr * bs_eval: (eval_itr + 1) * bs_eval]
            if eval_itr == num_eval_batches - 1:
                val_samples = val_samples_all[eval_itr * bs_eval:]
            sample_size = float(val_samples.shape[0])
            val_feed_dict = {
                cmd_ind: cmd_np[val_samples],
                act_ind: act_np[val_samples],
                mask_ph: mask_np[val_samples],
                act_lengths: np.clip(struct_np[val_samples], a_min=1, a_max=None),
                cmd_lengths: cmd_lengths_np[val_samples]
            }
            val_loss_i, acc_val_i = sess.run([loss, percent_fully_correct], val_feed_dict)
            #print(eval_itr, sample_size, num_eval_samples, acc_val_i)
            val_loss = val_loss + val_loss_i * sample_size / num_eval_samples
            acc_val = acc_val + acc_val_i * sample_size / num_eval_samples
        if total_eval_itr == 0:
            val_loss_avg = val_loss
            acc_val_avg = acc_val
        else:
            val_loss_avg = val_loss_avg * .9 + val_loss * .1
            acc_val_avg = acc_val_avg * .9 + acc_val * .1
        print('itr:', itr, 'trn_loss', trn_loss_avg, 'val_loss', val_loss_avg)
        print('itr:', itr, 'trn_acc', acc_trn_avg,
              'trn_single_acc', acc_trn_single_avg, 'val_acc', acc_val_avg)

