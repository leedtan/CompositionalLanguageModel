"""
Models and training functions

"""
import tensorflow as tf
from tensorflow.python.framework import ops
import util
import numpy as np
import pickle

#=========== Data =============
def parseFile(fname, delimiter = ':::'):
    commands, actions, structures = [], [], []
    with open(fname) as f:
        for line in f:
            _, command, raw_actions, raw_structure = line.split(delimiter)
            actions.append(raw_actions.split())
            structures.append(list(map(int, raw_structure.split())))
            commands.append(command.split())
    return commands, actions, structures


def file2np(commands, actions, structures, command_map, action_map, max_cmd_len, max_actions_per_subprogram, max_num_subprograms):
    # Convert to np
    commands_ind = [[command_map[c] for c in cmd] + [0] * (max_cmd_len - len(cmd)) for cmd in commands]
    cmd_np = np.array(commands_ind)
    cmd_lengths_list = [len(s) + 1 for s in commands]
    cmd_lengths_np = np.array(cmd_lengths_list)

    actions_ind = [[action_map[a] for a in act]  for act in actions]
    actions_structured = []

    for row in range(len(structures)):
        action_row = []
        act = actions_ind[row]
        struct = structures[row]
        start = 0
        for step in struct:
            end = start + step
            a = act[start:end]
            padding = max_actions_per_subprogram - step - 2 # Add start and end action to each sub-program
            action_row.append([action_map['start_action']] + a + [action_map['end_action']] + [0] * padding)
            start = end
        actions_structured.append(
            action_row + [[action_map['end_subprogram']] + [0] * (max_actions_per_subprogram - 1)] +
            [[0] * max_actions_per_subprogram] * (max_num_subprograms - len(struct) - 1)
        )
    act_np = np.array(actions_structured)
    struct_padded = [[sa + 1 for sa in s] + [1] + [0] * (max_num_subprograms - len(s) - 1) for s in structures] # Add end
    struct_np = np.array(struct_padded)

    mask_list = [[np.concatenate((np.ones(st), np.zeros(max_actions_per_subprogram - st)), 0) for st in s] for s in struct_np]
    mask_np = np.array(mask_list)

    return cmd_np, act_np, mask_np, struct_np, cmd_lengths_np


class DataSet(object):

    def __init__(self, fname, command_map, action_map, max_cmd_len, max_actions_per_subprogram, max_num_subprograms, delimiter = ':::', seed=100):
        """
        Construct a DataSet from input files
        """
        commands, actions, structures = parseFile(fname, delimiter = delimiter)
        self.cmd_np, self.act_np, self.mask_np, self.struct_np, self.cmd_lengths_np = \
            file2np(commands, actions, structures, command_map, action_map, max_cmd_len, max_actions_per_subprogram, max_num_subprograms)

        self._dataSize = self.cmd_np.shape[0]

        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.seed = seed
        self._allIdx = np.arange(self._dataSize)
        self._perm = np.arange(self._dataSize)
        np.random.seed(seed)

    def next_batch(self, batch_size, isTrain = True):
        """Return the next `batch_size` examples from this data set. If isTrain = False, do not shuffle"""
        if batch_size >= self._dataSize:
            cmd, act, mask, struct, cmd_length = self.cmd_np, self.act_np, self.mask_np, self.struct_np, self.cmd_lengths_np
            idx = self._allIdx
        else:
            start = self._index_in_epoch
            self._index_in_epoch += batch_size
            if start >= self._dataSize:
                # Finished epoch
                self._epochs_completed += 1
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size
                # Re-Shuffle the data indices
                if isTrain:
                    np.random.shuffle(self._perm)
            elif self._index_in_epoch > self._dataSize:
                self._index_in_epoch = self._dataSize

            if (self._index_in_epoch == batch_size) & (self._epochs_completed == 0) & isTrain:
                # Re-Shuffle the data indices at initial epoch
                np.random.shuffle(self._perm)

            batch_idx = self._perm[start:self._index_in_epoch]
            cmd, act, mask, struct, cmd_length = self.cmd_np[batch_idx], self.act_np[batch_idx], self.mask_np[batch_idx], \
                                                 self.struct_np[batch_idx], self.cmd_lengths_np[batch_idx]
            idx = self._allIdx[batch_idx]

        return cmd, act, mask, struct, cmd_length, idx

    def reset(self):
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._perm = np.arange(self._dataSize)
        np.random.seed(self.seed)


#=========== Models ===============
def encode(x, num_layers, cells, initial_states, lengths, name=''):
    """
    General encoding function
    :param x: input
    :param num_layers:
    :param cells: list of rnn cells
    :param lengths: cmd_length
    """
    prev_layer = x
    shortcut = x
    hiddenlayers = []
    returncells = []
    cell_fw, cell_bw = cells
    bs = tf.shape(x)[0]
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
        if idx != num_layers - 1:
            prev_layer = tf.concat((prev_layer, shortcut), 2)
    return prev_layer, returncells, hiddenlayers, output, fw, stacked



def subprogram(x, num_layers, cells, initial_states, lengths, reuse, name='',):
    """
    RNN of each subprogram, shared across subprograms
    :param x: last encoder output
    :param num_layers:
    :param cells: decoder cells
    :param lengths: struct_np
    :param reuse:
    :param name:
    :return:
    """
    prev_layer = x
    shortcut = x
    hiddenlayers = []
    returncells = []
    bs = tf.shape(x)[0]
    for idx in range(num_layers):
        with tf.variable_scope(name + 'subprogram' + str(idx), reuse=reuse):
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
            if idx != num_layers - 1:
                prev_layer = tf.concat((prev_layer, shortcut), 2)
    output = tf.gather_nd(prev_layer, tf.stack([tf.range(bs), lengths], 1), name=None)
    return prev_layer, returncells, hiddenlayers, output



class m1():
    def __init__(self, model_paras):
        self.num_cmd = model_paras.get('num_cmd', 14)
        self.num_act = model_paras.get('num_act', 9)
        self.max_cmd_len = model_paras.get('max_cmd_len', 10)
        self.max_num_subprograms = model_paras.get('max_num_subprograms', 7)
        self.max_actions_per_subprogram = model_paras.get('max_actions_per_subprogram', 10)

        self.hidden_filters = model_paras.get('hidden_filters', 128)
        self.num_layers_encoder = model_paras.get('num_layers_encoder', 2)
        self.size_emb = model_paras.get('size_emb', 64)
        self.init_mag = model_paras.get('init_mag', 1e-3)
        self.hidden_filters_subprogram = model_paras.get('hidden_filters_subprogram', 128)
        self.num_layers_subprogram = model_paras.get('num_layers_subprogram', 3)
        self.l2_lambda = model_paras.get('l2_lambda', 1e-3) # coefficient of L2 penalization


    def _buildGraph(self):
        # Inputs
        self.output_keep_prob = tf.placeholder_with_default(1.0, ())
        self.state_keep_prob = tf.placeholder_with_default(1.0, ())
        self.cmd_ind = tf.placeholder(tf.int32, shape=(None, self.max_cmd_len))
        self.act_ind = tf.placeholder(tf.int32, shape=(None, self.max_num_subprograms, self.max_actions_per_subprogram))
        self.mask_ph = tf.placeholder(tf.float32, shape=(None, self.max_num_subprograms, self.max_actions_per_subprogram))
        self.cmd_lengths = tf.placeholder(tf.int32, shape=(None,))
        self.act_lengths = tf.placeholder(tf.int32, shape=(None, self.max_num_subprograms))
        self.learning_rate = tf.placeholder(tf.float32, shape=(None))

        ## Embeddings
        self.cmd_mat = tf.Variable(self.init_mag * tf.random_normal([self.num_cmd, self.size_emb]))
        self.act_mat = tf.Variable(self.init_mag * tf.random_normal([self.num_act, self.size_emb]))
        cmd_emb = tf.nn.embedding_lookup(self.cmd_mat, self.cmd_ind)
        act_emb = tf.nn.embedding_lookup(self.act_mat, self.act_ind)

        ## Encoder
        first_cell_encoder = [tf.nn.rnn_cell.LSTMCell(self.hidden_filters, forget_bias=1., name='layer1_' + d) for d in ['f', 'b']]
        if self.num_layers_encoder > 1:
            hidden_cells_encoder = [[tf.nn.rnn_cell.LSTMCell(self.hidden_filters, forget_bias=1., name='layer' + str(lidx) + '_' + d) for d in ['f', 'b']]
                for lidx in range(self.num_layers_encoder - 1)]
            hidden_cells_encoder = [[tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.output_keep_prob,
                                                                   state_keep_prob=self.state_keep_prob, variational_recurrent=True, dtype=tf.float32) for
                                     cell in cells] for cells in hidden_cells_encoder[:-1]] + [hidden_cells_encoder[-1]]  # No dropout on last layer

            cells_encoder = [first_cell_encoder] + hidden_cells_encoder
        else:
            cells_encoder = [first_cell_encoder]
        c1, c2 = zip(*cells_encoder)
        cells_encoder = [c1, c2]

        encoding_last_layer, encoding_final_cells, encoding_hidden_layers, encoding_last_timestep, dbg1, dbg2 = \
            encode(cmd_emb, self.num_layers_encoder, cells_encoder, None, lengths=self.cmd_lengths, name='encoder')
        hidden_filters_encoder = encoding_last_timestep.shape[-1].value

        ## Decoder
        first_cell_subprogram = tf.nn.rnn_cell.LSTMCell(self.hidden_filters_subprogram, forget_bias=1., name='subpogramlayer1_')
        if self.num_layers_subprogram > 1:
            hidden_cells_subprogram = [tf.nn.rnn_cell.LSTMCell(self.hidden_filters_subprogram, forget_bias=1., name='subpogramlayer' + str(lidx))
                for lidx in range(self.num_layers_subprogram - 1)]
            cells_subprogram_rnn = [first_cell_subprogram] + hidden_cells_subprogram
        else:
            cells_subprogram_rnn = [first_cell_subprogram]

        #-- Add attention ---
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=hidden_filters_encoder // 2, memory=encoding_last_layer, memory_sequence_length=self.cmd_lengths)
        cells_subprogram = [tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism, attention_layer_size=self.hidden_filters_subprogram)
                            for cell in cells_subprogram_rnn]

        encodings = [encoding_last_timestep]
        last_encoding = encoding_last_timestep
        initial_cmb_encoding = last_encoding

        action_probabilities_presoftmax = []
        # Iterate through each subprogram
        for sub_idx in range(self.max_num_subprograms):
            from_last_layer = tf.tile(tf.expand_dims(tf.concat((initial_cmb_encoding, last_encoding), 1), 1), [1, self.max_actions_per_subprogram, 1])  # Shape bs x (self.max_actions_per_subprogram) x LSTM_outx2
            autoregressive = act_emb[:, sub_idx, :, :]
            x_input = tf.concat((from_last_layer, autoregressive), -1)  # bs x (self.max_actions_per_subprogram) x (input dim)
            subprogram_last_layer, _, subprogram_hidden_layers, subprogram_output = subprogram(x_input, self.num_layers_subprogram, cells_subprogram, None,
                lengths=self.act_lengths[:, sub_idx], reuse=(sub_idx > 0), name='subprogram')  # Reuse the same network across subprogram
            action_prob_flat = util.mlp(tf.reshape(subprogram_last_layer, [-1, self.hidden_filters_subprogram]), [], output_size=self.num_act, name='action_choice_mlp', reuse=(sub_idx > 0))  # (bs x num_act_sub) x num_act
            action_prob_expanded = tf.reshape(action_prob_flat, [-1, self.max_actions_per_subprogram, self.num_act])
            action_probabilities_layer = tf.nn.softmax(action_prob_expanded, axis=-1)
            action_probabilities_presoftmax.append(action_prob_expanded)
            # Recurrent across subprograms
            delta1, delta2 = [util.mlp(subprogram_output, [256, ], output_size=hidden_filters_encoder, name='global_transform' + str(idx), reuse=(sub_idx > 0) ) for idx in range(2)]
            remember = tf.sigmoid(delta1)
            insert = tf.tanh(delta2) + delta2 / 100
            last_encoding = last_encoding * remember + insert
            encodings.append(last_encoding)

        ## Loss
        #act_presoftmax = tf.stack(action_probabilities_presoftmax, 1)[:, :, 1:-1, :]
        act_presoftmax = tf.stack(action_probabilities_presoftmax, 1)
        # batch, subprogram, timestep, action_selection
        self.logprobabilities = tf.nn.log_softmax(act_presoftmax, -1)
        act_presoftmax_flat = tf.reshape(act_presoftmax, [-1, self.max_actions_per_subprogram, self.num_act])
        mask_ph_flat = tf.reshape(self.mask_ph, [-1, self.max_actions_per_subprogram])
        act_ind_flat = tf.reshape(self.act_ind, [-1, self.max_actions_per_subprogram])

        ppl_loss = tf.contrib.seq2seq.sequence_loss(logits=act_presoftmax_flat, targets=act_ind_flat, weights=mask_ph_flat,
            average_across_timesteps=False, average_across_batch=False, softmax_loss_function=None, name='ppl_loss')
        ppl_loss_avg = tf.reduce_mean(tf.pow(ppl_loss, 2.0)) * 10000

        tfvars = tf.trainable_variables()
        weight_norm = tf.reduce_mean([tf.reduce_sum(tf.square(var)) for var in tfvars]) * self.l2_lambda

        action_taken = tf.argmax(act_presoftmax, -1, output_type=tf.int32)
        correct_mat = tf.cast(tf.equal(action_taken, self.act_ind), tf.float32) * self.mask_ph
        correct_percent = tf.reduce_sum(correct_mat, [1, 2]) / tf.reduce_sum(self.mask_ph, [1, 2])
        self.percent_correct = tf.reduce_mean(correct_percent)
        self.percent_fully_correct = tf.reduce_mean(tf.cast(tf.equal(correct_percent, 1), tf.float32))

        self.loss = ppl_loss_avg + weight_norm

        opt_fcn = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # opt_fcn = tf.train.MomentumOptimizer(learning_rate=learning_rate, use_nesterov=True, momentum=.8)
        self.optimizer, grad_norm_total = util.apply_clipped_optimizer(opt_fcn, self.loss)


#=========== Training module =============



class trainModel:
    def __init__(self, model, train_paras, trainset, testset, flgEval = False):
        self.train_paras = train_paras
        self.batchSize = train_paras.get('batchSize', 8)
        self.nIter = train_paras.get('nIter', 50)
        self.seed = train_paras.get('seed', 1)
        self.testIter = train_paras.get('testIter', 50)  # Print validation every testIter iterations
        self.flgSave = train_paras.get('flgSave', True)
        self.savePath = train_paras.get('savePath')

        self.trainset = trainset
        self.testset = testset
        self.flgEval = flgEval
        self.flgTest = False

        if self.testset is not None:
            self.flgTest = True
            self.nBatchTest = max(1, np.ceil(self.testset._dataSize / self.batchSize).astype('int32'))

        self.m = model
        tf.set_random_seed(self.seed)
        np.random.seed(self.seed)

    def run(self):
        lsTrainAcc, lsTestAcc, self.bestAcc = [], [], 0
        self.m._buildGraph()
        saver = tf.train.Saver()

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        trn_loss_avg, acc_trn_avg, acc_trn_single_avg = 0, 0, 0

        for itr in range(self.nIter):
            trn_loss_avg, acc_trn_avg, acc_trn_single_avg = self._trainEpoch(sess, itr, trn_loss_avg, acc_trn_avg, acc_trn_single_avg)
            lsTrainAcc.append(acc_trn_avg)

            if (itr % self.testIter == 0) & (self.flgTest) :
                val_loss_avg, acc_val_avg, pred = self._test(sess)
                print('itr:', itr, 'trn_loss', round(trn_loss_avg, 4), 'val_loss', round(val_loss_avg, 4))
                print('itr:', itr, 'trn_acc', round(acc_trn_avg, 4), 'trn_single_acc', round(acc_trn_single_avg,4), 'val_acc', round(acc_val_avg, 4))
                lsTestAcc.append(acc_val_avg)
                if (acc_val_avg > self.bestAcc) or (itr == 0):
                    self.bestAcc = acc_val_avg
                    if self.flgSave:
                        save_path = saver.save(sess, self.savePath + "/model")
                        pickle.dump(pred, open(self.savePath + 'pred.p', 'wb'))
        return self.m, lsTrainAcc, lsTestAcc


    def _trainEpoch(self, sess, iter, trn_loss_avg, acc_trn_avg, acc_trn_single_avg):
        cmd, act, mask, struct, cmd_length, idx = self.trainset.next_batch(self.batchSize, isTrain=True)
        trn_feed_dict = {self.m.cmd_ind: cmd, self.m.act_ind: act, self.m.mask_ph: mask, self.m.act_lengths: np.clip(struct, a_min=1, a_max=None), self.m.cmd_lengths: cmd_length}
        trn_feed_dict[self.m.learning_rate] = .02 / (np.power(iter + 10, .6))
        _, trn_loss, acc_trn_single, acc_trn = sess.run([self.m.optimizer, self.m.loss, self.m.percent_correct, self.m.percent_fully_correct], trn_feed_dict)

        if iter == 0:
            trn_loss_avg = trn_loss
            acc_trn_avg = acc_trn
            acc_trn_single_avg = acc_trn_single
        else:
            trn_loss_avg = trn_loss_avg * .9 + trn_loss * .1
            acc_trn_avg = acc_trn_avg * .9 + acc_trn * .1
            acc_trn_single_avg = acc_trn_single_avg * .9 + acc_trn_single * .1
        return trn_loss_avg, acc_trn_avg, acc_trn_single_avg

    def _test(self, sess):
        val_loss_all, acc_val_all = 0, 0
        pred = []
        for i in range(self.nBatchTest):
            cmd, act, mask, struct, cmd_length, idx = self.testset.next_batch(self.batchSize, isTrain=False)
            val_feed_dict = {self.m.cmd_ind: cmd, self.m.act_ind: act, self.m.mask_ph: mask, self.m.act_lengths: np.clip(struct, a_min=1, a_max=None), self.m.cmd_lengths: cmd_length,}
            val_loss, acc_val, pred_i = sess.run([self.m.loss, self.m.percent_fully_correct, self.m.logprobabilities], val_feed_dict)
            bs = len(cmd)
            val_loss_all += val_loss
            acc_val_all += acc_val * bs
            pred.append(pred_i)
        return val_loss_all/self.nBatchTest, acc_val_all/self.testset._dataSize, pred

    def _loadModel(self):
        pass


    def _delGraph(self):
        tf.reset_default_graph()







