# Largely based on Tree-Structured Decoding with Doublyrecurrent Neural Networks
# (https://openreview.net/pdf?id=HkYhZDqxg)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import time
import json
import os

import numpy as np
import tensorflow as tf

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "test",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_string("data_path", None,
                    "XXX")
flags.DEFINE_string("dependencies", None,
                    "XXX")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS

possible_dependencies = {
    'parent': 'forward',
    'left_sibling': 'forward',
    'left_prior': 'forward',
    'children': 'reverse',
    'right_sibling': 'reverse',
    'right_prior': 'reverse'
}

SmallConfig = {
  "init_scale" : 0.1,
  "learning_rate" : 1.0,
  "max_grad_norm" : 5,
  "num_layers" : 2,
  "num_steps" : 20, # this isn't used at all in this file, since we aren't doing any truncated backpropagation
  "hidden_size" : 200,
  "max_epoch" : 4,
  "max_max_epoch" : 6,
  "keep_prob" : 1.0,
  "lr_decay" : 0.5,
  "batch_size" : 40, # currently, this is just 1
}

MediumConfig = {
  "init_scale" : 0.05,
  "learning_rate" : 1.0,
  "max_grad_norm" : 5,
  "num_layers" : 2,
  "num_steps" : 35,
  "hidden_size" : 650,
  "max_epoch" : 6,
  "max_max_epoch" : 39,
  "keep_prob" : 0.5,
  "lr_decay" : 0.8,
  "batch_size" : 20,
}

LargeConfig = {
  "init_scale" : 0.04,
  "learning_rate" : 1.0,
  "max_grad_norm" : 10,
  "num_layers" : 2,
  "num_steps" : 35,
  "hidden_size" : 1500,
  "max_epoch" : 14,
  "max_max_epoch" : 55,
  "keep_prob" : 0.35,
  "lr_decay" : 1 / 1.15,
  "batch_size" : 20,
}

TestConfig = {
  "init_scale" : 0.1,
  "learning_rate" : 1.0,
  "max_grad_norm" : 1,
  "num_layers" : 1,
  "num_steps" : 2,
  "hidden_size" : 4,
  "max_epoch" : 1,
  "max_max_epoch" : 1,
  "keep_prob" : 1.0,
  "lr_decay" : 0.5,
  "batch_size" : 20,
}

# store the LSTM state in a TensorArray
def save_multi_lstm_state(state_array, state, position):
    copy = []
    for i, (c, h) in enumerate(state):
        copy.append([])
        # c is in position 0, h is in position 1
        copy[i].append(state_array[i][0].write(position, state[i].c))
        copy[i].append(state_array[i][1].write(position, state[i].h))
    return copy

# reconstruct the LSTM state from a TensorArray
# initial_state is needed to use as a template for restoration
def restore_multi_lstm_state(state_array, initial_state, position, division_scalar=None, add_state=None):
    state = []
    for i, (c, h) in enumerate(initial_state):
        c_state = state_array[i][0].read(position)
        h_state = state_array[i][1].read(position)
        if add_state is not None:
            c_state += add_state[i].c
            h_state += add_state[i].h
        elif division_scalar is not None:
            c_state /= division_scalar
            h_state /= division_scalar

        # TensorArray returns shape <unknown>, which breaks things when passed to LSTM cell()
        c_state.set_shape(initial_state[i].c.shape)
        h_state.set_shape(initial_state[i].h.shape)

        state.append(tf.contrib.rnn.LSTMStateTuple(c_state, h_state))

    return tuple(state)

# initialize an array of TensorArrays to store an LSTM, based on the initial_state template
# XXX don't we know how big the tensor array will need to be?
def initialize_lstm_array(initial_state):
    states = []
    for i, (c, h) in enumerate(initial_state):
        states.append([])
        for k in [c, h]:
            states[i].append(tf.TensorArray(
                tf.float32,
                size=0,
                dynamic_size=True,
                clear_after_read=False,
                infer_shape=False))
    return states

# store the LSTM state in a TensorArray
def save_lstm_state(state_array, state, position):
    return [state_array[0].write(position, state.c),
            state_array[1].write(position, state.h)]

# reconstruct the LSTM state from a TensorArray
# initial_state is needed to use as a template for restoration
def restore_lstm_state(state_array, initial_state, position, add_state=None):
    c = state_array[0].read(position)
    c.set_shape(initial_state.c.shape)
    h = state_array[1].read(position)
    h.set_shape(initial_state.h.shape)
    if add_state is not None:
        c += add_state.c
        h += add_state.h

    return tf.contrib.rnn.LSTMStateTuple(c, h)


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32

# we no longer store (c,h), but rather, "c" is the part of the memory cell already past through the forget gate
class TreeLSTMCell(tf.contrib.rnn.RNNCell):

    def __init__(self, num_units, forget_bias=1.0, input_size=None, activation=tf.tanh, reuse=None):
        super(TreeLSTMCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation

        # TODO: get _linear equivalent to work

        # XXX XXX initializers??
        self.W_i = tf.get_variable("W_i", [num_units, num_units], dtype=data_type())
        self.U_i = tf.get_variable("U_i", [num_units, num_units], dtype=data_type())
        self.b_i = tf.get_variable("b_i", [1, num_units], dtype=data_type())

        self.W_f = tf.get_variable("W_f", [num_units, num_units], dtype=data_type())
        self.U_f = tf.get_variable("U_f", [num_units, num_units], dtype=data_type())
        self.b_f = tf.get_variable("b_f", [1, num_units], dtype=data_type())

        self.W_o = tf.get_variable("W_o", [num_units, num_units], dtype=data_type())
        self.U_o = tf.get_variable("U_o", [num_units, num_units], dtype=data_type())
        self.b_o = tf.get_variable("b_o", [1, num_units], dtype=data_type())

        self.W_u = tf.get_variable("W_u", [num_units, num_units], dtype=data_type())
        self.U_u = tf.get_variable("U_u", [num_units, num_units], dtype=data_type())
        self.b_u = tf.get_variable("b_u", [1, num_units], dtype=data_type())

    @property
    def state_size(self):
        return tf.contrib.rnn.LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units


    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):

            current_inputs, parent_inputs = tf.unstack(inputs)
            f, h = state

            i = tf.sigmoid(tf.matmul(current_inputs, self.W_i) + tf.matmul(h, self.U_i) + self.b_i)
            o = tf.sigmoid(tf.matmul(current_inputs, self.W_o) + tf.matmul(h, self.U_o) + self.b_o)
            u = self._activation(tf.matmul(current_inputs, self.W_u) + tf.matmul(h, self.U_u) + self.b_u)

            new_c = f + i * u
            new_h = self._activation(new_c) * o

            # XXX double bias?????
            new_f = tf.sigmoid(tf.matmul(parent_inputs, self.W_f) +
                               tf.matmul(new_h, self.U_f) +
                               self.b_f) * new_c

            new_state = tf.contrib.rnn.LSTMStateTuple(new_f, new_h)
            return new_h, new_state


class TRNNModel(object):

  def __init__(self, is_training, config):
    self.size = size = config['hidden_size']
    self.label_size = label_size = config['label_vocab_size']
    self.attr_size = attr_size = config['attr_vocab_size']
    self.dependencies = config['dependencies']
    num_layers = config['num_layers']

    # declare a bunch of parameters that will be reused later
    with tf.variable_scope('Parameters'):
        # the second dimension doesn't have to be "size", but does have to match softmax_w's first dimension
        for dependency in self.dependencies:
            tf.get_variable('U_' + dependency, [size, size], dtype=tf.float32)

        tf.get_variable('u_last', [size], dtype=tf.float32)
        tf.get_variable('u_first', [size], dtype=tf.float32)

        tf.get_variable("attr_w", [size, attr_size], dtype=data_type())
        tf.get_variable("attr_b", [attr_size], dtype=data_type())
        tf.get_variable("v_attr", [label_size, attr_size], dtype=data_type())

        tf.get_variable("softmax_w", [size, label_size], dtype=data_type())
        tf.get_variable("softmax_b", [label_size], dtype=data_type())

        tf.get_variable("v_first", [label_size], dtype=data_type())
        tf.get_variable("v_last", [label_size], dtype=data_type())

    with tf.device("/cpu:0"):
      label_embedding = tf.get_variable(
          "label_embedding", [label_size, size / 2], dtype=data_type())
      attr_embedding = tf.get_variable(
          "attr_embedding", [attr_size, size / 2], dtype=data_type())

    def lstm_cell(dependency):
        if dependency == 'children':
            return TreeLSTMCell(size, forget_bias=0.0, reuse=tf.get_variable_scope().reuse)
        else:
            return tf.contrib.rnn.BasicLSTMCell(
                size, forget_bias=0.0, state_is_tuple=True,
                reuse=tf.get_variable_scope().reuse)
    attn_cell = lstm_cell
    if is_training and config['keep_prob'] < 1:
      def attn_cell(dependency):
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(dependency), output_keep_prob=config['keep_prob'])

    self.placeholders = {}
    for k in config['placeholders']:
        self.placeholders[k] = tf.placeholder(tf.int32, [None], name=k+'_placeholder')

    self.dependency_initial_states = dict()
    self.dependency_cells = dict()

    # dependency_states can't be a dictionary on sibling/parent, since it needs to be convertible to a tensor (required
    # for the tf.while_loop). The first index in dependency_states is thus aligned with the dependency in the
    # dependencies array
    dependency_states = []
    dependency_outputs = []

    def embed(index):
        label_index = tf.gather(self.placeholders['label_index'], index,
                                        name="LabelIndexGather")
        attr_index = tf.gather(self.placeholders['attr_index'], index,
                                        name="AttrIndexGather")
        node_label_embedding = tf.gather(label_embedding, label_index,
                                            name="LabelEmbedGather")
        node_attr_embedding = tf.gather(attr_embedding, attr_index,
                                        name="AttrEmbedGather")
        node_embedding = tf.concat([node_label_embedding, node_attr_embedding], 0)
        return tf.expand_dims(node_embedding, 0)

    for i in range(len(self.dependencies)):
        dependency = self.dependencies[i]

        # XXX best I can tell, there is a tensorflow bug that causes basic_lstm_cell/basic_lstm_cell_1 issues...
        self.dependency_cells[dependency] = []
        self.dependency_initial_states[dependency] = []
        for j in range(num_layers):
            with tf.variable_scope("RNN_{}_cell_{}".format(dependency,j)):
                self.dependency_cells[dependency].append(attn_cell(dependency))
                self.dependency_initial_states[dependency].append(self.dependency_cells[dependency][j].zero_state(1, data_type()))
                inp = embed(0) if dependency != 'children' else tf.stack([embed(0), embed(0)])
                self.dependency_cells[dependency][j](inp, self.dependency_initial_states[dependency][j])

        # Need to manually handle LSTM states. This is gross.
        dependency_states.append(initialize_lstm_array(self.dependency_initial_states[dependency]))
        dependency_outputs.append([])
        for j in range(num_layers):
            dependency_outputs[-1].append(tf.TensorArray(
                tf.float32,
                size=0,
                dynamic_size=True,
                clear_after_read=False,
                infer_shape=False))

        # since we only use the TensorArray below, write the initial state in position 0. children needs to write
        # final output there, though
        if self.dependencies[i] != 'children':
            cur_input = embed(0)
            for j in range(num_layers):
                with tf.variable_scope("RNN_{}_cell_{}".format(dependency,j), reuse=True):
                    (cur_input, next_state) = self.dependency_cells[dependency][j](cur_input,
                                                self.dependency_initial_states[dependency][j])
                dependency_states[i][j] =  save_lstm_state(dependency_states[i][j],
                                                           next_state, 0)
                dependency_outputs[i][j] = dependency_outputs[i][j].write(0, cur_input)



    if 'children' in self.dependencies:
        # separate RNN just to predict a parent label given children embedding sequence
        with tf.variable_scope("Children_RNN"):
            # XXX hackish way to get a basic lstm cell
            self.children_predictor = \
                    tf.contrib.rnn.MultiRNNCell([attn_cell('parent') for i in range(num_layers)], state_is_tuple=True)
            # same tensorflow bug as above... have to actually call it (since we only use one multirnn, shouldn't need
            # to do anything with scoping)
            self.children_predictor(tf.zeros([1, size]), self.children_predictor.zero_state(1, data_type()))

        children_predictor_states = initialize_lstm_array(self.children_predictor.zero_state(1, data_type()))

        # TODO: split embedding size from hidden layer size
        # XXX Is this right?
        children_predictor_states = save_multi_lstm_state(children_predictor_states,
                                                          self.children_predictor.zero_state(1, data_type()), 0)

        # extra stuff needed by the children dependency
        # 0 is the <nil> token
        initial_embedding = embed(0)


        cur_input = initial_embedding
        for i in range(num_layers):
            child_idx = self.dependencies.index('children')
            # the second cur_input doesn't do anything -- it's just to get the output from the TreeLSTM
            # XXX could this one just be a regular LSTM instead of a TreeLSTM?
            with tf.variable_scope("RNN_{}_cell_{}".format(dependency,j), reuse=True):
                (output, new_state) = self.dependency_cells['children'][i](
                        tf.stack([cur_input, cur_input]), self.dependency_initial_states['children'][i])
            # XXX XXX XXX parent needs to write here? idk... ugh
            #dependency_states[child_idx][i] = save_lstm_state(dependency_states[child_idx][i], new_state, 0)
            dependency_outputs[child_idx][i] = dependency_outputs[child_idx][i].write(0, output)
            cur_input = output

        # extra TensorArrays for children, hurray!!
        children_tmp_states = initialize_lstm_array(self.dependency_initial_states['children'])
        children_output = tf.TensorArray(
            tf.float32,
            size=0,
            dynamic_size=True,
            clear_after_read=False,
            infer_shape=False)
    else:
        children_tmp_states = 0
        children_output = 0
        children_predictor_states = 0

    # TODO: dropout??
    #if is_training and config['keep_prob'] < 1:
    #  inputs = tf.nn.dropout(inputs, config['keep_prob'])

    # this returns true as long as the loop counter is less than the length of the example
    total_data = tf.squeeze(tf.shape(self.placeholders['is_leaf']))
    def loop_cond_wrapper(direction):
        def loop_cond (loss, ctr, h_pred, dependency_states, dependency_outputs,
                       children_tmp_states, children_output, children_predictor_states,
                       label_probabilities, attr_probabilities, predicted_p_first, predicted_p_last):
            if direction == 'forward':
                return tf.less(ctr, total_data)
            else:
                return tf.greater(ctr, 0)
        return loop_cond

    def loop_body_wrapper(direction, do_loss, layer):
        def loop_body(loss, ctr, h_pred, dependency_states, dependency_outputs,
                      children_tmp_states, children_output, children_predictor_states,
                      label_probabilities, attr_probabilities, predicted_p_first, predicted_p_last):

            # to be accurate
            is_leaf = tf.cast(tf.gather(self.placeholders['is_leaf'], ctr, name="IsLeafGather"), tf.float32)
            first_sibling = tf.cast(tf.gather(self.placeholders['first_sibling'], ctr), tf.float32)
            last_sibling = tf.cast(tf.gather(self.placeholders['last_sibling'], ctr, name="LastSiblingGather"), tf.float32)
            label_index = tf.gather(self.placeholders['label_index'], ctr, name="NodeIndexGather")
            attr_index = tf.gather(self.placeholders['attr_index'], ctr, name="AttrIndexGather")
            parent = tf.gather(self.placeholders['parent'], ctr)
            left_sibling = tf.gather(self.placeholders['left_sibling'], ctr)
            right_sibling = tf.gather(self.placeholders['right_sibling'], ctr)
            num_children = tf.cast(tf.gather(self.placeholders['num_children'], ctr,
                                             name="AttrIndexGather"), tf.float32)
            h_pred_ctr = tf.squeeze(tf.gather(h_pred, [ctr]))




            # Generate both the sibling and parent RNN states for the current node, based on the previous sibling and parent
            for i in range(len(self.dependencies)):
                dependency = self.dependencies[i]
                if possible_dependencies[dependency] != direction:
                    continue
                # During inference, we want to use the directly-supplied label for the parent, since each node is passed in
                # one-by-one and we won't have access to the parent when the child is passed in. During training, we have
                # all nodes in the example at once, so can directly grab the parent's label
                dependency_node = tf.gather(self.placeholders[dependency], ctr, name=(dependency+"Gather")) if dependency != 'children' else None

                #handle_inference = lambda: (self.placeholders['inference'][dependency]['label'], \
                #                            self.placeholders['inference'][dependency]['attr'])
                #handle_training = lambda: (tf.gather(self.placeholders['label_index'], dependency_node,
                #                                    name=(dependency+"TokenGatherLabel")), \
                #                          tf.gather(self.placeholders['attr_index'], dependency_node,
                #                                    name=(dependency+"TokenGatherAttr")))

                #dependency_label_token, dependency_attr_token = tf.cond(self.placeholders['is_inference'],
                #                                                        handle_inference, handle_training)

                #dependency_label_embedding = tf.gather(label_embedding, dependency_label_token,
                #                                       name=(dependency+"LabelEmbedGather"))
                #dependency_attr_embedding = tf.gather(attr_embedding, dependency_attr_token,
                #                                      name=(dependency+"AttrEmbedGather"))
                #dependency_embedding = tf.concat([dependency_label_embedding, dependency_attr_embedding], 0)
                #dependency_embedding = tf.expand_dims(dependency_embedding, 0)
                node_embedding = embed(ctr)


                if layer == 0:
                    inp = node_embedding
                else:
                    old_output = dependency_outputs[i][layer-1].read(ctr)
                    old_output.set_shape([1, size])
                    inp = old_output

                if dependency != 'children':
                    # reconstruct the LSTM state of the parent/sibling from the TensorArray
                    state = restore_lstm_state(dependency_states[i][layer],
                                                self.dependency_initial_states[dependency][layer],
                                                dependency_node)

                    with tf.variable_scope("RNN_{}_cell_{}".format(dependency,layer), reuse=True):
                        (output, new_state) = self.dependency_cells[dependency][layer](inp, state)

                    dependency_states[i][layer] = save_lstm_state(dependency_states[i][layer], new_state, ctr)
                    dependency_outputs[i][layer] = dependency_outputs[i][layer].write(ctr, output)

                else:
                    # TODO: could eventually do something with attention over the children?
                    # TODO: have a decoder, or just give the output directly from this RNN to h_pred?
                    if layer == 0:
                        leaf_embedding = initial_embedding
                    else:
                        leaf_embedding = dependency_outputs[i][layer-1].read(0)
                        leaf_embedding.set_shape([1, size])

                    # XXX XXX XXX XXX XXX fix tf.zeros here
                    bootstrap_leaf = lambda: \
                        (self.dependency_cells['children'][layer](
                            tf.stack([leaf_embedding, inp]),
                            self.dependency_initial_states['children'][layer])[1],
                         children_output if layer != num_layers - 1 else \
                            children_output.write(ctr, tf.zeros([1,size])))

                    bootstrap_else = lambda: \
                        (restore_lstm_state(dependency_states[i][layer],
                                            self.dependency_initial_states[dependency][layer], ctr),
                         children_output)

                    state, children_output = tf.cond(tf.cast(num_children, tf.bool), bootstrap_else,
                                                bootstrap_leaf, strict=True)

                    # XXX XXX this needs to be fixed for inference :-X
                    if layer == 0:
                        parent_embedding = embed(parent)
                    else:
                        parent_embedding = dependency_outputs[i][layer-1].read(parent)
                        parent_embedding.set_shape([1, size])

                    joint_embedding = tf.stack([inp, parent_embedding])
                    with tf.variable_scope("RNN_{}_cell_{}".format(dependency,layer), reuse=True):
                        (future_output, future_state) = self.dependency_cells['children'][layer](joint_embedding, state)

                    if layer == num_layers-1:
                        # for now, ignore output of all but last (could eventually use with attention)?
                        cp_state = restore_multi_lstm_state(children_predictor_states,
                                                            self.children_predictor.zero_state(1, data_type()),
                                                            right_sibling)
                        with tf.variable_scope("Children_RNN", reuse=True):
                            (cp_output, next_cp_state) = self.children_predictor(future_output, cp_state)
                        children_predictor_states = save_multi_lstm_state(children_predictor_states,
                                                                          next_cp_state, ctr)

                        # XXX XXX something is wrong with cp_output name?
                        handle_first = lambda: children_output.write(parent, cp_output)
                        handle_not_first = lambda: children_output

                        children_output = tf.cond(tf.cast(left_sibling, tf.bool),
                                                    handle_not_first,
                                                    handle_first)


                    # this is basically doing a +=. If we are the last sibling (on the right), we need to start
                    # from 0. If we are the first sibling (on the left), we need to write the sum to the parent
                    dependency_outputs[i][layer] = dependency_outputs[i][layer].write(ctr, future_output)

                    # update state
                    handle_last = lambda: future_state
                    handle_not_last = lambda: \
                            restore_lstm_state(children_tmp_states[layer],
                                                self.dependency_initial_states[dependency][layer],
                                                ctr, add_state=future_state)

                    new_state = tf.cond(tf.cast(right_sibling, tf.bool),
                                        handle_not_last, handle_last, strict=True)

                    # save to parent
                    handle_first = lambda state: (children_tmp_states[layer],
                                                    save_lstm_state(state, new_state, parent))
                    # save to sibling
                    handle_not_first = lambda state: (save_lstm_state(children_tmp_states[layer], new_state, left_sibling),
                                                        state)
                    children_tmp_states[layer], dependency_states[i][layer] = \
                            tf.cond(tf.cast(left_sibling, tf.bool),
                                    lambda: handle_not_first(dependency_states[i][layer]),
                                    lambda: handle_first(dependency_states[i][layer]))

                if layer == num_layers - 1:
                    with tf.variable_scope('Parameters', reuse=True):
                        U_dependency = tf.get_variable('U_' + dependency, [size, size], dtype=tf.float32)
                    predicted_output = dependency_outputs[i][layer].read(dependency_node) \
                                       if dependency != 'children' else children_output.read(ctr)
                    predicted_output.set_shape([1, size])

                    h_pred_ctr += tf.matmul(predicted_output, U_dependency)


            if layer == num_layers - 1:
                # only calculate loss after we handle both directions
                if do_loss:
                    loss, label_probabilities, attr_probabilities, predicted_p_first, predicted_p_last = \
                            self.calculate_loss(loss, h_pred_ctr, label_index, attr_index,
                                                first_sibling, last_sibling)
                else:
                    h_pred = tf.concat(axis=0, values=[h_pred[:ctr,:,:], [h_pred_ctr], h_pred[ctr+1:,:,:]])


            #ctr = tf.Print(ctr, [ctr])
            ctr = tf.add(ctr, 1) if direction == 'forward' else tf.subtract(ctr, 1)
            ctr = tf.Print(ctr, [ctr])


            return loss, ctr, h_pred, dependency_states, dependency_outputs, \
                   children_tmp_states, children_output, children_predictor_states, \
                   label_probabilities, attr_probabilities, predicted_p_first, predicted_p_last
        return loop_body

    directions = set()
    for k in self.dependencies:
        directions.add(possible_dependencies[k])

    loss = 0.0
    # tensor array vs regular array??
    h_pred = tf.zeros([total_data, 1, size])

    #h_pred = tf.TensorArray(
    #    tf.float32,
    #    size=0,
    #    dynamic_size=True,
    #    clear_after_read=False,
    #    infer_shape=False)
    for layer in range(num_layers):
        loops = 0
        for direction in directions:
            loops += 1

            # forward starts iterating from 1, since 0 is the "empty" parent/sibling
            ctr = 1 if direction == 'forward' else total_data - 1
            # The last 3 arguments we need to "return" from the while loop, so that inference can use them directly
            # XXX the last three args are just there because we need a tensor of the correct size. better way?
            loss, ctr, h_pred, dependency_states, dependency_outputs, \
                  children_tmp_states, children_output, children_predictor_states, \
                  label_probabilities, attr_probabilities, predicted_p_first, predicted_p_last = \
                tf.while_loop(loop_cond_wrapper(direction), loop_body_wrapper(direction, loops == len(directions) and layer == num_layers - 1, layer),
                            [loss,
                            ctr,
                            h_pred,
                            dependency_states,
                            dependency_outputs,
                            children_tmp_states,
                            children_output,
                            children_predictor_states,
                            tf.zeros([1,label_size], tf.float32), # label_probabilities
                            tf.zeros([1,attr_size], tf.float32), # attr_probabilities
                            0.0, 0.0], # predicted_p_{first/last}
                            parallel_iterations=1)

    # tensors we might want to have access to during inference
    self.fetches = {
        'predicted_p_first': predicted_p_first.name,
        'predicted_p_last': predicted_p_last.name,
        'label_probabilities': label_probabilities.name,
        'attr_probabilities': attr_probabilities.name,
        'initial_outputs': {},
        'states': {},
        'outputs': {}
    }
    for i in range(len(self.dependencies)):
        dep = self.dependencies[i]
        self.fetches['states'][dep] = []
        position = 1 if dep != 'children' else 0
        for j in range(len(dependency_states[i])):
            # for inference, we only care about the "root" node's state
            # children will end up writing the result to the parent
            self.fetches['states'][dep].append({
                'c': dependency_states[i][j][0].read(position).name,
                'h': dependency_states[i][j][1].read(position).name,
            })
        self.fetches['outputs'][dep] = dependency_outputs[i][num_layers-1].read(1).name
        self.fetches['initial_outputs'][dep] = dependency_outputs[i][num_layers-1].read(0).name
    if 'children' in self.dependencies:
        self.fetches['children_output'] = children_output.read(0).name

    # Record the names of the LSTM states, so later when we want to do inference we can use them in
    # the feed_dict
    self.feed = {
        'initial_states': {},
        'initial_outputs': {}
    }
    for i in range(len(self.dependencies)):
        dependency = self.dependencies[i]
        self.feed['initial_states'][dependency] = []
        # save this so we can manipulate the initial state when trying to perform inference
        for j, (c, h) in enumerate(self.dependency_initial_states[dependency]):
            self.feed['initial_states'][dependency].append({
                'c': self.dependency_initial_states[dependency][j].c.name,
                'h': self.dependency_initial_states[dependency][j].h.name
            })
        self.feed['initial_outputs'][dependency] = dependency_outputs[i][num_layers-1].read(0).name


    self._cost = cost = tf.reduce_sum(loss)

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config['max_grad_norm'])
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.train.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)


  def calculate_loss(self, loss, h_pred, label_index, attr_index, first_sibling, last_sibling):
      size = self.size
      label_size = self.label_size
      attr_size = self.attr_size

      # grab all of the projection paramaters, now that we have the current node's LSTM state
      with tf.variable_scope('Parameters', reuse=True):
          u_last = tf.get_variable('u_last', [size], dtype=tf.float32)
          u_first = tf.get_variable('u_first', [size], dtype=tf.float32)

          attr_w = tf.get_variable("attr_w", [size, attr_size], dtype=data_type())
          attr_b = tf.get_variable("attr_b", [attr_size], dtype=data_type())
          v_attr = tf.get_variable("v_attr", [label_size, attr_size], dtype=data_type())

          softmax_w = tf.get_variable("softmax_w", [size, label_size], dtype=data_type())
          softmax_b = tf.get_variable("softmax_b", [label_size], dtype=data_type())

          v_first = tf.get_variable("v_first", [label_size], dtype=data_type())
          v_last = tf.get_variable("v_last", [label_size], dtype=data_type())


      # predict where there is a sibling node (f = fraternal)
      logits_p_last = tf.reduce_sum(tf.multiply(u_last, h_pred))
      predicted_p_last = tf.sigmoid(logits_p_last)
      logits_p_last = tf.expand_dims(logits_p_last, 0)
      actual_p_last = tf.expand_dims(last_sibling, 0)
      # TODO: paper uses sigmoid. How does this compare to cross entropy?
      loss_p_last = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_p_last, labels=actual_p_last, name="p_last_loss")

      logits_p_first = tf.reduce_sum(tf.multiply(u_first, h_pred))
      predicted_p_first = tf.sigmoid(logits_p_first)
      logits_p_first = tf.expand_dims(logits_p_first, 0)
      actual_p_first = tf.expand_dims(first_sibling, 0)
      # TODO: paper uses sigmoid. How does this compare to cross entropy?
      loss_p_first = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_p_first, labels=actual_p_first, name="p_first_loss")

      # XXX Testing shouldn't necessarily use is_leaf and last_sibling directly, according to paper?
      # TODO: The paper doesn't seem to have a bias term. Could compare with and without
      #label_logits = tf.matmul(h_pred, softmax_w) + softmax_b + tf.multiply(v_a, is_leaf) + tf.multiply(v_f, last_sibling)
      label_logits = tf.matmul(h_pred, softmax_w) + softmax_b #+ tf.Print(tf.multiply(v_first, first_sibling), [tf.multiply(v_first, first_sibling)], "blah")

      # XXX name things...
      label_probabilities = tf.nn.softmax(label_logits)
      actual_label = tf.one_hot(label_index, label_size)
      # TODO: switch this to use sparse_softmax?
      label_loss = tf.nn.softmax_cross_entropy_with_logits(logits=label_logits, labels=actual_label, name="label_loss")

      attr_logits = tf.matmul(h_pred, attr_w) + attr_b + tf.matmul(tf.expand_dims(actual_label, 0), v_attr)
      attr_probabilities = tf.nn.softmax(attr_logits)
      actual_attr = tf.one_hot(attr_index, attr_size)
      attr_loss = tf.nn.softmax_cross_entropy_with_logits(logits=attr_logits, labels=actual_attr, name="attr_loss")


      # TODO: could differ the weights for structural predictions vs the label predictions when calculating loss
      # XXX XXX XXX somehow only include the correct directions (like, forward direction shouldn't affect loss_p_first?)
      loss = tf.add(loss, tf.reduce_sum(loss_p_last))
      loss = tf.add(loss, tf.reduce_sum(loss_p_first))
      loss = tf.add(loss, tf.reduce_sum(attr_loss))
      loss = tf.add(loss, tf.reduce_sum(label_loss))

      return loss, label_probabilities, attr_probabilities, predicted_p_first, predicted_p_last

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def probabilities(self):
    return self._probabilities

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return tf.Print(self._cost, [self._cost])

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


def run_epoch(session, model, data, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0

    fetches = {
        "cost": model.cost
    }

    if eval_op is not None:
        fetches["eval_op"] = eval_op

    epoch_size = len(data['is_leaf'])

    for step in range(epoch_size):
        feed_dict = {}

        # TODO: some of these placeholders might be unused, if the data isn't used. can filter out
        for k in data:
            feed_dict[model.placeholders[k]] = data[k][step]

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        print(cost)

        costs += cost
        iters += 1

        #if verbose: #and step % (epoch_size // 10) == 10:
        #  print("%.3f perplexity: %.3f speed: %.0f wps" %
        #        (step * 1.0 / epoch_size, np.exp(costs / iters),
        #         #iters * model.input.batch_size / (time.time() - start_time)))
        #         iters / (time.time() - start_time)))

    #return np.exp(costs / iters)
    return costs


def get_config():
    if FLAGS.model == "small":
        return SmallConfig
    elif FLAGS.model == "medium":
        return MediumConfig
    elif FLAGS.model == "large":
        return LargeConfig
    elif FLAGS.model == "test":
        return TestConfig
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)


def main(_):
    # load in all the data
    raw_data = dict()
    with open(os.path.join(FLAGS.data_path, 'tree_train.json')) as f:
        raw_data['train'] = json.load(f)
    with open(os.path.join(FLAGS.data_path,'tree_valid.json')) as f:
        raw_data['valid'] = json.load(f)
    with open(os.path.join(FLAGS.data_path,'tree_test.json')) as f:
        raw_data['test'] = json.load(f)
    with open(os.path.join(FLAGS.data_path, 'tree_tokens.json')) as f:
        token_ids = json.load(f)
        raw_data['token_to_id'] = token_ids['ast_labels']
        raw_data['attr_to_id'] = token_ids['label_attrs']

    config = get_config()
    config['label_vocab_size'] = len(raw_data['token_to_id'])
    config['attr_vocab_size'] = len(raw_data['attr_to_id'])

    config['possible_dependencies'] = possible_dependencies
    config['dependencies'] = FLAGS.dependencies.split()

    config['placeholders'] = {}
    # this needs to be populated so model initialization can quickly create the appropriate placeholders
    for k in raw_data['train']:
        config['placeholders'][k] = None

    eval_config = config.copy()
    #eval_config['batch_size'] = 1
    #eval_config['num_steps'] = 1

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config['init_scale'],
                                                    config['init_scale'])

        with tf.name_scope("Train"):
            with tf.variable_scope("TRNNModel", reuse=None, initializer=initializer):
                m = TRNNModel(is_training=True, config=config)#, input_=raw_data['train'])
            tf.summary.scalar("Training_Loss", m.cost)
            tf.summary.scalar("Learning_Rate", m.lr)

        # TODO: Can remove Valid and Training nodes from the saved model?
        with tf.name_scope("Valid"):
            with tf.variable_scope("TRNNModel", reuse=True, initializer=initializer):
                mvalid = TRNNModel(is_training=False, config=config)
            tf.summary.scalar("Validation_Loss", mvalid.cost)

        with tf.name_scope("Test"):
            with tf.variable_scope("TRNNModel", reuse=True, initializer=initializer):
                mtest = TRNNModel(is_training=False, config=eval_config)

        # save stuff to be used later in inference
        for k in mtest.placeholders:
            config['placeholders'][k] = mtest.placeholders[k].name
        config['fetches'] = mtest.fetches
        config['feed'] = mtest.feed

        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            for i in range(config['max_max_epoch']):
                lr_decay = config['lr_decay'] ** max(i + 1 - config['max_epoch'], 0.0)
                m.assign_lr(session, config['learning_rate'] * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, raw_data['train'], eval_op=m.train_op,
                                            verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

                valid_perplexity = run_epoch(session, mvalid, raw_data['valid'])
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            test_perplexity = run_epoch(session, mtest, raw_data['test'])
            print("Test Perplexity: %.3f" % test_perplexity)

            config['train_perplexity'] = train_perplexity
            config['valid_perplexity'] = valid_perplexity
            config['test_perplexity'] = test_perplexity

            if FLAGS.save_path:
                if not os.path.isdir(FLAGS.save_path):
                    os.makedirs(FLAGS.save_path)
                saver.save(session, os.path.join(FLAGS.save_path, 'tree_model'))
                with open(os.path.join(FLAGS.save_path, 'tree_training_config.json'), 'w') as f:
                    json.dump(config, f)

if __name__ == "__main__":
    tf.app.run()
