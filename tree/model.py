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
  "max_max_epoch" : 10,
  "keep_prob" : 1.0,
  "lr_decay" : 0.5,
  "batch_size" : 40, # currently, this is just 1
  "dependencies" : ['children']
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
  "dependencies" : ['left_sibling', 'parent']
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
  "dependencies" : ['left_sibling', 'parent']
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
  "dependencies" : ['right_sibling']
}


# initialize an array of TensorArrays to store an LSTM, based on the initial_state template
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
    copy = []
    for i, (c, h) in enumerate(state):
        copy.append([])
        # c is in position 0, h is in position 1
        copy[i].append(state_array[i][0].write(position, state[i].c))
        copy[i].append(state_array[i][1].write(position, state[i].h))
    return copy

# reconstruct the LSTM state from a TensorArray
# initial_state is needed to use as a template for restoration
def restore_lstm_state(state_array, initial_state, position, division_scalar=None, add_state=None):
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


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32

class TRNNModel(object):

  def __init__(self, is_training, config):
    size = config['hidden_size']
    vocab_size = config['vocab_size']
    attr_size = config['attr_vocab_size']
    self.dependencies = config['dependencies']

    # declare a bunch of parameters that will be reused later
    with tf.variable_scope('Parameters', reuse=False):
        # the second dimension doesn't have to be "size", but does have to match softmax_w's first dimension
        for dependency in self.dependencies:
            tf.get_variable('U_' + dependency, [size, size], dtype=tf.float32)
        u_first = tf.get_variable('u_first', [size], dtype=tf.float32)
        u_last = tf.get_variable('u_last', [size], dtype=tf.float32)
        u_a = tf.get_variable('u_a', [size], dtype=tf.float32)
        attr_w = tf.get_variable("attr_w", [size, attr_size], dtype=data_type())
        attr_b = tf.get_variable("attr_b", [attr_size], dtype=data_type())
        v_attr = tf.get_variable("v_attr", [vocab_size, attr_size], dtype=data_type())

        softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
        #v_a = tf.get_variable("v_a", [vocab_size], dtype=data_type())
        v_first = tf.get_variable("v_first", [vocab_size], dtype=data_type())
        v_last = tf.get_variable("v_last", [vocab_size], dtype=data_type())

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())

    def lstm_cell():
        return tf.contrib.rnn.BasicLSTMCell(
            size, forget_bias=0.0, state_is_tuple=True,
            reuse=tf.get_variable_scope().reuse)
    attn_cell = lstm_cell
    if is_training and config['keep_prob'] < 1:
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(), output_keep_prob=config['keep_prob'])

    self.placeholders = { 'data': {}, 'inference': {} }
    for k in config['placeholders']['data']:
        self.placeholders['data'][k] = tf.placeholder(tf.int32, [None], name=k+'_placeholder')

    # XXX XXX XXX better way of doing this? Basically, when doing inference, we want to be able to have different nodes
    # for each dependency, but we only use the Initial States as a place to write, which all are
    # associated with node 0

    self.placeholders['is_inference'] = tf.placeholder(tf.bool, [], name='is_inference_placeholder')
    for k in possible_dependencies:
        self.placeholders['inference'][k] = tf.placeholder(tf.int32, [], name='inference_' + k + '_placeholder')

    self.dependency_initial_states = dict()
    self.dependency_cells = dict()

    # dependency_states can't be a dictionary on sibling/parent, since it needs to be convertible to a tensor (required
    # for the tf.while_loop). The first index in dependency_states is thus aligned with the dependency in the
    # dependencies array
    dependency_states = []

    # Record the names of the LSTM states, so later when we want to do inference we can use them in
    # the feed_dict
    self.feed = {
        'initial_states': {},
        'states': {}
    }
    for i in range(len(self.dependencies)):
        dependency = self.dependencies[i]
        self.feed['initial_states'][dependency] = []
        #self.feed['states'][dependency] = []

        with tf.variable_scope("RNN", reuse=None):
            with tf.variable_scope(dependency, reuse=None):
                self.dependency_cells[dependency] = tf.contrib.rnn.MultiRNNCell(
                    [attn_cell() for _ in range(config['num_layers'])], state_is_tuple=True)
                # XXX 1 is the batch_size
                self.dependency_initial_states[dependency] = self.dependency_cells[dependency].zero_state(1, data_type())


        # Need to manually handle LSTM states. This is gross.
        dependency_states.append(initialize_lstm_array(self.dependency_initial_states[dependency]))

        # since we only use the TensorArray below, write the initial state in position 0. children needs to write
        # final output there, though
        if self.dependencies[i] != 'children':
            dependency_states[i] = save_lstm_state(dependency_states[i], self.dependency_initial_states[dependency], 0)


        # save this so we can manipulate the initial state when trying to perform inference
        for j, (c, h) in enumerate(self.dependency_initial_states[dependency]):
            self.feed['initial_states'][dependency].append({
                'c': self.dependency_initial_states[dependency][j].c.name,
                'h': self.dependency_initial_states[dependency][j].h.name
            })

    if 'children' in self.dependencies:
        # extra stuff needed by the children dependency
        # 0 is the <nil> token
        initial_embedding = tf.expand_dims(tf.gather(embedding, 0,
                                                        name=("InitialEmbedGather")), 0)
        # XXX need to make this into a variable in some way?
        initial_output, leaf_state = self.dependency_cells['children'](initial_embedding,
                                                                       self.dependency_initial_states['children'])
        self.feed['initial_output'] = initial_output.name
        # extra TensorArrays for children, hurray!!
        children_tmp_states = initialize_lstm_array(self.dependency_initial_states['children'])
        children_output = tf.TensorArray(
            tf.float32,
            size=0,
            dynamic_size=True,
            clear_after_read=False,
            infer_shape=False)
        children_tmp_output = tf.TensorArray(
            tf.float32,
            size=0,
            dynamic_size=True,
            clear_after_read=False,
            infer_shape=False)

    # TODO: dropout??
    #if is_training and config['keep_prob'] < 1:
    #  inputs = tf.nn.dropout(inputs, config['keep_prob'])

    # this returns true as long as the loop counter is less than the length of the example
        #tf.less(ctr, tf.squeeze(tf.shape(self.placeholders['data']['is_leaf'])))
    loop_cond = lambda loss, ctr, dependency_states, children_tmp_states, children_output, children_tmp_output, label_prob, attr_prob, lpf: \
        tf.greater(ctr, 0)

    def loop_body(loss, ctr, dependency_states, children_tmp_states, children_output, children_tmp_output, label_prob, attr_prob, lpf):
        # TODO: can get rid of last_sibling and first_sibling by just using left/right-sibling != 0
        # XXX correction, no we can't! for inference, those are set to zero, but we still want, e.g., is_leaf
        # to be accurate
        is_leaf = tf.cast(tf.gather(self.placeholders['data']['is_leaf'], ctr, name="IsLeafGather"), tf.float32)
        first_sibling = tf.cast(tf.gather(self.placeholders['data']['first_sibling'], ctr), tf.float32)
        last_sibling = tf.cast(tf.gather(self.placeholders['data']['last_sibling'], ctr, name="LastSiblingGather"), tf.float32)
        node_index = tf.gather(self.placeholders['data']['label_index'], ctr, name="NodeIndexGather")
        attr_index = tf.gather(self.placeholders['data']['attr_index'], ctr, name="AttrIndexGather")
        parent = tf.gather(self.placeholders['data']['parent'], ctr)
        left_sibling = tf.gather(self.placeholders['data']['left_sibling'], ctr)
        right_sibling = tf.gather(self.placeholders['data']['right_sibling'], ctr)
        num_children = tf.cast(tf.gather(self.placeholders['data']['num_children'], ctr,
                                         name="AttrIndexGather"), tf.float32)

        outputs = {}

        # Generate both the sibling and parent RNN states for the current node, based on the previous sibling and parent
        for i in range(len(self.dependencies)):

            dependency = self.dependencies[i]
            # During inference, we want to use the directly-supplied label for the parent, since each node is passed in
            # one-by-one and we won't have access to the parent when the child is passed in. During training, we have
            # all nodes in the example at once, so can directly grab the parent's label
            dependency_node = tf.gather(self.placeholders['data'][dependency], ctr, name=(dependency+"Gather")) if dependency != 'children' else ctr

            handle_inference = lambda: [self.placeholders['inference'][dependency]]
            handle_training = lambda: tf.gather(self.placeholders['data']['label_index'], dependency_node,
                                                name=(dependency+"TokenGather"))

            dependency_token = tf.cond(self.placeholders['is_inference'], handle_inference, handle_training)
            dependency_embedding = tf.expand_dims(tf.gather(embedding, dependency_token,
                                                            name=(dependency+"EmbedGather")), 0)
            with tf.variable_scope("RNN", reuse=None):
                with tf.variable_scope(dependency, reuse=None):
                    if dependency != 'children':
                        # reconstruct the LSTM state of the parent/sibling from the TensorArray
                        state = restore_lstm_state(dependency_states[i], self.dependency_initial_states[dependency], dependency_node)

                        # TODO: this technically gets recalculated over and over from the parent for all children, so could
                        # optimize by just doing it once
                        (output, new_state) = self.dependency_cells[dependency](dependency_embedding, state)
                        outputs[dependency] = output
                        dependency_states[i] = save_lstm_state(dependency_states[i], new_state, ctr)
                    else:
                        # update num children so we don't get a division by 0
                        bootstrap_leaf = lambda state: (initial_output,
                                                        save_lstm_state(state, leaf_state, ctr),
                                                        num_children+1)
                        bootstrap_else = lambda state: (children_output.read(ctr),
                                                        state,
                                                        num_children)
                        output, dependency_states[i], num_children = tf.cond(tf.cast(num_children, tf.bool),
                                                       lambda: bootstrap_else(dependency_states[i]),
                                                       lambda: bootstrap_leaf(dependency_states[i]), strict=True)
                        output.set_shape(initial_output.shape)
                        outputs['children'] = output / num_children
                        state = restore_lstm_state(dependency_states[i], leaf_state, ctr, division_scalar=num_children)
                        (future_output, future_state) = self.dependency_cells['children'](dependency_embedding, state)
                        # this is basically doing a +=. If we are the last sibling (on the right), we need to start
                        # from 0. If we are the first sibling (on the left), we need to write the sum to the parent

                        # update state
                        handle_last = lambda: (future_output, future_state)
                        handle_not_last = lambda: (children_tmp_output.read(ctr) + future_output,
                                                   restore_lstm_state(children_tmp_states, leaf_state, ctr,
                                                                      add_state=future_state))

                        new_output, new_state = tf.cond(tf.cast(right_sibling, tf.bool), handle_not_last, handle_last, strict=True)
                        new_output.set_shape(initial_output.shape)

                        ## save to parent
                        handle_first = lambda state: (children_tmp_states,
                                                      save_lstm_state(state, new_state, parent),
                                                      children_tmp_output,
                                                      children_output.write(parent, new_output))
                        # save to sibling
                        handle_not_first = lambda state: (save_lstm_state(children_tmp_states, new_state, left_sibling),
                                                          state,
                                                          children_tmp_output.write(left_sibling, new_output),
                                                          children_output)
                        children_tmp_states, dependency_states[i], children_tmp_output, children_output = \
                                tf.cond(tf.cast(left_sibling, tf.bool),
                                        lambda: handle_not_first(dependency_states[i]),
                                        lambda: handle_first(dependency_states[i]))




        # grab all of the projection paramaters, now that we have the current node's LSTM state
        U = {}
        with tf.variable_scope('Parameters', reuse=True):
            for dependency in self.dependencies:
                U[dependency] = tf.get_variable('U_' + dependency, [size, size], dtype=tf.float32)
            u_last = tf.get_variable('u_last', [size], dtype=tf.float32)
            u_first = tf.get_variable('u_first', [size], dtype=tf.float32)
            u_a = tf.get_variable('u_a', [size], dtype=tf.float32)
            attr_w = tf.get_variable("attr_w", [size, attr_size], dtype=data_type())
            attr_b = tf.get_variable("attr_b", [attr_size], dtype=data_type())
            v_attr = tf.get_variable("v_attr", [vocab_size, attr_size], dtype=data_type())

            softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=data_type())
            softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
            #v_a = tf.get_variable("v_a", [vocab_size], dtype=data_type())
            v_first = tf.get_variable("v_first", [vocab_size], dtype=data_type())
            v_last = tf.get_variable("v_last", [vocab_size], dtype=data_type())

        # this is the vector that combines the sibling and parent hidden states to be directly used in prediction
        h_pred = tf.zeros([1, size])
        for dependency in self.dependencies:
            h_pred += tf.matmul(outputs[dependency], U[dependency])

        # predict whether there is a child node (a = ancestor)
        logits_p_a = tf.reduce_sum(tf.multiply(u_a, h_pred))
        predicted_p_a = tf.sigmoid(logits_p_a)
        logits_p_a = tf.expand_dims(logits_p_a, 0)
        actual_p_a = tf.expand_dims(is_leaf, 0)
        # XXX paper uses sigmoid. How does this compare to cross entropy?
        loss_p_a = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_p_a, labels=actual_p_a, name="p_a_loss")

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
        actual_label = tf.one_hot(node_index, vocab_size)
        # TODO: switch this to use sparse_softmax?
        label_loss = tf.nn.softmax_cross_entropy_with_logits(logits=label_logits, labels=actual_label, name="label_loss")

        attr_logits = tf.matmul(h_pred, attr_w) + attr_b + tf.matmul(tf.expand_dims(actual_label, 0), v_attr)
        attr_probabilities = tf.nn.softmax(attr_logits)
        actual_attr = tf.one_hot(attr_index, attr_size)
        attr_loss = tf.nn.softmax_cross_entropy_with_logits(logits=attr_logits, labels=actual_attr, name="attr_loss")


        # TODO: could differ the weights for structural predictions vs the label predictions when calculating loss
        #loss = tf.add(loss, tf.reduce_sum(loss_p_a))
        #loss = tf.add(loss, tf.reduce_sum(loss_p_last))
        loss = tf.add(loss, tf.reduce_sum(loss_p_first))
        loss = tf.add(loss, tf.reduce_sum(attr_loss))
        loss = tf.add(loss, tf.reduce_sum(label_loss))

        #ctr = tf.add(ctr, 1)
        ctr = tf.subtract(ctr, 1)

        return loss, ctr, dependency_states, children_tmp_states, children_output, children_tmp_output, label_probabilities, attr_probabilities, predicted_p_first

    # start iterating from 1, since 0 is the "empty" parent/sibling
    # The last four arguments we need to "return" from the while loop, so that inference can use them directly
    # XXX the last three args are just there because we need a tensor of the correct size. better way?
    # do children_tmp_* arrays not have to be passed in here?
    loss, _, dependency_states, children_tmp_states, children_output, children_tmp_output, label_probabilities, attr_probabilities, predicted_p_first = tf.while_loop(loop_cond, loop_body,
        [0.0, # loss
         #1, # ctr
         tf.squeeze(tf.shape(self.placeholders['data']['is_leaf'])) - 1,
         dependency_states, # dependency_states
         children_tmp_states if 'children' in self.dependencies else 0,
         children_output if 'children' in self.dependencies else 0,
         children_tmp_output if 'children' in self.dependencies else 0,
         tf.zeros([1,vocab_size], tf.float32), # label_probabilities
         tf.zeros([1,attr_size], tf.float32), # attr_probabilities
         0.0], # predicted_p_first
         parallel_iterations=1)

    # tensors we might want to have access to during inference
    self.fetches = {
        #'predicted_p_a': predicted_p_a.name,
        'predicted_p_first': predicted_p_first.name,
        'label_probabilities': label_probabilities.name,
        'attr_probabilities': attr_probabilities.name,
        'states': {}
    }
    for i in range(len(self.dependencies)):
        self.fetches['states'][self.dependencies[i]] = []
        for j in range(len(dependency_states[i])):
            # for inference, we only care about the "root" node's state
            # children will end up writing the result to the parent
            position = 1 if self.dependencies[i] == 'children' else 0
            self.fetches['states'][self.dependencies[i]].append({
                'c': dependency_states[i][j][0].read(position).name,
                'h': dependency_states[i][j][1].read(position).name,
            })
    if 'children' in self.dependencies:
        self.fetches['children_output'] = children_output.read(0).name

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
        global_step=tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

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
    return self._cost

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
        feed_dict = { model.placeholders['is_inference']: False }

        # TODO: some of these placeholders might be unused, if the data isn't used. can filter out
        for k in data:
            feed_dict[model.placeholders['data'][k]] = data[k][step]

        # these aren't used if is_inference is False, but it seems we still need
        # to feed them in evidently :-\
        for k in model.dependencies:
            feed_dict[model.placeholders['inference'][k]] = 0

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
    config['vocab_size'] = len(raw_data['token_to_id'])
    config['attr_vocab_size'] = len(raw_data['attr_to_id'])

    config['possible_dependencies'] = possible_dependencies

    config['placeholders'] = {
        'data': {},
        'inference': {}
    }
    # this needs to be populated so model initialization can quickly create the appropriate placeholders
    for k in raw_data['train']:
        config['placeholders']['data'][k] = None

    eval_config = config.copy()
    #eval_config['batch_size'] = 1
    #eval_config['num_steps'] = 1

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config['init_scale'],
                                                    config['init_scale'])

        with tf.name_scope("Train"):
            with tf.variable_scope("TRNNModel", reuse=None, initializer=initializer):
                m = TRNNModel(is_training=True, config=config)#, input_=raw_data['train'])
            tf.summary.scalar("Training Loss", m.cost)
            tf.summary.scalar("Learning Rate", m.lr)

        # TODO: Can remove Valid and Training nodes from the saved model?
        with tf.name_scope("Valid"):
            with tf.variable_scope("TRNNModel", reuse=True, initializer=initializer):
                mvalid = TRNNModel(is_training=False, config=config)
            tf.summary.scalar("Validation Loss", mvalid.cost)

        with tf.name_scope("Test"):
            with tf.variable_scope("TRNNModel", reuse=True, initializer=initializer):
                mtest = TRNNModel(is_training=False, config=eval_config)

        # save stuff to be used later in inference
        for k in mtest.placeholders['data']:
            config['placeholders']['data'][k] = mtest.placeholders['data'][k].name
        for k in mtest.placeholders['inference']:
            config['placeholders']['inference'][k] = mtest.placeholders['inference'][k].name
        config['placeholders']['is_inference'] = mtest.placeholders['is_inference'].name
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

            if FLAGS.save_path:
                if not os.path.isdir(FLAGS.save_path):
                    os.makedirs(FLAGS.save_path)
                saver.save(session, os.path.join(FLAGS.save_path, 'tree_model'))
                with open(os.path.join(FLAGS.save_path, 'tree_training_config.json'), 'w') as f:
                    json.dump(config, f)

if __name__ == "__main__":
    tf.app.run()
