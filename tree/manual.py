# Copyright 2015 The TensorFlow Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import time
import json

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


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32

#class PTBInput(object):
#  """The input data."""
#
#  def __init__(self, config, data=None, test=False, name=None):
#    self.data = data
#    self.batch_size = batch_size = config['batch_size']
#    self.num_steps = num_steps = config['num_steps']
#    #self.input_data =
#    #self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
#    #if test:
#    #    self.input_data, self.targets = tf.placeholder(tf.int32, [1, 1], name="input"), tf.placeholder(tf.int32, [1, 1], name="target")
#    #else:
#    #    self.input_data, self.targets = ptb_producer(
#    #        data, batch_size, num_steps, name=name)


# XXX XXX removing Validation and Testing stuff from Graph?
class TRNNModel(object):

  def __init__(self, is_training, config):
    #self._input = input_

    #batch_size = input_.batch_size
    #num_steps = input_.num_steps
    size = config['hidden_size']
    vocab_size = config['vocab_size']

    with tf.variable_scope('Parameters', reuse=False):
        U_f = tf.get_variable('U_f', [size, size], dtype=tf.float32)
        U_a = tf.get_variable('U_a', [size, size], dtype=tf.float32)
        u_f = tf.get_variable('u_f', [size], dtype=tf.float32)
        u_a = tf.get_variable('u_a', [size], dtype=tf.float32)
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
        v_a = tf.get_variable("v_a", 1, dtype=data_type())
        v_f = tf.get_variable("v_f", 1, dtype=data_type())

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def lstm_cell():
        return tf.contrib.rnn.BasicLSTMCell(
            size, forget_bias=0.0, state_is_tuple=True,
            reuse=tf.get_variable_scope().reuse)
    attn_cell = lstm_cell
    if is_training and config['keep_prob'] < 1:
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(), output_keep_prob=config['keep_prob'])

    self.is_leaf_placeholder = tf.placeholder(
        tf.int32, [None], name='is_leaf_placeholder')
    self.last_sibling_placeholder = tf.placeholder(
        tf.int32, [None], name='last_sibling_placeholder')

    self.node_index_placeholder = tf.placeholder(
        tf.int32, [None], name='node_index_placeholder')

    config['dependencies'] = dependencies = ['sibling', 'parent']

    # XXX XXX XXX better way of doing this? Basically, when doing inference, we want to be able to have different nodes
    # for each dependency, but we only use the Initial States as a place to write, which theoretically all are
    # associated with node 0

    self.is_inference = tf.placeholder(tf.bool, [], name='is_inference_placeholder')
    self.inference_placeholders = {}
    for k in config['dependencies']:
        self.inference_placeholders[k] = tf.placeholder(tf.int32, [], name='inference_' + k + '_placeholder')

    # XXX XXX XXX change this to is_testing?
    if is_training:
        config['placeholders'] = {
            'is_leaf': self.is_leaf_placeholder.name,
            'last_sibling': self.last_sibling_placeholder.name,
            'node_index': self.node_index_placeholder.name,
            'is_inference': self.is_inference.name
        }
        for k in config['dependencies']:
            config['placeholders']['inference_' + k] = self.inference_placeholders[k].name


    self.dependency_placeholders = dict()
    self.dependency_initial_states = dict()
    self.dependency_cells = dict()
    # XXX dependency_states needs to be COMPLETELY ORDERED
    dependency_states = []
    for i in range(len(dependencies)):
        dependency = dependencies[i]

        with tf.variable_scope("RNN", reuse=None):
            with tf.variable_scope(dependency, reuse=None):
                self.dependency_cells[dependency] = tf.contrib.rnn.MultiRNNCell(
                    [attn_cell() for _ in range(config['num_layers'])], state_is_tuple=True)
                # XXX 1 is the batch_size
                self.dependency_initial_states[dependency] = self.dependency_cells[dependency].zero_state(1, data_type())

        self.dependency_placeholders[dependency] = tf.placeholder(tf.int32, [None], name=(dependency + '_placeholder'))
        # XXX XXX XXX change this to is_testing?
        if is_training:
            config['placeholders'][dependency] = self.dependency_placeholders[dependency].name

        # manually handle LSTM states
        dependency_states.append([])
        for j, (c, h) in enumerate(self.dependency_initial_states[dependency]):
            dependency_states[i].append([])
            dependency_states[i][-1].append(tf.TensorArray(
                tf.float32,
                size=0,
                dynamic_size=True,
                clear_after_read=False,
                infer_shape=False))
            dependency_states[i][-1].append(tf.TensorArray(
                tf.float32,
                size=0,
                dynamic_size=True,
                clear_after_read=False,
                infer_shape=False))

            # the nil parent/sibling should take on the initial state
            # c is in position 0, h is in position 1
            dependency_states[i][-1][0] = dependency_states[i][-1][0].write(0,
                                                        self.dependency_initial_states[dependency][j].c)
            dependency_states[i][-1][1] = dependency_states[i][-1][1].write(0,
                                                        self.dependency_initial_states[dependency][j].h)

            # Record the names in config, so later when we want to do inference we can use them in the
            # feed_dict
            # XXX XXX XXX change this to is_testing?
            if is_training:
                if 'initial_states' not in config:
                    config['initial_states'] = dict()
                    config['states'] = dict()
                if dependency not in config['initial_states']:
                    config['initial_states'][dependency] = []
                    config['states'][dependency] = []
                config['initial_states'][dependency].append({
                    'c': self.dependency_initial_states[dependency][j].c.name,
                    'h': self.dependency_initial_states[dependency][j].h.name
                })

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())
      #inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
      #inputs = tf.nn.embedding_lookup(embedding, self.node_index_placeholder)

    # XXX dropout??
    #if is_training and config['keep_prob'] < 1:
    #  inputs = tf.nn.dropout(inputs, config['keep_prob'])

    #is_leaf_array = tf.TensorArray(
    #    tf.float32,
    #    size=0,
    #    dynamic_size=True,
    #    clear_after_read=False,
    #    infer_shape=False)

    #last_sibling_array = tf.TensorArray(
    #    tf.float32,
    #    size=0,
    #    dynamic_size=True,
    #    clear_after_read=False,
    #    infer_shape=False)


    loop_cond = lambda loss, ctr, dependency_states, label_prob, lpa, lpf: \
        tf.less(ctr, tf.squeeze(tf.shape(self.is_leaf_placeholder)))

    def loop_body(loss, ctr, dependency_states, label_prob, lpa, lpf):
        is_leaf = tf.gather(self.is_leaf_placeholder, ctr, name="IsLeafGather")
        last_sibling = tf.gather(self.last_sibling_placeholder, ctr, name="LastSiblingGather")
        node_index = tf.gather(self.node_index_placeholder, ctr, name="NodeIndexGather")

        outputs = {}
        for i in range(len(dependencies)):
            dependency = dependencies[i]
            dependency_node = tf.gather(self.dependency_placeholders[dependency], ctr, name=(dependency+"Gather"))

            handle_inference = lambda: [self.inference_placeholders[dependency]]
            handle_training = lambda: tf.gather(self.node_index_placeholder, dependency_node, name=(dependency+"TokenGather"))

            dependency_token = tf.cond(self.is_inference, handle_inference, handle_training)
            dependency_embedding = tf.expand_dims(tf.gather(embedding, dependency_token, name=(dependency+"EmbedGather")), 0)

            with tf.variable_scope("RNN", reuse=None):
                with tf.variable_scope(dependency, reuse=None):
                    # reconstruct the LSTM state from the TensorArray
                    state = []
                    for j, (c, h) in enumerate(self.dependency_initial_states[dependency]):
                        c_state = dependency_states[i][j][0].read(dependency_node)
                        h_state = dependency_states[i][j][1].read(dependency_node)
                        # TensorArray returns shape <unknown>, which breaks things when passed to LSTM cell()
                        c_state.set_shape(self.dependency_initial_states[dependency][j].c.shape)
                        h_state.set_shape(self.dependency_initial_states[dependency][j].h.shape)
                        state.append(tf.contrib.rnn.LSTMStateTuple(c_state, h_state))

                    state = tuple(state)

                    # XXX this technically gets recalculated over and over from the parent
                    (output, new_state) = self.dependency_cells[dependency](dependency_embedding, state)
                    outputs[dependency] = output

                    # record new LSTM state in the TensorArray
                    for j, (c, h) in enumerate(new_state):
                        dependency_states[i][j][0] = dependency_states[i][j][0].write(ctr, new_state[j].c)
                        dependency_states[i][j][1] = dependency_states[i][j][1].write(ctr, new_state[j].h)

        with tf.variable_scope('Parameters', reuse=True):

            # the second size doesn't have to be "size"? But has to match softmax_w
            U_f = tf.get_variable('U_f', [size, size], dtype=tf.float32)
            U_a = tf.get_variable('U_a', [size, size], dtype=tf.float32)
            u_f = tf.get_variable('u_f', [size], dtype=tf.float32)
            u_a = tf.get_variable('u_a', [size], dtype=tf.float32)
            softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=data_type())
            softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
            v_a = tf.get_variable("v_a", 1, dtype=data_type())
            v_f = tf.get_variable("v_f", 1, dtype=data_type())

        # XXX generalize this in terms of "dependencies"
        h_pred = tf.matmul(outputs['sibling'], U_f) + tf.matmul(outputs['parent'], U_a)


        # XXX paper uses sigmoid
        #p_a = tf.sigmoid(tf.reduce_sum(tf.multiply(u_a, h_pred)))
        #p_f = tf.sigmoid(tf.reduce_sum(tf.multiply(u_f, h_pred)))

        # XXX XXX XXX are these between zero and 1?
        logits_p_a = tf.expand_dims(tf.reduce_sum(tf.multiply(u_a, h_pred)), 0)
        logits_p_f = tf.expand_dims(tf.reduce_sum(tf.multiply(u_f, h_pred)), 0)


        actual_p_a = tf.expand_dims(is_leaf, 0)#tf.one_hot(is_leaf, 2), 0)
        loss_p_a = tf.nn.softmax_cross_entropy_with_logits(logits=logits_p_a, labels=actual_p_a, name="p_a_loss")

        actual_p_f = tf.expand_dims(last_sibling, 0)#tf.one_hot(last_sibling, 2), 0)
        loss_p_f = tf.nn.softmax_cross_entropy_with_logits(logits=logits_p_f, labels=actual_p_f, name="p_f_loss")


        # XXX paper seemingly doesn't have softmax_b
        # XXX XXX XXX need to add in post-comment
        label_logits = tf.matmul(h_pred, softmax_w) + softmax_b # + v_a . is_leaf + v_b . last_sibling

        # XXX name things...
        label_probabilities = tf.nn.softmax(label_logits)

        # XXX why doesn't this need some kind of tf.expand_dims, like the above?
        actual_label = tf.one_hot(node_index, vocab_size)
        # XXX use sparse
        label_loss = tf.nn.softmax_cross_entropy_with_logits(logits=label_logits, labels=actual_label, name="label_loss")

        # put some kind of weights on these?
        #loss = tf.add(loss, tf.reduce_sum(loss_p_a))
        #loss = tf.add(loss, tf.reduce_sum(loss_p_f))
        loss = tf.add(loss, tf.reduce_sum(label_loss))

        ctr = tf.add(ctr, 1)

        return loss, ctr, dependency_states, label_probabilities, logits_p_a, logits_p_f

    # start iterating from 1, since 0 is the "empty" parent/sibling
    # XXX the tf.zeros are artificially because they have to have the correct size. better way?
    loss, _, dependency_states, label_probabilities, logits_p_a, logits_p_f = tf.while_loop(loop_cond, loop_body, [0.0, 1, dependency_states, tf.zeros([1,vocab_size], tf.float32), tf.ones(1, tf.float32), tf.zeros(1, tf.float32)], parallel_iterations=1)
    if is_training:
        config['fetches'] = {
            'logits_p_a': logits_p_a.name,
            'logits_p_f': logits_p_f.name,
            'label_probabilities': label_probabilities.name,
            'states': {}
        }
        for i in range(len(dependencies)):
            config['fetches']['states'][dependencies[i]] = []
            for j in range(len(dependency_states[i])):
                # for inference, we only care about the "root" node's state
                config['fetches']['states'][dependencies[i]].append({
                    'c': dependency_states[i][j][0].read(1).name,
                    'h': dependency_states[i][j][1].read(1).name,
                })

    #with tf.variable_scope("RNN"):
    #  for time_step in range(num_steps):
    #    if time_step > 0: tf.get_variable_scope().reuse_variables()
    #    (cell_output, state) = cell(inputs[:, time_step, :], state)
    #    outputs.append(cell_output)

    #output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, size])
    #softmax_w = tf.get_variable(
    #    "softmax_w", [size, vocab_size], dtype=data_type())
    #softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    #logits = tf.matmul(output, softmax_w) + softmax_b
    #self._probabilities = tf.nn.softmax(logits, name='Probabilities') #probabilities.append(tf.nn.softmax(logits))
    #loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
    #    [logits],
    #    [tf.reshape(input_.targets, [-1])],
    #    [tf.ones([batch_size * num_steps], dtype=data_type())])
    #self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._cost = cost = tf.reduce_sum(loss)
    #self._final_state = tf.identity(state, name="FinalState")

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

  #@property
  #def input(self):
  #  return self._input

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


SmallConfig = {
  "init_scale" : 0.1,
  "learning_rate" : 1.0,
  "max_grad_norm" : 5,
  "num_layers" : 2,
  "num_steps" : 20,
  "hidden_size" : 200,
  "max_epoch" : 4,
  "max_max_epoch" : 10,
  "keep_prob" : 1.0,
  "lr_decay" : 0.5,
  "batch_size" : 40,
  "vocab_size" : 10000
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
  "vocab_size" : 10000
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
  "vocab_size" : 10000
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
  "vocab_size" : 10000
}

def run_epoch(session, model, data, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  #state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
  #    "final_state": model.final_state,
  }

  if eval_op is not None:
    fetches["eval_op"] = eval_op

  #epoch_size = ((len(data) // model.input.batch_size) - 1) // model.input.num_steps
  epoch_size = len(data['leaf_node'])

  for step in range(epoch_size):
    feed_dict = {
        model.is_leaf_placeholder: data['leaf_node'][step],
        model.last_sibling_placeholder: data['last_sibling'][step],
        model.node_index_placeholder: data['token'][step],
        model.dependency_placeholders['sibling']: data['sibling'][step],
        model.dependency_placeholders['parent']: data['parent'][step],

        # XXX make these variables??
        model.is_inference: False,
        model.inference_placeholders['parent']: 0,
        model.inference_placeholders['sibling']: 0

    }
    #for i, (c, h) in enumerate(model.initial_state):
    #  feed_dict[c] = state[i].c
    #  feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    #final_state = vals["final_state"]
    #state = []

    #for i in range(len(final_state)):
    #    state.append(tf.contrib.rnn.LSTMStateTuple(final_state[i][0], final_state[i][1]))
    #state = tuple(state)

    print(cost)
    costs += cost
    iters += 1
    #iters += model.input.num_steps

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

  raw_data = dict()
  with open(FLAGS.data_path + '/tree_train.json') as f:
    raw_data['train'] = json.load(f)
    #raw_data['train_ids'].reverse()
  with open(FLAGS.data_path + '/tree_valid.json') as f:
    raw_data['valid'] = json.load(f)
    #raw_data['valid_ids'].reverse()
  with open(FLAGS.data_path + '/tree_test.json') as f:
    raw_data['test'] = json.load(f)
    #raw_data['test_ids'].reverse()
  with open(FLAGS.data_path + '/tree_tokens.json') as f:
    raw_data['token_to_id'] = json.load(f)

  config = get_config()
  config['vocab_size'] = len(raw_data['token_to_id'])

  eval_config = config.copy()
  eval_config['batch_size'] = 1
  eval_config['num_steps'] = 1

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config['init_scale'],
                                                config['init_scale'])

    with tf.name_scope("Train"):
      with tf.variable_scope("TRNNModel", reuse=None, initializer=initializer):
        m = TRNNModel(is_training=True, config=config)#, input_=raw_data['train'])
      tf.summary.scalar("Training Loss", m.cost)
      tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
      with tf.variable_scope("TRNNModel", reuse=True, initializer=initializer):
        mvalid = TRNNModel(is_training=False, config=config)
      tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
      with tf.variable_scope("TRNNModel", reuse=True, initializer=initializer):
        mtest = TRNNModel(is_training=False, config=eval_config)

    saver = tf.train.Saver()

    with tf.Session() as session:
      # ugh, Supervisor used to do this
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
        saver.save(session, FLAGS.save_path + '/tree_model')
        with open(FLAGS.save_path + '/tree_training_config.json', 'w') as f:
            json.dump(config, f)

if __name__ == "__main__":
  tf.app.run()
