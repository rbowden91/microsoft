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
from tensorflow.python import debug as tf_debug


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
  "dependencies" : ['children']
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

def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32

class TRNNModel(object):

  def __init__(self, is_training, config):
    size = config['hidden_size']
    vocab_size = config['vocab_size']
    attr_size = config['attr_vocab_size']
    self.dependencies = config['dependencies']

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
    for k in config['dependencies']:
        self.placeholders['inference'][k] = tf.placeholder(tf.int32, [], name='inference_' + k + '_placeholder')

    self.dependency_initial_states = dict()
    self.dependency_cells = dict()

    # Record the names of the LSTM states, so later when we want to do inference we can use them in
    # the feed_dict
    self.feed = {
        'initial_states': {},
        'states': {}
    }
    for i in range(len(self.dependencies)):
        dependency = self.dependencies[i]
        self.feed['initial_states'][dependency] = []
        self.feed['states'][dependency] = []

        with tf.variable_scope("RNN", reuse=None):
            with tf.variable_scope(dependency, reuse=None):
                self.dependency_cells[dependency] = tf.contrib.rnn.MultiRNNCell(
                    [attn_cell() for _ in range(config['num_layers'])], state_is_tuple=True)
                # XXX 1 is the batch_size
                self.dependency_initial_states[dependency] = self.dependency_cells[dependency].zero_state(1, data_type())


        # Need to manually handle LSTM states. This is gross.
        dependency_states = (initialize_lstm_array(self.dependency_initial_states[dependency]))

        # extra stuff needed by the children dependency
        # 0 is the <nil> token
        initial_embedding = tf.expand_dims(tf.gather(embedding, 0,
                                                        name=("InitialEmbedGather")), 0)
        initial_output, leaf_state = self.dependency_cells['children'](initial_embedding,
                                                                       self.dependency_initial_states['children'])
    loop_cond = lambda ctr, dependency_states: \
        tf.greater(ctr, 2)

    def loop_body(ctr, dependency_states):
        left_sibling = tf.gather(self.placeholders['data']['left_sibling'], ctr)
        is_leaf = tf.cast(tf.gather(self.placeholders['data']['is_leaf'], ctr), tf.bool)

        # save to sibling
        def handle_not_first(state):
            state[0] = state[0].write(ctr, tf.zeros([1,4]))
            return state[0]

        dependency_states[0][0] = tf.cond(is_leaf,
                lambda: handle_not_first(dependency_states[0]),
                lambda: handle_not_first(dependency_states[0]), strict=True)

        ctr = tf.subtract(ctr, 1)

        return ctr, dependency_states

    # start iterating from 1, since 0 is the "empty" parent/sibling
    # The last four arguments we need to "return" from the while loop, so that inference can use them directly
    # XXX the last three args are just there because we need a tensor of the correct size. better way?
    # do children_tmp_* arrays not have to be passed in here?
    _, dependency_states = tf.while_loop(loop_cond, loop_body,
        [
         #1, # ctr
         tf.squeeze(tf.shape(self.placeholders['data']['is_leaf'])) - 1,
         dependency_states], # dependency_states
        parallel_iterations=1)
    self.dependency = tf.Print(dependency_states[0][0].read(4), [dependency_states[0][0].read(4)], "dep")

def run_epoch(session, model, data, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0

    fetches = {
        "dependency": model.dependency
    }

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

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config['init_scale'],
                                                    config['init_scale'])

        with tf.name_scope("Train"):
            with tf.variable_scope("TRNNModel", reuse=None, initializer=initializer):
                m = TRNNModel(is_training=True, config=config)#, input_=raw_data['train'])

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            for i in range(config['max_max_epoch']):
                train_perplexity = run_epoch(session, m, raw_data['train'],
                                            verbose=True)

if __name__ == "__main__":
    tf.app.run()
