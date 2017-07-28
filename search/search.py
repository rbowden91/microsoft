# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
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
import read
import os
from check_correct import check_vigenere

import numpy as np
import tensorflow as tf

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

# XXX ROB
flags.DEFINE_string("data_path", None,
                    "XXX")

FLAGS = flags.FLAGS

def generate_program(program):
  #print(program)
  output = ""
  # skip <sof> and <eof>?
  for i in range(0, len(program) - 0):
    output += program[i]['real_token'] + ' '
    if program[i]['real_token'] == '}' or program[i]['real_token'] == '{' or program[i]['real_token'] == ';':
        output += "\n"

  #print(output)
  return output



def search(session, graph, program):
  pass

# returns probabilities, new_state
def run_step(session, graph, input, state):
  fetches = {
      "final_state": graph.get_tensor_by_name("Test/Model/FinalState:0"),
      "probabilities": graph.get_tensor_by_name("Test/Model/Probabilities:0"),
  }

  input_tensor = graph.get_tensor_by_name("Test/input:0")

  feed_dict = {input_tensor: [[input]]}
  feed_dict['Test/Model/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros:0'] = state[0].c
  feed_dict['Test/Model/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros_1:0'] = state[0].h
  feed_dict['Test/Model/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros:0'] = state[1].c
  feed_dict['Test/Model/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros_1:0'] = state[1].h
  #feed_dict[model.input.targets] = [[data[step+1]]]
  #for i, (c, h) in enumerate(model.initial_state):
  #  feed_dict[c] = state[i].c
  #  feed_dict[h] = state[i].h

  vals = session.run(fetches, feed_dict)

  final_state = vals["final_state"]
  state = []

  for i in range(len(final_state)):
      state.append(tf.contrib.rnn.LSTMStateTuple(final_state[i][0], final_state[i][1]))
  state = tuple(state)

  probabilities = vals['probabilities'][0]

  return probabilities, state


def run_epoch(session, graph, data, id_to_token):
  """Runs the model on the given data."""

  future_tokens = 0

  epoch_size = len(data) - 1
  #state = session.run(model.initial_state)

  # XXX
  state = []
  # hidden layers == 200?
  z = np.zeros([1,200])
  # num layers?
  for i in range(2):
      state.append(tf.contrib.rnn.LSTMStateTuple(z, z))
  state = tuple(state)

  output = []
  for step in range(epoch_size):

    probabilities, new_state = run_step(session, graph, data[step][0], state)

    m = np.argmax(probabilities)

    target_probability = probabilities[data[step+1][0]]
    max_probability = probabilities[m]
    output.append({
        'token': id_to_token[data[step][0]],
        'real_token': data[step][1], # if <unk>, what is it actually?
        'target': id_to_token[data[step + 1][0]],
        'expected': id_to_token[m],
        'future': [],
        'target_probability': float(target_probability),
        'expected_probability': float(max_probability),
        'ratio': float(target_probability / max_probability),
        #'probabilities': probabilities,
        #'state': new_state
    })

    # XXX very duplicated...
    #for j in range(future_tokens):
    #  feed_dict = {}
    #  feed_dict[model.input.input_data] = [[m]]
    #  for i, (c, h) in enumerate(model.initial_state):
    #    feed_dict[c] = state[i].c
    #    feed_dict[h] = state[i].h

    #  vals = session.run(fetches, feed_dict)
    #  state = vals["final_state"]

    #  probabilities = vals['probabilities'][0]
    #  # XXX ROB: do this in tensorflow
    #  m2 = np.argmax(probabilities)
    #  #test_input = PTBInput(config=config, data=[m], name="TestInput")
    #  max_probability = probabilities[m2]
    #  output[-1]['future'].append({
    #      'token': id_to_token[m],
    #      'expected': id_to_token[m2],
    #      'expected_probability': float(max_probability),
    #  })
    #  m = m2

    #state = remember_state

  #print(output)
  return output


def main(_):

  directory = os.path.dirname(os.path.realpath(__file__))
  os.chdir(directory)

  raw_data = dict()
  with open(FLAGS.data_path + '/tokens.json') as f:
    raw_data['token_to_id'] = json.load(f)

  raw_data['id_to_token'] = dict()
  for k in raw_data['token_to_id']:
    raw_data['id_to_token'][raw_data['token_to_id'][k]] = k

  with open(os.path.join(FLAGS.save_path, "training_config.json")) as f:
      config = json.load(f)
  config['batch_size'] = 1
  config['num_steps'] = 1

  with tf.Graph().as_default() as graph:
    initializer = tf.random_uniform_initializer(-config['init_scale'],
                                                config['init_scale'])

    saver = tf.train.import_meta_graph(os.path.join(FLAGS.save_path, "model.meta"))

    coord = tf.train.Coordinator()

    with tf.Session() as session:
      # ugh, Supervisor used to do this
      threads = tf.train.start_queue_runners(sess=session, coord=coord)
      saver.restore(session, os.path.join(FLAGS.save_path, 'model'))

      while True:
        for filename in os.listdir('../../server_tasks'):
          if filename.startswith('.'):
            continue
          test_ids = read.read('../../server_tasks/' + filename, token_to_id=raw_data['token_to_id'], include_token=True)
          #test_ids.reverse()

          output = run_epoch(session, graph, test_ids, raw_data['id_to_token'])
          generate_program(output)
          #output.reverse()

          with open('../../server_tasks/.' + filename + '-results-tmp', 'w') as f:
            json.dump(output, f)

          # make the output file appear atomically
          os.rename('../../server_tasks/.' + filename + '-results-tmp', '../../server_tasks/.' + filename + '-results');

          try:
            os.unlink('../../server_tasks/' + filename)
          except FileNotFoundError:
            pass
        time.sleep(0.01)

    # XXX this isn't fixing the error messages about cancelled enqueue attempts...
    #coord.request_stop()
    #coord.join(threads)

if __name__ == "__main__":
    tf.app.run()
