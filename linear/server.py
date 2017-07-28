from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import time
import json
import read
import os

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
flags.DEFINE_string("task_path", None,
                    "Task directory")

FLAGS = flags.FLAGS

def run_epoch(session, graph, data, id_to_token):
  """Runs the model on the given data."""

  future_tokens = 0

  epoch_size = len(data) - 1
  #state = session.run(model.initial_state)

  fetches = {
      #"cost": model.cost,
      "final_state": graph.get_tensor_by_name("Test/Model/FinalState:0"),
      # can conditionally add this in
      "probabilities": graph.get_tensor_by_name("Test/Model/Probabilities:0"),
      #"input": model.input.input_data,
      #"target": model.input.targets

  }
  # XXX
  state = []
  # hidden layers == 200?
  z = np.zeros([1,200])
  # num layers?
  for i in range(2):
      state.append(tf.contrib.rnn.LSTMStateTuple(z, z))
  state = tuple(state)

  #input = tf.get_variable('Test/input', reuse=True, shape=[1, 1])
  input = graph.get_tensor_by_name("Test/input:0")

  output = []
  for step in range(epoch_size):
    feed_dict = {input: [[data[step][0][0]]]}
    # test, not train, yeah?
    feed_dict['Test/Model/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros:0'] = state[0].c
    feed_dict['Test/Model/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros_1:0'] = state[0].h
    feed_dict['Test/Model/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros:0'] = state[1].c
    feed_dict['Test/Model/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros_1:0'] = state[1].h
    #feed_dict[model.input.targets] = [[data[step+1]]]
    #for i, (c, h) in enumerate(model.initial_state):
    #  feed_dict[c] = state[i].c
    #  feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)

    #cost = vals["cost"]
    final_state = vals["final_state"]
    #state=final_state
    state = []

    #print(final_state.shape)

    for i in range(len(final_state)):
        state.append(tf.contrib.rnn.LSTMStateTuple(final_state[i][0], final_state[i][1]))
    state = tuple(state)

    remember_state = state

    probabilities = vals['probabilities'][0]
    ## XXX ROB: do this in tensorflow?
    m = np.argmax(probabilities)

    target_probability = probabilities[data[step+1][0][0]]
    max_probability = probabilities[m]
    output.append({
        'token': id_to_token[data[step][0][0]],
        'target': id_to_token[data[step + 1][0][0]],
        'expected': id_to_token[m],
        'future': [],
        'target_probability': float(target_probability),
        'expected_probability': float(max_probability),
        'ratio': float(target_probability / max_probability)
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

    state = remember_state

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
    saver = tf.train.import_meta_graph(os.path.join(FLAGS.save_path, "model.meta"))

    coord = tf.train.Coordinator()

    with tf.Session() as session:
      # ugh, Supervisor used to do this
      threads = tf.train.start_queue_runners(sess=session, coord=coord)
      saver.restore(session, os.path.join(FLAGS.save_path, 'model'))

      while True:
        for filename in os.listdir(FLAGS.task_path):
          if filename.startswith('.'):
            continue
          test_ids = read.read(FLAGS.task_path + filename, token_to_id=raw_data['token_to_id'], include_token=True)
          if test_ids:
            #test_ids.reverse()

            output = run_epoch(session, graph, test_ids, raw_data['id_to_token'])
            #output.reverse()

            with open(FLAGS.task_path + '.' + filename + '-results-tmp', 'w') as f:
              json.dump(output, f)

            # make the output file appear atomically
            os.rename(FLAGS.task_path + '.' + filename + '-results-tmp', FLAGS.task_path + '.' + filename + '-results');

          try:
            os.unlink(FLAGS.task_path + filename)
          except FileNotFoundError:
            pass
        time.sleep(0.01)

    # XXX this isn't fixing the error messages about cancelled enqueue attempts...
    #coord.request_stop()
    #coord.join(threads)

if __name__ == "__main__":
    tf.app.run()
