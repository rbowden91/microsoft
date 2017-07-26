from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import time
import json
import os

from pycparser import c_parser, c_ast, parse_file

import numpy as np
import tensorflow as tf

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

# XXX ROB
# XXX USE OS.JOIN EVERYWHERE
flags.DEFINE_string("data_path", None,
                    "XXX")
flags.DEFINE_string("task_path", None,
                    "Task directory")

FLAGS = flags.FLAGS

def print_ast(ast, node_properties):
    # ignore dumb typedefs for now
    if ast.__class__.__name__ == 'Typedef' or ast.__class__.__name__ == 'Decl' or ast.__class__.__name__ == 'TypeDecl' or ast.__class__.__name__ == 'DeclList':
        return False

    output = {
        "name": ast.__class__.__name__,
        "children": []
    }

    for k in ['expected_probability', 'target_probability', 'ratio', 'p_a', 'p_f']:
        output[k] = float(node_properties[ast][k])
    output['expected'] = node_properties[ast]['expected']

    children = ast.children()
    for i in range(len(children)):
        ret = print_ast(children[i][1], node_properties)
        if ret is not False:
            output['children'].append(ret)
    return output


def run_epoch(session, graph, config, ast, raw_data):
    fetches = {}
    for k in config['fetches']:
        # XXX fetches[k] = graph.get_tensor_by_name(config['fetches'][k])
        fetches[k] = config['fetches'][k]

    # XXX do this once, instead of in each run_epoch
    initial_states = {}
    for k in config['initial_states']:
        initial_states[k] = []
        for i in range(len(config['initial_states'][k])):
            c, h = session.run([config['initial_states'][k][i]['c'], config['initial_states'][k][i]['h']])
            # don't really need to wrap this up in an LSTM tuple
            initial_states[k].append(tf.contrib.rnn.LSTMStateTuple(c, h))


    # Can't add attributes directly to Nodes because the class uses __slots__, so use this dictionary to
    # extend objects
    node_properties = {}
    def visit_tree(node, last_sibling, sibling, parent):
        if node.__class__.__name__ == 'Typedef' or node.__class__.__name__ == 'Decl' or node.__class__.__name__ == 'TypeDecl' or node.__class__.__name__ == 'DeclList':
            return False
        children = node.children()

        feed_dict = {
            # XXX that first index?
            config['placeholders']['is_leaf']: [0, 1 if len(children) == 0 else 0],
            config['placeholders']['last_sibling']: [0, 1 if last_sibling else 0],
            # XXX should have an <unk>
            # need the parent token to feed it into the LSTM Cell
            config['placeholders']['node_index']: [0, raw_data['token_to_id'][node.__class__.__name__]],

            # these are specific to inference
            config['placeholders']['is_inference']: True,
            config['placeholders']['inference_parent']: raw_data['token_to_id'][parent.__class__.__name__] if parent is not None else 0,
            config['placeholders']['inference_sibling']: raw_data['token_to_id'][sibling.__class__.__name__] if sibling is not None else 0,

            # these should always be zero
            config['placeholders']['parent']: [0, 0],
            config['placeholders']['sibling']: [0, 0]
        }

        for i in range(len(config['initial_states']['sibling'])):
            feed_dict[config['initial_states']['sibling'][i]['c']] = node_properties[sibling]['states']['sibling'][i]['c'] if sibling is not None else initial_states['sibling'][i].c
            feed_dict[config['initial_states']['sibling'][i]['h']] = node_properties[sibling]['states']['sibling'][i]['h'] if sibling is not None else initial_states['sibling'][i].h

        for i in range(len(config['initial_states']['parent'])):
            feed_dict[config['initial_states']['parent'][i]['c']] = node_properties[parent]['states']['parent'][i]['c'] if parent is not None else initial_states['parent'][i].c
            feed_dict[config['initial_states']['parent'][i]['h']] = node_properties[parent]['states']['parent'][i]['h'] if parent is not None else initial_states['parent'][i].h


        vals = session.run(fetches, feed_dict)

        node_properties[node] = {}
        node_properties[node]['probabilities'] = probabilities = vals['label_probabilities'][0]
        #### XXX ROB: do this in tensorflow?
        expected_id = np.argmax(probabilities)
        node_properties[node]['expected'] = raw_data['id_to_token'][expected_id]
        node_properties[node]['expected_probability'] = probabilities[expected_id]

        # XXX again, possibly <unk>
        target_id = raw_data['token_to_id'][node.__class__.__name__]
        node_properties[node]['target_probability'] = probabilities[target_id]

        node_properties[node]['ratio'] = probabilities[target_id] / probabilities[expected_id]

        # XXX these can be negative. how to make them probabilities?
        node_properties[node]['p_a'] = vals['logits_p_a'][0]
        node_properties[node]['p_f'] = vals['logits_p_f'][0]
        node_properties[node]['states'] = vals['states']

        next_sibling = None
        for i in range(len(children)):
            result = visit_tree(children[i][1], i == len(children) - 1, next_sibling, node)
            if result is True:
                next_sibling = children[i][1]
        return True

    # XXX Can add initial state to parent_states and sibling_states
    visit_tree(ast, True, None, None)
    return node_properties




    #target_probability = probabilities[data[step+1][0]]
    #max_probability = probabilities[m]
    #output.append({
    #    'token': id_to_token[data[step][0]],
    #    'target': id_to_token[data[step + 1][0]],
    #    'expected': id_to_token[m],
    #    'future': [],
    #    'target_probability': float(target_probability),
    #    'expected_probability': float(max_probability),
    #    'ratio': float(target_probability / max_probability)
    #})

    #print(output)
    #return output


def main(_):

  directory = os.path.dirname(os.path.realpath(__file__))
  os.chdir(directory)

  raw_data = dict()
  with open(FLAGS.data_path + '/tree_tokens.json') as f:
    raw_data['token_to_id'] = json.load(f)

  raw_data['id_to_token'] = dict()
  for k in raw_data['token_to_id']:
    raw_data['id_to_token'][raw_data['token_to_id'][k]] = k

  with open(os.path.join(FLAGS.save_path, "tree_training_config.json")) as f:
      config = json.load(f)

  #config['batch_size'] = 1
  #config['num_steps'] = 1

  with tf.Graph().as_default() as graph:
    saver = tf.train.import_meta_graph(os.path.join(FLAGS.save_path, "tree_model.meta"))

    with tf.Session() as session:
      saver.restore(session, os.path.join(FLAGS.save_path, 'tree_model'))

      while True:
        for filename in os.listdir(FLAGS.task_path):
          if filename.startswith('.'):
            continue
          # XXX XXX add try-catch
          ast = parse_file(FLAGS.task_path + filename, use_cpp=True, cpp_path='gcc', cpp_args=['-E', r'-I../utils/fake_libc_include'])
          #test_ids.reverse()

          node_properties = run_epoch(session, graph, config, ast, raw_data)
          output = print_ast(ast, node_properties)
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
