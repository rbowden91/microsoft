from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import time
import json
import os
import tree_read

from pycparser import c_parser, c_ast, parse_file

import numpy as np
import tensorflow as tf

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

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

    for k in ['expected_probability', 'target_probability', 'ratio', 'p_a', 'p_f', 'attr_expected_probability',
              'attr_target_probability', 'attr_ratio']:
        output[k] = float(node_properties[ast][k])
    output['expected'] = node_properties[ast]['expected']
    output['attr_expected'] = node_properties[ast]['attr_expected']
    output['attr_actual'] = node_properties[ast]['attr_actual']

    children = ast.children()
    for i in range(len(children)):
        ret = print_ast(children[i][1], node_properties)
        if ret is not False:
            output['children'].append(ret)
    return output


def run_epoch(session, graph, config, ast, raw_data):
    fetches = {}
    for k in config['fetches']:
        fetches[k] = config['fetches'][k]

    # TODO: do this once, instead of in each run_epoch
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
            # the 0th index represents the empty sibling/parent, so the actual data for this node should
            # go in the 1st
            config['placeholders']['is_leaf']: [0, 1 if len(children) == 0 else 0],
            config['placeholders']['last_sibling']: [0, 1 if last_sibling else 0],

            # XXX should have an <unk>. This will fail if something unexpected is found
            # need the parent token to feed it into the LSTM Cell
            config['placeholders']['node_index']: [0, raw_data['token_to_id'][node.__class__.__name__]],

            # 2nd 0 is a placeholder, changed below
            config['placeholders']['attr_index']: [0, 0],

            # these are specific to inference
            config['placeholders']['is_inference']: True,
            config['placeholders']['inference_parent']: raw_data['token_to_id'][parent.__class__.__name__] if parent is not None else 0,
            config['placeholders']['inference_sibling']: raw_data['token_to_id'][sibling.__class__.__name__] if sibling is not None else 0,

            # these should always be zero
            config['placeholders']['parent']: [0, 0],
            config['placeholders']['sibling']: [0, 0]
        }

        nvlist = [(n, getattr(node, n)) for n in node.attr_names]
        for (name, val) in nvlist:
            if name in ['value', 'op', 'name']:
                feed_dict[config['placeholders']['attr_index']][1] = tree_read.tokens_to_ids([[val]],
                        raw_data['attr_to_id'], False)[0][0]
                break
        else:
            feed_dict[config['placeholders']['attr_index']][1] = raw_data['attr_to_id']['<no_attr>']


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
        #target_id = raw_data['token_to_id'][node.__class__.__name__]
        target_id = feed_dict[config['placeholders']['node_index']][1]
        node_properties[node]['target_probability'] = probabilities[target_id]

        node_properties[node]['ratio'] = probabilities[target_id] / probabilities[expected_id]

        node_properties[node]['attr_probabilities'] = attr_probabilities = vals['attr_probabilities'][0]
        attr_expected_id = np.argmax(attr_probabilities)
        node_properties[node]['attr_expected'] = raw_data['id_to_attr'][attr_expected_id]
        node_properties[node]['attr_expected_probability'] = attr_probabilities[attr_expected_id]
        attr_target_id = feed_dict[config['placeholders']['attr_index']][1]
        node_properties[node]['attr_actual'] = raw_data['id_to_attr'][attr_target_id]
        node_properties[node]['attr_target_probability'] = attr_probabilities[attr_target_id]

        node_properties[node]['attr_ratio'] = attr_probabilities[attr_target_id] / attr_probabilities[attr_expected_id]


        # XXX these can be negative. how to make them probabilities?
        node_properties[node]['p_a'] = vals['predicted_p_a']
        node_properties[node]['p_f'] = vals['predicted_p_f']
        node_properties[node]['states'] = vals['states']

        next_sibling = None
        for i in range(len(children)):
            result = visit_tree(children[i][1], i == len(children) - 1, next_sibling, node)
            if result is True:
                next_sibling = children[i][1]
        return True

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
    with open(os.path.join(FLAGS.data_path, 'tree_tokens.json')) as f:
        token_ids = json.load(f)
        raw_data['token_to_id'] = token_ids['ast_labels']
        raw_data['attr_to_id'] = token_ids['label_attrs']

    raw_data['id_to_token'] = dict()
    for k in raw_data['token_to_id']:
        raw_data['id_to_token'][raw_data['token_to_id'][k]] = k

    raw_data['id_to_attr'] = dict()
    for k in raw_data['attr_to_id']:
        raw_data['id_to_attr'][raw_data['attr_to_id'][k]] = k

    with open(os.path.join(FLAGS.save_path, "tree_training_config.json")) as f:
        config = json.load(f)

    with tf.Graph().as_default() as graph:
        saver = tf.train.import_meta_graph(os.path.join(FLAGS.save_path, "tree_model.meta"))

        with tf.Session() as session:
            saver.restore(session, os.path.join(FLAGS.save_path, 'tree_model'))

            while True:
                for filename in os.listdir(FLAGS.task_path):
                    if filename.startswith('.'):
                        continue
                    try:
                        ast = parse_file(os.path.join(FLAGS.task_path, filename), use_cpp=True, cpp_path='gcc', cpp_args=['-E', r'-I../fake_libc_include'])
                    except Exception:
                        # TODO: return some kind of error about failure to parse
                        continue

                    node_properties = run_epoch(session, graph, config, ast, raw_data)
                    output = print_ast(ast, node_properties)

                    with open(os.path.join(FLAGS.task_path, '.' + filename + '-results-tmp'), 'w') as f:
                        json.dump(output, f)

                    # make the output file appear atomically
                    os.rename(os.path.join(FLAGS.task_path, '.' + filename + '-results-tmp'),
                              os.path.join(FLAGS.task_path, '.' + filename + '-results'));

                    try:
                        os.unlink(os.path.join(FLAGS.task_path, filename))
                    except FileNotFoundError:
                        pass
                time.sleep(0.01)

if __name__ == "__main__":
    tf.app.run()
