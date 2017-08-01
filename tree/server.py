from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import inspect
import time
import json
import os
import tree_read
import re

from pycparser import c_parser, c_ast, parse_file, c_generator

import numpy as np
import tensorflow as tf
import queue as Q
import check_correct

import dump_ast

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

max_changes = 3
generator = c_generator.CGenerator()

# side effect: populate node_properties with parent pointers (not yet used?)
def fill_queue(node, node_properties, q, parent=None):
    if node.__class__.__name__ in dump_ast.ignore:
        return
    node_properties[node]['parent'] = parent

    score = node_properties[node]['attr_ratio']
    # XXX for now, time.time() is supposed to make sure that we never get to comparing nodes
    q.put((score, time.time(), node))
    children = node.children()
    for i in range(len(children)):
        fill_queue(children[i][1], node_properties, q, node)

# XXX heuristics about class name?
def search_changes(ast, node_properties, list_q, max_changes, filename, directives, start = 0, num_changes = 0):
    for i in range(start, len(list_q)):
        node = list_q[i][2]
        # adjust this cutoff?
        if node_properties[node]['attr_ratio'] == 1.0:
            break
        # XXX for now, don't deal with IDs...what about functions though? ugh...
        if node.__class__.__name__ in dump_ast.ignore or node.__class__.__name__ == 'ID':
            continue
        nvlist = [(n, getattr(node, n)) for n in node.attr_names]
        for (name, val) in nvlist:
            if name in ['value', 'op', 'name']:
                setattr(node, name, node_properties[node]['attr_expected'])
                if num_changes == max_changes - 1:
                    #try:
                        code = directives + generator.visit(ast)
                        path = os.path.join(FLAGS.task_path, '.' + filename + '.c')
                        with open(path, 'w') as f:
                            f.write(code)
                        ret = check_correct.check_vigenere(path)
                        os.unlink(path)
                        if ret == 0:
                            return code
                    #except Exception:
                    #    #print('uh ohhh')
                    #    pass
                else:
                    ret = search_changes(ast, node_properties, list_q, max_changes, filename, directives, start=i+1, num_changes=num_changes+1)
                    # Success! The ast is now repaired
                    if ret is not False:
                        return ret
                # no luck, revert to the old value
                setattr(node, name, val)
                break
    # didn't find a working tree
    return False



def search(ast, node_properties, filename, directives):
    # XXX check if code already works?
    #code = generator.visit(ast)
    #path = os.path.join(FLAGS.task_path, '.' + filename + '.c')
    #with open(path, 'w') as f:
    #    f.write(code)
    #ret = check_correct.check_vigenere(path)
    #os.unlink(path)
    #if ret == 0:
    #    return code
    q = Q.PriorityQueue()
    fill_queue(ast, node_properties, q)
    list_q = []
    while not q.empty():
        list_q.append(q.get())
    for i in range(max_changes):
        code = search_changes(ast, node_properties, list_q, i+1, filename, directives)
        if code is not False:
            return code
    return False


def print_ast(ast, node_properties):
    # ignore dumb typedefs for now
    if ast.__class__.__name__ in dump_ast.ignore:
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


def run_epoch(session, graph, config, ast, node_properties, raw_data, initial_states):
    fetches = {}
    for k in config['fetches']:
        fetches[k] = config['fetches'][k]

    def visit_tree(node):
        if node.__class__.__name__ in dump_ast.ignore:
            return False
        props = node_properties[node]

        label_index = raw_data['token_to_id'][props['label']] \
                        if props['label'] in raw_data['token_to_id'] \
                        else raw_data['token_to_id']['<unk_label>']
        for (name, val) in props['attrs']:
            if name in ['value', 'op', 'name']:
                attr_index = tree_read.tokens_to_ids([[val]],
                        raw_data['attr_to_id'], False)[0][0]
                break
        else:
            attr_index = raw_data['attr_to_id']['<no_attr>']

        props['label_index'] = label_index
        props['attr_index'] = label_index

        feed_dict = {
            config['placeholders']['is_inference']: True,
            config['placeholders']['data']['label_index']: [0, label_index],
            config['placeholders']['data']['attr_index']: [0, attr_index]
        }

        for k in config['placeholders']['data']:
            if config['placeholders']['data'][k] in feed_dict:
                continue

            # the 0th index represents the empty sibling/parent, so the actual data for this node should
            # go in the 1st
            if k in config['placeholders']['inference']:
                feed_dict[config['placeholders']['data'][k]] = [0, 0]
                # should exist
                dependency_node = props['dependencies'][k]
                feed_dict[config['placeholders']['inference'][k]] = node_properties[dependency_node]['label_index'] if dependency_node in node_properties else 0
            else:
                feed_dict[config['placeholders']['data'][k]] = [0, props[k]]

        for dependency in config['dependencies']:
            for i in range(len(config['feed']['initial_states'][dependency])):
                state = config['feed']['initial_states'][dependency][i]
                dependency_node = props['dependencies'][dependency]
                if dependency_node is not None:
                    dependency_props = node_properties[dependency_node]
                    feed_dict[state['c']] = dependency_props['states'][dependency][i]['c']
                    feed_dict[state['h']] = dependency_props['states'][dependency][i]['h']
                else:
                    feed_dict[state['c']] = initial_states[dependency][i].c
                    feed_dict[state['h']] = initial_states[dependency][i].h

        vals = session.run(fetches, feed_dict)

        node_properties[node]['probabilities'] = probabilities = vals['label_probabilities'][0]
        #### XXX ROB: do this in tensorflow?
        expected_id = np.argmax(probabilities)
        # XXX again, possibly <unk>
        node_properties[node]['expected'] = raw_data['id_to_token'][expected_id]
        node_properties[node]['expected_probability'] = probabilities[expected_id]

        #target_id = raw_data['token_to_id'][node.__class__.__name__]
        target_id = feed_dict[config['placeholders']['data']['label_index']][1]
        node_properties[node]['target_probability'] = probabilities[target_id]

        node_properties[node]['ratio'] = probabilities[target_id] / probabilities[expected_id]

        node_properties[node]['attr_probabilities'] = attr_probabilities = vals['attr_probabilities'][0]
        attr_expected_id = np.argmax(attr_probabilities)
        # check <unk>
        node_properties[node]['attr_expected'] = raw_data['id_to_attr'][attr_expected_id]
        node_properties[node]['attr_expected_probability'] = attr_probabilities[attr_expected_id]
        attr_target_id = feed_dict[config['placeholders']['data']['attr_index']][1]
        node_properties[node]['attr_actual'] = raw_data['id_to_attr'][attr_target_id]
        node_properties[node]['attr_target_probability'] = attr_probabilities[attr_target_id]

        node_properties[node]['attr_ratio'] = attr_probabilities[attr_target_id] / attr_probabilities[attr_expected_id]

        node_properties[node]['p_a'] = vals['predicted_p_a']
        node_properties[node]['p_f'] = vals['predicted_p_f']
        node_properties[node]['states'] = vals['states']

        children = node.children()
        for i in range(len(children)):
            visit_tree(children[i][1])
        return True

    visit_tree(ast)

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

# https://stackoverflow.com/questions/2319019/using-regex-to-remove-comments-from-source-files
def remove_comments(string):
    pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*$)"
    # first group captures quoted strings (double or single)
    # second group captures comments (//single-line or /* multi-line */)
    regex = re.compile(pattern, re.MULTILINE|re.DOTALL)
    def _replacer(match):
        # if the 2nd group (capturing comments) is not None,
        # it means we have captured a non-quoted (real) comment string.
        if match.group(2) is not None:
            return "" # so we will return empty to remove the comment
        else: # otherwise, we will return the 1st group
            return match.group(1) # captured quoted-string
    return regex.sub(_replacer, string)

def grab_directives(string):
    # not perfect...
    pattern = r"(^\s*#[^\r\n]*[\r\n])"
    regex = re.compile(pattern, re.MULTILINE|re.DOTALL)

    directives = ''.join(regex.findall(string))
    def _replacer(match):
        return ""
    sub = regex.sub(_replacer, string)
    return directives, sub


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

            initial_states = {}
            for k in config['feed']['initial_states']:
                initial_states[k] = []
                for i in range(len(config['feed']['initial_states'][k])):
                    c, h = session.run([config['feed']['initial_states'][k][i]['c'], config['feed']['initial_states'][k][i]['h']])
                    # don't really need to wrap this up in an LSTM tuple
                    initial_states[k].append(tf.contrib.rnn.LSTMStateTuple(c, h))

            parser = c_parser.CParser()
            while True:
                for filename in os.listdir(FLAGS.task_path):
                    if filename.startswith('.'):
                        continue
                    try:
                        with open(os.path.join(FLAGS.task_path, filename)) as f:
                            text = f.read()
                        directives, preprocessed = grab_directives(remove_comments(text))
                        # XXX XXX XXX TYPES ARE AWFUL
                        ast = parser.parse('typedef char* string;\n' + preprocessed)
                        #ast = parse_file(os.path.join(FLAGS.task_path, filename), use_cpp=True, cpp_path='gcc', cpp_args=['-E', r'-I../fake_libc_include'])
                    except ValueError:
                        # TODO: return some kind of error about failure to parse
                        #print(str(sys.exc_info()[0]))
                        continue

                    # Can't add attributes directly to Nodes because the class uses __slots__, so use this dictionary to
                    # extend objects
                    node_properties = {}
                    ret = dump_ast.linearize_ast(ast, node_properties=node_properties) # XXX how slow is this?
                    run_epoch(session, graph, config, ast, node_properties, raw_data, initial_states)
                    code = search(ast, node_properties, filename, directives)
                    output = {
                        'ast': print_ast(ast, node_properties),
                        'fixed_code': code
                    }

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
