from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import inspect
from pprint import pprint
import time
import json
import os
import preprocess
import re

from pycparser import c_parser, c_ast, parse_file

import numpy as np
import tensorflow as tf
import queue as Q
#import check_correct

import dump_ast

flags = tf.flags
logging = tf.logging

flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

flags.DEFINE_string("data_path", None,
                    "XXX")
flags.DEFINE_string("task_path", '../web/tasks',
                    "Task directory")

FLAGS = flags.FLAGS

max_changes = 3

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
                        #code = directives + generator.visit(ast)
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
    if ast not in node_properties:
        return False

    output = {
        "children": []
    }
    props = node_properties[ast]
    for k in props:
        if k in ['self', 'dependencies']:
            continue
        output[k] = props[k]

    children = ast.children()
    for i in range(len(children)):
        ret = print_ast(children[i][1], node_properties)
        if ret is not False:
            output['children'].append(ret)
    return output

def run_epoch(session, graph, config, tokens, raw_data, node_properties=None):

    data_dict = {}
    if config['model'] == 'linear':
        for k in config['features']:
            p = config['placeholders']['features'][k]
            if k == 'left_sibling':
                data_dict[p] = [[0] + list(range(len(tokens) - 1))]
            elif k == 'right_sibling':
                data_dict[p] = [[0] + list(range(2, len(tokens))) + [0]]
            elif k == 'label_index':
                data_dict[p] = [[0] + [token[0] for token in tokens]]
            elif k == 'mask':
                data_dict[p] = [[0] + [1] * (len(tokens)-1)]
            else:
                data_dict[p] = [[0] * len(tokens)]
    else:
        for k in config['features']:
            p = config['placeholders']['features'][k]
            data_dict[p] = [[0]]
        for token in tokens:
            for (name, val) in token['attrs']:
                if name in ['value', 'op', 'name']:
                    # TODO just look up directly, and do the string check, too
                    token['attr'] = val
                    token['attr_index'] = preprocess.tokens_to_ids([val], raw_data['attr_to_id'],
                                                True, False)[0]
                    break
            else:
                token['attr_index'] = raw_data['attr_to_id']['<no_attr>']
            token['label_index'] = raw_data['label_to_id'][token['label']]
            token['mask'] = 1
            for k in config['features']:
                p = config['placeholders']['features'][k]
                data_dict[p][0].append(token[k])


    session.run(config['ops']['node_iter'], data_dict)

    feed_dict = {
        config['placeholders']['is_inference']: False
    }
    vals = session.run(config['fetches']['loss'], feed_dict)

    props = node_properties if node_properties is not None else []
    for i in range(len(tokens)):
        token = tokens[i][1] if config['model'] == 'linear' else tokens[i]['label']
        prop = {}
        for direction in vals:
            prop[direction] = p = {'token': token}
            for k in vals[direction]:
                if i == 0:
                    print('{} {} perplexity: {}'.format(direction, k, np.exp(vals[direction][k]['loss'])))
                probs = vals[direction][k]['probabilities'][0][i+1]
                if direction == 'joint':
                    p['alpha'] = alpha = float(probs[0])
                    probs = alpha * vals['forward'][k]['probabilities'][0][i+1] + \
                            (1-alpha) * vals['reverse'][k]['probabilities'][0][i+1]

                target = data_dict[config['placeholders']['features'][k]][0]
                rank = np.flip(np.argsort(probs), 0)
                if k in ['label_index', 'attr_index']:
                    p[k + '_expected'] = raw_data[k + '_to_token'][rank[0]]
                    p[k + '_actual'] = raw_data[k + '_to_token'][target[i+1]]
                    p[k + '_expected_probability'] = float(probs[rank[0]])
                    p[k + '_actual_probability'] = float(probs[target[i+1]])
                    p[k + '_ratio'] = float(probs[target[i+1]] / (probs[rank[0]]))
                    p['probabilities'] = [(float(probs[j]), raw_data[k+'_to_token'][j]) for j in rank]
                else:
                    p[k + '_expected'] = target[i+1]
                    p[k + '_actual'] = float(probs)

        if node_properties is None:
            props.append(prop)
        else:
            props[tokens[i]['self']].update(prop)

    return props

def main(_):
    directory = os.path.dirname(os.path.realpath(__file__))
    os.chdir(directory)

    best_dir = os.path.join(FLAGS.data_path, 'best')
    with open(os.path.join(best_dir, 'config.json')) as f:
        config = json.load(f)
    # fix windows path separators
    config['data_path'] = os.path.join(*config['data_path'].split('\\'))

    raw_data = dict()
    with open(os.path.join(config['data_path'], config['model'] + '_lexicon.json')) as f:
        token_ids = json.load(f)
    raw_data['label_to_id'] = token_ids['label_ids']
    raw_data['attr_to_id'] = token_ids['attr_ids']

    raw_data['label_index_to_token'] = dict()
    for k in raw_data['label_to_id']:
        raw_data['label_index_to_token'][raw_data['label_to_id'][k]] = k

    raw_data['attr_index_to_token'] = dict()
    for k in raw_data['attr_to_id']:
        raw_data['attr_index_to_token'][raw_data['attr_to_id'][k]] = k

    with tf.Graph().as_default() as graph:
        saver = tf.train.import_meta_graph(os.path.join(best_dir, "model.meta"))

        with tf.Session() as session:
            saver.restore(session, os.path.join(best_dir, 'model'))

            parser = c_parser.CParser()
            while True:
                task_path = os.path.join(FLAGS.task_path, config['model'])
                for filename in os.listdir(task_path):
                    if filename.startswith('.'):
                        continue
                    filepath = os.path.join(task_path, filename)
                    with open(filepath) as f:
                        text = f.read()
                    directives, _ = preprocess.grab_directives(text)
                    # XXX this can return None
                    ast_nodes, ast, node_properties, tokens = preprocess.preprocess_c(filepath,
                            include_dependencies=True)

                    if config['model'] == 'ast':
                        # Can't add attributes directly to Nodes because the class uses __slots__, so use this dictionary to
                        # extend objects
                        #run_ast_epoch(session, graph, config, ast, ast_nodes, node_properties, raw_data)
                        run_epoch(session, graph, config, ast_nodes, raw_data, node_properties)
                        code = "blah"#search(ast, node_properties, filename, directives)
                        output = {
                            'ast': print_ast(ast, node_properties),
                            #'code': generator.visit(ast),
                            'fixed_code': code
                        }
                    else:
                        tokens = preprocess.tokens_to_ids(tokens, raw_data['label_to_id'], True, True)
                        props = run_epoch(session, graph, config, tokens, raw_data)
                        code = "blah"#search(ast, node_properties, filename, directives)
                        output = {
                            'linear': props,
                            'fixed_code': code
                        }

                    with open(os.path.join(task_path, '.' + filename + '-results-tmp'), 'w') as f:
                        json.dump(output, f)

                    # make the output file appear atomically
                    os.rename(os.path.join(task_path, '.' + filename + '-results-tmp'),
                              os.path.join(task_path, '.' + filename + '-results'));

                    try:
                        os.unlink(os.path.join(task_path, filename))
                    except FileNotFoundError:
                        pass
                time.sleep(0.01)

if __name__ == "__main__":
    tf.app.run()
