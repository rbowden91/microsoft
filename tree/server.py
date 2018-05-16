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
flags.DEFINE_string("task_path", None,
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

def convert(i):
    if isinstance(i, np.float32):
        return np.asscalar(i)
    elif isinstance(i, np.ndarray):
        return i.tolist()
    elif isinstance(i, dict):
        if 'forward' in i or 'reverse' in i:
            out = {}
            for k in i:
                out[k] = convert(i[k])
            return out
        else:
            return False
    else:
        return i

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
        conversion = convert(props[k])
        if conversion:
            output[k] = conversion

    children = ast.children()
    for i in range(len(children)):
        ret = print_ast(children[i][1], node_properties)
        if ret is not False:
            output['children'].append(ret)
    return output

def run_linear_epoch(session, graph, config, tokens, raw_data):
    directions = set()
    for k in config['dependencies']:
        directions.add(config['dependencies'][k])

    props = [{d: {} for d in directions} for i in tokens]
    for direction in directions:
        data_dict = {}
        for k in config['features']:
            p = config['placeholders']['features'][k]
            if k == 'left_sibling':
                data_dict[p] = [[0] + list(range(len(tokens) - 1))]
            elif k == 'right_sibling':
                data_dict[p] = [[0] + list(range(2, len(tokens))) + [0]]
            elif k == 'label_index':
                data_dict[p] = [[token[0] for token in tokens]]
            elif k == 'mask':
                data_dict[p] = [[0] + [1] * (len(tokens)-1)]
            else:
                data_dict[p] = [[0] * len(tokens)]
        session.run(config['ops']['node_iter'], data_dict)

        feed_dict = {
            config['placeholders']['is_inference']: False
        }
        vals = session.run(config['fetches']['loss'], feed_dict)
        #cost = 0
        #r = range(1, len(tokens)) if direction == 'forward' else range(len(tokens)-1,0,-1)
        #for i in r:
        #    if props[i] is None:
        #        props[i] = {
        #            'token': tokens[i][1]
        #        }
        #    token = tokens[i][0]
        #    for k in config['placeholders']['data']:
        #        feed_dict[config['placeholders']['data'][k]] = [0, 0]
        #    feed_dict[config['placeholders']['data']['label_index']] = [0, token]

        #    feed_dict[config['placeholders']['inference']['self']['label']] = token
        #    feed_dict[config['placeholders']['inference']['self']['attr']] = 0
        #    feed_dict[config['placeholders']['inference']['right_sibling']['label']] = tokens[i+1][0] if i+1<len(tokens) else 0
        #    feed_dict[config['placeholders']['inference']['right_sibling']['attr']] = 0
        #    feed_dict[config['placeholders']['inference']['left_sibling']['label']] = tokens[i-1][0] if i != 0 else 0
        #    feed_dict[config['placeholders']['inference']['left_sibling']['attr']] = 0

        #    sibling = i-1 if direction == 'forward' else i+1
        #    dependent = 'left_sibling' if direction == 'forward' else 'right_sibling'
        #    for layer in range(config['num_layers']):
        #        state = config['feed']['states'][dependent][layer]
        #        output = config['feed']['outputs'][dependent][layer]
        #        if direction == 'forward' and i-1 == 0 or direction == 'reverse' and i+1 == len(props):
        #            feed_dict[state['c']] = initial['states'][dependent][layer].c
        #            feed_dict[state['h']] = initial['states'][dependent][layer].h
        #            feed_dict[output] = initial['outputs'][dependent][layer]
        #        else:
        #            feed_dict[state['c']] = props[sibling]['states'][dependent][layer]['c']
        #            feed_dict[state['h']] = props[sibling]['states'][dependent][layer]['h']
        #            feed_dict[output] = props[sibling]['outputs'][dependent][layer]

        #    fetches = {
        #        'states': {dependent: {}},
        #        'outputs': {dependent: {}},
        #        'cost': config['fetches']['cost'][direction],
        #        'label_probabilities': config['fetches']['label_probabilities'][direction]
        #    }
        #    for layer in range(config['num_layers']):
        #        fetches['states'][dependent][layer] = config['fetches']['states'][dependent][layer]
        #        fetches['outputs'][dependent][layer] = config['fetches']['outputs'][dependent][layer]
        #    vals = session.run(fetches, feed_dict)

        #    if 'states' not in props[i]:
        #        props[i]['states'] = {}
        #    if 'outputs' not in props[i]:
        #        props[i]['outputs'] = {}
        #    props[i]['states'].update(vals['states'])
        #    props[i]['outputs'].update(vals['outputs'])

        #for k in vals:
            #props[i][k] = vals[k]
            # get ids of expected and actual labels

        probs = vals[direction]['label']['probabilities'][0]
        target_ids = data_dict[config['placeholders']['features']['label_index']][0]
        for i in range(len(tokens)):
            rank = np.flip(np.argsort(probs[i]), 0)
            props[i][direction]['token'] = tokens[i][1]
            props[i][direction]['label_expected'] = raw_data['id_to_label'][rank[0]]
            props[i][direction]['label_expected_probability'] = float(probs[i][rank[0]])
            props[i][direction]['label_actual'] = raw_data['id_to_label'][target_ids[i]]
            props[i][direction]['label_actual_probability'] = float(probs[i][target_ids[i]])
            props[i][direction]['label_ratio'] = float(probs[i][target_ids[i]] /
                    (probs[i][rank[0]]))
                    #(probabilities[i][expected_ids[i]] + probabilities[i][target_ids[i]]))
            props[i][direction]['probabilities'] = [(float(probs[i][j]), raw_data['id_to_label'][j]) for j in rank]
        for k in vals[direction]:
            print('{} {} perplexity: {}'.format(direction, k, np.exp(vals[direction][k]['loss'])))

        #cost += vals['cost']

    props.pop(0)
    #for i in range(len(props)):
    #    del(props[i]['states'])
    #    del(props[i]['outputs'])
    return props


def run_ast_epoch(session, graph, config, ast, node_properties, raw_data):
    directions = set()
    for k in config['dependencies']:
        directions.add(config['dependencies'][k])

    def visit_tree(node, direction, layer):
        # the node was removed from the tree
        if node not in node_properties:
            return False

        cost = 0
        if direction == 'reverse':
            children = node.children()
            for i in range(len(children) - 1, -1, -1):
                cost += visit_tree(children[i][1], direction, layer)

        is_last_layer = layer == config['num_layers'] - 1

        props = node_properties[node]

        # only need to do once
        if direction == 'forward' or len(directions) == 1:
            label_index = raw_data['label_to_id'][props['label']] \
                            if props['label'] in raw_data['label_to_id'] \
                            else raw_data['label_to_id']['<unk_label>']
            for (name, val) in props['attrs']:
                if name in ['value', 'op', 'name']:
                    attr_index = preprocess.tokens_to_ids([val], raw_data['attr_to_id'], False, False)[0]
                    break
            else:
                attr_index = raw_data['attr_to_id']['<no_attr>']

            # remember the index for the future
            props['label_index'] = label_index
            props['attr_index'] = attr_index

        fetches = {
            'states': {},
            'outputs': {}
        }
        for k in config['fetches']['outputs']:
            if k != 'children':
                fetches['states'][k] = { layer: config['fetches']['states'][k][layer] }
            fetches['outputs'][k] = { layer: config['fetches']['outputs'][k][layer] }

        if direction == 'reverse' and 'children' in config['dependencies']:
            if is_last_layer:
                fetches['children_predictor_states'] = { layer: config['fetches']['children_predictor_states'][layer] }
                fetches['predicted_right_hole'] = config['fetches']['predicted_right_hole']
            if props['dependencies']['left_sibling'] is not None:
                fetches['children_tmp_states'] = { layer: config['fetches']['children_tmp_states'][layer] }
            else:
                fetches['states']['children'] = { layer: config['fetches']['states']['children'][layer] }
                if is_last_layer:
                    fetches['children_output'] = config['fetches']['children_output']

        if is_last_layer:
            for k in ['cost', 'predicted_end', 'label_probabilities', 'attr_probabilities']:
                fetches[k] = config['fetches'][k][direction]

        if is_last_layer:
            fetches['h_pred'] = config['fetches']['h_pred']

        feed_dict = {
            config['placeholders']['is_inference']: True,
        }

        if 'children' in config['dependencies']:
            feed_dict[config['feed']['leaf_input']] = initial['leaf_input']

        for k in config['placeholders']['data']:
            feed_dict[config['placeholders']['data'][k]] = [0, props[k]]

        feed_dict[config['placeholders']['inference']['self']['label']] = props['label_index']
        feed_dict[config['placeholders']['inference']['self']['attr']] = props['attr_index']

        for dependency in config['dependencies']:

            # the children lstm needs to look at the parent
            key = dependency if dependency != 'children' else 'parent'
            dependency_node = props['dependencies'][key]
            if dependency_node is not None and 'states' in node_properties[dependency_node]:
                dependency_props = node_properties[dependency_node]
                feed_dict[config['placeholders']['inference'][key]['label']] = dependency_props['label_index']
                feed_dict[config['placeholders']['inference'][key]['attr']] = dependency_props['attr_index']
            else:
                feed_dict[config['placeholders']['inference'][key]['label']] = 0
                feed_dict[config['placeholders']['inference'][key]['attr']] = 0

            if is_last_layer:
                dependency_node = props['dependencies']['right_hole']
                if dependency_node is not None and 'h_pred' in node_properties[dependency_node]:
                    dependency_props = node_properties[dependency_node]
                    #for i in range(len(config['dependencies'])):
                    #    feed_dict[config['feed']['h_pred'][i]] = dependency_props['h_pred'][config['dependencies'][i]]

            for i in range(config['num_layers']):
                state = config['feed']['states'][dependency][i]
                output = config['feed']['outputs'][dependency][i]
                dependency_node = props['dependencies'][dependency] if dependency != 'children' else node
                # the second condition checks for the reverse dependencies while traveling
                # in the forward direction
                if dependency_node is not None:
                    dependency_props = node_properties[dependency_node]
                    if 'states' in dependency_props and dependency in dependency_props['states'] \
                                                    and i in dependency_props['states'][dependency]:
                        feed_dict[state['c']] = dependency_props['states'][dependency][i]['c']
                        feed_dict[state['h']] = dependency_props['states'][dependency][i]['h']
                    else:
                        feed_dict[state['c']] = initial['states'][dependency][i].c
                        feed_dict[state['h']] = initial['states'][dependency][i].h
                    if 'outputs' in dependency_props and dependency in dependency_props['outputs'] \
                                                    and i in dependency_props['outputs'][dependency]:
                        feed_dict[output] = dependency_props['outputs'][dependency][i]
                    else:
                        feed_dict[output] = initial['outputs'][dependency][i]

                if dependency == 'children':
                    if 'children_tmp_states' in props and i in props['children_tmp_states']:
                        feed_dict[config['feed']['children_tmp_states'][i]['c']] = props['children_tmp_states'][i]['c']
                        feed_dict[config['feed']['children_tmp_states'][i]['h']] = props['children_tmp_states'][i]['h']
                    else:
                        # this doesn't really do anything?
                        feed_dict[config['feed']['children_tmp_states'][i]['c']] = initial['children_tmp_states'][i].c
                        feed_dict[config['feed']['children_tmp_states'][i]['h']] = initial['children_tmp_states'][i].h

                    if 'children_predictor_states' in props and i in props['children_predictor_states']:
                        feed_dict[config['feed']['children_predictor_states'][i]['c']] = \
                                props['children_predictor_states'][i]['c']
                        feed_dict[config['feed']['children_predictor_states'][i]['h']] = \
                                props['children_predictor_states'][i]['h']
                    else:
                        feed_dict[config['feed']['children_predictor_states'][i]['c']] = \
                                initial['children_predictor_states'][i].c
                        feed_dict[config['feed']['children_predictor_states'][i]['h']] = \
                                initial['children_predictor_states'][i].h

            if dependency == 'children':
                if 'children_predictor_output' in props:
                    feed_dict[config['feed']['children_predictor_output']] = props['children_predictor_output']
                else:
                    feed_dict[config['feed']['children_predictor_output']] = initial['children_predictor_output']


        # TODO: this seems to run through the entire while loop each time, despite the fact that
        # we only want to do one layer at a time. Oh, that might actually be because
        # "dependency_outputs[i][layer-1].read(ctr)" requires processing earlier layers. But why do later ones
        # run for earlier layers, then??
        vals = session.run(fetches, feed_dict)

        if 'states' not in props:
            props['states'] = {}
        if 'outputs' not in props:
            props['outputs'] = {}
        if 'h_pred' not in props:
            props['h_pred'] = {}

        if 'children' in config['dependencies'] and direction == 'reverse':
            if props['dependencies']['left_sibling'] is not None:
                sibling_props = node_properties[props['dependencies']['left_sibling']]
                if 'children_tmp_states' not in sibling_props:
                    sibling_props['children_tmp_states'] = {}
                sibling_props['children_tmp_states'].update(vals['children_tmp_states'])
                if is_last_layer:
                    sibling_props['children_predictor_states'] = vals['children_predictor_states']
            elif props['dependencies']['parent'] is not None:
                parent_props = node_properties[props['dependencies']['parent']]
                if is_last_layer:
                    parent_props['children_predictor_output'] = vals['children_output']
                if 'states' not in parent_props:
                    parent_props['states'] = {}
                if 'children' not in parent_props['states']:
                    parent_props['states']['children'] = {}
                parent_props['states']['children'].update(vals['states']['children'])
                del(vals['states']['children'])

        #for k in vals:
        #    if k in props:
        #        props[k].update(vals[k])
        #    else:
        #        props[k] = vals[k]
        props['states'].update(vals['states'])
        props['outputs'].update(vals['outputs'])

        if is_last_layer:
            props['h_pred'].update(vals['h_pred'])
            # get ids of expected and actual labels
            for k in ['attr', 'label']:
                if k + '_expected' not in props:
                    props[k + '_expected_probability'] = {}
                    props[k + '_actual_probability'] = {}
                    props[k + '_ratio'] = {}

                probabilities = vals[k + '_probabilities']
                expected_id = np.argmax(probabilities)
                target_id = feed_dict[config['placeholders']['data'][k + '_index']][1]

                # should also indicate if it was cast to <unk> or something?
                props[k + '_expected'] = raw_data['id_to_' + k][expected_id]
                props[k + '_actual'] = raw_data['id_to_' + k][target_id]

                props[k + '_expected_probability'][direction] = probabilities[expected_id]
                props[k + '_actual_probability'][direction] = probabilities[target_id]
                props[k + '_ratio'][direction] = probabilities[target_id] / probabilities[expected_id]
            if 'predicted_end' not in props:
                props['predicted_end'] = {}
            props['predicted_end'][direction] = vals['predicted_end']

            cost += vals['cost']

        if direction == 'forward':
            children = node.children()
            for i in range(len(children)):
                cost += visit_tree(children[i][1], direction, layer)

        return cost

    # remove the nil token
    props.pop(0)
    for direction in directions:
        for j in range(config['num_layers']):
            cost = visit_tree(ast, direction, j)
    print(cost)


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

    raw_data['id_to_label'] = dict()
    for k in raw_data['label_to_id']:
        raw_data['id_to_label'][raw_data['label_to_id'][k]] = k

    raw_data['id_to_attr'] = dict()
    for k in raw_data['attr_to_id']:
        raw_data['id_to_attr'][raw_data['attr_to_id'][k]] = k

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
                        run_ast_epoch(session, graph, config, ast, node_properties, raw_data)
                        code = "blah"#search(ast, node_properties, filename, directives)
                        output = {
                            'ast': print_ast(ast, node_properties),
                            #'code': generator.visit(ast),
                            'fixed_code': code
                        }
                    else:
                        tokens.insert(0, '<nil>')
                        tokens = preprocess.tokens_to_ids(tokens, raw_data['label_to_id'], True, True)
                        props = run_linear_epoch(session, graph, config, tokens, raw_data)
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
