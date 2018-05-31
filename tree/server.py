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
import numpy as np
import tensorflow as tf
import queue as Q

#import check_correct
import dump_ast
from model import joint_configs
from pycparser import c_parser, c_ast, parse_file

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
        if k in ['self', 'dependencies', 'cells']:
            continue

        output[k] = props[k]

    children = ast.children()
    for i in range(len(children)):
        ret = print_ast(children[i][1], node_properties)
        if ret is not False:
            output['children'].append(ret)
    return output

def feed_dict_filler(feed_dict, dependency, initial_names, initial_values):
    if isinstance(initial_names, dict):
        for k in initial_names:
            feed_dict_filler(feed_dict, dependency[k] if dependency is not None else None, initial_names[k], initial_values[k])
    elif isinstance(initial_names, list):
        for k in range(len(initial_names)):
            feed_dict_filler(feed_dict, dependency[k] if dependency is not None else None, initial_names[k], initial_values[k])
    else:
        feed_dict[initial_names] = dependency if dependency is not None else initial_values

def subtree(session, config, raw_data, dependencies):

    # TODO: clean this up. really, it's three steps. first, we need to find the label index. then attr_index. then, we need
    # to calculate the state and output given that index

    # TODO: can we get rid of this if all we fetch are probabilities?
    data_dict = {}
    for k in config['features']:
        data_dict[config['features'][k]] = [[0, 0]]
    session.run(config['tensor_iter'], data_dict)

    feed_dict = config['feed_dict'].copy()
    feed_dict_filler(feed_dict, dependencies, config['initials'], config['initial_values'])
    vals = session.run(config['fetches'], feed_dict)

    props = { 'children': [], 'cells': {} }

    # TODO: this should use a specific direction, but also check if things are fine coming from the
    # other directions?
    for dconfig in vals:
        for dependency in vals[dconfig]:
            probs = vals[dconfig][dependency]['probabilities']['label_index'][0][1]
            rank = np.flip(np.argsort(probs), 0)
            props['label_index'] = {}
            props['label_index']['expected'] = raw_data['label_index_to_token'][rank[0]]
            props['label_index']['expected_index'] = rank[0]
            props['label_index']['expected_probability'] = float(probs[rank[0]])
            props['label_index']['probabilities'] = [(float(probs[j]), raw_data['label_index_to_token'][j]) for j in rank]

    data_dict = {}
    for k in config['features']:
        val = props[k]['expected_index'] if k in ['label_index'] else 0
        data_dict[config['features'][k]] = [[0, val]]
    session.run(config['tensor_iter'], data_dict)
    vals = session.run(config['fetches'], feed_dict)

    for dconfig in vals:
        for dependency in vals[dconfig]:
            probs = vals[dconfig][dependency]['probabilities']['attr_index'][0][1]
            rank = np.flip(np.argsort(probs), 0)
            props['attr_index'] = {}
            props['attr_index']['expected'] = raw_data['attr_index_to_token'][rank[0]]
            props['attr_index']['expected_index'] = rank[0]
            props['attr_index']['expected_probability'] = float(probs[rank[0]])
            props['attr_index']['probabilities'] = [(float(probs[j]), raw_data['attr_index_to_token'][j]) for j in rank]

    data_dict = {}
    for k in config['features']:
        val = props[k]['expected_index'] if k in ['label_index', 'attr_index'] else 0
        data_dict[config['features'][k]] = [[0, val]]
    session.run(config['tensor_iter'], data_dict)
    vals = session.run(config['fetches'], feed_dict)

    for dconfig in vals:
        props['cells'][dconfig] = {}
        for dependency in vals[dconfig]:
            props['cells'][dconfig][dependency] = prop = {}
            for k in vals[dconfig][dependency]['cells']:
                prop[k] = {
                    'output': [vals[dconfig][dependency]['cells'][k]['output'][0][1]],
                    'states': []
                }
                for layer in range(config['num_layers']):
                    prop[k]['states'].append({
                        'c': [vals[dconfig][dependency]['cells'][k]['states'][layer]['c'][0][1]],
                        'h': [vals[dconfig][dependency]['cells'][k]['states'][layer]['h'][0][1]]
                    })
            for k in vals[dconfig][dependency]['probabilities']:
                probs = vals[dconfig][dependency]['probabilities'][k][0][1]
                rank = np.flip(np.argsort(probs), 0)
                if k not in ['label_index', 'attr_index']:
                    props[k] = {'expected': float(probs)}


    # TODO: return can also fit here, but only if type is void
    # TODO: gather these from code rather than hard-coding
    print(props['label_index']['expected'], props['attr_index']['expected'], props['label_index']['expected_probability'])
    if props['label_index']['expected'] not in ['Break', 'Continue', 'ID', 'Constant', 'IdentifierType']:
        while True:
            child_dependencies = {}
            for dconfig in vals:
                child_dependencies[dconfig] = {}
                for dependency in vals[dconfig]:
                    child_dependencies[dconfig][dependency] = {
                        'parent': props['cells'][dconfig][dependency]['parent'],
                        'left_sibling': props['children'][-1]['cells'][dconfig][dependency]['left_sibling'] \
                                        if len(props['children']) > 0 else None
                    }
            print('down')
            child_props = subtree(session, config, raw_data, child_dependencies)
            print('up')
            props['children'].append(child_props)
            if child_props['last_sibling']['expected'] > 0.5:
                break

    return props

def step(session, config, tokens, raw_data, node_properties=None):

    token = tokens[2]

    dependencies = {}

    feed_dict = config['feed_dict'].copy()
    for dconfig in config['initials']:
        dependencies[dconfig] = {}
        for dependency in config['initials'][dconfig]:
            dependencies[dconfig][dependency] = {}
            for k in config['initials'][dconfig][dependency]:
                d = node_properties[token['self']]['dependencies'][k]
                if d is not None:
                    d = node_properties[d]['cells'][dconfig][dependency][k]
                dependencies[dconfig][dependency][k] = d

    subtree(session, config, raw_data, dependencies)





def run_epoch(session, config, tokens, raw_data, node_properties=None):

    data_dict = {}
    if config['model'] == 'linear':
        for k in config['features']:
            p = config['features'][k]
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
            p = config['features'][k]
            data_dict[p] = [[0]]
        for token in tokens:
            for (name, val) in token['attrs']:
                if name in ['value', 'op', 'name']:
                    # TODO just look up directly, and do the string check, too
                    token['attr'] = val
                    break
            else:
                token['attr'] = '<no_attr>'
            token['attr_index'] = preprocess.tokens_to_ids([token['attr']], raw_data['attr_to_id'], True, False)[0]
            token['label_index'] = raw_data['label_to_id'][token['label']]
            token['mask'] = 1
            for k in config['features']:
                p = config['features'][k]
                data_dict[p][0].append(token[k])


    session.run(config['tensor_iter'], data_dict)

    vals = session.run(config['fetches'], config['feed_dict'])
    for k in config['feed_dict']:
        config['feed_dict'][k] = True

    props = node_properties if node_properties is not None else []
    for i in range(len(tokens)):
        token = tokens[i][1] if config['model'] == 'linear' else tokens[i]['label']
        prop = {'label': token, 'cells': {}}
        for dconfig in vals:
            prop[dconfig] = {}
            prop['cells'][dconfig] = {}
            for dependency in vals[dconfig]:
                prop['cells'][dconfig][dependency] = {}
                for k in vals[dconfig][dependency]['cells']:
                    prop['cells'][dconfig][dependency][k] = {
                        'output': [vals[dconfig][dependency]['cells'][k]['output'][0][i+1]],
                        'states': []
                    }
                    for layer in range(config['num_layers']):
                        prop['cells'][dconfig][dependency][k]['states'].append({
                            'c': [vals[dconfig][dependency]['cells'][k]['states'][layer]['c'][0][i+1]],
                            'h': [vals[dconfig][dependency]['cells'][k]['states'][layer]['h'][0][i+1]]
                        })

                prop[dconfig][dependency] = {}
                for k in vals[dconfig][dependency]['loss']:
                    if i == 0:
                        print('{} {} {} perplexity: {}'.format(dconfig, dependency,
                                k, np.exp(vals[dconfig][dependency]['loss'][k])))

                    prop[dconfig][dependency][k] = p = {}

                    probs = vals[dconfig][dependency]['probabilities'][k][0]
                    target = data_dict[config['features'][k]][0]
                    if dconfig == 'joint_configs':
                        p['alpha'] = alpha = probs[i+1].tolist()
                        sum_probs = 0
                        for jd in range(len(joint_configs[config['model']][dependency])):
                            joint_dependency = joint_configs[config['model']][dependency][jd]
                            probs = vals['dependency_configs'][joint_dependency][k]['probabilities'][0][i+1]
                            sum_probs += alpha[jd] * probs
                        probs = sum_probs
                        #p['actual'] = target[i+1]
                        #p['expected'] = float(probs)
                    if k in ['label_index', 'attr_index']:
                        if dconfig != 'joint_configs':
                            probs = probs[i+1]
                        rank = np.flip(np.argsort(probs), 0)
                        p['expected'] = raw_data[k + '_to_token'][rank[0]]
                        p['actual'] = raw_data[k + '_to_token'][target[i+1]]
                        p['expected_probability'] = float(probs[rank[0]])
                        p['actual_probability'] = float(probs[target[i+1]])
                        p['ratio'] = float(probs[target[i+1]] / (probs[rank[0]]))
                        p['probabilities'] = [(float(probs[j]), raw_data[k+'_to_token'][j]) for j in rank]
                    else:
                        p['actual'] = target[i+1]
                        p['expected'] = float(probs[i+1])

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

    # we have to find the model that we can feed...
    config['fetches'] = fetches = {}
    config['initials'] = initials = {}
    config['feed_dict'] = feed = {}
    for d in config['models']:
        fetches[d] = {}
        initials[d] = {}
        for i in config['models'][d]:
            feed[config['models'][d][i]['placeholders']['is_inference']] = False
            fetches[d][i] = config['models'][d][i]['fetches']
            initials[d][i] = {}
            for j in config['models'][d][i]['initials']:
                initials[d][i][j] = config['models'][d][i]['initials'][j]
            for j in config['models'][d][i]['placeholders']:
                if 'features' == j:
                    config['features'] = config['models'][d][i]['placeholders'][j]
                    config['tensor_iter'] = config['models'][d][i]['ops']['tensor_iter']

    if 'features' not in config:
        print('yikes')
        sys.exit(0)

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

            config['initial_values'] = session.run(config['initials'])

            parser = c_parser.CParser()
            while True:
                task_path = os.path.join(FLAGS.task_path, config['model'])
                for filename in os.listdir(task_path):
                    if filename.startswith('.'):
                        continue
                    filepath = os.path.join(task_path, filename)
                    try:
                        with open(filepath) as f:
                            text = f.read()
                        directives, _ = preprocess.grab_directives(text)
                        # XXX this can return None
                        ast_nodes, ast, node_properties, tokens = preprocess.preprocess_c(filepath,
                                include_dependencies=True)
                    except Exception(e):
                        print(e)
                        try:
                            os.unlink(filepath)
                        except FileNotFoundError:
                            pass
                        # report some kind of error!
                        continue

                    if config['model'] == 'ast':
                        # Can't add attributes directly to Nodes because the class uses __slots__, so use this dictionary to
                        # extend objects
                        #run_ast_epoch(session, graph, config, ast, ast_nodes, node_properties, raw_data)
                        run_epoch(session, config, ast_nodes, raw_data, node_properties)
                        step(session, config, ast_nodes, raw_data, node_properties)
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
