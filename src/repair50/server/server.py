import sys
import os
import json
from typing import Dict, Union

import numpy as np # type:ignore
import tensorflow as tf #type:ignore
import collections

from ..default_dict import data_dict
from ..model.config import joint_configs, dependency_configs, valid_dependencies #type:ignore
from ..wrangler.wrangle import finish_row, wrangle, process_ast #type:ignore
from .c_generator import CGenerator



#import check_correct
#import queue as Q
#max_changes = 3
# side effect: populate node_properties with parent pointers (not yet used?)
#def fill_queue(node, node_properties, q, parent=None):
#    node_properties[node]['parent'] = parent
#
#    score = node_properties[node]['attr_ratio']
#    # XXX for now, time.time() is supposed to make sure that we never get to comparing nodes
#    q.put((score, time.time(), node))
#    children = node.children()
#    for i in range(len(children)):
#        fill_queue(children[i][1], node_properties, q, node)
#
## XXX heuristics about class name?
#def search_changes(ast, node_properties, list_q, max_changes, filename, directives, start = 0, num_changes = 0):
#    for i in range(start, len(list_q)):
#        node = list_q[i][2]
#        # adjust this cutoff?
#        if node_properties[node]['attr_ratio'] == 1.0:
#            break
#        nvlist = [(n, getattr(node, n)) for n in node.attr_names]
#        for (name, val) in nvlist:
#            if name in ['value', 'op', 'name']:
#                setattr(node, name, node_properties[node]['attr_expected'])
#                if num_changes == max_changes - 1:
#                    #try:
#                        #code = directives + generator.visit(ast)
#                        path = os.path.join(FLAGS.task_path, '.' + filename + '.c')
#                        with open(path, 'w') as f:
#                            f.write(code)
#                        ret = check_correct.check_vigenere(path)
#                        os.unlink(path)
#                        if ret == 0:
#                            return code
#                    #except Exception:
#                    #    #print('uh ohhh')
#                    #    pass
#                else:
#                    ret = search_changes(ast, node_properties, list_q, max_changes, filename, directives, start=i+1, num_changes=num_changes+1)
#                    # Success! The ast is now repaired
#                    if ret is not False:
#                        return ret
#                # no luck, revert to the old value
#                setattr(node, name, val)
#                break
#    # didn't find a working tree
#    return False
#
#
#
#def search(ast, node_properties, filename, directives):
#    # XXX check if code already works?
#    #code = generator.visit(ast)
#    #path = os.path.join(FLAGS.task_path, '.' + filename + '.c')
#    #with open(path, 'w') as f:
#    #    f.write(code)
#    #ret = check_correct.check_vigenere(path)
#    #os.unlink(path)
#    #if ret == 0:
#    #    return code
#    q = Q.PriorityQueue()
#    fill_queue(ast, node_properties, q)
#    list_q = []
#    while not q.empty():
#        list_q.append(q.get())
#    for i in range(max_changes):
#        code = search_changes(ast, node_properties, list_q, i+1, filename, directives)
#        if code is not False:
#            return code
#    return False

class Server(object):

    # TODO: close self.session in Server deconstructor?
    # TODO: pass in the config directly instead of data_path
    def __init__(self, data_path, subtests) -> None:
        self.subtests = subtests

        with open(os.path.join(data_path, 'config.json')) as f:
            self.config = json.load(f)

        self.test_conf = data_dict(lambda: False) #type:ignore
        root_file = os.path.join(data_path, 'config.json')
        with open(root_file, 'r') as f:
            self.root_config = json.load(f)
        for test in self.config['tests']:
            if subtests and test not in subtests:
                for i in range(len(self.config['unit_tests'])):
                    if self.config['unit_tests'][i]['name'] == test:
                        #self.config['unit_tests'].pop(i)
                        break
                continue
            for root_idx in self.config['tests'][test]:
                for transitions in self.config['tests'][test][root_idx]:
                    model_file = os.path.join(data_path, 'tests', test, root_idx, transitions, 'best', 'config.json')
                    if not os.path.isfile(model_file): continue
                    print('Loading model {} {} {}'.format(test, root_idx, transitions))
                    with open(model_file, 'r') as f:
                        self.test_conf[test][root_idx][transitions] = test_conf = json.load(f)
                    self.dependency_configs = [dc for dc in test_conf['initials']['dependency_configs']]

                    test_conf['graph'] = tf.Graph()
                    with test_conf['graph'].as_default():
                        saver = tf.train.import_meta_graph(os.path.join(test_conf['best_dir'], "model.meta"))
                        test_conf['session'] = tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0}))
                        saver.restore(test_conf['session'], os.path.join(test_conf['best_dir'], 'model'))

        print('Server initialized')

    def gather_props(self, vals, data_dict, config, lexicon, node_id=None, node=None):
        revlex = lexicon['index_to_token']
        lex = lexicon['token_to_index']
        prop = collections.defaultdict(lambda: collections.defaultdict(lambda: {}))

        for dconfig in vals:
            for cdependency in vals[dconfig]:
                v = vals[dconfig][cdependency]
                direction = 'forward' if dconfig == 'joint_configs' or \
                            dependency_configs[cdependency][-1][0] else 'reverse'
                prop[dconfig][cdependency]['direction'] = direction
                if node is not None:
                    idx = node[direction]['self']
                else:
                    idx = 1
                if node_id is None:
                    node_id = idx

                if config['transitions'] == 'true':
                    token_target = data_dict[config['features'][direction + '-transitions_index']][0][idx]
                else:
                    token_target = data_dict[config['features'][direction + '-label_index']][0][idx]
                    attr_target = data_dict[config['features'][direction + '-attr_index']][0][idx]

                for dependency in v['cells']:
                    for k in v['cells'][dependency]:
                        if node_id not in config['cells'][dconfig][cdependency][dependency][k]:
                            config['cells'][dconfig][cdependency][dependency][k][node_id] = {}
                        cells = config['cells'][dconfig][cdependency][dependency][k][node_id]
                        if config['transitions']:
                            cells[token_target] = v['cells'][dependency][k][0][idx].tolist()
                        else:
                            if token_target not in cells:
                                cells[token_target] = {}
                            cells[token_target][attr_target] = v['cells'][dependency][k][0][idx].tolist()

                for k in vals[dconfig][cdependency]['loss']:
                    # attr_index handled within label_index
                    if k == 'attr_index': continue

                    prop[dconfig][cdependency][k] = p = {}

                    probs = v['probabilities'][k][0]
                    if dconfig == 'joint_configs':
                        p['alpha'] = alpha = probs[idx].tolist()
                        sum_probs = 0
                        for jd in range(len(joint_configs[cdependency])):
                            joint_dependency = joint_configs[cdependency][jd]
                            probs = vals['dependency_configs'][joint_dependency][k]['probabilities'][0][idx]
                            sum_probs += alpha[jd] * probs
                        probs = sum_probs
                    else:
                        probs = probs[idx]

                    if k == 'label_index' or k == 'transitions_index':

                        p['actual_probability'] = float(probs[token_target])
                        if config['transitions'] == 'false':
                            p['actual_probability'] *= float(v['probabilities']['attr_index'][0][idx][attr_target])

                        if k == 'label_index':
                            new_probs = []
                            for token_idx in range(len(probs)):
                                token = (float(probs[token_idx]), revlex['label'][str(token_idx)])

                                attr_probs = v['attr_all'][0][idx][token_idx]
                                new_probs.extend([(token[0] * float(attr_probs[j]),
                                                    token[1], revlex['attr'][str(j)]) for j in range(len(attr_probs))])
                            probs = new_probs
                            p['actual_attr'] = revlex['attr'][str(attr_target)]
                            p['actual_label'] = revlex['label'][str(token_target)]
                        else:
                            probs = [(float(probs[j]), revlex['transitions'][str(j)]) for j in range(len(probs))]
                            #probs = probs.tolist()
                            #print(config['transitions_groups'])
                            p['actual_transitions'] = revlex['transitions'][str(token_target)]
                            p['transitions_groups'] = config['transitions_groups'][p['actual_transitions']] if p['actual_transitions'] in config['transitions_groups'] else False

                        probs.sort(key=lambda x: x[0], reverse=True)
                        p['probabilities'] = [x for x in probs if x[0] > .001]
                        expected_probability = float(probs[0][0])
                        p['ratio'] = p['actual_probability'] / expected_probability
                    elif k == 'pointers':
                        p['actual'] = []
                        for q in range(20):
                            target = data_dict[config['features'][direction + '-pointers-mask-' + str(q)]][0][idx]
                            p['actual'].append(target)
                        p['expected'] = probs.tolist()
                    else:
                        target = data_dict[config['features'][direction + '-' + k]][0][idx]
                        p['actual'] = target
                        p['expected'] = probs.tolist()

        return prop

    def get_cells(self, node_id, prop, data_dict, initials_dict, test):
        for dconfig in prop:
            for cdependency in prop[dconfig]:
                # TODO: this has different labels/attrs for each dependency config
                for k in ['forward', 'reverse']:
                    for j in ['attr', 'label']:
                        token = prop[dconfig][cdependency]['label_index']['expected_' + j]
                        index = self.config['lexicon'][test][j][token]
                        data_dict[self.config['features'][k + '-' + j + '_index']] = [[0, index]]

                config['session'].run(self.config['tensor_iter'], data_dict)
                vals = config['session'].run(self.config['fetches'], initials_dict)
                prop = self.gather_props(vals, data_dict, node_id=node_id)
                return prop

    def beam_step(self, props, test=None):#, label, attr, transition, dependencies):
        config = self.config

        data_dict = {}
        initials_dict = {}
        for k in props['forward']:
            for dc in config['initials']['dependency_configs']:
                if k in config['initials']['dependency_configs'][dc]:
                    dep_id = props['forward'][k]
                    if dep_id == 0:
                        dep_label_idx = 0
                        dep_attr_idx = 0
                    else:
                        dep_prop = self.data.prop_map[dep_id][test]
                        dep_label = dep_prop['dependency_configs'][dc]['label_index']['actual_label']
                        dep_attr = dep_prop['dependency_configs'][dc]['label_index']['actual_attr']
                        dep_label_idx = self.config['lexicon'][test]['label'][dep_label]
                        dep_attr_idx = self.config['lexicon'][test]['attr'][dep_attr]
                    for q in config['initials']['dependency_configs'][dc][k]:
                        initials_dict[config['initials']['dependency_configs'][dc][k][q]] = \
                            [config['cells']['dependency_configs'][dc][k][q][dep_id][dep_label_idx][dep_attr_idx]]
            for direction in ['forward', 'reverse']:
                key = direction + '-' + k
                if key not in config['features']: continue
                if k in valid_dependencies.keys():
                    data_dict[config['features'][key]] = [[0, 0]]
                else:
                    data_dict[config['features'][key]] = [[0, props[direction][k]]]

        for direction in ['forward', 'reverse']:
            data_dict[config['features'][direction + '-mask']] = [[0, 1]]

        for k in config['features']:
            if config['features'][k] not in data_dict:
                data_dict[config['features'][k]] = [[0, 0]]
        config['session'].run(config['tensor_iter'], data_dict)
        # need to pass in inital cell values here
        vals = config['session'].run(config['fetches'], initials_dict)
        node_id = self.data.num_nodes
        self.data.num_nodes+=1
        prop = self.gather_props(vals, data_dict, node_id=node_id)
        prop = self.get_cells(node_id, prop, data_dict, initials_dict, test)
        return node_id, prop

    def beam_row(self, parent_node_id, direction, test, dconfig, cdependency):
        row = []
        while True:
            props = { 'forward': {}, 'reverse': {} }
            props[direction]['parent'] = parent_node_id
            props[direction]['left_sibling' if direction == 'forward' else 'right_sibling'] = \
                    row[-1] if len(row) > 0 else 0
            node_id, prop = self.beam_step(props)
            props.update(prop)
            if node_id not in self.data.prop_map:
                self.data.prop_map[node_id] = {}
            self.data.prop_map[node_id]['props'][test] = props
            row.append(node_id)
            if prop[dconfig][cdependency]['last_sibling' if direction == 'forward' else 'first_sibling']['expected'] > 0.5:
                break
        if direction == 'reverse':
            row.reverse()
        return row

    def beam(self, dconfig, cdependency, node_id=39, test=None):
        direction = 'forward' if dconfig == 'joint_configs' or \
                    dependency_configs[cdependency][-1][0] else 'reverse'

        queue = [node_id]
        while len(queue) > 0:
            node_id = queue.pop(0)
            row = self.beam_row(node_id, direction, test, dconfig, cdependency)
            # TODO: learn these from the data
            # ExpressionList? Have an attr that is if it is empty or not
            for node_id in row:
                prop = self.data.prop_map[node_id]['props'][test][dconfig][cdependency]
                if prop['label_index']['actual_label'] not in ['Constant', 'IdentifierType', 'ID', 'ExpressionList', 'Exprlist']:
                    queue.append(node_id)



    def infer(self, data):
        rows = process_ast(data)
        import pprint

        for test in self.test_conf:
            for root_node in rows[test]:
                for transitions in rows[test][root_node]:
                    if len(rows[test][root_node][transitions]) == 0: continue

                    root_transitions = data.prop_map[root_node]['props'][test][root_node][transitions]['transitions']
                    if root_transitions == '<unk>': continue
                    transitions_ = 'true' if transitions else 'false'
                    root_lex = self.root_config['root_lexicon'][test]['token_to_index']
                    if root_transitions not in root_lex['transitions']: continue
                    root_idx = str(root_lex['transitions'][root_transitions])

                    test_conf = self.test_conf[test][root_idx]
                    if not test_conf or transitions_ not in test_conf: continue
                    test_conf = test_conf[transitions_]

                    print('Running model {} {} {}'.format(test, root_idx, transitions))
                    lexicon = test_conf['lexicon']
                    row = finish_row(rows[test][root_node][transitions], lexicon['token_to_index'], root_lex)
                    data_dict = {}
                    for k in test_conf['features']:
                        data_dict[test_conf['features'][k]] = [row[k]]

                    test_conf['session'].run(test_conf['tensor_iter'], data_dict)

                    vals = test_conf['session'].run(test_conf['fetches'])

                    nodes = data.nodes[test][root_node][transitions]['forward']
                    for i in range(len(nodes)):
                        nodes[i].update(self.gather_props(vals, data_dict, test_conf, lexicon, node=nodes[i]))


    def process_code(self, code):
        data = wrangle(code, tests=self.config['unit_tests'], is_file=False)
        try:
            data = wrangle(code, tests=self.config['unit_tests'], is_file=False)
        except Exception as e:
            print(str(e))
            # TODO
            return {'error': "Couldn't parse code." + str(e)}

        self.data = data

        self.infer(data)
        #self.beam('dependency_configs', 'd2')

        output = {
            'code': CGenerator(data).code,
            'test_results': data.results,
            'dependency_configs': self.dependency_configs,
            'props': data.prop_map
        }
        return output
