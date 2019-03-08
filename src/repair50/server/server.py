import sys
import os
import json
from typing import Dict, Union

import numpy as np # type:ignore
import tensorflow as tf #type:ignore
import collections

from ..model.config import joint_configs, dependency_configs, valid_dependencies #type:ignore
from ..wrangler import finish_row, wrangle, process_ast #type:ignore
from .c_generator import CGenerator

nested_dict = lambda: collections.defaultdict(nested_dict) #type:ignore


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
    def __init__(self, data_path) -> None:
        best_dir = os.path.join(data_path, 'best')
        with open(os.path.join(best_dir, 'config.json')) as f:
            self.config = json.load(f)
        self.config['best_dir'] = best_dir

        # we have to find the model that we can feed...
        self.config['fetches'] = fetches = {} # type: ignore
        self.config['initials'] = initials = {} # type: ignore
        #self.config['feed_dict'] = feed = {} # type: ignore
        for d in self.config['models']:
            fetches[d] = {}
            initials[d] = {}
            for i in self.config['models'][d]:
                #feed[self.config['models'][d][i]['placeholders']['is_inference']] = False
                fetches[d][i] = self.config['models'][d][i]['fetches']
                initials[d][i] = self.config['models'][d][i]['initials']

                for j in self.config['models'][d][i]['placeholders']:
                    if 'features' == j:
                        self.config['features'] = self.config['models'][d][i]['placeholders'][j]
                        self.config['tensor_iter'] = self.config['models'][d][i]['ops']['tensor_iter']

        if 'features' not in self.config:
            print('yikes')
            return

        # fix windows path separators TODO: this isn't particularly portable?
        self.config['data_path'] = os.path.join(*self.config['data_path'].split('\\'))

        with open(os.path.join(self.config['data_path'], self.config['model'] + '_lexicon.json')) as f:
            token_ids = json.load(f)
        token_ids[None] = token_ids['null']
        del(token_ids['null'])


        for test in token_ids:
            for k in list(token_ids[test].keys()):
                token_ids[test][k + '_index_to_token'] = {}
                for token in token_ids[test][k]:
                    token_ids[test][k + '_index_to_token'][token_ids[test][k][token]] = token
        self.config['lexicon'] = token_ids

        self.graph = tf.Graph()
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(os.path.join(self.config['best_dir'], "model.meta"))
            self.session = tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0}))
            saver.restore(self.session, os.path.join(self.config['best_dir'], 'model'))

        initial_vals = self.session.run(self.config['initials'])
        self.cells = {} # type:ignore
        for dconfig in initial_vals:
            self.cells[dconfig] = {}
            for cdependency in initial_vals[dconfig]:
                self.cells[dconfig][cdependency] = {}
                for dependency in initial_vals[dconfig][cdependency]:
                    self.cells[dconfig][cdependency][dependency] = {}
                    for k in initial_vals[dconfig][cdependency][dependency]:
                        self.cells[dconfig][cdependency][dependency][k] = {
                                0: { 0: { 0: initial_vals[dconfig][cdependency][dependency][k][0] } }
                        }




    def gather_props(self, vals, data_dict, node_id=None, node=None):
        config = self.config
        lexicon = config['lexicon'][None]
        prop = nested_dict()

        for dconfig in vals:
            for cdependency in vals[dconfig]:
                v = vals[dconfig][cdependency]
                direction = 'forward' if dconfig == 'joint_configs' or \
                            dependency_configs[config['model']][cdependency][-1][0] else 'reverse'
                prop[dconfig][cdependency]['direction'] = direction
                if node is not None:
                    idx = node[direction]['self']
                else:
                    idx = 1
                if node_id is None:
                    node_id = idx

                label_target = data_dict[config['features'][direction + '-label_index']][0][idx]
                attr_target = data_dict[config['features'][direction + '-attr_index']][0][idx]

                for dependency in v['cells']:
                    for k in v['cells'][dependency]:
                        if node_id not in self.cells[dconfig][cdependency][dependency][k]:
                            self.cells[dconfig][cdependency][dependency][k][node_id] = {}
                        cells = self.cells[dconfig][cdependency][dependency][k][node_id]
                        if label_target not in cells:
                            cells[label_target] = {}
                        cells[label_target][attr_target] = v['cells'][dependency][k][0][idx].tolist()

                for k in vals[dconfig][cdependency]['loss']:
                    # attr_index handled within label_index
                    if k == 'attr_index': continue

                    p = {}

                    probs = v['probabilities'][k][0]
                    if dconfig == 'joint_configs':
                        p['alpha'] = alpha = probs[idx].tolist()
                        sum_probs = 0
                        for jd in range(len(joint_configs[config['model']][cdependency])):
                            joint_dependency = joint_configs[config['model']][cdependency][jd]
                            probs = vals['dependency_configs'][joint_dependency][k]['probabilities'][0][idx]
                            sum_probs += alpha[jd] * probs
                        probs = sum_probs
                    else:
                        probs = probs[idx]

                    if k == 'label_index':

                        new_probs = []
                        for label_idx in range(len(probs)):
                            label = (float(probs[label_idx]), lexicon['label_index_to_token'][label_idx])
                            attr_probs = v['attr_all'][0][idx][label_idx]
                            new_probs.extend([(label[0] * float(attr_probs[j]),
                                                label[1], lexicon['attr_index_to_token'][j]) for j in range(len(attr_probs))])

                        new_probs.sort(key=lambda x: x[0], reverse=True)
                        p['probabilities'] = [x for x in new_probs if x[0] > .0001]

                        #p['expected_subtree'] = self.generate_subtree(
                        p['expected_label'] = new_probs[0][1]
                        p['expected_attr'] = new_probs[0][2]
                        p['actual_label'] = lexicon['label_index_to_token'][label_target]
                        p['actual_attr'] = lexicon['attr_index_to_token'][attr_target]
                        p['expected_probability'] = float(new_probs[0][0])
                        p['actual_probability'] = float(probs[label_target]) * \
                                                    float(v['probabilities']['attr_index'][0][idx][attr_target])
                        p['ratio'] = p['actual_probability'] / p['expected_probability']
                    elif k == 'pointers':
                        p['actual'] = []
                        for q in range(20):
                            target = data_dict[config['features'][direction + '-pointers-mask-' + str(q)]][0][idx]
                            p['actual'].append(target)
                        p['expected'] = probs[idx].tolist()
                    else:
                        target = data_dict[config['features'][direction + '-' + k]][0][idx]
                        p['actual'] = target
                        p['expected'] = float(probs)

                    prop[dconfig][cdependency][k] = p
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

                self.session.run(self.config['tensor_iter'], data_dict)
                vals = self.session.run(self.config['fetches'], initials_dict)
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
                            [self.cells['dependency_configs'][dc][k][q][dep_id][dep_label_idx][dep_attr_idx]]
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
        self.session.run(config['tensor_iter'], data_dict)
        # need to pass in inital cell values here
        vals = self.session.run(config['fetches'], initials_dict)
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
            self.data.prop_map[node_id][test] = props
            row.append(node_id)
            if prop[dconfig][cdependency]['last_sibling' if direction == 'forward' else 'first_sibling']['expected'] > 0.5:
                break
        if direction == 'reverse':
            row.reverse()
        return row

    def beam(self, dconfig, cdependency, node_id=39, test=None):
        direction = 'forward' if dconfig == 'joint_configs' or \
                    dependency_configs[self.config['model']][cdependency][-1][0] else 'reverse'

        queue = [node_id]
        while len(queue) > 0:
            node_id = queue.pop(0)
            row = self.beam_row(node_id, direction, test, dconfig, cdependency)
            # TODO: learn these from the data
            # ExpressionList? Have an attr that is if it is empty or not
            for node_id in row:
                prop = self.data.prop_map[node_id][test][dconfig][cdependency]
                if prop['label_index']['actual_label'] not in ['Constant', 'IdentifierType', 'ID', 'ExpressionList', 'Exprlist']:
                    queue.append(node_id)



    def infer(self, data):
        config = self.config

        rows = process_ast(data, lexicon={'ast':config['lexicon']})
        rows = finish_row(rows, config['lexicon'])

        for test in config['lexicon']:
            for i in range(len(data.nodes[test]['forward'])):
                transition = data.nodes[test]['forward'][i]['transitions']
                data.nodes[test]['forward'][i]['transitions_percentage'] = config['lexicon'][test]['transition_percentages'][transition] if transition in config['lexicon'][test]['transition_percentages'] else 0

        test = None
        data_dict = {}
        for k in config['features']:
            data_dict[config['features'][k]] = [rows[test][k]]

        self.session.run(config['tensor_iter'], data_dict)

        vals = self.session.run(config['fetches'])

        nodes = data.nodes[test]['forward']
        for i in range(len(nodes)):
            nodes[i].update(self.gather_props(vals, data_dict, node=nodes[i]))


    def process_code(self, code):
        try:
            ast_data, linear_data = wrangle(code, tests=self.config['unit_tests'], is_file=False)
        except Exception as e:
            # TODO
            return {'error': "Couldn't parse code." + str(e)}

        data = ast_data if self.config['model'] == 'ast' else linear_data
        self.data = data

        self.infer(data)
        self.beam('dependency_configs', 'd2')

        output = {
            'code': CGenerator(data).code,
            'test_results': data.results,
            'props': data.prop_map
        }
        return output
