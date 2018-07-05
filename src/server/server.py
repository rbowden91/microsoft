from ..my_env import sys
from ..my_env import os
from ..my_env import json
from ..my_env.typing import Dict, Union

from ..my_env.packages import numpy as np
from ..my_env.packages import tensorflow as tf

from ..model.config import joint_configs, dependency_configs
from ..wrangler.preprocess import finish_row, wrangle, process_ast


# TODO: better thing to extend from than Exception?
class ServerException(Exception): pass

class ServerInput(object):
    pass

class ServerOutput(object):
    def __init__(self, error : bool) -> None:
        self.error = error

def server_json_decoder(obj : Dict[str, str]) -> ServerInput:
    if 'type' not in obj:
        raise ServerException("JSON input must have a 'type' field.")
    return ServerInput()

def server_input(code) -> ServerInput:
    try:
        return json.loads(code, object_hook=server_json_decoder)
    except json.JSONDecodeError as err:
        raise ServerException("Invalid command. Failed to parse command as JSON. {}".format(err))
    except ServerException as err:
        raise ServerException(str(err))

def server_serializer(obj: Union[ServerOutput, dict]) -> dict:
    return obj.__dict__

def server_output(response : ServerOutput):
    return json.dumps(response, default=server_serializer)


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

def print_ast(ast, node_properties):

    # TODO: is this possible anymore, given how we're doing normalizations???
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
            feed_dict_filler(feed_dict, dependency[k] if dependency is not None else None,
                             initial_names[k], initial_values[k])
    elif isinstance(initial_names, list):
        for k in range(len(initial_names)):
            feed_dict_filler(feed_dict, dependency[k] if dependency is not None else None,
                             initial_names[k], initial_values[k])
    else:
        feed_dict[initial_names] = dependency if dependency is not None else initial_values

class Server(object):
    def __init__(self, config, session, graph) -> None:
        self.config = config
        self.session = session
        self.graph = graph

    #def subtree(self, dependencies):

    #    # TODO: clean this up. really, it's three steps. first, we need to find the label index. then attr_index. then, we need
    #    # to calculate the state and output given that index

    #    # TODO: can we get rid of this if all we fetch are probabilities?
    #    data_dict = {}
    #    for k in config['features']:
    #        data_dict[config['features'][k]] = [[0, 0]]
    #    session.run(config['tensor_iter'], data_dict)

    #    feed_dict = config['feed_dict'].copy()
    #    feed_dict_filler(feed_dict, dependencies, config['initials'], config['initial_values'])
    #    vals = session.run(config['fetches'], feed_dict)

    #    props = { 'children': [], 'cells': {} }

    #    # TODO: this should use a specific direction, but also check if things are fine coming from the
    #    # other directions?
    #    for dconfig in vals:
    #        for dependency in vals[dconfig]:
    #            probs = vals[dconfig][dependency]['probabilities']['label_index'][0][1]
    #            rank = np.flip(np.argsort(probs), 0)
    #            props['label_index'] = {}
    #            props['label_index']['expected'] = raw_data['label_index_to_token'][rank[0]]
    #            props['label_index']['expected_index'] = rank[0]
    #            props['label_index']['expected_probability'] = float(probs[rank[0]])
    #            props['label_index']['probabilities'] = [(float(probs[j]), raw_data['label_index_to_token'][j]) for j in rank]

    #    data_dict = {}
    #    for k in config['features']:
    #        val = props[k]['expected_index'] if k in ['label_index'] else 0
    #        data_dict[config['features'][k]] = [[0, val]]
    #    session.run(config['tensor_iter'], data_dict)
    #    vals = session.run(config['fetches'], feed_dict)

    #    for dconfig in vals:
    #        for dependency in vals[dconfig]:
    #            probs = vals[dconfig][dependency]['probabilities']['attr_index'][0][1]
    #            rank = np.flip(np.argsort(probs), 0)
    #            props['attr_index'] = {}
    #            props['attr_index']['expected'] = raw_data['attr_index_to_token'][rank[0]]
    #            props['attr_index']['expected_index'] = rank[0]
    #            props['attr_index']['expected_probability'] = float(probs[rank[0]])
    #            props['attr_index']['probabilities'] = [(float(probs[j]), raw_data['attr_index_to_token'][j]) for j in rank]

    #    data_dict = {}
    #    for k in config['features']:
    #        val = props[k]['expected_index'] if k in ['label_index', 'attr_index'] else 0
    #        data_dict[config['features'][k]] = [[0, val]]
    #    session.run(config['tensor_iter'], data_dict)
    #    vals = session.run(config['fetches'], feed_dict)

    #    for dconfig in vals:
    #        props['cells'][dconfig] = {}
    #        for dependency in vals[dconfig]:
    #            props['cells'][dconfig][dependency] = prop = {}
    #            for k in vals[dconfig][dependency]['cells']:
    #                prop[k] = {
    #                    'output': [vals[dconfig][dependency]['cells'][k]['output'][0][1]],
    #                    'states': []
    #                }
    #                for layer in range(config['num_layers']):
    #                    prop[k]['states'].append({
    #                        'c': [vals[dconfig][dependency]['cells'][k]['states'][layer]['c'][0][1]],
    #                        'h': [vals[dconfig][dependency]['cells'][k]['states'][layer]['h'][0][1]]
    #                    })
    #            for k in vals[dconfig][dependency]['probabilities']:
    #                probs = vals[dconfig][dependency]['probabilities'][k][0][1]
    #                rank = np.flip(np.argsort(probs), 0)
    #                if k not in ['label_index', 'attr_index']:
    #                    props[k] = {'expected': float(probs)}


    #    # TODO: return can also fit here, but only if type is void
    #    # TODO: gather these from code rather than hard-coding
    #    print(props['label_index']['expected'], props['attr_index']['expected'], props['label_index']['expected_probability'])
    #    if props['label_index']['expected'] not in ['Break', 'Continue', 'ID', 'Constant', 'IdentifierType']:
    #        while True:
    #            child_dependencies = {}
    #            for dconfig in vals:
    #                child_dependencies[dconfig] = {}
    #                for dependency in vals[dconfig]:
    #                    child_dependencies[dconfig][dependency] = {
    #                        'parent': props['cells'][dconfig][dependency]['parent'],
    #                        'left_sibling': props['children'][-1]['cells'][dconfig][dependency]['left_sibling'] \
    #                                        if len(props['children']) > 0 else None
    #                    }
    #            print('down')
    #            child_props = subtree(session, config, raw_data, child_dependencies)
    #            print('up')
    #            props['children'].append(child_props)
    #            if child_props['last_sibling']['expected'] > 0.5:
    #                break

    #    return props


    #def step(self, tokens, raw_data, node_properties=None):

    #    token = tokens[2]

    #    dependencies = {}

    #    feed_dict = config['feed_dict'].copy()
    #    for dconfig in config['initials']:
    #        dependencies[dconfig] = {}
    #        for dependency in config['initials'][dconfig]:
    #            dependencies[dconfig][dependency] = {}
    #            for k in config['initials'][dconfig][dependency]:
    #                d = node_properties[token['self']]['dependencies'][k]
    #                if d is not None:
    #                    d = node_properties[d]['cells'][dconfig][dependency][k]
    #                dependencies[dconfig][dependency][k] = d

    #    self.subtree(raw_data, dependencies)





    def infer(self, ast_data, linear_data):
        config = self.config

        if config['model'] == 'linear':
            # XXX XXX pass in the include_token here
            tokens = finish_row(data, config['raw_data'], config['features'].keys())
        else:
            node_properties = ast_data.node_properties
            tokens = ast_data.nodes['forward']
            rows = process_ast(ast_data)
            rows = finish_row(rows, config['raw_data'])

        data_dict = {}
        for k in rows:
            # skip over dependencies
            if k in config['features']:
                data_dict[config['features'][k]] = [rows[k]]

        self.session.run(config['tensor_iter'], data_dict)

        vals = self.session.run(config['fetches'], config['feed_dict'])
        for k in config['feed_dict']:
            config['feed_dict'][k] = True

        props = tokens if tokens is not None else []
        for i in range(len(tokens)):
            token = tokens[i][1] if config['model'] == 'linear' else tokens[i]['label']
            prop = {'label': token, 'cells': {}}
            for dconfig in vals:
                prop[dconfig] = {}
                prop['cells'][dconfig] = {}
                for dependency in vals[dconfig]:
                    prop['cells'][dconfig][dependency] = {}
                    for k in vals[dconfig][dependency]['cells']:
                        direction = 'forward' if dconfig == 'joint_configs' or \
                                    dependency_configs[config['model']][dependency][-1][0] else 'reverse'
                        idx = tokens[i][direction]['self']
                        prop['cells'][dconfig][dependency][k] = {
                            'output': [vals[dconfig][dependency]['cells'][k]['output'][0][idx]],
                            'states': []
                        }
                        for layer in range(config['num_layers']):
                            prop['cells'][dconfig][dependency][k]['states'].append({
                                'c': [vals[dconfig][dependency]['cells'][k]['states'][layer]['c'][0][idx]],
                                'h': [vals[dconfig][dependency]['cells'][k]['states'][layer]['h'][0][idx]]
                            })

                    prop[dconfig][dependency] = {}
                    for k in vals[dconfig][dependency]['loss']:
                        #if i == 0:
                        #    print('{} {} {} perplexity: {}'.format(dconfig, dependency,
                        #            k, np.exp(vals[dconfig][dependency]['loss'][k])))

                        prop[dconfig][dependency][k] = p = {}

                        probs = vals[dconfig][dependency]['probabilities'][k][0]
                        target = data_dict[config['features'][direction + '_' + k]][0]
                        if dconfig == 'joint_configs':
                            p['alpha'] = alpha = probs[idx].tolist()
                            sum_probs = 0
                            for jd in range(len(joint_configs[config['model']][dependency])):
                                joint_dependency = joint_configs[config['model']][dependency][jd]
                                probs = vals['dependency_configs'][joint_dependency][k]['probabilities'][0][idx]
                                sum_probs += alpha[jd] * probs
                            probs = sum_probs
                        if k in ['label_index', 'attr_index']:
                            if dconfig != 'joint_configs':
                                probs = probs[idx]
                            rank = np.flip(np.argsort(probs), 0)
                            p['expected'] = config['raw_data'][k + '_to_token'][rank[0]]
                            p['actual'] = config['raw_data'][k + '_to_token'][target[idx]]
                            p['expected_probability'] = float(probs[rank[0]])
                            p['actual_probability'] = float(probs[target[idx]])
                            p['ratio'] = float(probs[target[idx]] / (probs[rank[0]]))
                            p['probabilities'] = [(float(probs[j]), config['raw_data'][k+'_to_token'][j]) for j in rank]
                        else:
                            p['actual'] = target[idx]
                            p['expected'] = float(probs[idx])

            if node_properties is None:
                props.append(prop)
            else:
                props[i].update(prop)

        return props


    def process_code(self, code):
        #directives, _ = preprocess.grab_directives(code)


        # can raise exception
        ast_data, linear_data = wrangle(code)

        self.infer(ast_data, linear_data)
        output = {
            'ast': print_ast(ast_data.ast, ast_data.node_properties),
            #'code': generator.visit(ast),
            'fixed_code': 'blah'
        }
        return output

       #if self.config['model'] == 'ast':
       #    # Can't add attributes directly to Nodes because the class uses __slots__, so use this dictionary to
       #    # extend objects
       #    #run_ast_epoch(session, graph, config, ast, ast_nodes, node_properties, raw_data)
       #    self.run_epoch(linearizer)
       #    #step(session, config, linearizer, raw_data)
       #    code = "blah"#search(ast, node_properties, filename, directives)
       #    output = {
       #        'ast': print_ast(ast, linearizer.node_properties),
       #        #'code': generator.visit(ast),
       #        'fixed_code': code
       #    }
       #else:
       #    props = run_epoch(session, config, tokens, raw_data)
       #    code = "blah"#search(ast, node_properties, filename, directives)
       #    output = {
       #        'linear': props,
       #        'fixed_code': code
       #    }
       #server_send_output(ServerOutput(output))

def load_data(data_path : str):
    # change to the directory of this file, so that the data paths are accurate
    directory = os.path.dirname(os.path.realpath(__file__))
    os.chdir(directory)

    best_dir = os.path.join(data_path, 'best')
    with open(os.path.join(best_dir, 'config.json')) as f:
        config = json.load(f)
    config['best_dir'] = best_dir

    # we have to find the model that we can feed...
    config['fetches'] = fetches = {} # type: ignore
    config['initials'] = initials = {} # type: ignore
    config['feed_dict'] = feed = {} # type: ignore
    for d in config['models']:
        fetches[d] = {}
        initials[d] = {}
        for i in config['models'][d]:
            #feed[config['models'][d][i]['placeholders']['is_inference']] = False
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

    # fix windows path separators TODO: this isn't particularly portable?
    config['data_path'] = os.path.join(*config['data_path'].split('\\'))

    raw_data = dict()
    # TODO: this data path is hard-coded
    with open(os.path.join('..', config['data_path'], config['model'] + '_lexicon.json')) as f:
        token_ids = json.load(f)
    raw_data['label'] = token_ids['label']
    raw_data['attr'] = token_ids['attr']

    raw_data['label_index_to_token'] = dict()
    for k in raw_data['label']:
        raw_data['label_index_to_token'][raw_data['label'][k]] = k

    raw_data['attr_index_to_token'] = dict()
    for k in raw_data['attr']:
        raw_data['attr_index_to_token'][raw_data['attr'][k]] = k
    config['raw_data'] = raw_data
    return config

def create_server(config) -> Server:
    graph = tf.Graph()
    with graph.as_default():
        saver = tf.train.import_meta_graph(os.path.join(config['best_dir'], "model.meta"))
        session = tf.Session()
        saver.restore(session, os.path.join(config['best_dir'], 'model'))
        config['initial_values'] = session.run(config['initials'])

    # TODO: close session in Server deconstructor?
    return Server(config, session, graph)
