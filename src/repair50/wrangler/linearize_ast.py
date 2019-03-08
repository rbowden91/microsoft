import sys
from copy import deepcopy
from pycparser import c_ast # type:ignore



def canonicalize_snapshots(node, test):
    if 'snapshots' not in node.node_properties or test not in node.node_properties['snapshots']:
        return None

    transitions = []
    for snap in node.node_properties['snapshots'][test]:
        ret = ''
        if test not in ['test7', 'test8', 'test9']:
            ret += snap['stdout'] + '\n'
            ret += snap['stderr'] + '\n'
        ret += str(snap['return']) + '\n'
        for func_scope in snap['scope']:
            for scope in func_scope:
                for key in scope:
                    ret += str(scope[key]['before']) + ' ' + str(scope[key]['after']) + '\n'
        for array in snap['memory']:
            for key in snap['memory'][array]:
                vals = snap['memory'][array][key]
                ret += str(vals['before']) + ' ' + str(vals['after']) + '\n'
        if ret not in transitions:
            transitions.append(ret)
    transitions.sort()
    return '\n'.join(transitions)


class WrangledAST(object):
    def __init__(self, ast, results, include_dependencies = True):
        self.include_dependencies = include_dependencies
        self.ast = ast
        self.results = results
        self.tests = set()
        self.prop_map = {}
        self.num_nodes = 1

        for result in results:
            self.tests.add(result['name'])
        self.tests.add(None)

        #if self.include_dependencies:
        #    self.default_props['dependencies'] = { 'self': None }

        self.hole = {}
        self.nodes = {}
        self.pointer_memory = {}

        nil_props = {'label': None, 'attr': None, 'transitions': None,
                'forward':{'self': 0, 'left_child': 0}, 'reverse':{'self': 0, 'right_child': 0}}
        for test in self.tests:
            self.hole[test] = 0
            self.nodes[test] = { 'forward': [nil_props], 'reverse': [nil_props] }
            self.pointer_memory[test] = []

        self.generic_visit()

    def handle_pointers(self, node, direction, test):
        props = node.node_properties['props'][test]

        pointers = []
        if 'pointers' in node.node_properties:
            for n in node.node_properties['pointers']:
                # TODO: for some reason, a node is encountered more than once???
                if isinstance(n, int):
                    return
                # removed typedecls and what not
                if 'props' in n.node_properties and \
                        n.node_properties['props'][test] and \
                        direction in n.node_properties['props'][test] and \
                        n.node_properties['props'][test][direction]['self'] != 0 and \
                        n != node:
                    pointers.append(n.node_properties['props'][test][direction]['self'])
        props[direction]['pointers']['ids'] = pointers

        # TODO: filter by scope???
        if props['attr'] == 'LOCAL_ID':
            props[direction]['pointers']['memory'] = deepcopy(self.pointer_memory[test])
            props[direction]['pointers']['mask'] = [(1 if pointer in props[direction]['pointers']['ids'] else 0) for pointer in props[direction]['pointers']['memory']]
            #self.pointer_memory.pop()
            self.pointer_memory[test].insert(0, props[direction]['self'])

        if 'pointers' in node.node_properties and direction == 'reverse':
            node.node_properties['pointers'] = [n.node_properties['node_num'] for n in node.node_properties['pointers'] if 'node_num' in n.node_properties]

    def generic_visit_reverse(self, node):
        if node.__class__.__name__ == 'NodeWrapper':
            return self.generic_visit_reverse(node.new) if node.new is not None else {test: False for test in self.tests}

        ret = {}
        for test in self.tests:
            props = node.node_properties['props'][test]
            if not props:
                ret[test] = False
                continue

            my_node_num = len(self.nodes[test]['reverse'])
            props['reverse'] = {
                'self': my_node_num,
                'parent': self.nodes[test]['forward'][props['forward']['parent']]['reverse']['self'],
                'right_sibling': self.nodes[test]['forward'][props['forward']['right_sibling']]['reverse']['self'],
                'right_prior': my_node_num - 1,
                'right_hole': self.hole[test],

                'pointers': {'memory': [], 'mask': []},
                # these are all set by the relevant dependency if they exist
                'left_hole': 0,
                'left_child': 0,
                'left_sibling': 0,
                'left_prior': 0,
                'right_child': 0,
            }
            props['last_sibling'] = props['forward']['right_sibling'] == 0

            self.nodes[test]['reverse'].append(props)

            if self.nodes[test]['reverse'][props['reverse']['parent']]['reverse']['right_child'] == 0:
                self.nodes[test]['reverse'][props['reverse']['parent']]['reverse']['right_child'] = my_node_num
            self.nodes[test]['reverse'][props['reverse']['parent']]['reverse']['left_child'] = my_node_num
            self.nodes[test]['forward'][props['forward']['right_sibling']]['reverse']['left_sibling'] = my_node_num
            self.nodes[test]['reverse'][my_node_num-1]['reverse']['left_prior'] = my_node_num

            self.handle_pointers(node, 'reverse', test)
            ret[test] = node

        children = node.children()
        #self.hole = 0
        for i in range(len(children)-1, -1, -1):
            child_ret = self.generic_visit_reverse(children[i][1])
            for test in self.tests:
                if child_ret[test]:
                    self.hole[test] = child_ret[test].node_properties['props'][test]['reverse']['self']

        for test in self.tests:
            props = node.node_properties['props'][test]
            if props:
                props['forward']['right_hole'] = self.nodes[test]['reverse'][props['reverse']['right_hole']]['forward']['self']
                # TODO: fix this!
                #props['reverse']['left_hole'] = self.nodes[test]['forward'][props['forward']['left_hole']]['reverse']['self']

        return ret




    def generic_visit_forward(self, node, parent, left_sibling):
        if node.__class__.__name__ == 'NodeWrapper':
            return self.generic_visit_forward(node.new, parent, left_sibling) \
                    if node.new is not None else {test: False for test in self.tests}

        node.node_properties['props'] = {}
        node.node_properties['node_num'] = self.num_nodes
        self.prop_map[node.node_properties['node_num']] = node.node_properties['props'];
        self.num_nodes += 1

        # strongly assumes there is at most one attribute we care about
        nvlist = [(n, getattr(node, n)) for n in node.attr_names]
        if 'replace_name' in node.node_properties:
            attr = node.node_properties['replace_name']
        else:
            attr = None
            for (name, val) in nvlist:
                if name in ['value', 'op', 'name', 'declname']:
                    attr = val
                elif name == 'names':
                    attr = ' '.join(val)
                else:
                    #print(name, val)
                    pass

        ret = {}
        props = {}
        new_parent = {}
        new_left_sibling = {}
        for test in self.tests:
            if test is not None and node.__class__.__name__ != 'FileAST' and node.__class__.__name__ != 'FuncDef' and \
                    (parent[test] is False or \
                    'visited' not in node.node_properties or test not in node.node_properties['visited']):
                new_parent[test] = False
                ret[test] = False
                node.node_properties['props'][test] = False
                continue

            nodes = self.nodes[test]['forward']
            my_node_num = len(nodes)

            props = {
                'label': node.__class__.__name__,
                'first_sibling': left_sibling[test] == 0,
                'num_children': 0,
                'is_leaf': True,
                'attr': attr,
                'transitions': canonicalize_snapshots(node, test),

                # this is the only dependency we need for the TreeLSTMs
                'parent_transitions': nodes[parent[test]]['transitions'],
                'parent_label': nodes[parent[test]]['label'],
                'parent_attr': nodes[parent[test]]['attr'],

                'forward': {
                    'pointers': {'memory': [], 'mask': []},
                    'self': my_node_num,
                    'parent': parent[test],
                    'left_sibling': left_sibling[test],
                    'left_hole': self.hole[test],
                    'left_prior': my_node_num - 1,
                    # these are all set by the relevant dependency if they exist
                    'left_child': 0,
                    'right_hole': 0,
                    'right_sibling': 0,
                    'right_prior': 0,
                    'right_child': 0,
                }
            }

            #if self.include_dependencies:
            #    props['dependencies']['self'] = node
            nodes.append(props)
            node.node_properties['props'][test] = props

            if nodes[parent[test]]['forward']['left_child'] == 0:
                nodes[parent[test]]['forward']['left_child'] = my_node_num

            nodes[parent[test]]['forward']['right_child'] = my_node_num
            nodes[left_sibling[test]]['forward']['right_sibling'] = my_node_num
            nodes[my_node_num - 1]['forward']['right_prior'] = my_node_num

            self.handle_pointers(node, 'forward', test)

            new_parent[test] = my_node_num
            new_left_sibling[test] = 0
            ret[test] = node

        children = node.children()
        for i in range(len(children)):
            child_ret = self.generic_visit_forward(children[i][1], new_parent, new_left_sibling)
            for test in child_ret:
                if child_ret[test]:
                    new_left_sibling[test] = self.hole[test] = child_ret[test].node_properties['props'][test]['forward']['self']
                    node.node_properties['props'][test]['num_children'] += 1
                    node.node_properties['props'][test]['is_leaf'] = False


        #if self.include_dependencies:
        #    for i in props['forward']:
        #        if i == 'pointers': continue
        #        props['dependencies'][i] = nodes[props['forward'][i]]['dependencies']['self']


        return ret


    def generic_visit(self):
        self.generic_visit_forward(self.ast, {test: 0 for test in self.tests}, {test: 0 for test in self.tests})
        for test in self.tests:
            self.hole[test] = 0
            self.pointer_memory[test] = []
        self.generic_visit_reverse(self.ast)
        # delete the nil slot
        for test in self.tests:
            self.nodes[test]['forward'][0] = None
            self.nodes[test]['reverse'][0] = None
