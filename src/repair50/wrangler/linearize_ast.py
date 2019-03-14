import sys
from copy import deepcopy
from pycparser import c_ast # type:ignore

transition_classes = ['FileAST', 'FuncDef', 'Compound', 'ExpressionList', 'If', 'For', 'While', 'DoWhile', 'Switch', 'Case', 'Default']


def canonicalize_snapshots(node, test):
    if 'snapshots' not in node.node_properties or test not in node.node_properties['snapshots']:
        return None

    transitions = []
    for snap in node.node_properties['snapshots'][test]:
        ret = ''
        if test not in ['test7', 'test8', 'test9']:
            ret += snap['stdout'] + '\n'
            ret += snap['stderr'] + '\n'
        ret += (str(snap['return']) if snap['return'] is not False else '') + '\n'
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
    def __init__(self, ast, results):
        self.ast = ast
        self.results = results
        self.prop_map = {}
        self.tests = set()
        self.num_nodes = 1

        for result in results:
            self.tests.add(result['name'])
        self.tests.add('null')

        nil_props = {'label': None, 'attr': None, 'transitions': None,
                'forward':{'self': 0, 'left_child': 0}, 'reverse':{'self': 0, 'right_child': 0}}

        self.hole = {}
        self.nodes = {}
        self.pointer_memory = {}
        for transitions in [False, True]:
            self.hole[transitions] = {}
            self.nodes[transitions] = {}
            self.pointer_memory[transitions] = {}
            for test in self.tests:
                self.hole[transitions][test] = 0
                self.nodes[transitions][test] = { 'forward': [nil_props], 'reverse': [nil_props] }
                self.pointer_memory[transitions][test] = []
        self.generic_visit()

        self.transitions_groups = {test: {} for test in self.tests}
        self.get_transitions_groups(ast)

        self.transitions_trees = []

    def get_transitions_groups(self, node):
        if node.__class__.__name__ == 'NodeWrapper':
            return self.get_transitions_groups(node.new) if node.new is not None else None
        for test in self.tests:
            transitions = node.node_properties['props'][True][test]
            if transitions:
                transitions = transitions['transitions']
            if transitions not in self.transitions_groups[test]:
                self.transitions_groups[test][transitions] = {}
            tg = self.transitions_groups[test][transitions]
            for test2 in self.tests:
                transitions2 = node.node_properties['props'][True][test2]
                if transitions2:
                    transitions2 = transitions2['transitions']
                if test2 not in tg:
                    tg[test2] = {}
                if transitions2 not in tg:
                    tg[test2][transitions2] = 0
                tg[test2][transitions2] += 1

        children = node.children()
        for i in range(len(children)):
            self.get_transitions_groups(children[i][1])


    def handle_pointers(self, node, direction, transitions, test):
        props = node.node_properties['props'][transitions][test]

        pointers = []
        if 'pointers' in node.node_properties:
            for n in node.node_properties['pointers']:
                # TODO: for some reason, a node is encountered more than once???
                if isinstance(n, int):
                    return
                # removed typedecls and what not
                if 'props' in n.node_properties and \
                        n.node_properties['props'][transitions][test] and \
                        direction in n.node_properties['props'][transitions][test] and \
                        n.node_properties['props'][transitions][test][direction]['self'] != 0 and \
                        n != node:
                    pointers.append(n.node_properties['props'][transitions][test][direction]['self'])
        props[direction]['pointers']['ids'] = pointers

        # TODO: filter by scope???
        if props['attr'] == 'LOCAL_ID':
            props[direction]['pointers']['memory'] = deepcopy(self.pointer_memory[transitions][test])
            props[direction]['pointers']['mask'] = [(1 if pointer in props[direction]['pointers']['ids'] else 0) for pointer in props[direction]['pointers']['memory']]
            #self.pointer_memory.pop()
            self.pointer_memory[transitions][test].insert(0, props[direction]['self'])

        if 'pointers' in node.node_properties and direction == 'reverse' and not transitions:
            node.node_properties['pointers'] = [n.node_properties['node_num'] for n in node.node_properties['pointers'] if 'node_num' in n.node_properties]

    def generic_visit_reverse(self, node):
        className = node.__class__.__name__
        if className == 'NodeWrapper':
            return self.generic_visit_reverse(node.new) if node.new is not None else \
                    { transitions: {test: False for test in self.tests} for transitions in [True, False] }

        ret = {}

        for transitions in [True, False]:
            ret[transitions] = {}
            for test in self.tests:
                props = node.node_properties['props'][transitions][test]
                if not props:
                    ret[transitions][test] = False
                    continue

                my_node_num = len(self.nodes[transitions][test]['reverse'])
                props['reverse'] = {
                    'self': my_node_num,
                    'parent': self.nodes[transitions][test]['forward'][props['forward']['parent']]['reverse']['self'],
                    'right_sibling': self.nodes[transitions][test]['forward'][props['forward']['right_sibling']]['reverse']['self'],
                    'right_prior': my_node_num - 1,
                    'right_hole': self.hole[transitions][test],

                    'pointers': {'memory': [], 'mask': []},
                    # these are all set by the relevant dependency if they exist
                    'left_hole': 0,
                    'left_child': 0,
                    'left_sibling': 0,
                    'left_prior': 0,
                    'right_child': 0,
                }
                props['last_sibling'] = props['forward']['right_sibling'] == 0

                self.nodes[transitions][test]['reverse'].append(props)

                if self.nodes[transitions][test]['reverse'][props['reverse']['parent']]['reverse']['right_child'] == 0:
                    self.nodes[transitions][test]['reverse'][props['reverse']['parent']]['reverse']['right_child'] = my_node_num
                self.nodes[transitions][test]['reverse'][props['reverse']['parent']]['reverse']['left_child'] = my_node_num
                self.nodes[transitions][test]['forward'][props['forward']['right_sibling']]['reverse']['left_sibling'] = my_node_num
                self.nodes[transitions][test]['reverse'][my_node_num-1]['reverse']['left_prior'] = my_node_num

                self.handle_pointers(node, 'reverse', transitions, test)
                ret[transitions][test] = node

        children = node.children()
        for i in range(len(children)-1, -1, -1):
            child_ret = self.generic_visit_reverse(children[i][1])
            for transitions in child_ret:
                for test in self.tests:
                    if child_ret[transitions][test]:
                        self.hole[transitions][test] = child_ret[transitions][test].node_properties['props'][transitions][test]['reverse']['self']

        for transitions in node.node_properties['props']:
            for test in self.tests:
                props = node.node_properties['props'][transitions][test]
                if props:
                    props['forward']['right_hole'] = self.nodes[transitions][test]['reverse'][props['reverse']['right_hole']]['forward']['self']
                    # TODO: fix this!
                    #props['reverse']['left_hole'] = self.nodes[test]['forward'][props['forward']['left_hole']]['reverse']['self']

        return ret




    def generic_visit_forward(self, node, parent, left_sibling, transitions_ancestry):
        className = node.__class__.__name__
        if className == 'NodeWrapper':
            return self.generic_visit_forward(node.new, parent, left_sibling, transitions_ancestry) \
                    if node.new is not None else \
                    { transitions: {test: False for test in self.tests} for transitions in [True, False] }

        node.node_properties['props'] = {True: {}, False: {}}
        node.node_properties['node_num'] = self.num_nodes
        self.num_nodes += 1
        self.prop_map[node.node_properties['node_num']] = node.node_properties['props']

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
        new_parent = {}
        new_left_sibling = {}
        new_transitions_ancestry = {}

        for transitions in [True, False]:
            ret[transitions] = {}
            new_left_sibling[transitions] = {}
            new_parent[transitions] = {}
            new_transitions_ancestry[transitions] = {}

            for test in self.tests:

                if (transitions and className not in transition_classes) or \
                    (test != 'null' and className not in ['FileAST', 'FuncDef'] and (parent[transitions][test] is False or 'visited' not in node.node_properties or test not in node.node_properties['visited'])):
                    ret[transitions][test] = False
                    new_left_sibling[transitions][test] = False
                    new_parent[transitions][test] = False
                    new_transitions_ancestry[transitions][test] = False
                    node.node_properties['props'][transitions][test] = False
                    continue

                nodes = self.nodes[transitions][test]['forward']
                my_node_num = len(nodes)

                transitions_ancestry[transitions][test].

                my_transition = canonicalize_snapshots(node, test)
                if className in transition_classes:
                    transitions_ancestry[transitions][test].append(my_transition)

                props = {
                    'label': node.__class__.__name__,
                    'first_sibling': left_sibling[transitions][test] == 0,
                    'num_children': 0,
                    'is_leaf': True,
                    'attr': attr,
                    'snapshots':  node.node_properties['snapshots'][test] if 'snapshots' in node.node_properties and test in node.node_properties['snapshots'] else [],
                    'transitions': canonicalize_snapshots(node, test),
                    'transitions_subtree': transitions_ancestry[transitions][test],

                    # this is the only dependency we need for the TreeLSTMs
                    'parent_transitions': nodes[parent[transitions][test]]['transitions'],
                    'parent_label': nodes[parent[transitions][test]]['label'],
                    'parent_attr': nodes[parent[transitions][test]]['attr'],

                    'forward': {
                        'pointers': {'memory': [], 'mask': []},
                        'self': my_node_num,
                        'parent': parent[transitions][test],
                        'left_sibling': left_sibling[transitions][test],
                        'left_hole': self.hole[transitions][test],
                        'left_prior': my_node_num - 1,
                        # these are all set by the relevant dependency if they exist
                        'left_child': 0,
                        'right_hole': 0,
                        'right_sibling': 0,
                        'right_prior': 0,
                        'right_child': 0,
                    }
                }

                nodes.append(props)
                node.node_properties['props'][transitions][test] = props

                if nodes[parent[transitions][test]]['forward']['left_child'] == 0:
                    nodes[parent[transitions][test]]['forward']['left_child'] = my_node_num

                nodes[parent[transitions][test]]['forward']['right_child'] = my_node_num
                nodes[left_sibling[transitions][test]]['forward']['right_sibling'] = my_node_num
                nodes[my_node_num - 1]['forward']['right_prior'] = my_node_num

                self.handle_pointers(node, 'forward', transitions, test)

                new_parent[transitions][test] = my_node_num
                new_left_sibling[transitions][test] = 0
                ret[transitions][test] = node

        children = node.children()
        for i in range(len(children)):
            child_ret = self.generic_visit_forward(children[i][1], new_parent, new_left_sibling)
            for transitions in child_ret:
                for test in child_ret[transitions]:
                    if child_ret[transitions][test]:
                        new_left_sibling[transitions][test] = self.hole[transitions][test] = child_ret[transitions][test].node_properties['props'][transitions][test]['forward']['self']
                        node.node_properties['props'][transitions][test]['num_children'] += 1
                        node.node_properties['props'][transitions][test]['is_leaf'] = False


        return ret


    def generic_visit(self):
        parent = { transitions: {test: 0 for test in self.tests} for transitions in [True, False] }
        left_sibling = { transitions: {test: 0 for test in self.tests} for transitions in [True, False] }
        self.generic_visit_forward(self.ast, parent, left_sibling)
        for transitions in self.hole:
            for test in self.hole[transitions]:
                    self.hole[transitions][test] = 0
                    self.pointer_memory[transitions][test] = []
        self.generic_visit_reverse(self.ast)
        # delete the nil slot
        for transitions in self.nodes:
            for test in self.nodes[transitions]:
                self.nodes[transitions][test]['forward'][0] = None
                self.nodes[transitions][test]['reverse'][0] = None
