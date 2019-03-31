import sys
import collections
from ..default_dict import data_dict
from pycparser import c_ast # type:ignore

transition_classes = ['FileAST', 'FuncDef', 'ExpressionList', 'If', 'For', 'While', 'DoWhile', 'Switch', 'Case', 'Default']

default_dependencies = lambda: {
    'pointers': {'memory': [], 'mask': []},
    'self': 0,
    'parent': 0,
    'left_sibling': 0,
    'left_hole': 0,
    'left_prior': 0,
    'left_child': 0,
    'right_hole': 0,
    'right_sibling': 0,
    'right_prior': 0,
    'right_child': 0,
}

default_props = lambda: {
    'label': None,
    'first_sibling': True,
    'num_children': 0,
    'is_leaf': True,
    'attr': None,
    'snapshots': [],
    'transitions': None,

    # this is the only dependency we need for the TreeLSTMs
    'parent_transitions': None,
    'parent_label': None,
    'parent_attr': None,

    'forward': default_dependencies(),
    'reverse': default_dependencies()
}

# ancestor -> transitions -> test -> key -> 0
nodes_list = lambda: { 'forward': [default_props()],
                       'reverse': [default_props()] }

arg_dict = lambda: data_dict(lambda: collections.defaultdict(int, nodes=nodes_list(), pointers=[])) #type:ignore


def canonicalize_snapshots(node, test):
    if node.__class__.__name__ == 'FileAST': return '<FileAST>'

    if 'snapshots' not in node.node_properties or test not in node.node_properties['snapshots']:
        return '<unk>'

    transitions = []
    for snap in node.node_properties['snapshots'][test]:
        locals_ = []
        for func_scope in snap['scope']:
            for scope in func_scope:
                for key in scope:
                    locals_.append(str(scope[key]['before']) + '->' + str(scope[key]['after']))
        memory = []
        for array in snap['memory']:
            for key in snap['memory'][array]:
                vals = snap['memory'][array][key]
                memory.append(str(vals['before']) + '->' + str(vals['after']))
        memory.extend(locals_)
        # make changes to variables agnostic of order
        memory.sort()
        ret = 'locals: ' + ', '.join(memory)
        ret += ' stdout: ' + snap['stdout']
        ret += ' stderr: ' + snap['stderr']
        ret += ' return: ' + (str(snap['return']) if snap['return'] is not False else '') + ' '
        if ret not in transitions:
            transitions.append(ret)
    transitions.sort()
    return '\n '.join(transitions)


class WrangledAST(object):

    def __init__(self, ast, results):
        self.ast = ast
        self.results = results
        self.prop_map = {}
        self.tests = set(['null'])
        self.num_nodes = 1

        for result_group in results:
            for result_name in result_group:
                self.tests.add(result_name)

        self.generic_visit()
        self.transitions_groups = collections.defaultdict(lambda: collections.defaultdict(lambda:
            collections.defaultdict(lambda: collections.defaultdict(int))))
        self.get_transitions_groups(ast)

    def update_args(self, args, args_update):
        for test in args_update:
            for ancestor in args_update[test]:
                for transitions in args_update[test][ancestor]:
                    args[test][ancestor][transitions].update(args_update[test][ancestor][transitions])

    def get_transitions_groups(self, node):
        if node.__class__.__name__ == 'NodeWrapper':
            return self.get_transitions_groups(node.new) if node.new is not None else None
        if node.__class__.__name__ in transition_classes:
            for test in self.tests:
                transitions = node.node_properties['props'][test][node.node_properties['node_num']]['true']
                if transitions:
                    tg = self.transitions_groups[test][transitions['transitions']]
                    for test2 in self.tests:
                        transitions2 = node.node_properties['props'][test2][node.node_properties['node_num']]['true']
                        if transitions2:
                            tg[test2][transitions2['transitions']] += 1

        children = node.children()
        for i in range(len(children)):
            self.get_transitions_groups(children[i][1])

    # This is currently O(n^2)
    def handle_pointers(self, args):
        for test in args:
            for ancestor in args[test]:
                ancestor_node_num = ancestor.node_properties['node_num']
                for transitions in args[test][ancestor]:
                    pointers = args[test][ancestor][transitions]['pointers']
                    for node in pointers:
                        props = node.node_properties['props'][test][ancestor_node_num][transitions]
                        for node2 in pointers:
                            if node == node2: continue
                            props2 = node2.node_properties['props'][test][ancestor_node_num][transitions]
                            direction = 'forward' if node.node_properties['node_num'] < node2.node_properties['node_num'] else 'reverse'
                            props[direction]['pointers']['mask'].append(1 if node2 in node.node_properties['pointers'] else 0)
                            props[direction]['pointers']['memory'].append(props2[direction]['self'])

    def generic_visit_reverse(self, node, args, transitions_ancestry):
        className = node.__class__.__name__
        if className == 'NodeWrapper':
            return self.generic_visit_reverse(node.new, args, transitions_ancestry) \
                   if node.new is not None else None

        if className in transition_classes:
            transitions_ancestry.insert(0, node)
            idx = 2
        else:
            idx = 1


        update_args = arg_dict()
        for test in args:
            for ancestor in (transitions_ancestry[:idx] if test != 'null' else [self.ast]):
                ancestor_node_num = ancestor.node_properties['node_num']
                for transitions in args[test][ancestor]:
                    props = node.node_properties['props'][test][ancestor_node_num][transitions]
                    if not props:
                        continue

                    arg = args[test][ancestor][transitions]
                    up_arg = args[test][ancestor][transitions]
                    fprops = props['forward']
                    fnodes = arg['nodes']['forward']
                    rnodes = arg['nodes']['reverse']

                    my_node_num = len(rnodes)
                    props['reverse'].update({
                        'self': my_node_num,
                        'parent': fnodes[fprops['parent']]['reverse']['self'],
                        'right_sibling': fnodes[fprops['right_sibling']]['reverse']['self'],
                        'right_prior': my_node_num - 1,
                        'right_hole': arg['right_hole'],
                    })
                    props['last_sibling'] = fprops['right_sibling'] == 0
                    rnodes.append(props)

                    rprops = props['reverse']


                    if rnodes[rprops['parent']]['reverse']['right_child'] == 0:
                        rnodes[rprops['parent']]['reverse']['right_child'] = my_node_num
                    rnodes[rprops['parent']]['reverse']['left_child'] = my_node_num
                    fnodes[fprops['right_sibling']]['reverse']['left_sibling'] = my_node_num
                    rnodes[my_node_num-1]['reverse']['left_prior'] = my_node_num
                    fprops['right_hole'] = rnodes[rprops['right_hole']]['forward']['self']
                    # TODO: fix this
                    #props['reverse']['left_hole'] = self.nodes[test]['forward'][props['forward']['left_hole']]['reverse']['self']
                    #rprops['left_hole'] = fnodes[fprops['left_hole']]['reverse']['self']

                    arg['right_sibling'] = 0
                    up_arg['right_hole'] = my_node_num
                    up_arg['right_sibling'] = my_node_num

        if 'pointers' in node.node_properties:
            node.node_properties['pointer_ids'] = [n.node_properties['node_num'] for n in node.node_properties['pointers'] if 'node_num' in n.node_properties]

        children = node.children()
        for i in range(len(children)-1, -1, -1):
            self.generic_visit_reverse(children[i][1], args, transitions_ancestry)

        # cleanup/setup for next node
        self.update_args(args, update_args)

        if className in transition_classes:
            transitions_ancestry.pop(0)



    def get_attr(self, node):
        # strongly assumes there is at most one attribute we care about
        nvlist = [(n, getattr(node, n)) for n in node.attr_names]
        if 'replace_name' in node.node_properties:
            return node.node_properties['replace_name']
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
            return attr


    def generic_visit_forward(self, node, args, transitions_ancestry):
        className = node.__class__.__name__
        if className == 'NodeWrapper':
            return self.generic_visit_forward(node.new, args, transitions_ancestry) \
                    if node.new is not None else None

        nprops = node.node_properties
        nprops['node_num'] = self.num_nodes
        self.prop_map[nprops['node_num']] = nprops
        self.num_nodes += 1
        nprops['props'] = collections.defaultdict(lambda: collections.defaultdict(lambda:
                    collections.defaultdict(lambda: False)))

        attr = self.get_attr(node)

        # we don't want the subtree from every node.
        # only from transition-relevant nodes (for loops, compounds, etc.)
        if className in transition_classes:
            transitions_ancestry.insert(0, node)
            idx = 2
        else:
            idx = 1

        update_args = arg_dict()
        for test in self.tests:
            for ancestor in (transitions_ancestry[:idx] if test != 'null' else [self.ast]):
                ancestor_node_num = ancestor.node_properties['node_num']
                for transitions in args[test][ancestor]:
                    arg = args[test][ancestor][transitions]
                    up_arg = update_args[test][ancestor][transitions] = {'parent': arg['parent']}

                    if (transitions == 'true' and className not in transition_classes) or \
                            (test != 'null' and (arg['parent'] is False or \
                            test not in nprops['visited'])):
                        arg['parent'] = False
                        continue

                    nodes = arg['nodes']['forward']
                    my_node_num = len(nodes)

                    parent_idx = arg['parent']
                    parent_node = nodes[parent_idx]
                    left_idx = arg['left_sibling']
                    left_node = nodes[left_idx]

                    nprops['props'][test][ancestor_node_num][transitions] = props = default_props()

                    props['forward'].update({
                        'self': my_node_num,
                        'parent': parent_idx,
                        'left_sibling': left_idx,
                        'left_hole': arg['left_hole'],
                        'left_prior': my_node_num - 1,
                    })
                    props.update({
                        'label': className,
                        'first_sibling': left_idx == 0,
                        'attr': attr,
                        'snapshots':  nprops['snapshots'][test] if 'snapshots' in nprops and test in nprops['snapshots'] else [],
                        'root_transitions': canonicalize_snapshots(ancestor, test),
                        'transitions': canonicalize_snapshots(node, test),
                        'parent_transitions': parent_node['transitions'],
                        'parent_label': parent_node['label'],
                        'parent_attr': parent_node['attr'],
                    })

                    nodes.append(props)

                    if parent_node['forward']['left_child'] == 0:
                        parent_node['forward']['left_child'] = my_node_num

                    parent_node['forward']['right_child'] = my_node_num
                    parent_node['num_children'] += 1
                    parent_node['is_leaf'] = False
                    left_node['forward']['right_sibling'] = my_node_num
                    nodes[my_node_num - 1]['forward']['right_prior'] = my_node_num

                    arg['parent'] = my_node_num
                    arg['left_sibling'] = 0
                    up_arg['left_sibling'] = my_node_num
                    up_arg['left_hole'] = my_node_num

                    if props['attr'] == 'LOCAL_ID' and 'pointers' in node.node_properties:
                        arg['pointers'].append(node)

        children = node.children()
        for i in range(len(children)):
            self.generic_visit_forward(children[i][1], args, transitions_ancestry)

        # cleanup/setup for next node
        self.update_args(args, update_args)

        if className in transition_classes:
            transitions_ancestry.pop(0)

    def generic_visit(self):
        args = arg_dict()
        self.generic_visit_forward(self.ast, args, [])
        self.generic_visit_reverse(self.ast, args, [])
        self.handle_pointers(args)

        self.nodes = data_dict(lambda: False)
        for test in args:
            for ancestor in args[test]:
                ancestor_node_num = ancestor.node_properties['node_num']
                for transitions in args[test][ancestor]:
                    if transitions == 'true' and test == 'null': continue
                    arg = args[test][ancestor][transitions]['nodes']
                    if len(arg['forward']) <= 1: continue
                    arg['forward'][0] = arg['reverse'][0] = None
                    self.nodes[test][ancestor_node_num][transitions] = arg
