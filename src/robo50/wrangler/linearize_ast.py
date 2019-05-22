import sys
import collections
from ..default_dict import get_dict, get_dict_default
from pycparser import c_ast # type:ignore

transition_classes = ['FileAST', 'FuncDef', 'ExpressionList', 'If', 'For', 'While', 'DoWhile', 'Switch', 'Case', 'Default']

default_dependencies = lambda: {
    'self': 0,
    'parent': 0,
    'end_sibling': 0,
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
    'num_children': 0,
    'is_leaf': True,
    'attr': None,
    'transitions': None,

    # this is the only dependency we need for the TreeLSTMs
    'parent_transitions': None,
    'parent_label': None,
    'parent_attr': None,

    'forward': default_dependencies(),
    'reverse': default_dependencies()
}

def canonicalize_snapshots(node, test):
    if node.__class__.__name__ == 'FileAST': return '<FileAST>'
    elif test == 'null': return '<unk>'

    transitions = []
    for snap in node.node_properties['test_data'][test]['snapshots']:
        locals_ = []
        for func_scope in snap['scope']:
            for scope in func_scope:
                for key in scope:
                    if 'after' not in scope[key]:
                        # the variable was only read, not written
                        continue
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
        self.node_map = {}
        self.prop_map = {}
        self.num_nodes = 1

        for test, tresults in results.items():
            for node, changes in tresults['node_changes'].items():
                props = get_dict(node.node_properties, 'test_data', test)
                props['snapshots'] = changes
                props['passed'] = tresults['passed']
            del(tresults['node_changes'])

        self.visit()

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

    def handle_pointers_directional(self, direction):
        pointer_memory = []
        pointer_map = {}
        # only for the null "test" case
        mds = self.args['null'][self.ast.node_properties['node_num']]['false'][direction]['model_data'][1:]
        for md in mds:
            node_num = md['node_num']
            node = self.prop_map[node_num]
            if 'pointers' not in node:
                md[direction]['pointers'] = {'mask': [], 'memory': []}
                continue
            if direction == 'forward':
                # filter out any nodes that might have been normalized out (like typedefs)
                node['pointers'] = [p.node_properties['node_num'] for p in node['pointers'] if 'node_num' in p.node_properties]

            if node['attr'] == 'LOCAL_ID':
                memory_length = len(pointer_memory)
                md[direction]['pointers'] = {'mask': [0]*memory_length, 'memory': pointer_memory.copy()}
                for ptr in node['pointers']:
                    if ptr in pointer_map:
                        md[direction]['pointers']['mask'][pointer_map[ptr]] = 1
                pointer_map[node_num] = memory_length
                pointer_memory.append(node_num)
            else:
                md[direction]['pointers'] = {'mask': [], 'memory': []}


    def visit_directional(self, node, direction):
        className = node.__class__.__name__
        if className == 'NodeWrapper':
            return self.visit_directional(node.new, direction) \
                    if node.new is not None else None

        args = self.args
        transitions_ancestry = self.transitions_ancestry

        nprops = node.node_properties
        if 'node_num' not in nprops:
            attr = self.get_attr(node)
            nprops.update({
                'node_num': self.num_nodes,
                'label': className,
                'attr': attr,
                'is_root': className in transition_classes
            })
            if 'test_data' not in nprops:
                nprops['test_data'] = {}
            nprops['test_data']['null'] = {}
            self.num_nodes += 1
            self.node_map[nprops['node_num']] = node
            self.prop_map[nprops['node_num']] = nprops

        # we don't want the subtree from every node.
        # only from transition-relevant nodes (for loops, compounds, etc.)
        if className in transition_classes:
            transitions_ancestry.insert(0, nprops['node_num'])
            idx = 2
        else:
            idx = 1

        update_args = []
        for test in nprops['test_data']:
            if 'transitions' not in nprops['test_data'][test]:
                nprops['test_data'][test]['transitions'] = canonicalize_snapshots(node, test)

            for ancestor in (transitions_ancestry[:idx] if test != 'null' else [self.ast.node_properties['node_num']]):
                for transitions in ['true', 'false']:
                    if transitions == 'true' and (test == 'null' or className not in transition_classes): continue
                    arg = get_dict_default(args, test, ancestor, transitions, {
                        'forward': {'parent': 0, 'sibling': 0, 'hole': 0, 'model_data': [default_props()]},
                        'reverse': {'parent': 0, 'sibling': 0, 'hole': 0, 'model_data': [default_props()]}
                    })[transitions]

                    key,rkey,rdir = ('left','right','reverse') if direction == 'forward' else ('right','left','forward')
                    rmodel_data = arg[rdir]['model_data']
                    arg = arg[direction]
                    model_data = arg['model_data']
                    model_node_num = len(model_data)
                    parent_model_data = model_data[arg['parent']]
                    all_props = get_dict_default(nprops['test_data'], test, 'model_data', ancestor, transitions, {
                        'node_num': nprops['node_num'],
                        'label': nprops['label'],
                        'attr': nprops['attr'],
                        'transitions': nprops['test_data'][test]['transitions'],
                        'parent_transitions': parent_model_data['transitions'],
                        'parent_label': parent_model_data['label'],
                        'parent_attr': parent_model_data['attr'],
                        'num_children': 0,
                        'is_leaf': True,
                        'forward': default_dependencies(),
                        'reverse': default_dependencies()})[transitions]

                    props = all_props[direction]
                    rprops = all_props[rdir]
                    sibling_idx = arg['sibling']
                    props.update({
                        'self': model_node_num,
                        'parent': arg['parent'],
                        key + '_sibling': sibling_idx,
                        key + '_hole': arg['hole'],
                        key + '_prior': model_node_num - 1,
                        'end_sibling': sibling_idx == 0
                    })


                    sibling = model_data[sibling_idx]
                    sibling[direction][rkey + '_sibling'] = model_node_num

                    if parent_model_data[direction][key + '_child'] == 0:
                        parent_model_data[direction][key + '_child'] = model_node_num
                    parent_model_data[direction][rkey + '_child'] = model_node_num
                    parent_model_data['is_leaf'] = False
                    model_data[model_node_num - 1][direction][rkey + '_prior'] = model_node_num

                    if direction == 'forward':
                        parent_model_data['num_children'] += 1
                    else:
                        rmodel_data[rprops['right_sibling']]['reverse']['left_sibling'] = model_node_num
                        # TODO: fix this (left hole and right hole aren't set correctly for the inverse directions)
                        #props['reverse']['left_hole'] = self.nodes[test]['forward'][props['forward']['left_hole']]['reverse']['self']
                        #rprops['left_hole'] = fnodes[fprops['left_hole']]['reverse']['self']


                    model_data.append(all_props)

                    update_args.append((arg, 'hole', model_node_num))
                    update_args.append((arg, 'sibling', model_node_num))
                    update_args.append((arg, 'parent', arg['parent']))
                    arg['sibling'] = 0
                    arg['parent'] = model_node_num

        children = node.children()
        for i in range(len(children)):
            idx = i if direction == 'forward' else len(children) - i - 1
            self.visit_directional(children[idx][1], direction)

        # cleanup/setup for next node
        for (arg, key, val) in update_args:
            arg[key] = val

        if className in transition_classes:
            transitions_ancestry.pop(0)

    def visit(self):
        self.args = args = {}
        self.transitions_ancestry = []
        self.visit_directional(self.ast, 'forward')
        self.visit_directional(self.ast, 'reverse')
        self.handle_pointers_directional('forward')
        self.handle_pointers_directional('reverse')

        self.nodes = {}
        for test in args:
            for ancestor in args[test]:
                for transitions in args[test][ancestor]:
                    arg = args[test][ancestor][transitions]
                    if len(arg['forward']['model_data']) <= 1: continue
                    n = get_dict(self.nodes, test, ancestor, transitions)
                    for direction in arg:
                        arg[direction]['model_data'][0] = None
                        n[direction] = arg[direction]['model_data']
