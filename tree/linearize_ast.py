import sys
from copy import deepcopy
from pycparser import c_generator, c_ast, c_lexer, c_parser, preprocess_file

# doesn't really use anything from c_ast.NodeVisitor
class LinearizeAST(c_ast.NodeVisitor):
    def __init__(self, include_dependencies=True):
        self.include_dependencies = include_dependencies

        # for the empty dependency in slot 0
        self.default_props = {
            'label': '<nil>',
            'attr': '<nil>',
            'forward': {
                'self': 0,
                # these are all set by the relevant dependency if they exist
                'left_child': 0,
                'right_sibling': 0,
                'right_prior': 0,
                'right_child': 0
            },

            'reverse': {
                'self': 0,
                # these are all set by the relevant dependency if they exist
                'left_child': 0,
                'left_sibling': 0,
                'left_prior': 0,
                'right_child': 0
            },
        }
        if self.include_dependencies:
            self.default_props['dependencies'] = { 'self': None }

        self.nodes = {
            'forward': [deepcopy(self.default_props)],
            'reverse': [deepcopy(self.default_props)]
        }
        self.node_properties = {}

        # dependencies
        self.parent = 0
        self.left_sibling = 0

    def generic_visit_reverse(self, node):
        props = self.node_properties[node]
        my_node_num = len(self.nodes['reverse'])
        props['reverse'].update({
            'self': my_node_num,
            'parent': self.nodes['forward'][props['forward']['parent']]['reverse']['self'],
            'right_sibling': self.nodes['forward'][props['forward']['right_sibling']]['reverse']['self'],
            'right_prior': my_node_num - 1,
        })
        props['last_sibling'] = props['forward']['right_sibling'] == 0

        self.nodes['reverse'].append(props)

        if self.nodes['reverse'][props['reverse']['parent']]['reverse']['right_child'] == 0:
            self.nodes['reverse'][props['reverse']['parent']]['reverse']['right_child'] = my_node_num
        self.nodes['reverse'][props['reverse']['parent']]['reverse']['left_child'] = my_node_num
        self.nodes['forward'][props['forward']['right_sibling']]['reverse']['left_sibling'] = my_node_num
        self.nodes['forward'][props['forward']['right_prior']]['reverse']['left_prior'] = my_node_num

        children = node.children()
        for i in range(len(children)-1, -1, -1):
            self.generic_visit_reverse(children[i][1])

    def generic_visit_forward(self, node):
        # strongly assumes there is at most one attribute we care about
        nvlist = [(n, getattr(node, n)) for n in node.attr_names]
        attr = None
        for (name, val) in nvlist:
            if name in ['value', 'op', 'name', 'declname']:
                attr = val
            else:
                #print(name, val)
                pass

        nodes = self.nodes['forward']
        my_node_num = len(nodes)

        # assumes that nodes we want to ignore have already been filtered out
        children = node.children()

        props = deepcopy(self.default_props)
        props.update({
            'label': node.__class__.__name__,
            'first_sibling': self.left_sibling == 0,
            'num_children': len(children),
            'is_leaf': len(children) == 0,
            'attr': attr,
            # this is the only dependency we need for the TreeLSTMs
            'parent_label': nodes[self.parent]['label'],
            'parent_attr': nodes[self.parent]['attr']
        })
        props['forward'].update({
            'self': my_node_num,
            'parent': self.parent,
            'left_sibling': self.left_sibling,
            'left_prior': my_node_num - 1,
        })
        if self.include_dependencies:
            props['dependencies']['self'] = node
        nodes.append(props)
        self.node_properties[node] = props

        if nodes[self.parent]['forward']['left_child'] == 0:
            nodes[self.parent]['forward']['left_child'] = my_node_num

        nodes[self.parent]['forward']['right_child'] = my_node_num
        nodes[self.left_sibling]['forward']['right_sibling'] = my_node_num
        nodes[my_node_num - 1]['forward']['right_prior'] = my_node_num

        for i in range(len(children)):
            self.parent = props['forward']['self']
            self.left_sibling = self.node_properties[children[i-1][1]]['forward']['self'] if i != 0 else 0
            self.generic_visit_forward(children[i][1])

        if self.include_dependencies:
            for i in props['forward']:
                props['dependencies'][i] = nodes[props['forward'][i]]['dependencies']['self']


    def generic_visit(self, node):
        self.generic_visit_forward(node)
        self.generic_visit_reverse(node)
        # delete the nil slot
        self.nodes['forward'][0] = None
        self.nodes['reverse'][0] = None