from ..my_env import sys
from ..my_env.copy import deepcopy
from ..my_env.packages.pycparser import c_ast

from ..my_env.typing import Dict, Any

class WrangledAST(object):
    def __init__(self, ast : c_ast.Node, orig_ast : c_ast.Node, name_map, results, visited, include_dependencies : bool = True) -> None:
        self.include_dependencies = include_dependencies
        self.ast = ast
        self.orig_ast = orig_ast
        self.name_map = name_map
        self.results = results
        self.visited = visited

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
        self.node_properties : Dict[c_ast.Node, Any] = {}

        # dependencies
        self.parent = 0
        self.left_sibling = 0
        self.hole = 0

        self.generic_visit()

    def generic_visit_reverse(self, node):
        props = self.node_properties[node]
        my_node_num = len(self.nodes['reverse'])
        props['reverse'].update({
            'self': my_node_num,
            'parent': self.nodes['forward'][props['forward']['parent']]['reverse']['self'],
            'right_sibling': self.nodes['forward'][props['forward']['right_sibling']]['reverse']['self'],
            'right_prior': my_node_num - 1,
            'right_hole': self.hole
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
            self.hole = self.node_properties[children[i][1]]['reverse']['self']

        props['forward']['right_hole'] = self.nodes['reverse'][props['reverse']['right_hole']]['forward']['self']
        props['reverse']['left_hole'] = self.nodes['forward'][props['forward']['left_hole']]['reverse']['self']

    def generic_visit_forward(self, node):
        # strongly assumes there is at most one attribute we care about
        nvlist = [(n, getattr(node, n)) for n in node.attr_names]
        attr = None
        for (name, val) in nvlist:
            if name in ['value', 'op', 'name', 'declname']:
                attr = val
            elif name == 'names':
                attr = ' '.join(val)
            else:
                #print(name, val)
                pass

        nodes = self.nodes['forward']
        my_node_num = len(nodes)

        # assumes that nodes we want to ignore have already been filtered out
        children = node.children()

        props = deepcopy(self.default_props)
        props.update({
            'visited': {},
            'label': node.__class__.__name__,
            'first_sibling': self.left_sibling == 0,
            'num_children': len(children),
            'is_leaf': len(children) == 0,
            'attr': attr,
            # this is the only dependency we need for the TreeLSTMs
            'parent_label': nodes[self.parent]['label'],
            'parent_attr': nodes[self.parent]['attr']
        })
        for i in range(len(self.visited)):
            if node in self.visited[i]:
                props['visited'][i] = True
                self.visited[i][node] = my_node_num

        props['forward'].update({
            'self': my_node_num,
            'parent': self.parent,
            'left_sibling': self.left_sibling,
            'left_hole': self.hole,
            'left_prior': my_node_num - 1
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
            self.hole = self.node_properties[children[i][1]]['forward']['self']

        if self.include_dependencies:
            for i in props['forward']:
                props['dependencies'][i] = nodes[props['forward'][i]]['dependencies']['self']


    def generic_visit(self):
        self.generic_visit_forward(self.ast)
        self.hole = 0
        self.generic_visit_reverse(self.ast)
        # delete the nil slot
        self.nodes['forward'][0] = None
        self.nodes['reverse'][0] = None
