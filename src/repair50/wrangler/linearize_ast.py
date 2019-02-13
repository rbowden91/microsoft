from ..my_env import sys
from ..my_env.copy import deepcopy
from ..my_env.packages.pycparser import c_ast

from ..my_env.typing import Dict, Any

POINTER_MEMORY_SIZE = 20

class WrangledAST(object):
    def __init__(self, ast : c_ast.Node, node_properties, results, visited, include_dependencies : bool = True) -> None:
        self.pointer_memory = [0] * POINTER_MEMORY_SIZE
        self.include_dependencies = include_dependencies
        self.ast = ast
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
                'right_child': 0,
                'pointers': {'memory': [0] * POINTER_MEMORY_SIZE, 'mask': [0] * POINTER_MEMORY_SIZE, 'ids': []}
            },

            'reverse': {
                'self': 0,
                # these are all set by the relevant dependency if they exist
                'left_child': 0,
                'left_sibling': 0,
                'left_prior': 0,
                'right_child': 0,
                'pointers': {'memory': [0] * POINTER_MEMORY_SIZE, 'mask': [0] * POINTER_MEMORY_SIZE, 'ids': []}
            },
        }
        if self.include_dependencies:
            self.default_props['dependencies'] = { 'self': None }

        self.nodes = {
            'forward': [deepcopy(self.default_props)],
            'reverse': [deepcopy(self.default_props)]
        }
        self.node_properties : Dict[c_ast.Node, Any] = node_properties

        # dependencies
        self.parent = 0
        self.left_sibling = 0
        self.hole = 0

        self.generic_visit()

    def handle_pointers(self, node, direction):
        props = self.node_properties[node]
        pointers = []
        if 'pointers' in props:
            for n in props['pointers']:
                # removed typedecls and what not
                if direction in self.node_properties[n] and self.node_properties[n][direction]['self'] != 0 and n != node:
                    pointers.append(self.node_properties[n][direction]['self'])
        props[direction]['pointers']['ids'] = pointers
        # TODO: filter by scope???
        if props['attr'] == 'LOCAL_ID':
            props[direction]['pointers']['memory'] = deepcopy(self.pointer_memory)
            props[direction]['pointers']['mask'] = [(1 if pointer in props[direction]['pointers']['ids'] else 0) for pointer in props[direction]['pointers']['memory']]
            self.pointer_memory.pop()
            self.pointer_memory.insert(0, props[direction]['self'])


        if 'pointers' in props and direction == 'reverse':
            props['pointers'] = [self.node_properties[n]['node_num'] for n in props['pointers']]

    def generic_visit_reverse(self, node):
        if 'removed' in self.node_properties[node]:
            return False

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

        self.handle_pointers(node, 'reverse')

        children = node.children()
        self.hole = 0
        for i in range(len(children)-1, -1, -1):
            if self.generic_visit_reverse(children[i][1]):
                self.hole = self.node_properties[children[i][1]]['reverse']['self']

        props['forward']['right_hole'] = self.nodes['reverse'][props['reverse']['right_hole']]['forward']['self']
        props['reverse']['left_hole'] = self.nodes['forward'][props['forward']['left_hole']]['reverse']['self']

        return True

    def generic_visit_forward(self, node):
        if 'removed' in self.node_properties[node]:
            self.node_properties[node]['pointers'] = []
            return False


        props = deepcopy(self.default_props)
        props.update(self.node_properties[node])

        # strongly assumes there is at most one attribute we care about
        nvlist = [(n, getattr(node, n)) for n in node.attr_names]
        attr = None
        for (name, val) in nvlist:
            if name in ['value', 'op', 'name', 'declname']:
                attr = val if 'replace_name' not in props else props['replace_name']
            elif name == 'names':
                attr = ' '.join(val)
            else:
                #print(name, val)
                pass

        nodes = self.nodes['forward']
        my_node_num = len(nodes)

        children = node.children()


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
        if self.visited is not None:
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

        self.handle_pointers(node, 'forward')

        self.left_sibling = 0
        for i in range(len(children)):
            self.parent = props['forward']['self']
            if self.generic_visit_forward(children[i][1]):
                self.left_sibling = self.hole = self.node_properties[children[i][1]]['forward']['self']

        if self.include_dependencies:
            for i in props['forward']:
                if i == 'pointers': continue
                props['dependencies'][i] = nodes[props['forward'][i]]['dependencies']['self']


        return True


    def generic_visit(self):
        self.generic_visit_forward(self.ast)
        self.hole = 0
        self.pointer_memory = [0] * POINTER_MEMORY_SIZE
        self.generic_visit_reverse(self.ast)
        # delete the nil slot
        self.nodes['forward'][0] = None
        self.nodes['reverse'][0] = None
