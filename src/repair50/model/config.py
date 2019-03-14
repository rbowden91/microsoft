from typing import NamedTuple

valid_dependencies = {
    'children': 'bottom',
    'parent': 'top',
    'left_prior': 'left',
    'left_sibling': 'left',
    'right_prior': 'right',
    'right_sibling': 'right',

    'right_hole': 'right',
    'left_hole': 'left',

    # TODO: these could technically be either left or right?
    'left_child': 'left',
    'right_child': 'right',
}

# 'top/bottom' and 'left/right' end up replaced by 'True/False'
dependency_configs = {
    # should never need to have a dependency appear twice in the list
    'left_sibling': [('top', 'left', ['left_sibling'])],
    'children': [('bottom', 'right', ['children'])],
    'right_sibling': [('bottom', 'right', ['right_sibling'])],
    'parent': [('top', 'left', ['parent'])],
    'd1': [('bottom', 'right', ['children', 'right_sibling'])],
    'd2': [('top', 'left', ['parent', 'left_sibling'])],
    'd3': [('top', 'right', ['parent', 'right_sibling'])],
    # TODO: validate that it makes sense to traverse in this order?
    # TODO: dependency configs in "server" should not read from this
    'd4': [('bottom', 'left', ['children']), ('top', 'left', ['parent', 'left_sibling', 'left_hole'])],
    'd5': [('bottom', 'right', ['children']), ('top', 'right', ['parent', 'right_sibling', 'right_hole'])],
    'd6': [('top', 'left', ['parent', 'left_sibling', 'left_prior'])],
    'd7': [('top', 'right', ['parent', 'right_sibling', 'right_prior'])],
    #'d3': [('bottom_right', [
    #{ 'bottom_right': [['children'], ['right_children'],
    #                   ['children', 'initial_right_children']]},
    #{ 'bottom_left': [['left_children'],
    #                   ['children', 'initial_left_children'],
    #                   ['children', 'initial_left_children', 'initial_right_children']]},
    #{ 'top_left': [['parent', 'left_sibling'],
    #               ['parent', 'left_sibling', 'initial_left_children'],
    #               ['parent', 'left_sibling', 'initial_left_children', 'initial_right_children']]},
    #{ 'top_right': [['parent', 'right_sibling'],
    #               ['parent', 'right_sibling', 'initial_right_children'],
    #               ['parent', 'right_sibling', 'initial_left_children', 'initial_right_children']]}
}

joint_configs = {
    'j1': ['d1', 'd2'],
    'j2': ['d2', 'd3']
}

# TODO: make actual exceptions for these

for config in dependency_configs:
    for i in range(len(dependency_configs[config])):
        top_down, left_right, dependencies = dependency_configs[config][i]
        # validate the dependency configs
        for d in dependencies:
            if d not in valid_dependencies or \
                    valid_dependencies[d] != top_down and valid_dependencies[d] != left_right:
                raise Exception('dependency_configs["{}"] is invalid'.format(config))

        # which way we traverse the tree
        forward_tree = top_down == 'top' and left_right == 'left' \
                    or top_down == 'bottom' and left_right == 'right'

        # which way we traverse the array
        forward_array = (left_right == 'left') == forward_tree

        dependency_configs[config][i] = (forward_tree, forward_array, dependencies)

# validate the joint configs
for config in joint_configs:
    for dconfig in joint_configs[config]:
        assert dconfig in dependency_configs
