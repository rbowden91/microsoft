from ..my_env.typing import NamedTuple

valid_dependencies = {
    'children': 'bottom',
    'parent': 'top',
    'left_prior': 'left',
    'left_child': 'left',
    'left_sibling': 'left',
    'right_prior': 'right',
    'right_child': 'right',
    'right_sibling': 'right',

    # FIXME
    'right_hole': 'left'
}

# 'top/bottom' and 'left/right' end up replaced by 'True/False'
dependency_configs = {
    'ast': {
        # should never need to have a dependency appear twice in the list
        'left_sibling': [('top', 'left', ['left_sibling'])],
        'children': [('bottom', 'right', ['children'])],
        'right_sibling': [('bottom', 'right', ['right_sibling'])],
        'parent': [('top', 'left', ['parent'])],
        'd1': [('bottom', 'right', ['children', 'right_sibling'])],
        'd2': [('top', 'left', ['parent', 'left_sibling'])],
        'd3': [('top', 'right', ['parent', 'right_sibling'])],
        # TODO: validate that it makes sense to traverse in this order?
        'd4': [('bottom', 'left', ['children']), ('top', 'left', ['parent', 'left_sibling', 'right_hole'])],
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
    },
    'linear': {
        'reverse': [('bottom', 'right', ['right_sibling'])],
        'forward': [('top', 'left', ['left_sibling'])],
        'both': [('top', 'left', ['left_sibling']), ('bottom', 'right', ['right_sibling'])]
    }
}

joint_configs = {
    'ast': {
        'j1': ['d1', 'd2'],
        'j2': ['d2', 'd3']
    },
    'linear': {
        'both': [ 'forward', 'reverse' ]
    }
}

# TODO: make actual exceptions for these

for model in dependency_configs:
    for config in dependency_configs[model]:
        for i in range(len(dependency_configs[model][config])):
            top_down, left_right, dependencies = dependency_configs[model][config][i]
            # validate the dependency configs
            for d in dependencies:
                if d not in valid_dependencies or \
                        valid_dependencies[d] != top_down and valid_dependencies[d] != left_right:
                    raise Exception('dependency_configs["{}"]["{}"] is invalid'.format(model, config))

            # which way we traverse the tree
            forward_tree = top_down == 'top' and left_right == 'left' \
                        or top_down == 'bottom' and left_right == 'right'

            # which way we traverse the array
            forward_array = (left_right == 'left') == forward_tree

            dependency_configs[model][config][i] = (forward_tree, forward_array, dependencies)

# validate the joint configs
for model in joint_configs:
    for config in joint_configs[model]:
        for dconfig in joint_configs[model][config]:
            assert dconfig in dependency_configs[model]
