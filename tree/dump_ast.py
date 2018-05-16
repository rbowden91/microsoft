#-----------------------------------------------------------------
# pycparser: dump_ast.py
#
# Basic example of parsing a file and dumping its parsed AST.
#
# Eli Bendersky [http://eli.thegreenplace.net]
# License: BSD
#-----------------------------------------------------------------
from __future__ import print_function
import argparse
import sys
import json

from pycparser import c_parser, c_ast, parse_file

#ignore = ['Typedef', 'Decl', 'TypeDecl', 'DeclList']
#ignore = ['Typedef', 'TypeDecl']
ignore = []

# right now, only does right holes
def fill_holes_helper(ast, generator, nodes, is_left_hole=True, hole=0, hole_node=None, include_dependencies=True):
    if ast.__class__.__name__ in ignore:
        return False
    if generator is not None and ast.__class__.__name__ == 'Typedef' and generator.visit(ast) == '':
        return False
    children = ast.children()

    if is_left_hole:
        nodes[ast]['left_hole'] = hole
        if include_dependencies:
            nodes[ast]['dependencies']['left_hole'] = hole_node
        enumerator = range(len(children))
    else:
        nodes[ast]['right_hole'] = hole
        if include_dependencies:
            nodes[ast]['dependencies']['right_hole'] = hole_node
        enumerator = range(len(children) - 1, -1, -1)

    for i in enumerator:
        (child_name, child) = children[i]
        if fill_holes_helper(child, generator, nodes, is_left_hole, hole, hole_node, include_dependencies):
            hole = nodes[child]['node_number']
            hole_node = child
    return True

def fill_holes(ast, generator, nodes, include_dependencies):
    fill_holes_helper(ast, generator, nodes, True, include_dependencies=include_dependencies)
    fill_holes_helper(ast, generator, nodes, False, include_dependencies=include_dependencies)

def linearize_ast_helper(ast, generator, ret, parent=0, sibling=0, prior=0, first_sibling=True, last_sibling=True, node_properties=None, include_dependencies=True):
    if ast.__class__.__name__ in ignore:
        return False
    if generator is not None and ast.__class__.__name__ == 'Typedef' and generator.visit(ast) == '':
        return False

    nvlist = [(n, getattr(ast,n)) for n in ast.attr_names]

    ## Fix up IDs...
    #for i in range(len(nvlist)):
    #    if nvlist[i][0]  == 'name':
    #        nvlist[i] = (nvlist[i][0], 'ID')

    my_node_num = len(ret) + 1

    children = ast.children()
    node = {
        'label': ast.__class__.__name__,
        'node_number': my_node_num, # should just match position in array, +1
        'parent': parent,

        'left_sibling': sibling,
        'right_sibling': 0, # if there is a right sibling, then that sibling will set this value
                            # on the current node when we reach it
        'left_prior': prior,
        'right_prior': 0, # same as with right sibling
        'left_child': 0, # same
        'right_child': 0, # same

        'first_sibling': first_sibling,
        'last_sibling': last_sibling,
        'is_leaf': True,
        'num_children': 0,
        'attrs': nvlist,
    }
    ret.append(node)

    node_properties[ast] = node
    if include_dependencies:
        node['self'] = ast
        node['dependencies'] = {
            'parent': ret[parent-1]['self'] if parent is not 0 else None,
            'left_sibling': ret[sibling-1]['self'] if sibling is not 0 else None,
            'left_prior': ret[prior-1]['self'] if prior is not 0 else None,
            'right_sibling': None,
            'right_prior': None,
            'left_child': None,
            'right_child': None
        }

    if parent != 0:
        if ret[parent-1]['left_child'] is None:
            ret[parent-1]['left_child'] = my_node_num
            if include_dependencies:
                ret[parent-1]['dependencies']['left_child'] = ast

        # further right children will overwrite this if we aren't actually the last
        ret[parent-1]['right_child'] = my_node_num
        if include_dependencies:
            ret[parent-1]['dependencies']['right_child'] = ast

    if sibling != 0:
        # need to subtract 1, because nodes are 1-indexed instead of 0-indexed
        ret[sibling-1]['right_sibling'] = my_node_num
        if include_dependencies:
            ret[sibling-1]['dependencies']['right_sibling'] = ast

    if prior != 0:
        ret[prior-1]['right_prior'] = my_node_num
        if include_dependencies:
            ret[prior-1]['dependencies']['right_prior'] = ast

    new_number = my_node_num+1
    sibling = 0
    # XXX compare this against "= 0", which wouldn't cause parent and prior to be equal
    # until then, this is always just current_node_num-1
    prior = my_node_num
    first_sibling = True
    num_children = 0
    last_child = None

    # actual children
    for i in range(len(children)):
        (child_name, child) = children[i]
        last_child_tmp = len(ret)
        child_result = linearize_ast_helper(
            child,
            generator,
            ret=ret,
            parent=my_node_num,
            sibling=sibling,
            prior=prior,
            first_sibling=first_sibling,
            last_sibling=False,
            node_properties=node_properties,
            include_dependencies=include_dependencies)
        if child_result is False:
            continue
        last_child = last_child_tmp
        first_sibling = False
        num_children += 1
        sibling = new_number
        new_number = ret[-1]['node_number'] + 1
        # XXX theoretically, sibling and prior could be equal.
        prior = ret[-1]['node_number']
    if num_children > 0:
        ret[last_child]['last_sibling'] = True
        node['is_leaf'] = False;
        node['num_children'] = num_children

    return ret

def linearize_ast(ast, generator=None, include_dependencies=True):
    node_properties = {}
    ret = linearize_ast_helper(ast, generator, [], node_properties=node_properties, include_dependencies=include_dependencies)
    fill_holes(ast, generator, node_properties, include_dependencies)
    return ret, node_properties

if __name__ == "__main__":
    argparser = argparse.ArgumentParser('Dump AST')
    argparser.add_argument('filename', help='name of file to parse')
    args = argparser.parse_args()

    ast = parse_file(args.filename, use_cpp=True, cpp_path='gcc', cpp_args=['-E', r'-I../fake_libc_include'])
    ret, node_properties = linearize_ast(ast, include_dependencies=False)
    print(json.dumps(ret))

