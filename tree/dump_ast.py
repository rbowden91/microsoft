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

ignore = ['Typedef', 'Decl', 'TypeDecl', 'DeclList']

def linearize_ast(ast, ret=[], parent=0, sibling=0, prior=0, first_sibling=True, last_sibling=True, node_properties=None):
    if ast.__class__.__name__ in ignore:
        return False

    nvlist = [(n, getattr(ast,n)) for n in ast.attr_names]

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
        'right_prior': 0, # same as with righ sibling

        'first_sibling': first_sibling,
        'last_sibling': last_sibling,
        'is_leaf': True,
        'num_children': 0,
        'attrs': nvlist,
    }
    ret.append(node)

    if node_properties is not None:
        node_properties[ast] = node
        node['self'] = ast
        node['dependencies'] = {
            'parent': ret[parent-1]['self'] if parent is not 0 else None,
            'left_sibling': ret[sibling-1]['self'] if sibling is not 0 else None,
            'right_sibling': None,
            'left_prior': ret[prior-1]['self'] if prior is not 0 else None,
            'right_prior': None
        }

    if sibling != 0:
        # need to subtract 1, because nodes are 1-indexed instead of 0-indexed
        ret[sibling-1]['right_sibling'] = my_node_num
        if node_properties is not None:
            ret[sibling-1]['dependencies']['right_sibling'] = ast

    if prior != 0:
        ret[prior-1]['right_prior'] = my_node_num
        if node_properties is not None:
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
        last_child = len(ret)
        child_result = linearize_ast(
            child,
            ret=ret,
            parent=my_node_num,
            sibling=sibling,
            prior=prior,
            first_sibling=first_sibling,
            last_sibling = False,
            node_properties=node_properties)
        if child_result is False:
            continue
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

if __name__ == "__main__":
    argparser = argparse.ArgumentParser('Dump AST')
    argparser.add_argument('filename', help='name of file to parse')
    args = argparser.parse_args()

    ast = parse_file(args.filename, use_cpp=True, cpp_path='gcc', cpp_args=['-E', r'-I../fake_libc_include'])
    print(json.dumps(linearize_ast(ast)))

