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

def print_ast(ast, ret=[], parent=0, sibling=0, prior=0, last_sibling=True):
    if ast.__class__.__name__ == 'Typedef' or ast.__class__.__name__ == 'Decl' or ast.__class__.__name__ == 'TypeDecl' or ast.__class__.__name__ == 'DeclList':
        return False

    nvlist = [(n, getattr(ast,n)) for n in ast.attr_names]

    my_node_num = len(ret)

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

        'last_sibling': last_sibling,
        'is_leaf': len(children) == 0,
        'num_children': len(children),
        'attrs': nvlist,
    }
    ret.append(node)

    if sibling != 0:
        ret[sibling]['right_sibling'] = my_node_num

    if prior != 0:
        ret[prior]['right_prior'] = my_node_num

    new_number = my_node_num+1
    sibling = 0
    # XXX compare this against "= 0", which wouldn't cause parent and prior to be equal
    # until then, this is always just current_node_num-1
    prior = my_node_num

    # actual children
    for i in range(len(children)):
        (child_name, child) = children[i]
        child_result = print_ast(
            child,
            ret=ret,
            parent=my_node_num,
            sibling=sibling,
            prior=prior,
            last_sibling = i == len(children) - 1)
        if child_result is False:
            continue
        sibling = new_number
        new_number = ret[-1]['node_number'] + 1
        # XXX theoretically, sibling and prior could be equal.
        prior = ret[-1]['node_number']
    return ret

if __name__ == "__main__":
    argparser = argparse.ArgumentParser('Dump AST')
    argparser.add_argument('filename', help='name of file to parse')
    args = argparser.parse_args()

    ast = parse_file(args.filename, use_cpp=True, cpp_path='gcc', cpp_args=['-E', r'-I../fake_libc_include'])
    print(json.dumps(print_ast(ast)))

