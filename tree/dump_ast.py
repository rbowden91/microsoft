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

def print_ast(ast, attrnames=False, parent=0, sibling=0, my_number=1, last_sibling=True):
    if ast.__class__.__name__ == 'Typedef' or ast.__class__.__name__ == 'Decl' or ast.__class__.__name__ == 'TypeDecl' or ast.__class__.__name__ == 'DeclList':
        return False

    children = ast.children()
    node = {
        'name': ast.__class__.__name__,
        'node_number': my_number,
        'parent': parent,
        'sibling': sibling,
        'last_sibling': last_sibling,
        'leaf_node': len(children) == 0
    }
    ret = [node]

    #if nodenames and _my_node_name is not None:
    #    buf.write(lead + ast.__class__.__name__+ ' <' + _my_node_name + '>: ')
    #else:
    #    buf.write(lead + ast.__class__.__name__+ ' ')

    if ast.attr_names:
        if attrnames:
            nvlist = [(n, getattr(ast,n)) for n in ast.attr_names]
            #attrstr = ', '.join('%s=%s' % nv for nv in nvlist)
            attrstr = ' '.join('%s=%s' % nv for nv in nvlist)
        else:
            vlist = [getattr(ast, n) for n in ast.attr_names]
            #attrstr = ', '.join('%s' % v for v in vlist)
            attrstr = ' '.join('%s' % v for v in vlist)

    sibling = 0
    new_number = my_number + 1
    for i in range(len(children)):
        (child_name, child) = children[i]
        child_result = print_ast(
            child,
            attrnames=attrnames,
            parent=my_number,
            sibling=sibling,
            my_number=new_number,
            last_sibling = i == len(children) - 1)
        if child_result is False:
            continue
        ret.extend(child_result)
        sibling = new_number
        new_number = ret[-1]['node_number'] + 1
    return ret

if __name__ == "__main__":
    argparser = argparse.ArgumentParser('Dump AST')
    argparser.add_argument('filename', help='name of file to parse')
    args = argparser.parse_args()

    ast = parse_file(args.filename, use_cpp=True, cpp_path='gcc', cpp_args=['-E', r'-I../fake_libc_include'])
    print(print_ast(ast))

