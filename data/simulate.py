import sys
import time
import json
import os
import preprocess
import re

from pycparser import c_parser, c_ast, parse_file

import numpy as np
import tensorflow as tf
import queue as Q
#import check_correct

import dump_ast

flags = tf.flags

flags.DEFINE_string("data_path", None,
                    "XXX")

FLAGS = flags.FLAGS

config = {
    'max_changes': 5,
    'types': ['swap', 'replace', 'insert', 'parent_delete', 'subtree_delete', 'attr_replace']
}

def search_changes(ast_nodes, node_properties):
    nvlist = [(n, getattr(node, n)) for n in node.attr_names]
    for (name, val) in nvlist:
        if name in ['value', 'op', 'name']:
            setattr(node, name, node_properties[node]['attr_expected'])
            if num_changes == max_changes - 1:
                #try:
                    #code = directives + generator.visit(ast)
                    path = os.path.join(FLAGS.task_path, '.' + filename + '.c')
                    with open(path, 'w') as f:
                        f.write(code)
                    ret = check_correct.check_vigenere(path)
                    os.unlink(path)
                    if ret == 0:
                        return code
                #except Exception:
                #    #print('uh ohhh')
                #    pass
            else:
                ret = search_changes(ast, node_properties, list_q, max_changes, filename, directives, start=i+1, num_changes=num_changes+1)
                # Success! The ast is now repaired
                if ret is not False:
                    return ret
            # no luck, revert to the old value
            setattr(node, name, val)
            break
    # didn't find a working tree

def preprocess_files(files):
    classes = {}
    for f in files:
        nodes = f['ast_nodes']
        for node in nodes:
            attrs = node['attrs']
            cname = node['self'].__class__.__name__
            for (name, val) in attrs:
                if name in ['value', 'op', 'name']:
                    if cname not in classes:
                        classes[cname] = set()
                    classes[cname].add(val)
                    break
    print(classes)

def main():
    parser = c_parser.CParser()
    files = []
    for filename in os.listdir(FLAGS.data_path):
        try:
            filename = os.path.join(FLAGS.data_path, filename, 'vigenere.c')
            #with open(filename) as f:
            #    text = f.read()
            #directives, _ = preprocess.grab_directives(text)

            # XXX this can return None
            ast_nodes, ast, node_properties, tokens = preprocess.preprocess_c(filename,
                    include_dependencies=True)
            files.append({
                'ast_nodes': ast_nodes,
                'ast': ast,
                'node_properties': node_properties,
                'tokens': tokens,
                'filename': filename
            })
        except:
            continue
    preprocess_files(files)

if __name__ == '__main__':
    main()

