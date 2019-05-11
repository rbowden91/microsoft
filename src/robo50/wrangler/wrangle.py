import os, sys
import collections
from typing import List, Tuple
from pycparser import c_parser, c_generator, c_lexer # type:ignore
from ..preprocessor.external import ExternalCPP # type:ignore

from centipyde.interpreter import run_tests # type:ignore
from .normalizers import normalize
from .linearize_ast import WrangledAST
from ..default_dict import get_dict, get_dict_default

def add_to_row(rows, key, val):
    feature_row = get_dict_default(rows, key, [])
    feature_row[key].append(int(val) if isinstance(val, bool) else val)

def process_ast(ast, lexicon=None):
    all_new_rows = {}

    for test in ast.nodes:
        for root_node in ast.nodes[test]:
            root_transitions = ast.prop_map[root_node]['test_data'][test]['transitions']
            for transitions in ast.nodes[test][root_node]:
                local_lexicon = {
                    'label': set(),
                    'attr': set(),
                    'transitions': set()
                }
                nodes_ = ast.nodes[test][root_node][transitions]

                new_rows = get_dict(all_new_rows, test, root_node, transitions)

                for k in ['reverse', 'forward']:
                    nodes = nodes_[k]
                    # skip the nil slot
                    nodes.pop(0)
                    for node in nodes:
                        for feature, val in node.items():
                            if val is None:
                                node[feature] = '<nil>'
                            if lexicon is not None and feature in local_lexicon and val not in local_lexicon[feature]:
                                l = get_dict_default(lexicon, test, root_transitions, transitions, feature, val, 0)
                                l[val] += 1
                                local_lexicon[feature].add(val)

                            if feature not in ['forward', 'reverse']:
                                add_to_row(new_rows, k + '-' + feature, val)
                            else:
                                if feature != k: continue
                                for dep, dep_val in val.items():
                                    if dep == 'pointers':
                                        mem_size = len(dep_val['memory'])
                                        dep_val['filter'] = [1] * mem_size
                                        for ptr_val in dep_val:
                                            # FIXME: release limit of pointer memory of 20
                                            if len(dep_val[ptr_val]) < 20:
                                                dep_val[ptr_val].extend([0] * (20 - len(dep_val[ptr_val])))
                                            dep_val[ptr_val] = dep_val[ptr_val][-20:]
                                            for q in range(len(dep_val[ptr_val])):
                                                # TODO: use sequence example instead or something?
                                                add_to_row(new_rows, '-'.join([k,dep,ptr_val,str(q)]), dep_val[ptr_val][q])
                                    else:
                                        add_to_row(new_rows, k + '-' + dep, int(dep_val) if isinstance(dep_val, bool) else dep_val)

    return all_new_rows

def tokens_to_ids(tokens, token_to_id):
    output = []
    for i in range(len(tokens)):
        token = tokens[i]
        if token in token_to_id:
            id = token_to_id[token]
        #elif token.startswith('"'):
            # TODO: do this for all types?
            #id = token_to_id['<unk_str>']
        else:
            id = token_to_id['<unk>']
        output.append(id)
    return output

def finish_row(row, lexicon):
    # all rows should have a forward label
    row_len = len(row['forward-label'])

    for i in ['forward-', 'reverse-']:
        for j in ['parent_', '']:
            for k in ['label', 'attr', 'transitions']:
                idx = i + j + k
                if idx in row:
                    row[idx + '_index'] = tokens_to_ids(row[idx], lexicon[k])
                    del(row[idx])
                else:
                    row[idx + '_index'] = [0] * row_len
        # the nil slot is the only slot that's masked out, other than potential padded batches
        row[i + 'mask'] = [1] * row_len

    # insert the nil slot
    for i in row:
        row[i] = [0] + row[i]
    return row

def wrangle(code : str, is_file=True, tests=None) -> WrangledAST:
    # these can raise exceptions that we'll let pass through
    cfile = ExternalCPP().preprocess(code, is_file)
    ast = c_parser.CParser().parse(cfile)
    ast = normalize(ast, ['ExpandAssignments', 'WrapExpressions', 'ReturnZero'])
    results = run_tests(ast, tests) if tests is not None else None
    # TODO: remove decls can't be interpreted while variables share the same name in different scopes
    ast = normalize(ast, ['RemoveTypedefs', 'RemoveDecls', 'IDRenamer'])


    ast_data = WrangledAST(ast, results)

    return ast_data

