import os, sys
import collections
from typing import List, Tuple
from pycparser import c_parser, c_generator, c_lexer # type:ignore
from ..preprocessor.external import ExternalCPP # type:ignore

from centipyde.interpreter import run_tests # type:ignore
from .normalizers import normalize
from .linearize_ast import WrangledAST
from ..default_dict import data_dict

def process_ast(ast, lexicon=None, transitions_groups=None):
    all_new_rows = data_dict(lambda: collections.defaultdict(list))

    for test in ast.nodes:
        for root_node in list(ast.nodes[test].keys()):
            for transitions in ast.nodes[test][root_node]:
                local_lexicon = {
                    'label': set(),
                    'attr': set(),
                    'transitions': set()
                }
                nodes_ = ast.nodes[test][root_node][transitions]
                if not nodes_: continue

                root_transitions = ast.prop_map[root_node]['props'][test][root_node][transitions]['transitions']
                new_row = all_new_rows[test][root_node][transitions]

                for k in ['reverse', 'forward']:
                    nodes = nodes_[k]
                    # skip the nil slot
                    nodes.pop(0)

                    for node in nodes:
                        for feature, val in node.items():
                            if feature in ['snapshots']: continue
                            if val is None:
                                node[feature] = '<nil>'
                            if lexicon is not None and feature in local_lexicon and val not in local_lexicon[feature]:
                                lexicon[test][root_transitions][transitions][feature][val] += 1
                                local_lexicon[feature].add(val)

                            if feature not in ['forward', 'reverse']:
                                new_row[k + '-' + feature].append(int(val) if isinstance(val, bool) else val)
                            else:
                                if feature != k: continue
                                for dep, dep_val in val.items():
                                    if dep == 'pointers':
                                        # FIXME: release limit of pointer memory of 20
                                        if len(dep_val['memory']) < 20:
                                            dep_val['memory'].extend([0] * (20 - len(dep_val['memory'])))
                                            dep_val['mask'].extend([0] * (20 - len(dep_val['mask'])))
                                        if k == 'forward':
                                            dep_val['memory'] = dep_val['memory'][-20:]
                                            dep_val['mask'] = dep_val['mask'][-20:]
                                        else:
                                            dep_val['memory'] = dep_val['memory'][:20]
                                            dep_val['mask'] = dep_val['mask'][:20]

                                        for q in range(len(dep_val['memory'])):
                                            # TODO: use sequence example instead or something?
                                            new_row['-'.join([k,dep,'memory',str(q)])].append(dep_val['memory'][q])
                                            new_row['-'.join([k,dep,'mask',str(q)])].append(dep_val['mask'][q])
                                    else:
                                        new_row['-'.join([k,dep])].append(int(dep_val) if isinstance(dep_val, bool) else dep_val)


    if transitions_groups:
        for test in ast.transitions_groups:
            for transition in ast.transitions_groups[test]:
                for test2 in ast.transitions_groups[test][transition]:
                    for transition2 in ast.transitions_groups[test][transition][test2]:
                        transitions_groups[test][transition][test2][transition2] += ast.transitions_groups[test][transition][test2][transition2]

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

def finish_row(row, lexicon, root_lexicon):
    # all rows should have a forward label
    row_len = len(row['forward-label'])

    for i in ['forward-', 'reverse-']:
        for j in ['parent_', '']:
            for k in ['label', 'attr', 'transitions', 'root_transitions']:
                idx = i + j + k
                if idx in row:
                    if k == 'root_transitions':
                        key = 'transitions'
                        lex = root_lexicon
                    else:
                        key = k
                        lex = lexicon

                    row[idx + '_index'] = tokens_to_ids(row[idx], lex[key])
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
    ast = normalize(ast, ['ExpandAssignments', 'WrapExpressions'])
    results = run_tests(ast, tests) if tests is not None else None
    # TODO: remove decls can't be interpreted while variables share the same name in different scopes
    ast = normalize(ast, ['RemoveTypedefs', 'RemoveDecls', 'IDRenamer'])


    ast_data = WrangledAST(ast, results)

    return ast_data

