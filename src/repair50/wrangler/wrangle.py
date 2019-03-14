import os, sys
from typing import List, Tuple
from pycparser import c_parser, c_generator, c_lexer # type:ignore
from ..preprocessor.external import ExternalCPP # type:ignore

from centipyde.interpreter import run_tests # type:ignore
from .normalizers import normalize
from .linearize_ast import WrangledAST

def process_ast(ast, key=None, lexicon=None, transitions_groups=None, lock=None):
    all_new_rows = {}

    for transitions in ast.nodes:
        all_new_rows[transitions] = {}
        for test in ast.tests:
            local_lexicon = {
                'label': set(),
                'attr': set(),
                'transitions': set()
            }
            # skip the nil slot
            ast.nodes[transitions][test]['reverse'].pop(0)
            nodes = ast.nodes[transitions][test]['forward']
            nodes.pop(0)
            for i in range(len(nodes)):
                data = nodes[i]
                for j in ['attr', 'parent_attr', 'parent_label', 'transitions']:
                    data[j] = data[j] if data[j] is not None else '<nil>'

                for j in ['label', 'attr', 'transitions']:
                    if key == "train" and data[j] not in local_lexicon[j]:
                        with lock:
                            if data[j] not in lexicon[transitions][test][j]:
                                lexicon[transitions][test][j][data[j]] = 0
                            lexicon[transitions][test][j][data[j]] += 1
                        local_lexicon[j].add(data[j])

            all_new_rows[transitions][test] = new_rows = {}
            for k in ['reverse', 'forward']:
                nodes = ast.nodes[transitions][test][k]
                for i in nodes[0]:
                    if i == k:
                        for j in nodes[0][i]:
                            if j == 'pointers':
                                for q in range(len(nodes[0][i][j]['memory'])):
                                    # TODO: use sequence example instead or something?
                                    new_rows['-'.join([k,j,'memory',str(q)])] = [n[i][j]['memory'][q] for n in nodes]
                                    new_rows['-'.join([k,j,'mask',str(q)])] = [n[i][j]['mask'][q] for n in nodes]
                            else:
                                new_rows['-'.join([i,j])] = [int(n[i][j]) if isinstance(n[i][j], bool) else n[i][j] for n in nodes]
                    elif i in ['forward', 'reverse']: continue
                    else:
                        new_rows[k + '-' + i] = [int(n[i]) if isinstance(n[i], bool) else n[i] for n in nodes]

    if key == 'train':
        for test in ast.transitions_groups:
            if test not in transitions_groups:
                transitions_groups[test] = {}
            for transition in ast.transitions_groups[test]:
                if transition not in transitions_groups[test]:
                    transitions_groups[test][transition] = {}
                for test2 in ast.transitions_groups[test][transition]:
                    if test2 not in transitions_groups[test][transition]:
                        transitions_groups[test][transition][test2] = {}
                    for transition2 in ast.transitions_groups[test][transition][test2]:
                        if transition2 not in transitions_groups[test][transition][test2]:
                            transitions_groups[test][transition][test2][transition2] = 0
                        transitions_groups[test][transition][test2][transition2] += ast.transitions_groups[test][transition][test2][transition2]

    return all_new_rows

def tokens_to_ids(tokens, token_to_id, include_token):
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
        output.append((id, token) if include_token else id)
    return output

def finish_row(row, lexicon):
    # all rows should have a forward label
    row_len = len(row['forward-label'])

    for i in ['forward-', 'reverse-']:
        for j in ['parent_', '']:
            for k in ['label', 'attr', 'transitions']:
                idx = i + j + k
                if idx in row:
                    row[idx + '_index'] = tokens_to_ids(row[idx], lexicon[k], False)
                    del(row[idx])
                else:
                    row[idx + '_index'] = [0] * row_len
        # the nil slot is the only slot that's masked out, other than potential padded batches
        row[i + 'mask'] = [1] * row_len
    del(row['forward-snapshots'])
    del(row['reverse-snapshots'])

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

