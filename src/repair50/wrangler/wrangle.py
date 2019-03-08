import os, sys
from typing import List, Tuple
from pycparser import c_parser, c_generator, c_lexer # type:ignore
from ..preprocessor.external import ExternalCPP # type:ignore

from centipyde.interpreter import run_tests # type:ignore
from .normalizers import normalize
from .linearize_ast import WrangledAST

def process_linear(tokens, key=None, lexicon=None, lock=None):
    local_lexicon = set()
    if key == 'train':
        for token in set(tokens):
            if token not in local_lexicon:
                with lock:
                    if token not in lexicon['linear']['label']:
                        lexicon['linear']['label'][token] = 0
                    lexicon['linear']['label'][token] += 1
                    local_lexicon.add(token)
    return { 'forward-label': tokens }


def process_ast(ast, key=None, lexicon=None, lock=None):
    all_new_rows = {}

    for test in ast.tests:
        local_lexicon = {
            'label': set(),
            'attr': set(),
            'transitions': set()
        }
        # skip the nil slot
        ast.nodes[test]['reverse'].pop(0)
        nodes = ast.nodes[test]['forward']
        nodes.pop(0)
        for i in range(len(nodes)):
            data = nodes[i]
            for j in ['attr', 'parent_attr', 'parent_label', 'transitions']:
                data[j] = data[j] if data[j] is not None else '<nil>'

            for j in ['label', 'attr', 'transitions']:
                if key == "train" and data[j] not in local_lexicon[j]:
                    with lock:
                        if data[j] not in lexicon['ast'][test][j]:
                            lexicon['ast'][test][j][data[j]] = 0
                        lexicon['ast'][test][j][data[j]] += 1
                    local_lexicon[j].add(data[j])
                #if data['label'] not in ['For', 'If', 'While', 'DoWhile', 'Switch']:#, 'ExpressionList']:
                #    continue

        all_new_rows[test] = new_rows = {}
        for k in ['reverse', 'forward']:
            nodes = ast.nodes[test][k]
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

def finish_row(rows, lexicon, features=None):
    # all rows should have a forward label
    for test in rows:
        row = rows[test]
        row_len = len(row['forward-label'])

        for i in ['parent_', '']:
            for j in ['label', 'attr', 'transitions']:
                for k in ['forward-', 'reverse-']:
                    idx = k + i + j
                    if idx in row:
                        row[idx + '_index'] = tokens_to_ids(row[idx], lexicon[test][j], False)
                        del(row[idx])
                    else:
                        row[idx + '_index'] = [0] * row_len

        # this must be a linear model
        if features is not None:
            row.update({
                'forward-left_sibling': list(range(row_len)),
                'forward-right_sibling': list(range(2, row_len+1)) + [0]
            })
            for key in features:
                if key not in row:
                    row[key] = [0] * row_len

        # the nil slot is the only slot that's masked out, other than potential padded batches
        row['forward-mask'] = [1] * row_len
        row['reverse-mask'] = [1] * row_len

        # insert the nil slot
        for i in row:
            row[i] = [0] + row[i]
    return rows

def lex_code(code : str) -> List[str]:
    lexer = c_lexer.CLexer(
            error_func=lambda x, y, z: True,
            type_lookup_func=lambda x: True,
            on_lbrace_func=lambda: True,
            on_rbrace_func=lambda: True
    )

    lexer.build()
    lexer.input(code)

    tokens = []
    while True:
        token = lexer.token()
        if token is None:
            break
        value = token.value
        tokens.append(value)

    tokens.insert(0,"<sof>")
    tokens.append("<eof>")
    return tokens

def wrangle(code : str, include_dependencies : bool = True, is_file=True, tests=None) -> Tuple[WrangledAST, List[str]] :
    # these can raise exceptions that we'll let pass through
    cfile = ExternalCPP().preprocess(code, is_file)
    ast = c_parser.CParser().parse(cfile)
    ast = normalize(ast, ['ExpandAssignments', 'WrapExpressions'])
    results = run_tests(ast, tests) if tests is not None else None
    # TODO: remove decls can't be interpreted while variables share the same name in different scopes
    ast = normalize(ast, ['RemoveTypedefs', 'RemoveDecls', 'IDRenamer'])


    linear_data : List[str] = [] #lex_code(c_generator.CGenerator().visit(ast))
    ast_data = WrangledAST(ast, results, include_dependencies)

    return ast_data, linear_data

