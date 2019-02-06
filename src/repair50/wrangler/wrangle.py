from ..my_env import os, sys
from ..my_env.typing import List, Tuple
from ..my_env.packages.pycparser import c_parser, c_generator, c_lexer
from ..preprocessor.external import ExternalCPP

from ..interpreter import Interpreter
from .normalize import RemoveDecls, RemoveTypedefs, IDRenamer
from .linearize_ast import WrangledAST

def process_linear(tokens, key=None, lexicon=None, lock=None):
    local_lexicon = set()
    if key == 'train':
        for token in set(tokens):
            if token not in local_lexicon:
                with lock:
                    if token not in lexicon['linear_tokens']:
                        lexicon['linear_tokens'][token] = 0
                    lexicon['linear_tokens'][token] += 1
                    local_lexicon.add(token)
    return { 'forward_label': tokens }

def process_ast(ast, key=None, lexicon=None, lock=None):
    local_lexicon = {
        'label': set(),
        'attr': set()
    }
    # skip the nil slot
    ast.nodes['reverse'].pop(0)
    nodes = ast.nodes['forward']
    nodes.pop(0)
    for i in range(len(nodes)):
        data = nodes[i]
        data['attr'] = data['attr'] if data['attr'] is not None else '<no_attr>'
        data['label'] = data['label'] if data['label'] is not None else '<nil>'
        data['parent_attr'] = data['parent_attr'] if data['parent_attr'] is not None else '<no_attr>'
        data['parent_label'] = data['parent_label'] if data['parent_label'] is not None else '<nil>'

        if lexicon is not None:
            for j in ['label', 'attr']:
                if key == "train" and data[j] not in local_lexicon[j]:
                    with lock:
                        if data[j] not in local_lexicon[j]:
                            lexicon['ast_' + j + 's'][data[j]] = 0
                        lexicon['ast_' + j + 's'][data[j]] += 1
                    local_lexicon[j].add(data[j])

    new_rows = {}
    for k in ['reverse', 'forward']:
        nodes = ast.nodes[k]
        for i in nodes[0]:
            if i in ['forward', 'reverse']:
                if i != k: continue
                for j in nodes[0][i]:
                    new_rows[k + '_' + j] = [int(n[i][j]) if isinstance(n[i][j], bool) else n[i][j] for n in nodes]
            else:
                new_rows[k + '_' + i] = [int(n[i]) if isinstance(n[i], bool) else n[i] for n in nodes]
    return new_rows

def tokens_to_ids(tokens, token_to_id, include_token):
    output = []
    for i in range(len(tokens)):
        token = tokens[i]
        if token in token_to_id:
            id = token_to_id[token]
        elif token.startswith('"'):
            # TODO: do this for all types?
            id = token_to_id['<unk_str>']
        else:
            id = token_to_id['<unk>']
        output.append((id, token) if include_token else id)
    return output

def finish_row(row, lexicon, features=None):
    # all rows should have a forward label
    row_len = len(row['forward_label'])

    for i in ['forward_', 'reverse_']:
        for j in ['parent_', '']:
            for k in ['label', 'attr']:
                idx = i + j + k
                if idx in row:
                    row[idx + '_index'] = tokens_to_ids(row[idx], lexicon[k], False)
                    del(row[idx])
                else:
                    row[idx + '_index'] = [0] * row_len

    # this must be a linear model
    if features is not None:
        row['forward_left_sibling'] = list(range(row_len))
        row['forward_right_sibling'] = list(range(2, row_len+1)) + [0]
        for key in features:
            if key not in row:
                row[key] = [0] * row_len

    # the nil slot is the only slot that's masked out, other than potential padded batches
    row['forward_mask'] = [1] * row_len
    row['reverse_mask'] = [1] * row_len

    # insert the nil slot
    for k in row:
        row[k] = [0] + row[k]
    return row

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

def wrangle(code : str, test_name : str,  include_dependencies : bool = True, is_file=True) -> Tuple[WrangledAST,
        List[str]] :
    # these can raise exceptions that we'll let pass through
    cfile = ExternalCPP().preprocess(code, is_file)
    orig_ast = c_parser.CParser().parse(cfile)
    interpreter = Interpreter(test_name)
    results, visited = interpreter.run_tests(orig_ast)

    #ast = RemoveDecls().visit(orig_ast)
    ast = RemoveTypedefs().visit(orig_ast)
    renamer = IDRenamer()
    ast = renamer.visit(ast)
    name_map = renamer.node_name_map

    linear_data = lex_code(c_generator.CGenerator().visit(ast))
    ast_data = WrangledAST(ast, orig_ast, name_map, results, visited, include_dependencies)

    return ast_data, linear_data

