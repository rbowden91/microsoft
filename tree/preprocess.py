from ..my_env import os, sys
from ..my_env.packages.pycparser import c_parser, c_generator, c_lexer
from .normalize import RemoveDecls, RemoveTypedefs, IDRenamer
from .linearize_ast import LinearizeAST

cpp_args = ['-E', '-nostdinc']
#, '-D__extension__=', '-D__attribute__(x)=', '-D__nonnull(x)=', '-D__restrict=',
#            '-D__THROW=', '-D__volatile__=', '-D__asm__(x)=', '-D__STRING_INLINE=', '-D__inline=']
            #"-D__builtin_va_list=char*"]

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


# taken from pycparser, but extended to also return stderr
# can also use 'scc' if it exists on thes system, to
def preprocess_file(filename, path='cpp', args=[]):
    path_list = [path]
    if isinstance(args, list):
        path_list += args
    elif args != '':
        path_list += [args]
    path_list += [filename]

    try:
        pipe = Popen(path_list,
                     stdout=PIPE,
                     stderr=PIPE,
                     universal_newlines=True)
        text = pipe.communicate()
    except OSError as e:
        raise RuntimeError("Unable to invoke '" + path + "'.  " +
            'Make sure its path was passed correctly\n' +
            ('Original error: %s' % e))

    return text

def grab_directives(string, defines=False):
    # not perfect...
    # XXX want to do something with potentially making the #define replacements, since those are what could
    # be breaking things...
    if defines:
        pattern = r"(^\s*#[^\r\n]*[\r\n])"
    else:
        pattern = r"(^\s*#\s*define\s[^\r\n]*[\r\n])"
    regex = re.compile(pattern, re.MULTILINE|re.DOTALL)

    directives = ''.join(regex.findall(string))
    def _replacer(match):
        return ""
    sub = regex.sub(_replacer, string)
    return directives, sub

def lex_code(code):
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

# does not handle digraphs/trigraphs
def preprocess_c(code, options=None, include_dependencies=True):
    #if options['remove_comments']:
    # code, stderr = preprocess_file(filename, path='scc')
    # does the lexer automatically remove comments for us? do we need a separate step for that?
    # we can use the preprocessor to remove comments:
    # gcc -fpreprocessed -dD -E test.c
    # but that will remove any invalid preprocessor lines as well
    #with open(filename, 'r') as content_file:
    #    content = content_file.read()
    #directives, _ = grab_directives(content)

    cfile, stderr = preprocess_file(code)
    parser = c_parser.CParser()
    try:
        ast = parser.parse(cfile)
    except Exception as e:
        print('uh oh', e, filename)
        return None

    ast = RemoveDecls().visit(ast)
    ast = RemoveTypedefs().visit(ast)
    ast = IDRenamer().visit(ast)

    linear_tokens = lex_code(c_generator.CGenerator().visit(ast))

    linearizer = LinearizeAST(include_dependencies)
    linearizer.visit(ast)

    return ast, linearizer, linear_tokens

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
