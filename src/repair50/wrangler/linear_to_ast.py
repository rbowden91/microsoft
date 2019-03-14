import os, sys
from typing import List, Tuple
from pycparser import c_parser, c_generator, c_lexer # type:ignore
from ..preprocessor.external import ExternalCPP # type:ignore

from centipyde.interpreter import run_tests # type:ignore
from .normalizers import normalize
from .linearize_ast import WrangledAST

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
