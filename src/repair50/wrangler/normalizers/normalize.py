from .remove_decls import RemoveDecls
from .expand_assignments import ExpandAssignments
from .remove_typedefs import RemoveTypedefs
from .id_renamer import IDRenamer
from .ssa import SSAify
from .wrap_expressions import WrapExpressions

all_normalizers = ['RemoveTypedefs', 'RemoveDecls', 'ExpandAssignments', 'WrapExpressions', 'IDRenamer']

def normalize(ast, normalizers=all_normalizers):
    for normalizer in normalizers:
        globals()[normalizer]().visit(ast)
    return ast
