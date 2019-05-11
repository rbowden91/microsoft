from pycparser.c_ast import Return, Compound, Constant # type:ignore
from .modifying_visitor import ModifyingVisitor
from .NodeWrapper import NodeWrapper

class ReturnZero(ModifyingVisitor):
    def visit_FuncDef(self, node):
        # TODO: this is naive. theoretically, there could be an "else { return 0 }" or something
        # TODO: can you "return", but not "return 0", from main??
        old_body = node.body
        if node.decl.name == 'main' and (len(old_body.block_items) == 0 or \
                not isinstance(old_body.block_items[-1], Return)):
            new_body = Compound(old_body.block_items[:] + [Return(Constant('int', "0"))])
            node.body = NodeWrapper(old_body, new_body)#, lambda: old_body)

        return node
