import sys, re
from pycparser import c_ast # type:ignore
from .modifying_visitor import ModifyingVisitor
from .NodeWrapper import NodeWrapper

class RemoveDecls(ModifyingVisitor):
    def visit_IdentifierType(self, node):
        return NodeWrapper(node, None)

    def visit_TypeDecl(self, node):
        return NodeWrapper(node, None)

    # TODO: casts?

    def visit_Decl(self, node):
        if node.init is None:
            return NodeWrapper(node, None)
        else:
            id_ = c_ast.ID(node.name)
            asst = c_ast.Assignment('=', id_, node.init)
            return NodeWrapper(node, asst)

    def visit_FuncDef(self, node):
        # don't remove a funcdef's declaration
        #decl = self.visit(n.decl)
        node.body = self.visit(node.body)
        if node.param_decls:
            node.param_decls = [self.visit(p) for p in node.param_decls]
        return node
