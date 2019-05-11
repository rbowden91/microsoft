from pycparser import c_ast # type:ignore
from .modifying_visitor import ModifyingVisitor
from .NodeWrapper import NodeWrapper

# TODO: what about pointer deref?
def copy_node(node):
    if isinstance(node, c_ast.ID):
        return c_ast.ID(node.name)
    elif isinstance(node, c_ast.ArrayRef):
        return c_ast.ArrayRef(copy_node(node.name), copy_node(node.subscript))
    elif isinstance(node, c_ast.StructRef):
        return c_ast.StructRef(copy_node(node.name), node.type, copy_node(node.field))
    elif isinstance(node, c_ast.Constant):
        return c_ast.Constant(node.type, node.value)
    else:
        print(type(node))
        assert False

class ExpandAssignments(ModifyingVisitor):
    def visit_Assignment(self, node):
        if node.op != '=':
            # TODO: need to make a copy of lvalue? e.g., for pointers
            new_lvalue = copy_node(node.lvalue)
            binop = c_ast.BinaryOp(node.op[:-1], node.lvalue, node.rvalue)
            asst = c_ast.Assignment('=', new_lvalue, binop)
            return NodeWrapper(node, asst)
        return node
