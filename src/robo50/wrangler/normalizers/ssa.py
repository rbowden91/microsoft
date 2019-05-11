from pycparser import c_ast # type:ignore
from .modifying_visitor import ModifyingVisitor

class SSAify(ModifyingVisitor):
    def __init__(self):
        self.visiting_rvalue = False
        self.tmps = []
        self.tmp_ctr = 0
        self.ids_visited = set()
        self.visiting_assignment = False
        pass

    def visit_Compound(self, n):
        items = []
        for item in n.block_items:
            ret = self.visit(item)
            items.extend(self.tmps)
            self.tmps = []
            items.append(ret)
        n.block_items = items
        return n

    def visit_Constant(self, n):
        return n.value

    def visit_ID(self, n):
        self.ids_visited.add(n.name)
        return n.name


    #def visit_ArrayRef(self, n):
    #    #self.op_list.append(self.visit(n.subscript))
    #    self.op_list.append(n.name)
    #    return n

    def assign_fresh(self, node):
        tmp = c_ast.Assignment('=', self.tmp_id, node)
        self.tmps.append(tmp)
        return self.tmp_id

    def visit_BinaryOp(self, node):
        if self.visiting_assignment:
            self.visit(node.left)
            self.visit(node.right)
            if isinstance(node.left, c_ast.BinaryOp):
                node.left = self.assign_fresh(node.left)
            if isinstance(node.right, c_ast.BinaryOp):
                node.right = self.assign_fresh(node.right)
        return node

    #def contains_ID(self, id_, node):
    #    if node.

    def generate_fresh(self):
        tmp = c_ast.ID('tmp_' + str(self.tmp_ctr))
        self.tmp_ctr += 1
        return tmp

    # TODO: handling assignments within expressions?
    def visit_Assignment(self, node):
        self.visiting_assignment = True
        # this removes all instances of "-=", etc.
        if node.op != '=':
            op = node.op[:-1]
            node = c_ast.Assignment('=', node.lvalue,
                        c_ast.BinaryOp(op, node.lvalue, node.rvalue))
        # TODO: it's only the array that we care about in making sure we declare a separate tmp
        left_ids_visited = self.ids_visited = set()
        self.visit(node.lvalue)
        self.ids_visited = set()

        self.tmp_id = self.generate_fresh()

        self.visiting_rvalue = True
        self.visit(node.rvalue)

        right_ids_visited = self.ids_visited = set()
        for i in range(1, len(self.tmps)):
            self.visit(self.tmps[i].rvalue)

        if len(right_ids_visited.intersection(left_ids_visited)) == 0:
            for i in range(len(self.tmps)):
                self.tmps[i].lvalue = node.lvalue
                self.tmps[i].rvalue.left = node.lvalue
            if len(self.tmps) > 0:
                node.rvalue.left = node.lvalue
        self.visiting_assignment = False
        return node

    #def visit_UnaryOp(self, node):
    #    self.visit(node.expr)
    #    return node

    def generic_visit(self, node):
        #print(node.__class__.__name__)
        for c_name, c in node.children():
            self.visit(c)
        return node
