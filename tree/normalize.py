import sys
from pycparser import c_generator, c_ast, c_lexer, c_parser, preprocess_file

class RemoveDecls(c_ast.NodeVisitor):
    def generic_visit(self, node):
        #print(node.__class__.__name__)
        for c_name, c in node.children():
            self.visit(c)
        return node

    # TODO: typedefs?

    def visit_Compound(self, node):
        if node.block_items is None:
            # handle empty blocks
            return node
        items = []
        for item in node.block_items:
            ret = self.visit(item)
            if ret is not None:
                items.append(ret)
        node.block_items = items
        return node

    def visit_FuncDef(self, node):
        #node.decl = None
        node.body = self.visit(node.body)
        return node

    #def visit_DeclList(self, node):
        # is it possible for a DeclList to only have 1?
        #if len(node.decls) > 1:

    # expand ExprLists?
    # move decls out of for loops?
    # at that point, make all for loops while loops?

    def visit_Decl(self, node):
        if node.init is not None:
            return c_ast.Assignment('=', c_ast.ID(node.name), node.init)
        else:
            return None


# list of typedefs from common header files that we don't want to output
class SSAify(c_ast.NodeVisitor):
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



if __name__=='__main__':
    generator = RemoveDecls()
    parser = c_parser.CParser()
    try:
        cfile = preprocess_file(sys.argv[1], cpp_path='cpp', cpp_args=[r'-I../fake_libc_include'])
        ast = parser.parse(cfile)
    except Exception as e:
        print('uh oh2', e)
        sys.exit(1)
    renamed_code = generator.visit(ast)
    cgen = c_generator.CGenerator()
    print(cgen.visit(renamed_code))
