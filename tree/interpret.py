import sys
import re
from collections import namedtuple
from contextlib import contextmanager
from pycparser import c_generator, c_ast, c_lexer, c_parser, preprocess_file


# TODO: cache results from particularly common subtrees?

class Interpreter(c_ast.NodeVisitor):
    def __init__(self, require_decls=True):
        self.require_decls = require_decls

        self.stdin = None
        self.stdout = ''
        self.stderr = ''

        self.func_map = {}
        self.id_map = [{}]
        self.type_map = [{}]

        # values are tuples of type, num_elements, and array of elements
        self.memory = {}
        self.context = [None]

    @contextmanager
    def scope(self):
        self.id_map.append(self.id_map[-1].copy())
        self.type_map.append(self.type_map[-1].copy())
        try:
            yield
        except:
            raise
        else:
            self.id_map.pop()
            self.type_map.pop()

    @contextmanager
    def context(self, context):
        self.context.append(context)
        try:
            yield
        except:
            raise
        else:
            self.context.pop()

    def execute(self, argv, stdin=''):
        # TODO: need to somehow handle NULL at the end of argv?
        self.id_map[-1]['argc'] = len(argv) + 1
        self.type_map[-1]['argc'] = 'int'
        self.id_map[-1]['argv'] = argv + ['<null>']
        self.type_map[-1]['argv'] = ['[]', 'string']
        self.stdin = stdin

        # validate argv?
        self.visit(self.func_map['main'][2])

    def visit_ArrayRef(self, n):
        # TODO: technically can flip order of these things?
        assignment = self.context[-1] == 'assigning'
        with self.context('array_ref'):
            sub = self.visit(n.subscript)
            arr = self.visit(n.name)
        assert sub < self.memory[arr][1]
        if assignment:
            return lambda val: self.memory[arr].set(sub, val)
        else:
            return self.memory[arr][0], self.memory[arr][2][sub]

    # TODO: pointer deref?
    def visit_Assignment(self, n):
        with self.context('assigning'):
            assignment_op = self.visit(n.lvalue)

        if n.op != '=':
            op = node.op[-1]
            node = c_ast.BinaryOp(op, n.lvalue, n.rvalue)
        else:
            node = n.rvalue
        assignment_op(self.visit(n.rvalue))


    def visit_BinaryOp(self, n):
        ltype, lval = self.visit(n.left)
        rtype, rval = self.visit(n.right)

        if n.op == '+':
            return lval + rval
        elif n.op == '-':
            return lval - rval
        elif n.op == '*':
            return lval * rval
        elif n.op == '/':
            # TODO: handle more types
            if ltype == 'int' and rtype == 'int':
                return lval // rval
            else:
                return lval / rval
        elif n.op == '%':
            return lval % rval
        elif n.op == '|':
            return lval | rval
        elif n.op == '&':
            return lval & rval
        elif n.op == '^':
            return lval ^ rval
        elif n.op == '>>':
            return lval >> rval
        elif n.op == '<<':
            return lval << rval

        elif n.op == '==':
            return 1 if lval == rval else 0
        elif n.op == '!=':
            return 1 if lval != rval else 0
        # TODO: any issues with falsiness?
        elif n.op == '&&':
            return 1 if lval and rval else 0
        elif n.op == '||':
            return 1 if lval or rval else 0
        elif n.op == '>=':
            return 1 if lval >= rval else 0
        elif n.op == '<=':
            return 1 if lval <= rval else 0
        elif n.op == '>':
            return 1 if lval > rval else 0
        elif n.op == '<':
            return 1 if lval < rval else 0
        assert False

        # TODO: need to account for types for integer division. anything about signed/unsigned for bitwise?

    def visit_Break(self, n):
        return 'break', None

    def visit_Compound(self, n):
        with self.scope():
            if n.block_items:
                for stmt in n.block_items:
                    ret, retval = self.visit(stmt)
                    if ret is not None:
                        return ret
        return None, None

    def visit_Constant(self, n):
        return n.type, n.value

    def visit_Continue(self, n):
        return 'continue', None

    # name: the variable being declared
    # quals: list of qualifiers (const, volatile)
    # funcspec: list function specifiers (i.e. inline in C99)
    # storage: list of storage specifiers (extern, register, etc.)
    # type: declaration type (probably nested with all the modifiers)
    # init: initialization value, or None
    # bitsize: bit field size, or None
    def visit_Decl(self, n):
        type_, val = self.visit(n.init) if n.init else None, None
        # TODO: compare n.type against type_ for validity
        self.id_map[-1][n.name] = val
        self.type_map[-1][n.name] = n.type

        # TODO: hmmm, this is stmt level
        return None, None


    # TODO: handle new variable declarations for scope better?
    def visit_ID(self, n):
        if self.context[-1] == 'assigning':
            if self.require_decls:
                assert n.name in self.id_map[-1]
            return lambda val: self.id_map[-1].set(n.name, val)
        elif self.context[-1] == 'arrayref':
            return n.name
        else:
            # check use of uninitialized values
            assert self.id_map[-1][n.name] is not None
            return self.type_map[-1][n.name], self.id_map[-1][n.name]

    def visit_FileAST(self, n):
        [self.visit(ext) for ext in n.ext]

    def visit_For(self, n):
        with self.scope():
            if n.init: self.visit(n.init)
            while not n.cond or self.visit(n.cond):
                ret, retval = self.visit(n.stmt)
                if ret == 'break' or ret == 'return':
                    return ret, retval
                self.visit(n.next)
        return None, None

    def visit_FuncCall(self, n):
        print(n.name, n.args)

    def visit_TypeDecl(self, n):
        return self.visit(n.type)

    # TODO: account for dims
    def visit_ArrayDecl(self, n):
        # self.visit(n.dim)
        return ['[]'] + [self.visit(n.type)]

    def visit_IdentifierType(self, n):
        # TODO: multiple names?
        # TODO: account for overflow in binop/unop?
        return n.names

    def visit_ParamList(self, n):
        # TODO: the decl vs the typedecl??
        # TODO: this is very hard-coded
        if n.params:
            return [(param.name, self.visit(param.type)) for param in n.params]
        else:
            return []

    def visit_FuncDecl(self, n):
        # make sure this matches any prior funcdecl?
        params = self.visit(n.args)
        ret_type = self.visit(n.type)
        name = n.type.declname
        print(params, ret_type, name)
        self.func_map[name] = [ret_type, params, None]

    def visit_FuncDef(self, n):
        # don't understand both the decl and the funcdecl under type?
        self.visit(n.decl.type)
        self.func_map[n.decl.name][2] = n.body

    def visit_If(self, n):
        # XXX can the condition actually be left off?
        cond = not n.cond or self.visit(n.cond)
        if cond:
            return self.visit(n.iftrue)
        elif n.iffalse:
            return self.visit(n.iffalse)
        else:
            return None, None


    def visit_UnaryOp(self, n):
        # TODO: use types
        type_, val = self.visit(n.expr)
        if n.op == 'p++' or n.op == 'p--' or n.op == '++p' or n.op == '--p':
            with self.context('assigning'):
                assignment_op = self.visit(n.expr)
            # TODO: this doesn't handle post-increment correctly
            if n.op == 'p++' or n.op == '++p':
                assignment_op(val + 1)
            else:
                assignment_op(val - 1)
            return val
        elif n.op == '!':
            return 0 if val else 1
        elif n.op == '~':
            return ~val
        assert False

    # TODO: detect infinite loop??
    def visit_While(self, n):
        while not n.cond or self.visit(n.cond):
            ret, retval = self.visit(n.stmt)
            if ret == 'break' or ret == 'return':
                return ret, retval
        return None, None

    def visit_Return(self, n):
        ret = self.visit(n.expr) if n.expr else None
        return 'return',  ret

if __name__=='__main__':
    interpret = Interpreter()
    parser = c_parser.CParser()
    try:
        cfile = preprocess_file(sys.argv[1], cpp_path='cpp', cpp_args=[r'-I../fake_libc_include'])
        ast = parser.parse(cfile)
    except Exception as e:
        print('uh oh2', e)
        sys.exit(1)

    # some kind of JIT after the first execution?
    interpret.visit(ast)
    interpret.execute(['hello'], 'world\n')
