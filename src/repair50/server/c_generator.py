# based heavily on the version from pycparser

from pycparser import c_ast # type:ignore
from ..wrangler.normalizers.NodeWrapper import NodeWrapper
from ..wrangler.normalizers.ExpressionList import ExpressionList


class CGenerator(object):
    def __init__(self, ast_data):
        # Statements start with indentation of self.indent_level spaces, using
        # the _make_indent method
        #
        self.indent_level = 0
        self.line = 0
        self.char = 0
        self.indent = 0
        self.ast_data = ast_data
        self.most_recent_node = 0
        self.roots = []

        code = self.visit(ast_data.ast)
        self.lines = []
        line = {'indentation': 0, 'root': None, 'chars':[]}
        for char in code:
            if char['class'] == 'newline':
                self.lines.append(line)
                line = {'indentation': 0, 'root': None, 'chars':[]}
                continue
            elif char['class'] == 'indent':
                line['indentation'] = char['value']
                continue
            if line['root'] is None and len(char['roots']) > 0:
                line['root'] = char['roots'][-1]
            line['chars'].append(char)
        self.lines.append(line)

    def _mp(self, class_, value):
        if class_ == 'indent':
            return [{'class': class_, 'value': value}]
        elif class_ == 'newline':
            self.line += 1
            self.char = 0
            return [{'class': class_}]
        else:
            ret = [{'class': class_, 'value': i, 'roots': self.roots[:]} for i in value]
            self.char += len(ret)
            return ret

    def _make_space(self):
        return self._mp('whitespace', ' ')

    def _make_indent(self):
        return self._mp('indent', self.indent_level)

    def _make_newline(self):
        return self._mp('newline', self.line)

    def _make_raw(self, raw):
        return self._mp('raw', raw)

    def _make_id(self, id_):
        if id_ in ('string', 'int', 'double', 'float', 'long', 'char', 'bool', 'short'):
            return self._mp('type', id_)
        elif id_ in ('printf', 'strlen', 'toupper', 'tolower', 'isupper', 'islower', 'argc', 'argv', 'isalpha', 'main'):
            return self._mp('keyword', id_)
        return self._make_raw(id_)

    def _paren(self, ret):
        ret.insert(0, self._make_raw('(')[0])
        ret.append(self._make_raw(')')[0])
        return ret

    def _bracket(self, ret):
        ret.insert(0, self._make_raw('[')[0])
        ret.append(self._make_raw(']')[0])
        return ret

    def _brace(self, ret):
        ret.insert(0, self._make_raw('{')[0])
        ret.insert(0, self._make_indent()[0])
        ret.append(self._make_indent()[0])
        ret.append(self._make_raw('}')[0])
        return ret


    def _get_id(self, node):
        self.most_recent_node = node.node_properties['node_num']
        return node.node_properties['node_num']

    def visit_NodeWrapper(self, n):
        assert False


    def visit(self, node):
        if isinstance(node, NodeWrapper):
            if node.new is None:
                return []
            else:
                return self.visit(node.new)
        if node is None:
            return []


        method = 'visit_' + node.__class__.__name__

        #if node in self.ast_data.node_properties and 'original_name' in self.ast_data.node_properties[node]:
        #    if hasattr(node, 'name'):
        #        node.name = self.ast_data.node_properties[node]['original_name']
        #    elif hasattr(node, 'declname'):
        #        node.declname = self.ast_data.name_map[node]
        #    #if 'names' in self.ast_data.name_map[node]:
        #    #    node.names = []
        #    #    for i in self.ast_data.name_map[node]['names']:
        #    #        node.names += i)
        node_num = self._get_id(node)
        lineno = self.line
        charno = self.char

        self.roots.append(node_num)
        ret = getattr(self, method)(node)
        self.roots.pop()

        assert ret is not None

        node.node_properties = {
            'starting_char': charno,
            'starting_line': lineno,
            'ending_line': self.line,
            'ending_char': self.char
        }

        return ret

    def visit_Constant(self, n):
        if n.value.startswith('"'):
            class_ = 'string'
        elif n.value.startswith("'"):
            class_ = 'char'
        else:
            class_ = 'number'
        return self._mp(class_, n.value)

    def visit_ID(self, n):
        if 'original_name' in n.node_properties:
            name = n.node_properties['original_name']
        else:
            name = n.name
        return self._make_id(name)

    def visit_Pragma(self, n):
        ret = self._make_raw('#') + self._mp('keyword', '#pragma')
        if n.string:
            ret.extend += self._make_space() + self._make_raw(n.string)
        return ret

    def visit_ArrayRef(self, n):
        ret = self._parenthesize_unless_simple(n.name)
        ret += self._bracket(self.visit(n.subscript))
        return ret

    def visit_StructRef(self, n):
        ret = self._parenthesize_unless_simple(n.name)
        ret += self._make_raw(n.type) + self.visit(n.field)
        return ret

    def visit_FuncCall(self, n):
        ret = self._parenthesize_unless_simple(n.name)
        ret += self._paren(self.visit(n.args))
        return ret

    def visit_UnaryOp(self, n):
        ret = self._parenthesize_unless_simple(n.expr)
        if n.op == 'p++':
            ret += self._make_raw('++')
        elif n.op == 'p--':
            ret += self._make_raw('--')
        elif n.op == 'sizeof':
            # Always parenthesize the argument of sizeof since it can be
            # a name.
            ret += self._mp('keyword', 'sizeof')
            ret += self._paren(self.visit(n.expr))
        else:
            ret = self._make_raw(n.op) + ret
        return ret

    def visit_BinaryOp(self, n):
        ret = self._parenthesize_if(n.left,
                            lambda d: not self._is_simple_node(d))
        ret += self._make_raw(n.op)
        ret += self._parenthesize_if(n.right,
                            lambda d: not self._is_simple_node(d))
        return ret

    def visit_Assignment(self, n):
        ret = self.visit(n.lvalue) + self._make_raw(n.op)
        ret += self._parenthesize_if(
                            n.rvalue,
                            lambda n: self.isinstance(n, c_ast.Assignment))
        return ret

    def type_(self, n):
        if isinstance(n, NodeWrapper):
            if n.new is None: return None
            return self.type_(n.new)
        return type(n)

    def isinstance(self, n, instance):
        type_ = self.type_(n)
        if not isinstance(instance, list):
            instance = [instance]
        return type_ in instance

    def visit_IdentifierType(self, n):
        ret = []
        for name in n.names:
            if len(ret) != 0:
                ret += self._make_space()
            ret += self._make_id(name)
        return ret

    def _visit_expr(self, n):
        ret = self.visit(n)
        if isinstance(n, c_ast.InitList):
            self._brace(ret)
        elif isinstance(n, c_ast.ExprList):
            self._paren(ret)
        return ret

    def visit_Decl(self, n, no_type=False):
        # no_type is used when a Decl is part of a DeclList, where the type is
        # explicitly only for the first declaration in a list.
        #
        ret = []
        props = n.node_properties
        if 'no_type' in props:
            name = n.name if 'original_name' not in props else props['original_name']
            ret += self._make_id(name)
        else:
            ret += self._generate_decl(n)

        if n.bitsize:
            ret += self._make_space() + self._make_raw(':') + self._make_space()
            ret += self.visit(n.bitsize)
        if n.init:
            ret += self._make_space() + self._make_raw('=') + self._make_space()
            ret += self._visit_expr(n.init)
        return ret

    def visit_DeclList(self, n):
        ret = self.visit(n.decls[0])
        for i in range(1, len(n.decls)):
            ret += self._make_raw(',') + self._make_space()
            n.decls[i].node_properties['no_type'] = True
            ret += self.visit(n.decls[i])
        return ret

    def visit_Typedef(self, n):
        ret = []
        if n.storage:
            for i in n.storage:
                ret += self._make_raw(i), self._make_space()
        ret += self._generate_type(n.type)
        return ret

    def visit_Cast(self, n):
        ret = self.generate_type(n.to_type)
        self._paren(ret)
        ret += self._make_space()
        ret += self._parenthesize_unless_simple(n.expr)
        return ret

    def visit_ExpressionList(self, n):
        ret = []
        [ret.extend(self._generate_stmt(x)) for x in n.expressions]
        return ret

    def visit_ExprList(self, n):
        ret = []
        for expr in n.exprs:
            if len(ret) > 0:
                ret += self._make_raw(',') + self._make_space()
            ret += self._visit_expr(expr)
        return ret

    def visit_InitList(self, n):
        ret = []
        for expr in n.exprs:
            if len(ret) > 0:
                ret += self._make_raw(',') + self._make_space()
            ret += self._visit_expr(expr)
        return ret

    def visit_Enum(self, n):
        return self._generate_struct_union_enum(n, 'enum')

    def visit_Enumerator(self, n):
        ret = self._make_indent() + self._make_raw(n.name)
        if n.value:
            ret += self._make_space() + self._make_raw('=') + self._make_space()
            ret += self.visit(n.value)
            ret += self._make_raw(',')
        ret += self._make_newline()
        return ret

    def visit_FuncDef(self, n):
        decl = self.visit(n.decl)
        nl1 = self._make_newline()
        self.indent_level = 0
        body = self.visit(n.body)
        nl2 = self._make_newline()
        if n.param_decls:
            return ''
            # TODO
            #knrdecls = ';\n'.join(self.visit(p) for p in n.param_decls)
            #return decl + '\n' + knrdecls + ';\n' + body + '\n'
        else:
            return decl + nl1 + body + nl2

    def visit_FileAST(self, n):
        ret = []
        for ext in n.ext:
            ret += self.visit(ext)
            if self.isinstance(ext, c_ast.Pragma):
                ret += self._make_newline()
            elif self.type_(ext) not in (None, c_ast.FuncDef):
                ret += self._make_raw(';') + self._make_newline()
        return ret

    def visit_Compound(self, n):
        ret = self._make_newline()
        self.indent_level += 1
        if n.block_items:
            [ret.extend(self._generate_stmt(stmt)) for stmt in n.block_items]
        self.indent_level -= 1
        self._brace(ret)
        ret += self._make_newline()
        return ret

    def visit_CompoundLiteral(self, n):
        ret = self._paren(self.visit(n.type))
        ret += self._brace(self.visit(n.init))
        return ret


    def visit_EmptyStatement(self, n):
        return self._make_raw(';')

    def visit_ParamList(self, n):
        ret = []
        for param in n.params:
            if len(ret) > 0:
                ret += self._make_raw(',') + self._make_space()
            ret += self.visit(param)
        return ret

    def visit_Return(self, n):
        ret = self._mp('keyword', 'return')
        if n.expr:
            ret += self._make_space()
            ret += self.visit(n.expr)
        ret += self._make_raw(';')
        return ret

    def visit_Break(self, n):
        return self._mp('keyword', 'break') + self._make_raw(';')

    def visit_Continue(self, n):
        return self._mp('keyword', 'continue') + self._make_raw(';')

    def visit_TernaryOp(self, n):
        return self._paren(self._visit_expr(n.cond)) + \
                self._make_space() + self.make_raw('?') + self._make_space() + \
                self._paren(self._visit_expr(n.iftrue)) + \
                self._make_space() + self.make_raw(':') + self._make_space() + \
                self._paren(self._visit_expr(n.iffalse))

    def visit_If(self, n):
        ret = self._mp('keyword', 'if') + self._make_space()
        cond = self.visit(n.cond) if n.cond else []
        ret += self._paren(cond)
        ret += self._make_newline()
        ret += self._generate_stmt(n.iftrue, add_indent=True)
        if n.iffalse:
            ret += self._make_indent() + self._mp('keyword', 'else') + self._make_newline()
            ret += self._generate_stmt(n.iffalse, add_indent=True)
        return ret

    def visit_For(self, n):
        ret = self._mp('keyword', 'for') + self._make_space()
        inner = []
        if n.init: inner += self.visit(n.init)
        inner += self._make_raw(';')
        if n.cond: inner += self._make_space() + self.visit(n.cond)
        inner += self._make_raw(';')
        if n.next: inner += self._make_space() + self.visit(n.next)
        ret += self._paren(inner)
        ret += self._make_newline()
        ret += self._generate_stmt(n.stmt, add_indent=True)
        return ret

    def visit_While(self, n):
        ret =self._mp('keyword', 'while') + self._make_space()
        inner = []
        if n.cond: inner += self._make_space() + self.visit(n.cond)
        ret += self._paren(inner)
        ret += self._make_newline()
        ret += self._generate_stmt(n.stmt, add_indent=True)
        return ret

    def visit_DoWhile(self, n):
        ret = self._mp('keyword', 'do'), self._make_newline()
        ret += self._generate_stmt(n.stmt, add_indent=True)
        ret += self._make_indent() + self._mp('keyword', 'while') + self._make_space()
        ret += self._paren(self.visit(n.cond) if n.cond else [])
        ret += self._make_raw(';')
        return ret

    def visit_Switch(self, n):
        ret = self._mp('keyword', 'switch') + self._make_space()
        ret += self._paren(self.visit(n.cond))
        ret += self._make_newline()
        ret += self._generate_stmt(n.stmt, add_indent=True)
        return ret

    def visit_Case(self, n):
        ret = self._mp('keyword', 'case') + self._make_space() + \
               self.visit(n.expr) + self._make_raw(':') + self._make_newline()
        for stmt in n.stmts:
            ret += self._generate_stmt(stmt, add_indent=True)
        return ret

    def visit_Default(self, n):
        ret = self._mp('keyword', 'default') + self._make_raw(':') + self._make_newline()
        for stmt in n.stmts:
            ret += self._generate_stmt(stmt, add_indent=True)
        return ret

    def visit_Label(self, n):
        ret = self._make_raw(n.name) + self._make_raw(':') + self._make_newline()
        ret += self._generate_stmt(n.stmt)
        return ret

    def visit_Goto(self, n):
        ret = self._mp('keyword', 'goto') + self._make_space() + self._make_raw(n.name) + self._make_raw(';')

    def visit_EllipsisParam(self, n):
        return self._make_raw('...')

    def visit_Struct(self, n):
        return self._generate_struct_union_enum(n, 'struct')

    def visit_Typename(self, n):
        return self._generate_type(n.type)

    def visit_Union(self, n):
        return self._generate_struct_union_enum(n, 'union')

    def visit_NamedInitializer(self, n):
        ret = []
        for name in n.name:
            if isinstance(name, c_ast.ID):
                ret += self._make_raw('.') + self.visit(name)
            else:
                ret += self._bracket(self.visit(name))
        ret += self._make_space() + self._make_raw('=') + self._make_space()
        ret += self._visit_expr(n.expr)
        return ret

    def visit_FuncDecl(self, n):
        return self._generate_type(n)

    def _generate_struct_union_enum(self, n, name):
        """ Generates code for structs, unions, and enums. name should be
            'struct', 'union', or 'enum'.
        """
        ret = []
        if name in ('struct', 'union'):
            members = n.decls
            body_function = self._generate_struct_union_body
        else:
            assert name == 'enum'
            members = None if n.values is None else n.values.enumerators
            body_function = self._generate_enum_body
        ret += name + ' ' + (n.name or '')
        ret += self._make_raw(name) + self._make_space()
        if n.name:
            ret += self._make_raw(n.name)
        if members is not None:
            # None means no members
            # Empty sequence means an empty list of members
            ret += self._make_newline() + self._make_indent()
            self.indent_level += 1
            inner = self._make_newline()
            inner += body_function(members)
            self.indent_level -= 1
            inner += self._make_indent()
            ret += self._brace(inner)
        return ret

    def _generate_struct_union_body(self, members):
        ret = []
        [ret.extend(self._generate_stmt(decl)) for decl in members]
        return ret

    def _generate_enum_body(self, members):
        ret = []
        [ret.extend(self.visit(value)) for value in members]
        # remove the final `,` from the enumerator list
        # XXX fix this...
        ret[-1]['value'].pop(-2)
        return ret

    def _generate_stmt(self, n, add_indent=False):
        """ Generation from a statement node. This method exists as a wrapper
            for individual visit_* methods to handle different treatment of
            some statements in this context.
        """
        typ = self.type_(n)
        if typ is None:
            return []

        if add_indent: self.indent_level += 1
        indent = self._make_indent()
        if add_indent: self.indent_level -= 1

        ret = self.visit(n)
        if typ in (c_ast.Compound, ExpressionList):
            return ret
        ret = indent + ret

        if typ in (
                c_ast.Decl, c_ast.Assignment, c_ast.Cast, c_ast.UnaryOp,
                c_ast.BinaryOp, c_ast.TernaryOp, c_ast.FuncCall, c_ast.ArrayRef,
                c_ast.StructRef, c_ast.Constant, c_ast.ID, c_ast.Typedef,
                c_ast.ExprList):
            # These can also appear in an expression context so no semicolon
            # is added to them automatically
            #
            ret += self._make_raw(';')
        ret += self._make_newline()
        return ret

    def _generate_decl(self, n):
        """ Generation from a Decl node.
        """
        s = []
        if n.funcspec:
            for i in n.funcspec:
                s += self._make_raw(i) + self._make_space
        if n.storage:
            for f in n.storage:
                s += self._make_raw(f) + self._make_space
        s += self._generate_type(n.type)
        return s

    def _generate_type(self, n, modifiers=[]):
        """ Recursive generation from a type node. n is the type node.
            modifiers collects the PtrDecl, ArrayDecl and FuncDecl modifiers
            encountered on the way down to a TypeDecl, to allow proper
            generation from it.
        """
        typ = self.type_(n)
        if typ is None:
            return []
        #~ print(n, modifiers)

        ret = []

        if typ == c_ast.TypeDecl:
            if n.quals:
                for q in n.quals:
                    ret += self._make_raw(q) + self._make_space()
            ret += self.visit(n.type)

            if n.declname:
                if 'original_name' in n.node_properties:
                    name = n.node_properties['original_name']
                else:
                    name = n.declname
            else:
                name = ''
            nstr = self._make_raw(name)
            # Resolve modifiers.
            # Wrap in parens to distinguish pointer to array and pointer to
            # function syntax.
            #
            for i, modifier in enumerate(modifiers):
                if self.isinstance(modifier, c_ast.ArrayDecl):
                    if (i != 0 and self.isinstance(modifiers[i - 1], c_ast.PtrDecl)):
                        self._paren(nstr)
                    nstr += self._bracket(self.visit(modifier.dim))
                elif self.isinstance(modifier, c_ast.FuncDecl):
                    if (i != 0 and self.isinstance(modifiers[i - 1], c_ast.PtrDecl)):
                        self._paren(nstr)
                    nstr += self._paren(self.visit(modifier.args))
                elif self.isinstance(modifier, c_ast.PtrDecl):
                    if modifier.quals:
                        mods = self._make_raw('*')
                        for q in modifier.quals:
                            mods += self._make_space() + self._make_raw(q)
                        nstr = mods + nstr
                    else:
                        nstr = self._make_raw('*') + nstr
            if nstr: nstr = self._make_space() + nstr
            ret += nstr
            return ret
        elif typ == c_ast.Decl:
            ret = self.generate_decl(n.type)
            return ret
        elif typ == c_ast.Typename:
            ret += self._generate_type(n.type)
            return ret
        elif typ == c_ast.IdentifierType:
            for name in n.names:
                ret += self._make_raw(name) + self._make_space()
            return ret
        elif typ in (c_ast.ArrayDecl, c_ast.PtrDecl, c_ast.FuncDecl):
            ret += self._generate_type(n.type, modifiers + [n])
            return ret
        else:
            return self.visit(n)

    def _parenthesize_if(self, n, condition):
        """ Visits 'n' and returns its string representation, parenthesized
            if the condition function applied to the node returns True.
        """
        s = self._visit_expr(n)
        if condition(n):
            self._paren(s)
        return s

    def _parenthesize_unless_simple(self, n):
        """ Common use case for _parenthesize_if
        """
        return self._parenthesize_if(n, lambda d: not self._is_simple_node(d))

    def _is_simple_node(self, n):
        """ Returns True for nodes that are "simple" - i.e. nodes that always
            have higher precedence than operators.  """
        return self.isinstance(n, [c_ast.Constant, c_ast.ID, c_ast.ArrayRef,
                              c_ast.StructRef, c_ast.FuncCall])
