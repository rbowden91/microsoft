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
        self.line = 1
        self.ast_data = ast_data

        self.code = self.visit(ast_data.ast)

    def _mp(self, class_, value):
        return { 'class': class_, 'value': value }

    def _make_space(self):
        return self._mp('whitespace', False)

    def _make_indent(self):
        return self._mp('indent', self.indent_level)

    def _make_newline(self):
        self.line += 1
        return self._mp('newline', self.line)

    def _make_raw(self, raw):
        return self._mp('raw', raw)

    def _make_id(self, id_):
        if id_ in ('string', 'int', 'double', 'float', 'long', 'char', 'bool', 'short'):
            return self._mp('keyword', id_)
        elif id_ in ('printf', 'strlen', 'toupper', 'tolower', 'isupper', 'islower', 'argc', 'argv', 'isalpha', 'main'):
            return self._mp('built_in', id_)
        return self._make_raw(id_)

    def _paren(self, ret):
        if not isinstance(ret, list):
            ret = [ret]
        ret.insert(0, self._make_raw('('))
        ret.append(self._make_raw(')'))
        return ret

    def _bracket(self, ret):
        if not isinstance(ret, list):
            ret = [ret]
        ret.insert(0, self._make_raw('['))
        ret.append(self._make_raw(']'))
        return ret

    def _brace(self, ret):
        if not isinstance(ret, list):
            ret = [ret]
        ret.insert(0, self._make_raw('{'))
        ret.insert(0, self._make_indent())
        ret.append(self._make_indent())
        ret.append(self._make_raw('}'))
        return ret


    def _get_id(self, node):
        return node.node_properties['node_num']

    def visit(self, node):
        if node is None:
            return {'class': 'empty'}
        if isinstance(node, NodeWrapper):
            return self.visit(node.new)

        method = 'visit_' + node.__class__.__name__

        #if node in self.ast_data.node_properties and 'original_name' in self.ast_data.node_properties[node]:
        #    if hasattr(node, 'name'):
        #        node.name = self.ast_data.node_properties[node]['original_name']
        #    elif hasattr(node, 'declname'):
        #        node.declname = self.ast_data.name_map[node]
        #    #if 'names' in self.ast_data.name_map[node]:
        #    #    node.names = []
        #    #    for i in self.ast_data.name_map[node]['names']:
        #    #        node.names.append(i)
        lineno = self.line
        ret = {'value': getattr(self, method)(node), 'class': 'node',
                'id': self._get_id(node)}
        ret['starting_line'] = lineno
        ret['ending_line'] = self.line
        #ret.update(node.node_properties)
        return ret

    #def generic_visit(self, node):
    #    return ''.join(self.visit(c) for c_name, c in node.children())

    def visit_Constant(self, n):
        if n.value.startswith('"'):
            class_ = 'string'
        elif n.value.startswith("'"):
            class_ = 'char'
        else:
            class_ = 'number'
        return [self._mp(class_, n.value)]

    def visit_ID(self, n):
        if 'original_name' in n.node_properties:
            name = n.node_properties['original_name']
        else:
            name = n.name
        return [self._make_id(name)]

    def visit_Pragma(self, n):
        ret = [self._make_raw('#'), self._mp('keyword', '#pragma')]
        if n.string:
            ret.extend([self._make_space(), self._make_raw(n.string)])
        return ret

    def visit_ArrayRef(self, n):
        ret = self._parenthesize_unless_simple(n.name)
        ret.extend(self._bracket(self.visit(n.subscript)))
        return ret

    def visit_StructRef(self, n):
        ret = self._parenthesize_unless_simple(n.name)
        ret.extend([self._make_raw(n.type), self.visit(n.field)])
        return ret

    def visit_FuncCall(self, n):
        ret = self._parenthesize_unless_simple(n.name)
        ret.extend(self._paren(self.visit(n.args)))
        return ret

    def visit_UnaryOp(self, n):
        ret = self._parenthesize_unless_simple(n.expr)
        if n.op == 'p++':
            ret.append(self._make_raw('++'))
        elif n.op == 'p--':
            ret.append(self._make_raw('--'))
        elif n.op == 'sizeof':
            # Always parenthesize the argument of sizeof since it can be
            # a name.
            ret.append(self._mp('keyword', 'sizeof'))
            ret.extend(self._paren(self.visit(n.expr)))
        else:
            ret.insert(0, self._make_raw(n.op))
        return ret

    def visit_BinaryOp(self, n):
        ret = self._parenthesize_if(n.left,
                            lambda d: not self._is_simple_node(d))
        ret.append(self._make_raw(n.op))
        ret.extend(self._parenthesize_if(n.right,
                            lambda d: not self._is_simple_node(d)))
        return ret

    def visit_Assignment(self, n):
        ret = [self.visit(n.lvalue), self._make_raw(n.op)]
        ret.extend(self._parenthesize_if(
                            n.rvalue,
                            lambda n: isinstance(n, c_ast.Assignment)))
        return ret

    def visit_IdentifierType(self, n):
        ret = []
        for name in n.names:
            if len(ret) != 0:
                ret.append(self._make_space())
            ret.append(self._make_id(name))
        return ret

    def _visit_expr(self, n):
        ret = [self.visit(n)]
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
            ret.append(self._make_id(name))
        else:
            ret.extend(self._generate_decl(n))

        if n.bitsize:
            ret.extend([self._make_space(), self._make_raw(':'), self._make_space()])
            ret.append(self.visit(n.bitsize))
        if n.init:
            ret.extend([self._make_space(), self._make_raw('='), self._make_space()])
            ret.extend(self._visit_expr(n.init))
        return ret

    def visit_DeclList(self, n):
        ret = [self.visit(n.decls[0])]
        for i in range(1, len(n.decls)):
            ret.extend([self._make_raw(','), self._make_space()])
            n.decls[i].node_properties['no_type'] = True
            ret.append(self.visit(n.decls[i]))
        return ret

    def visit_Typedef(self, n):
        ret = []
        if n.storage:
            for i in n.storage:
                ret.extend([self._make_raw(i), self._make_space()])
        ret.extend(self._generate_type(n.type))
        return ret

    def visit_Cast(self, n):
        ret = self.generate_type(n.to_type)
        self._paren(ret)
        ret.append(self._make_space())
        ret.extend(self._parenthesize_unless_simple(n.expr))
        return ret

    def visit_ExpressionList(self, n):
        ret = []
        [ret.append(self._generate_stmt(x)) for x in n.expressions]
        return ret

    def visit_ExprList(self, n):
        ret = []
        for expr in n.exprs:
            if len(ret) > 0:
                ret.extend([self._make_raw(','), self._make_space()])
            ret.extend(self._visit_expr(expr))
        return ret

    def visit_InitList(self, n):
        ret = []
        for expr in n.exprs:
            if len(ret) > 0:
                ret.extend([self._make_raw(','), self._make_space()])
            ret.extend(self._visit_expr(expr))
        return ret

    def visit_Enum(self, n):
        return self._generate_struct_union_enum(n, 'enum')

    def visit_Enumerator(self, n):
        ret = [self._make_indent(), self._make_raw(n.name)]
        if n.value:
            ret.extend([self._make_space(), self._make_raw('='), self._make_space()])
            ret.append(self.visit(n.value))
            ret.append(self._make_raw(','))
        ret.append(self._make_newline())
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
            return [decl, nl1, body, nl2]

    def visit_FileAST(self, n):
        ret = []
        for ext in n.ext:
            tmp = self.visit(ext)
            if tmp['class'] == 'empty':
                continue
            ret.append(self.visit(ext))
            if isinstance(ext, c_ast.Pragma):
                ret.append(self._make_newline())
            elif not isinstance(ext, c_ast.FuncDef):
                ret.extend([self._make_raw(';'), self._make_newline()])
        return ret

    def visit_Compound(self, n):
        ret = [self._make_newline()]
        self.indent_level += 1
        if n.block_items:
            [ret.append(self._generate_stmt(stmt)) for stmt in n.block_items]
        self.indent_level -= 1
        self._brace(ret)
        ret.append(self._make_newline())
        return ret

    def visit_CompoundLiteral(self, n):
        ret = self._paren(self.visit(n.type))
        ret.extend(self._brace(self.visit(n.init)))
        return ret


    def visit_EmptyStatement(self, n):
        return [self._make_raw(';')]

    def visit_ParamList(self, n):
        ret = []
        for param in n.params:
            if len(ret) > 0:
                ret.extend([self._make_raw(','), self._make_space()])
            ret.append(self.visit(param))
        return ret

    def visit_Return(self, n):
        ret = [self._mp('keyword', 'return')]
        if n.expr:
            ret.append(self._make_space())
            ret.append(self.visit(n.expr))
        ret.append(self._make_raw(';'))
        return ret

    def visit_Break(self, n):
        return [self._mp('keyword', 'break'), self._make_raw(';')]

    def visit_Continue(self, n):
        return [self._mp('keyword', 'continue'), self._make_raw(';')]

    def visit_TernaryOp(self, n):
        return [self._paren(self._visit_expr(n.cond)),
                self._make_space(), self.make_raw('?'), self._make_space(),
                self._paren(self._visit_expr(n.iftrue)),
                self._make_space(), self.make_raw(':'), self._make_space(),
                self._paren(self._visit_expr(n.iffalse))]

    def visit_If(self, n):
        ret = [self._mp('keyword', 'if'), self._make_space()]
        cond = self.visit(n.cond) if n.cond else []
        ret.extend(self._paren(cond))
        ret.append(self._make_newline())
        ret.append(self._generate_stmt(n.iftrue, add_indent=True))
        if n.iffalse:
            ret.extend([self._make_indent(), self._mp('keyword', 'else'), self._make_newline()])
            ret.append(self._generate_stmt(n.iffalse, add_indent=True))
        return ret

    def visit_For(self, n):
        ret = [self._mp('keyword', 'for'), self._make_space()]
        inner = []
        if n.init: inner.append(self.visit(n.init))
        inner.append(self._make_raw(';'))
        if n.cond: inner.extend([self._make_space(), self.visit(n.cond)])
        inner.append(self._make_raw(';'))
        if n.next: inner.extend([self._make_space(), self.visit(n.next)])
        ret.extend(self._paren(inner))
        ret.append(self._make_newline())
        ret.append(self._generate_stmt(n.stmt, add_indent=True))
        return ret

    def visit_While(self, n):
        ret = [self._mp('keyword', 'while'), self._make_space()]
        inner = []
        if n.cond: inner.extend([self._make_space(), self.visit(n.cond)])
        ret.extend(self._paren(inner))
        ret.append(self._make_newline())
        ret.append(self._generate_stmt(n.stmt, add_indent=True))
        return ret

    def visit_DoWhile(self, n):
        ret = [self._mp('keyword', 'do'), self._make_newline()]
        ret.append(self._generate_stmt(n.stmt, add_indent=True))
        ret.extend([self._make_indent(), self._mp('keyword', 'while'), self._make_space()])
        ret.extend(self._paren(self.visit(n.cond) if n.cond else []))
        ret.append(self._make_raw(';'))
        return ret

    def visit_Switch(self, n):
        ret = [self._mp('keyword', 'switch'), self._make_space()]
        ret.extend(self._paren(self.visit(n.cond)))
        ret.append(self._make_newline())
        ret.append(self._generate_stmt(n.stmt, add_indent=True))
        return ret

    def visit_Case(self, n):
        ret = [self._mp('keyword', 'case'), self._make_space(),
               self.visit(n.expr), self._make_raw(':'), self._make_newline()]
        for stmt in n.stmts:
            ret.append(self._generate_stmt(stmt, add_indent=True))
        return ret

    def visit_Default(self, n):
        ret = [self._mp('keyword', 'default'), self._make_raw(':'), self._make_newline()]
        for stmt in n.stmts:
            ret.append(self._generate_stmt(stmt, add_indent=True))
        return ret

    def visit_Label(self, n):
        ret = [self._make_raw(n.name), self._make_raw(':'), self._make_newline()]
        ret.append(self._generate_stmt(n.stmt))
        return ret

    def visit_Goto(self, n):
        ret = [self._mp('keyword', 'goto'), self._make_space(), self._make_raw(n.name), self._make_raw(';')]

    def visit_EllipsisParam(self, n):
        return [self._make_raw('...')]

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
                ret.extend([self._make_raw('.'), self.visit(name)])
            else:
                ret.extend(self._bracket(self.visit(name)))
        ret.extend([self._make_space(), self._make_raw('='), self._make_space()])
        ret.extend(self._visit_expr(n.expr))
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
        ret.append(name + ' ' + (n.name or ''))
        ret.extend(self._make_raw(name), self._make_space())
        if n.name:
            ret.append(self._make_raw(n.name))
        if members is not None:
            # None means no members
            # Empty sequence means an empty list of members
            ret.extend([self._make_newline(), self._make_indent()])
            self.indent_level += 1
            inner = [self._make_newline()]
            inner.extend(body_function(members))
            self.indent_level -= 1
            inner.append(self._make_indent())
            ret.extend(self._brace(inner))
        return ret

    def _generate_struct_union_body(self, members):
        return [self._generate_stmt(decl) for decl in members]

    def _generate_enum_body(self, members):
        ret = [self.visit(value) for value in members]
        # remove the final `,` from the enumerator list
        ret[-1]['value'].pop(-2)
        return ret

    def _generate_stmt(self, n, add_indent=False):
        """ Generation from a statement node. This method exists as a wrapper
            for individual visit_* methods to handle different treatment of
            some statements in this context.
        """
        typ = type(n)
        if add_indent: self.indent_level += 1
        indent = self._make_indent()
        if add_indent: self.indent_level -= 1

        ret = self.visit(n)
        if typ in (c_ast.Compound, ExpressionList) or ret['class'] == 'empty':
            return ret
        ret['value'].insert(0, indent)

        if typ in (
                c_ast.Decl, c_ast.Assignment, c_ast.Cast, c_ast.UnaryOp,
                c_ast.BinaryOp, c_ast.TernaryOp, c_ast.FuncCall, c_ast.ArrayRef,
                c_ast.StructRef, c_ast.Constant, c_ast.ID, c_ast.Typedef,
                c_ast.ExprList):
            # These can also appear in an expression context so no semicolon
            # is added to them automatically
            #
            ret['value'].append(self._make_raw(';'))
        ret['value'].append(self._make_newline())
        return ret

    def _generate_decl(self, n):
        """ Generation from a Decl node.
        """
        s = []
        if n.funcspec:
            for i in n.funcspec:
                s.extend([self._make_raw(i), self._make_space])
        if n.storage:
            for f in n.storage:
                s.extend([self._make_raw(f), self._make_space])
        s.extend(self._generate_type(n.type))
        return s

    def _generate_type(self, n, modifiers=[]):
        """ Recursive generation from a type node. n is the type node.
            modifiers collects the PtrDecl, ArrayDecl and FuncDecl modifiers
            encountered on the way down to a TypeDecl, to allow proper
            generation from it.
        """
        typ = type(n)
        #~ print(n, modifiers)

        ret = []
        #if n in self.ast_data.node_properties:
        #    ret.update(self.ast_data.node_properties[n])

        if typ == c_ast.TypeDecl:
            if n.quals:
                for q in n.quals:
                    ret.extend([self._make_raw(q), self._make_space()])
            ret.append(self.visit(n.type))

            if n.declname:
                if 'original_name' in n.node_properties:
                    name = n.node_properties['original_name']
                else:
                    name = n.declname
            else:
                name = ''
            nstr = [self._make_raw(name)]
            # Resolve modifiers.
            # Wrap in parens to distinguish pointer to array and pointer to
            # function syntax.
            #
            for i, modifier in enumerate(modifiers):
                if isinstance(modifier, c_ast.ArrayDecl):
                    if (i != 0 and isinstance(modifiers[i - 1], c_ast.PtrDecl)):
                        self._paren(nstr)
                    nstr.extend(self._bracket(self.visit(modifier.dim)))
                elif isinstance(modifier, c_ast.FuncDecl):
                    if (i != 0 and isinstance(modifiers[i - 1], c_ast.PtrDecl)):
                        self._paren(nstr)
                    nstr.extend(self._paren(self.visit(modifier.args)))
                elif isinstance(modifier, c_ast.PtrDecl):
                    if modifier.quals:
                        mods = [self._make_raw('*')]
                        for q in modifier.quals:
                            mods.extend([self._make_space(), self._make_raw(q)])
                        nstr = mods + nstr
                    else:
                        nstr.insert(0, self._make_raw('*'))
            if nstr: nstr.insert(0, self._make_space())
            ret.extend(nstr)
            return ret
        elif typ == c_ast.Decl:
            ret = self.generate_decl(n.type)
            return ret
        elif typ == c_ast.Typename:
            ret.extend(self._generate_type(n.type))
            return ret
        elif typ == c_ast.IdentifierType:
            for name in n.names:
                ret.extend([self._make_raw(name), self._make_space()])
            return ret
        elif typ in (c_ast.ArrayDecl, c_ast.PtrDecl, c_ast.FuncDecl):
            ret.extend(self._generate_type(n.type, modifiers + [n]))
            return ret
        else:
            return [self.visit(n)]

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
            have higher precedence than operators.
        """
        return isinstance(n, (c_ast.Constant, c_ast.ID, c_ast.ArrayRef,
                              c_ast.StructRef, c_ast.FuncCall))
