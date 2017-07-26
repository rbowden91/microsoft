from __future__ import print_function
import sys
import os
from pycparser import parse_file, c_parser, c_generator, c_ast

class IDRenamer(c_generator.CGenerator):
    # XXX do something with string constants???
    def __init__(self, rename_ids, truncate_strings, *args, **kwargs):
        super(IDRenamer, self).__init__(*args, **kwargs)
        # XXX use FID?
        self.rename = rename_ids
        self.truncate_strings = truncate_strings
        self.never_rename = set('main')
        self.func_strs = ''

    def visit_Constant(self, n):
        # for now, get rid of arbitrary strings
        if self.truncate_strings and n.value.startswith('"'):
            return '""'
        else:
            return n.value

    def visit_ID(self, n):
        if not self.rename or n.name in self.never_rename:
            self.never_rename.add(n.name)
            return n.name
        else:
            return "ID"

    def visit_FuncCall(self, n):
        old = self.rename
        self.rename = False
        fref = self._parenthesize_unless_simple(n.name)
        self.rename = old
        return fref + '(' + self.visit(n.args) + ')'

    def visit_Decl(self, n, no_type=False):
        # no_type is used when a Decl is part of a DeclList, where the type is
        # explicitly only for the first declaration in a list.
        #
        if no_type:
            if not self.rename or n.name in self.never_rename:
                return n.name
            else:
                return "ID"
        else:
            s = self._generate_decl(n)
        if n.bitsize: s += ' : ' + self.visit(n.bitsize)
        if n.init:
            s += ' = ' + self._visit_expr(n.init)
        return s

    #def visit_FuncDecl(self, n):
    #    old = self.rename
    #    self.rename = False
    #    tmp = self._generate_type(n)
    #    self.rename = old
    #    print("***" + tmp)
    #    return tmp

    def visit_FuncDef(self, n):
        old = self.rename
        self.rename = False
        decl = self.visit(n.decl)
        self.rename = old
        self.indent_level = 0
        body = self.visit(n.body)
        if n.param_decls:
            knrdecls = ';\n'.join(self.visit(p) for p in n.param_decls)
            ret = decl + '\n' + knrdecls + ';\n' + body + '\n'
        else:
            ret = decl + '\n' + body + '\n'
        # XXX don't print anything outside function definitions, for now
        self.func_strs += ret
        return ret

    def _generate_type(self, n, modifiers=[]):
        """ Recursive generation from a type node. n is the type node.
            modifiers collects the PtrDecl, ArrayDecl and FuncDecl modifiers
            encountered on the way down to a TypeDecl, to allow proper
            generation from it.
        """
        typ = type(n)
        #~ print(n, modifiers)

        if typ == c_ast.TypeDecl:
            s = ''
            if n.quals: s += ' '.join(n.quals) + ' '
            s += self.visit(n.type)

            if (n.declname):
                if not self.rename or n.declname in self.never_rename:
                    self.never_rename.add(n.declname)
                    nstr = n.declname
                else:
                    nstr = "ID"
            else:
                nstr = ''

            # Resolve modifiers.
            # Wrap in parens to distinguish pointer to array and pointer to
            # function syntax.
            #
            for i, modifier in enumerate(modifiers):
                if isinstance(modifier, c_ast.ArrayDecl):
                    if (i != 0 and isinstance(modifiers[i - 1], c_ast.PtrDecl)):
                        nstr = '(' + nstr + ')'
                    nstr += '[' + self.visit(modifier.dim) + ']'
                elif isinstance(modifier, c_ast.FuncDecl):
                    if (i != 0 and isinstance(modifiers[i - 1], c_ast.PtrDecl)):
                        nstr = '(' + nstr + ')'
                    nstr += '(' + self.visit(modifier.args) + ')'
                elif isinstance(modifier, c_ast.PtrDecl):
                    if modifier.quals:
                        nstr = '* %s %s' % (' '.join(modifier.quals), nstr)
                    else:
                        nstr = '*' + nstr
            if nstr: s += ' ' + nstr
            return s
        elif typ == c_ast.Decl:
            return self._generate_decl(n.type)
        elif typ == c_ast.Typename:
            return self._generate_type(n.type)
        elif typ == c_ast.IdentifierType:
            #if self.rename:
            #    return "ID " * len(n.names)
            #else:
            #    return ' '.join(n.names) + ' '
            return ' '.join(n.names) + ' '
        elif typ in (c_ast.ArrayDecl, c_ast.PtrDecl, c_ast.FuncDecl):
            return self._generate_type(n.type, modifiers + [n])
        else:
            return self.visit(n)

def preprocess_c(filename, rename_ids=False, truncate_strings=False):
    ast = parse_file(filename, use_cpp=True, cpp_path='gcc', cpp_args=['-E', r'-I../fake_libc_include'])
    generator = IDRenamer(rename_ids, truncate_strings)
    generator.visit(ast)
    return generator.func_strs

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(preprocess_c(sys.argv[1]))
    else:
        print("Please provide a filename as argument")
