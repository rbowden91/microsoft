import sys
from pycparser import c_generator, c_ast, c_lexer, c_parser, preprocess_file

# list of typedefs from common header files that we don't want to output
typedefs = { "size_t":"typedef int", "__builtin_va_list":"typedef int", "__gnuc_va_list":"typedef int", "__int8_t":"typedef int", "__uint8_t":"typedef int", "__int16_t":"typedef int", "__uint16_t":"typedef int", "__int_least16_t":"typedef int", "__uint_least16_t":"typedef int", "__int32_t":"typedef int", "__uint32_t":"typedef int", "__int64_t":"typedef int", "__uint64_t":"typedef int", "__int_least32_t":"typedef int", "__uint_least32_t":"typedef int", "__s8":"typedef int", "__u8":"typedef int", "__s16":"typedef int", "__u16":"typedef int", "__s32":"typedef int", "__u32":"typedef int", "__s64":"typedef int", "__u64":"typedef int", "_LOCK_T":"typedef int", "_LOCK_RECURSIVE_T":"typedef int", "_off_t":"typedef int", "__dev_t":"typedef int", "__uid_t":"typedef int", "__gid_t":"typedef int", "_off64_t":"typedef int", "_fpos_t":"typedef int", "_ssize_t":"typedef int", "wint_t":"typedef int", "_mbstate_t":"typedef int", "_flock_t":"typedef int", "_iconv_t":"typedef int", "__ULong":"typedef int", "__FILE":"typedef int", "ptrdiff_t":"typedef int", "wchar_t":"typedef int", "__off_t":"typedef int", "__pid_t":"typedef int", "__loff_t":"typedef int", "u_char":"typedef int", "u_short":"typedef int", "u_int":"typedef int", "u_long":"typedef int", "ushort":"typedef int", "uint":"typedef int", "clock_t":"typedef int", "time_t":"typedef int", "daddr_t":"typedef int", "caddr_t":"typedef int", "ino_t":"typedef int", "off_t":"typedef int", "dev_t":"typedef int", "uid_t":"typedef int", "gid_t":"typedef int", "pid_t":"typedef int", "key_t":"typedef int", "ssize_t":"typedef int", "mode_t":"typedef int", "nlink_t":"typedef int", "fd_mask":"typedef int", "_types_fd_set":"typedef int", "clockid_t":"typedef int", "timer_t":"typedef int", "useconds_t":"typedef int", "suseconds_t":"typedef int", "FILE":"typedef int", "fpos_t":"typedef int", "cookie_read_function_t":"typedef int", "cookie_write_function_t":"typedef int", "cookie_seek_function_t":"typedef int", "cookie_close_function_t":"typedef int", "cookie_io_functions_t":"typedef int", "div_t":"typedef int", "ldiv_t":"typedef int", "lldiv_t":"typedef int", "sigset_t":"typedef int", "__sigset_t":"typedef int", "_sig_func_ptr":"typedef int", "sig_atomic_t":"typedef int", "__tzrule_type":"typedef int", "__tzinfo_type":"typedef int", "mbstate_t":"typedef int", "sem_t":"typedef int", "pthread_t":"typedef int", "pthread_attr_t":"typedef int", "pthread_mutex_t":"typedef int", "pthread_mutexattr_t":"typedef int", "pthread_cond_t":"typedef int", "pthread_condattr_t":"typedef int", "pthread_key_t":"typedef int", "pthread_once_t":"typedef int", "pthread_rwlock_t":"typedef int", "pthread_rwlockattr_t":"typedef int", "pthread_spinlock_t":"typedef int", "pthread_barrier_t":"typedef int", "pthread_barrierattr_t":"typedef int", "jmp_buf":"typedef int", "rlim_t":"typedef int", "sa_family_t":"typedef int", "sigjmp_buf":"typedef int", "stack_t":"typedef int", "siginfo_t":"typedef int", "z_stream":"typedef int", "int8_t":"typedef int", "uint8_t":"typedef int", "int16_t":"typedef int", "uint16_t":"typedef int", "int32_t":"typedef int", "uint32_t":"typedef int", "int64_t":"typedef int", "uint64_t":"typedef int", "int_least8_t":"typedef int", "uint_least8_t":"typedef int", "int_least16_t":"typedef int", "uint_least16_t":"typedef int", "int_least32_t":"typedef int", "uint_least32_t":"typedef int", "int_least64_t":"typedef int", "uint_least64_t":"typedef int", "int_fast8_t":"typedef int", "uint_fast8_t":"typedef int", "int_fast16_t":"typedef int", "uint_fast16_t":"typedef int", "int_fast32_t":"typedef int", "uint_fast32_t":"typedef int", "int_fast64_t":"typedef int", "uint_fast64_t":"typedef int", "intptr_t":"typedef int", "uintptr_t":"typedef int", "intmax_t":"typedef int", "uintmax_t":"typedef int", "bool":"typedef _Bool", "va_list":"typedef int", "Display":"typedef struct Display", "XID":"typedef unsigned long", "VisualID":"typedef unsigned long", "Window":"typedef XID", "MirEGLNativeWindowType":"typedef void *", "MirEGLNativeDisplayType":"typedef void*", "MirConnection":"typedef struct MirConnection", "MirSurface":"typedef struct MirSurface", "MirSurfaceSpec":"typedef struct MirSurfaceSpec", "MirScreencast":"typedef struct MirScreencast", "MirPromptSession":"typedef struct MirPromptSession", "MirBufferStream":"typedef struct MirBufferStream", "MirPersistentId":"typedef struct MirPersistentId", "MirBlob":"typedef struct MirBlob", "MirDisplayConfig":"typedef struct MirDisplayConfig", "xcb_connection_t":"typedef struct xcb_connection_t", "xcb_window_t":"typedef uint32_t", "xcb_visualid_t":"typedef uint32_t", "string":"typedef char *"}

full_typedefs = set()
for k in typedefs:
    full_typedefs.add(typedefs[k] + ' ' + k)
# string is the only one that doesn't have a space between old and new type name
full_typedefs.add('typedef char *string')

no_replace = set(['main', 'argc', 'argv', 'int', 'char', 'signed char', 'unsigned char',
                  # best I can tell, the combo "short int", etc., are all handled as individual tokens
                  'short', 'short int', 'signed short', 'signed short int', 'unsigned short', 'unsigned short int',
                  'unsigned short', 'unsigned short int', 'int', 'signed', 'signed int', 'unsigned', 'unsigned int',
                  'long', 'long int', 'signed long', 'signed long int', 'unsigned long', 'unsigned long int',
                  'long long', 'long long int', 'signed long long', 'signed long long int', 'unsigned long long',
                  'unsigned long long int', 'float', 'double', 'long double', '_Bool', 'void'])

class IDRenamer(c_generator.CGenerator):
    def __init__(self, remove_typedefs, *args, **kwargs):
        super(IDRenamer, self).__init__(*args, **kwargs)
        self.remove_typedefs = remove_typedefs

        self.id_map = {}
        self.reverse_id_map = {}

        self.local_maps = [{}]
        self.reverse_local_maps = [{}]
        self.used_local_maps = []
        self.used_reverse_local_maps = []
        self.decl_now = None
        self.decl_sets = [set()]

        #self.str_map = {}
        self.type_ = ''
        # XXX global variables from headers, like errno?
        # also need to restrict variables within their scope, so that we don't consider local variables across functions
        # to be the same
        self.type_defs = {}
        self.visiting_funccall = False
        self.visiting_funcdef = False
        self.visiting_funcdecl = False
        self.visiting_args = False
        self.visiting_typedef = False
        self.visiting_struct_members = False
        self.visiting_union_members = False
        self.visiting_funcbody = False
        self.visiting_decl = False

    def save_id(self, n, type_=None, hungarian=''):
        if n in typedefs or n in no_replace:
            return n
        # XXX don't want to rename something like "struct tmp tmp" to "struct STRUCT_ID1 STRUCT_ID1"
        # XXX handle struct members? need to rename them
        elif n in self.local_maps[-1]:
            name = self.local_maps[-1][n]
            # a declaration with the same name shouldn't count as reusing the old variable
            if not self.visiting_decl and n not in self.decl_sets[-1]:
                self.used_local_maps[-1][n] = name
                self.used_reverse_local_maps[-1][name] = n
            elif self.visiting_decl:
                self.decl_now = n
            return name
        elif n in self.id_map:
            return self.id_map[n]

        if type_ in ['label', 'enum', 'struct', 'union']:
            # XXX currently does not distinguish between the enum type and its constants
            pass
        elif self.visiting_typedef:
            type_ = 'type'
        elif self.visiting_struct_members:
            type_ = 'struct_member'
        elif self.visiting_union_members:
            type_ = 'union_member'
        elif self.visiting_args:
            # don't put in names in function declarations
            # the second condition might be sufficient without the first?
            if self.visiting_funcdecl and not self.visiting_funcdef:
                return ''
            type_ = 'arg'
        # a function call that didn't come from a funcdef or funcdecl is presumably a library function
        elif self.visiting_funccall:
            return n
        elif self.visiting_funcdef or self.visiting_funcdecl:
            type_ = 'function'
        elif self.visiting_funcbody:
            type_ = 'local'
        else:
            type_ = 'global'
        name = type_.upper()
        if hungarian != "": name += '_' + hungarian.upper()
        name += '_ID'

        if type_ == 'local' or type_ == 'arg':
            for i in range(len(self.reverse_local_maps[-1]) + 1):
                if name + str(i) not in self.reverse_local_maps[-1]:
                    # XXX don't append a unique ID to it, for now
                    #name += str(i)
                    break
            else:
                print('uh oh')
        else:
            for i in range(len(self.reverse_id_map) + 1):
                if name + str(i) not in self.reverse_id_map:
                    #name += str(i)
                    break
            else:
                print('uh oh')

        if type_ == 'local' or type_ == 'arg':
            self.local_maps[-1][n] = name# + '___' + n
            self.reverse_local_maps[-1][name] = n
        else:
            self.id_map[n] = name# + '___' + n
            self.reverse_id_map[name] = n
        return name

    def push_scope(self):
        self.local_maps.append(dict(self.local_maps[-1]))
        self.reverse_local_maps.append(dict(self.reverse_local_maps[-1]))
        self.used_local_maps.append({})
        self.used_reverse_local_maps.append({})
        self.decl_sets.append(set())

    def update_scope(self):
        for var in self.used_local_maps[-1]:
            # we only want to pass "used" info back to the parent if the parent didn't declare the variable
            if var not in self.decl_sets[-2]:
                rename = self.used_local_maps[-1][var]
                self.used_local_maps[-2][var] = rename
                self.used_reverse_local_maps[-2][rename] = var

        self.local_maps[-1] = self.used_local_maps[-1]
        self.reverse_local_maps[-1] = self.used_reverse_local_maps[-1]

    def pop_scope(self):
        self.local_maps.pop()
        self.reverse_local_maps.pop()
        self.used_local_maps.pop()
        self.used_reverse_local_maps.pop()
        self.decl_sets.pop()


    def visit_Compound(self, n):
        self.push_scope()

        if n.block_items: [self._generate_stmt(stmt) for stmt in n.block_items]

        self.update_scope()

        s = self._make_indent() + '{\n'
        self.indent_level += 2
        if n.block_items:
            s += ''.join(self._generate_stmt(stmt) for stmt in n.block_items)
        self.indent_level -= 2
        s += self._make_indent() + '}\n'

        self.pop_scope()

        return s


    def visit_Enumerator(self, n):
        if not n.value:
            return '{indent}{name},\n'.format(
                indent=self._make_indent(),
                name=self.save_id(n.name, type_='enum'),
            )
        else:
            return '{indent}{name} = {value},\n'.format(
                indent=self._make_indent(),
                name=self.save_id(n.name, type_='enum'),
                value=self.visit(n.value),
            )

    def _generate_struct_union_body(self, members):
        tmp = ''.join(self._generate_stmt(decl) for decl in members)
        self.visiting_struct_members = False
        self.visiting_union_members = False
        return tmp

    def _generate_struct_union_enum(self, n, name):
        """ Generates code for structs, unions, and enums. name should be
            'struct', 'union', or 'enum'.
        """
        if name in ('struct', 'union'):
            members = n.decls
            if members is not None:
                if name == 'struct':
                    self.visiting_struct_members = True
                    # XXX THIS ISN'T 100% RIGHT!!! Two structs with members of the same name
                    # in different positions in the struct will conflict
                else:
                    self.visiting_union_members = True
            body_function = self._generate_struct_union_body
        else:
            assert name == 'enum'
            members = None if n.values is None else n.values.enumerators
            body_function = self._generate_enum_body
        if n.name:
            tmp = self.save_id(n.name, type_=name)
        else:
            tmp = ''
        s = name + ' ' + tmp
        if members is not None:
            # None means no members
            # Empty sequence means an empty list of members
            s += '\n'
            s += self._make_indent()
            self.indent_level += 2
            s += '{\n'
            s += body_function(members)
            self.indent_level -= 2
            s += self._make_indent() + '}'
        return s


    def visit_Label(self, n):
        return self.save_id(n.name, type_='label') + ':\n' + self._generate_stmt(n.stmt)

    def visit_Goto(self, n):
        return 'goto ' + self.save_id(n.name, type_='label') + ';'

    def visit_Constant(self, n):
        # for now, get rid of arbitrary strings
        #if n.value.startswith('"'):
        #    s = '"STR' + str(len(self.str_map)) + '"'
        #    self.str_map[s] = n.value
        #    return s
        #else:
        return n.value

    def visit_ID(self, n):
        return self.save_id(n.name)

    # XXX have to check this!!!
    def visit_NamedInitializer(self, n):
        s = ''
        for name in n.name:
            if isinstance(name, c_ast.ID):
                s += '.' + self.save_id(name.name)
            else:
                s += '[' + self.visit(name) + ']'
        s += ' = ' + self._visit_expr(n.expr)
        return s

    def visit_FuncCall(self, n):
        self.visiting_funccall = True
        fref = self._parenthesize_unless_simple(n.name)
        self.visiting_funccall = False
        tmp = self.visit(n.args)
        if fref + tmp == '':
            return ''
        return fref + '(' + self.visit(n.args) + ')'

    def visit_Decl(self, n, no_type=False):
        # no_type is used when a Decl is part of a DeclList, where the type is
        # explicitly only for the first declaration in a list.
        #
        if no_type:
            self.visiting_decl = True
            s = self.save_id(n.name, hungarian=self.type_)
            self.visiting_decl = False
        else:
            s = self._generate_decl(n)
        if n.bitsize: s += ' : ' + self.visit(n.bitsize)

        # the assignment shouldn't be included as a new variable
        if n.init:
            s += ' = ' + self._visit_expr(n.init)
        # only add the new declaration after the assignment completes (for something like "int x = x * 2")
        self.decl_sets[-1].add(self.decl_now)
        return s

    def _generate_decl(self, n):
        """ Generation from a Decl node."""
        s = ''
        if n.funcspec: s = ' '.join(n.funcspec) + ' '
        if n.storage: s += ' '.join(n.storage) + ' '
        s += self._generate_type(n.type)
        return s


    def visit_FuncDecl(self, n):
        self.visiting_funcdecl = True
        tmp = self._generate_type(n)
        self.visiting_funcdecl = False
        return tmp

    def visit_IdentifierType(self, n):
        return ' '.join([self.save_id(i) for i in n.names])

    def visit_For(self, n):
        self.push_scope()

        if n.init: self.visit(n.init)
        if n.cond: self.visit(n.cond)
        if n.next: self.visit(n.next)
        self._generate_stmt(n.stmt, add_indent=True)

        self.update_scope()

        s = 'for ('
        if n.init: s += self.visit(n.init)
        s += ';'
        if n.cond: s += ' ' + self.visit(n.cond)
        s += ';'
        if n.next: s += ' ' + self.visit(n.next)
        s += ')\n'
        s += self._generate_stmt(n.stmt, add_indent=True)

        self.pop_scope()
        return s


    def visit_FuncDef(self, n):
        self.push_scope()
        self.visiting_funcdef = True
        decl = self.visit(n.decl)
        self.visiting_funcdef = False
        self.indent_level = 0

        self.visiting_funcbody = True
        body = self.visit(n.body)
        self.visiting_funcbody = False
        self.pop_scope()
        if n.param_decls:
            knrdecls = ';\n'.join(self.visit(p) for p in n.param_decls)
            ret = decl + '\n' + knrdecls + ';\n' + body + '\n'
        else:
            ret = decl + '\n' + body + '\n'
        return ret

    def visit_Typedef(self, n):
        s = ''
        if n.storage: s += ' '.join(n.storage) + ' '
        # first check if this is a common typedef
        self.visiting_typedef = True
        s += self._generate_type(n.type)
        self.visiting_typedef = False
        if s in full_typedefs and self.remove_typedefs:
            return ""
        return s

    def _generate_type(self, n, modifiers=[]):
        """ Recursive generation from a type node. n is the type node.
            modifiers collects the PtrDecl, ArrayDecl and FuncDecl modifiers
            encountered on the way down to a TypeDecl, to allow proper
            generation from it.
        """
        typ = type(n)

        if typ == c_ast.TypeDecl:
            s = ''
            if n.quals: s += ' '.join(n.quals) + ' '
            s += self.visit(n.type)


            # XXX XXX XXX need to handle anonymous struct/enum/union typedefs

            # get the type for a form of hungarian notation renaming
            type_ = self.visit(n.type)
            for i, modifier in enumerate(modifiers):
                if isinstance(modifier, c_ast.ArrayDecl):
                    type_ += '_array'
                # XXX don't think about function pointers for now
                #elif isinstance(modifier, c_ast.FuncDecl):
                #    if (i != 0 and isinstance(modifiers[i - 1], c_ast.PtrDecl)):
                #        nstr = '(' + nstr + ')'
                #    self.visiting_args = True
                #    nstr += '(' + self.visit(modifier.args) + ')'
                #    self.visiting_args = False
                elif isinstance(modifier, c_ast.PtrDecl):
                    type_ += '_pointer'
                    # XXX don't think about modifiers, either
                    #if modifier.quals:
                    #    nstr = '* %s %s' % (' '.join(modifier.quals), nstr)
            # remember this, in case we are a part of a decllist
            type_ = type_.replace(' ', '_')
            # XXX this is to fix the anonymous enum issue
            if type_.find('{') != -1:
                type_ = 'BROKEN_TYPE'
            self.type_ = type_

            if (n.declname):
                self.visiting_decl = True
                nstr = self.save_id(n.declname, hungarian=type_)
                self.visiting_decl = False
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
                    self.visitng_decl = False
                    nstr += '[' + self.visit(modifier.dim) + ']'
                    self.visitng_decl = True
                elif isinstance(modifier, c_ast.FuncDecl):
                    if (i != 0 and isinstance(modifiers[i - 1], c_ast.PtrDecl)):
                        nstr = '(' + nstr + ')'
                    self.visiting_args = True
                    nstr += '(' + self.visit(modifier.args) + ')'
                    self.visiting_args = False
                elif isinstance(modifier, c_ast.PtrDecl):
                    if modifier.quals:
                        nstr = '* %s %s' % (' '.join(modifier.quals), nstr)
                    else:
                        nstr = '*' + nstr
            if nstr: s += ' ' + nstr
            # only add the new declaration after the assignment completes (for something like "int x = x * 2")
            self.decl_sets[-1].add(self.decl_now)
            return s
        elif typ == c_ast.Decl:
            return self._generate_decl(n.type)
        elif typ == c_ast.Typename:
            return self._generate_type(n.type)
        elif typ == c_ast.IdentifierType:
            return ' '.join([self.save_id(i) for i in n.names]) + ' '
        elif typ in (c_ast.ArrayDecl, c_ast.PtrDecl, c_ast.FuncDecl):
            if typ is c_ast.FuncDecl:
                self.visiting_funcdecl = True
                tmp = self._generate_type(n.type, modifiers + [n])
                self.visiting_funcdecl = False
                return tmp
            else:
                return self._generate_type(n.type, modifiers + [n])
        else:
            return self.visit(n)

    def visit_FileAST(self, n):
        s = ''
        for ext in n.ext:
            if isinstance(ext, c_ast.FuncDef):
                s += self.visit(ext)
            elif isinstance(ext, c_ast.Pragma):
                s += self.visit(ext) + '\n'
            else:
                # RTB: add this check to not insert semicolons for typedefs we don't care about
                tmp = self.visit(ext)
                if tmp != '':
                    s += tmp + ';\n'
        return s

if __name__=='__main__':
    generator = IDRenamer(False)
    parser = c_parser.CParser()
    try:
        cfile = preprocess_file(sys.argv[1], cpp_path='cpp', cpp_args=[r'-I../fake_libc_include'])
        ast = parser.parse(cfile)
    except Exception as e:
        print('uh oh2', e)
        sys.exit(1)
    renamed_code = generator.visit(ast)
