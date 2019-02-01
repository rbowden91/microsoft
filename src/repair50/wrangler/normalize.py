from ..my_env import sys, re
from ..my_env.packages.pycparser import c_ast

no_replace = set(['main', 'argc', 'argv', 'int', 'char', 'signed char', 'unsigned char',
                  # best I can tell, the combo "short int", etc., are all handled as individual tokens
                  'short', 'short int', 'signed short', 'signed short int', 'unsigned short', 'unsigned short int',
                  'unsigned short', 'unsigned short int', 'int', 'signed', 'signed int', 'unsigned', 'unsigned int',
                  'long', 'long int', 'signed long', 'signed long int', 'unsigned long', 'unsigned long int',
                  'long long', 'long long int', 'signed long long', 'signed long long int', 'unsigned long long',
                  'unsigned long long int', 'float', 'double', 'long double', '_Bool', 'void'])

typedefs = { "size_t":"typedef int", "__builtin_va_list":"typedef int", "__gnuc_va_list":"typedef int", "__int8_t":"typedef int", "__uint8_t":"typedef int", "__int16_t":"typedef int", "__uint16_t":"typedef int", "__int_least16_t":"typedef int", "__uint_least16_t":"typedef int", "__int32_t":"typedef int", "__uint32_t":"typedef int", "__int64_t":"typedef int", "__uint64_t":"typedef int", "__int_least32_t":"typedef int", "__uint_least32_t":"typedef int", "__s8":"typedef int", "__u8":"typedef int", "__s16":"typedef int", "__u16":"typedef int", "__s32":"typedef int", "__u32":"typedef int", "__s64":"typedef int", "__u64":"typedef int", "_LOCK_T":"typedef int", "_LOCK_RECURSIVE_T":"typedef int", "_off_t":"typedef int", "__dev_t":"typedef int", "__uid_t":"typedef int", "__gid_t":"typedef int", "_off64_t":"typedef int", "_fpos_t":"typedef int", "_ssize_t":"typedef int", "wint_t":"typedef int", "_mbstate_t":"typedef int", "_flock_t":"typedef int", "_iconv_t":"typedef int", "__ULong":"typedef int", "__FILE":"typedef int", "ptrdiff_t":"typedef int", "wchar_t":"typedef int", "__off_t":"typedef int", "__pid_t":"typedef int", "__loff_t":"typedef int", "u_char":"typedef int", "u_short":"typedef int", "u_int":"typedef int", "u_long":"typedef int", "ushort":"typedef int", "uint":"typedef int", "clock_t":"typedef int", "time_t":"typedef int", "daddr_t":"typedef int", "caddr_t":"typedef int", "ino_t":"typedef int", "off_t":"typedef int", "dev_t":"typedef int", "uid_t":"typedef int", "gid_t":"typedef int", "pid_t":"typedef int", "key_t":"typedef int", "ssize_t":"typedef int", "mode_t":"typedef int", "nlink_t":"typedef int", "fd_mask":"typedef int", "_types_fd_set":"typedef int", "clockid_t":"typedef int", "timer_t":"typedef int", "useconds_t":"typedef int", "suseconds_t":"typedef int", "FILE":"typedef int", "fpos_t":"typedef int", "cookie_read_function_t":"typedef int", "cookie_write_function_t":"typedef int", "cookie_seek_function_t":"typedef int", "cookie_close_function_t":"typedef int", "cookie_io_functions_t":"typedef int", "div_t":"typedef int", "ldiv_t":"typedef int", "lldiv_t":"typedef int", "sigset_t":"typedef int", "__sigset_t":"typedef int", "_sig_func_ptr":"typedef int", "sig_atomic_t":"typedef int", "__tzrule_type":"typedef int", "__tzinfo_type":"typedef int", "mbstate_t":"typedef int", "sem_t":"typedef int", "pthread_t":"typedef int", "pthread_attr_t":"typedef int", "pthread_mutex_t":"typedef int", "pthread_mutexattr_t":"typedef int", "pthread_cond_t":"typedef int", "pthread_condattr_t":"typedef int", "pthread_key_t":"typedef int", "pthread_once_t":"typedef int", "pthread_rwlock_t":"typedef int", "pthread_rwlockattr_t":"typedef int", "pthread_spinlock_t":"typedef int", "pthread_barrier_t":"typedef int", "pthread_barrierattr_t":"typedef int", "jmp_buf":"typedef int", "rlim_t":"typedef int", "sa_family_t":"typedef int", "sigjmp_buf":"typedef int", "stack_t":"typedef int", "siginfo_t":"typedef int", "z_stream":"typedef int", "int8_t":"typedef int", "uint8_t":"typedef int", "int16_t":"typedef int", "uint16_t":"typedef int", "int32_t":"typedef int", "uint32_t":"typedef int", "int64_t":"typedef int", "uint64_t":"typedef int", "int_least8_t":"typedef int", "uint_least8_t":"typedef int", "int_least16_t":"typedef int", "uint_least16_t":"typedef int", "int_least32_t":"typedef int", "uint_least32_t":"typedef int", "int_least64_t":"typedef int", "uint_least64_t":"typedef int", "int_fast8_t":"typedef int", "uint_fast8_t":"typedef int", "int_fast16_t":"typedef int", "uint_fast16_t":"typedef int", "int_fast32_t":"typedef int", "uint_fast32_t":"typedef int", "int_fast64_t":"typedef int", "uint_fast64_t":"typedef int", "intptr_t":"typedef int", "uintptr_t":"typedef int", "intmax_t":"typedef int", "uintmax_t":"typedef int", "bool":"typedef _Bool", "va_list":"typedef int", "Display":"typedef struct Display", "XID":"typedef unsigned long", "VisualID":"typedef unsigned long", "Window":"typedef XID", "MirEGLNativeWindowType":"typedef void *", "MirEGLNativeDisplayType":"typedef void*", "MirConnection":"typedef struct MirConnection", "MirSurface":"typedef struct MirSurface", "MirSurfaceSpec":"typedef struct MirSurfaceSpec", "MirScreencast":"typedef struct MirScreencast", "MirPromptSession":"typedef struct MirPromptSession", "MirBufferStream":"typedef struct MirBufferStream", "MirPersistentId":"typedef struct MirPersistentId", "MirBlob":"typedef struct MirBlob", "MirDisplayConfig":"typedef struct MirDisplayConfig", "xcb_connection_t":"typedef struct xcb_connection_t", "xcb_window_t":"typedef uint32_t", "xcb_visualid_t":"typedef uint32_t", "string":"typedef char *" }

class RemoveTypedefs(c_ast.NodeVisitor):
    def visit_Typedef(self, node):
        return node if node.name not in typedefs else None

    def generic_visit(self, node):
        c_names = {}
        for c_name, c in node.children():
            # for now, we can only really remove something if its from a list, since we can't just remove a
            # necessary part of an AST node
            m = re.match("([^\\[]*)\\[", c_name)
            if not m:
                continue

            name = m.groups()[0]
            if name not in c_names:
                c_names[name] = {
                    'list': [],
                    'removed': False
                }
            ret = self.visit(c)

            if ret is None:
                c_names[name]['removed'] = True
            else:
                c_names[name]['list'].append(c)
        for name in c_names:
            if c_names[name]['removed']:
                setattr(node, name, c_names[name]['list'])
        return node

class RemoveDecls(c_ast.NodeVisitor):
    def generic_visit(self, node):
        #print(node.__class__.__name__)
        for c_name, c in node.children():
            self.visit(c)
        return node

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

class IDRenamer(c_ast.NodeVisitor):
    def __init__(self, *args, **kwargs):
        super(IDRenamer, self).__init__(*args, **kwargs)

        self.node_name_map = {}

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
        self.visiting_enum = False
        self.visiting_struct_members = False
        self.visiting_union_members = False
        self.visiting_funcbody = False
        self.visiting_decl = False

    def save_id(self, n, node, context=None):
        self.node_name_map[node] = n
        if n in no_replace or n in typedefs.keys():
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
        # a function call that didn't come from a funcdef or funcdecl is presumably a library function
        elif self.visiting_funccall:
            return n

        if self.visiting_struct_members:
            context = 'struct_member'
        elif self.visiting_union_members:
            context = 'union_member'
        elif self.visiting_enum:
            context = 'enum_member'
        elif self.visiting_args:
            # don't put in names in function declarations
            if self.visiting_funcdecl:
                return ''
            context = 'arg'
        elif (self.visiting_funcdef and not self.visiting_funcbody) or self.visiting_funcdecl:
            context = 'function'
        elif self.visiting_funcbody:
            context = 'local'
        else:
            context = 'global'
        name = '_'.join(context.split(' ')).upper()
        name += '_ID'

        if context == 'local' or context == 'arg':
            for i in range(len(self.reverse_local_maps[-1]) + 1):
                if name + str(i) not in self.reverse_local_maps[-1]:
                    # XXX don't append a unique ID to it, for now
                    #name += str(i)
                    break
            else: assert False
        else:
            for i in range(len(self.reverse_id_map) + 1):
                if name + str(i) not in self.reverse_id_map:
                    #name += str(i)
                    break
            else: assert False

        if context == 'local' or context == 'arg':
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

        if n.block_items: n.block_items = [self.visit(stmt) for stmt in n.block_items]

        self.update_scope()

        self.pop_scope()
        return n

    def visit_Enum(self, n):
        n.name = self.save_id(n.name, n, context='enum')
        self.visiting_enum = n.name
        self.visit(n.values)
        self.visiting_enum = False
        return n

    def visit_Struct(self, n):
        n.name = self.save_id(n.name, n, context='struct')
        self.visiting_struct_members = n.name
        if n.decls: [self.visit(decl) for decl in n.decls]
        self.visiting_struct_members = False
        return n

    def visit_Union(self, n):
        n.name = self.save_id(n.name, n, context='union')
        self.visiting_union_members = n.name
        if n.decls: [self.visit(decl) for decl in n.decls]
        self.visiting_union_members = False
        return n


    def visit_Enumerator(self, n):
        n.name = self.save_id(n.name, n, context='enum_val')
        if n.value: self.visit(n.value)
        return n

    def visit_Label(self, n):
        n.name = self.save_id(n.name, n, context='label')
        self.visit(n.stmt)
        return n

    def visit_Goto(self, n):
        n.name = self.save_id(n.name, n, context='label')
        return n

    def visit_Constant(self, n):
        return n

    def visit_ID(self, n):
        n.name = self.save_id(n.name, n)
        return n

    def visit_FuncCall(self, n):
        self.visiting_funccall = True
        self.visit(n.name)
        self.visiting_funccall = False
        if n.args: self.visit(n.args)
        return n

    def visit_Decl(self, n):
        self.visit(n.type)
        self.visiting_decl = True
        # TODO: get type from n.type
        n.name = self.save_id(n.name, n)
        self.visiting_decl = False
        if n.init: self.visit(n.init)
        # only add the new declaration after the assignment completes (for something like "int x = x * 2")
        self.decl_sets[-1].add(self.decl_now)
        return n


    def visit_FuncDecl(self, n):
        self.visiting_funcdecl = True
        self.visit(n.type)
        self.visiting_args = True
        if n.args: self.visit(n.args)
        self.visiting_args = False
        self.visiting_funcdecl = False
        return n

    def visit_IdentifierType(self, n):
        # is this necessary? will it only ever be the default "int", etc.?
        # TODO: COPY NAMES INTO node_name_map for whole list?
        n.names = [self.save_id(i, n) for i in n.names]
        return n

    def visit_For(self, n):
        self.push_scope()

        if n.init: self.visit(n.init)
        if n.cond: self.visit(n.cond)
        if n.next: self.visit(n.next)
        self.visit(n.stmt)

        self.update_scope()

        #if n.init: self.visit(n.init)
        #if n.cond: self.visit(n.cond)
        #if n.next: self.visit(n.next)

        #self.visit(n.stmt)

        self.pop_scope()
        return n


    def visit_FuncDef(self, n):
        self.push_scope()
        self.visiting_funcdef = True
        self.visiting_args = True
        decl = self.visit(n.decl)
        if n.param_decls: [self.visit(p) for p in n.param_decls]
        self.visiting_args = False

        self.visiting_funcbody = True
        body = self.visit(n.body)
        self.visiting_funcbody = False
        self.visiting_funcdef = False
        self.pop_scope()
        return n

    def visit_Typedef(self, n):
        self.visiting_typedef = True
        n.name = self.save_id(n.name, n, context='type')
        self.visit(n.type)
        self.visiting_typedef = False
        return n

    # when is this used??
    def visit_TypeDecl(self, n):
        n.declname = self.save_id(n.declname, n, context='type')
        self.visit(n.type)
        return n

    # similarly, when is this used??
    def visit_Typename(self, n):
        n.name = self.save_id(n.name, n, context='type')
        self.visit(n.type)
        return n


    def generic_visit(self, n):
        for c_name, c in n.children():
            self.visit(c)
        return n

