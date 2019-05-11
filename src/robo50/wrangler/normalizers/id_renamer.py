from .remove_typedefs import typedefs
from .modifying_visitor import ModifyingVisitor

no_replace = set(['main', 'argc', 'argv', 'int', 'char', 'signed char', 'unsigned char',
                  # best I can tell, the combo "short int", etc., are all handled as individual tokens
                  'short', 'short int', 'signed short', 'signed short int', 'unsigned short', 'unsigned short int',
                  'unsigned short', 'unsigned short int', 'int', 'signed', 'signed int', 'unsigned', 'unsigned int',
                  'long', 'long int', 'signed long', 'signed long int', 'unsigned long', 'unsigned long int',
                  'long long', 'long long int', 'signed long long', 'signed long long int', 'unsigned long long',
                  'unsigned long long int', 'float', 'double', 'long double', '_Bool', 'void'])

class IDRenamer(ModifyingVisitor):
    def __init__(self, *args, **kwargs):
        super(IDRenamer, self).__init__(*args, **kwargs)

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
            self.reverse_local_maps[-1][n].add(node)
            node.node_properties['replace_name'] = name
            node.node_properties['pointers'] = self.reverse_local_maps[-1][n]
            return name
        elif n in self.id_map:
            self.reverse_id_map[n].add(node)
            node.node_properties['replace_name'] = self.id_map[n]
            return self.id_map[n]
        # a function call that didn't come from a funcdef or funcdecl is presumably a library function
        elif self.visiting_funccall:
            node.node_properties['replace_name'] = n
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

        node.node_properties['replace_name'] = name
        if context == 'local' or context == 'arg':
            self.local_maps[-1][n] = name# + '___' + n
            self.reverse_local_maps[-1][n] = set(node)
            node.node_properties['pointers'] = self.reverse_local_maps[-1][n]
        else:
            self.id_map[n] = name# + '___' + n
            self.reverse_id_map[n] = set(node)
            node.node_properties['pointers'] = self.reverse_id_map[n]
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
        self.save_id(n.name, n, context='enum')
        self.visiting_enum = n.name
        self.visit(n.values)
        self.visiting_enum = False
        return n

    def visit_Struct(self, n):
        self.save_id(n.name, n, context='struct')
        self.visiting_struct_members = n.name
        if n.decls: [self.visit(decl) for decl in n.decls]
        self.visiting_struct_members = False
        return n

    def visit_Union(self, n):
        self.save_id(n.name, n, context='union')
        self.visiting_union_members = n.name
        if n.decls: [self.visit(decl) for decl in n.decls]
        self.visiting_union_members = False
        return n


    def visit_Enumerator(self, n):
        self.save_id(n.name, n, context='enum_val')
        if n.value: self.visit(n.value)
        return n

    def visit_Label(self, n):
        self.save_id(n.name, n, context='label')
        self.visit(n.stmt)
        return n

    def visit_Goto(self, n):
        self.save_id(n.name, n, context='label')
        return n

    def visit_Constant(self, n):
        return n

    def visit_ID(self, n):
        self.save_id(n.name, n)
        return n

    def visit_FuncCall(self, n):
        # TODO: push_func_scope
        self.visiting_funccall = True
        self.visit(n.name)
        self.visiting_funccall = False
        if n.args: self.visit(n.args)
        return n

    def visit_Decl(self, n):
        self.visit(n.type)
        self.visiting_decl = True
        # TODO: get type from n.type
        self.save_id(n.name, n)
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
        n.names = [self.save_id(i, n) for i in n.names]
        return n

    def visit_For(self, n):
        self.push_scope()

        if n.init: self.visit(n.init)
        if n.cond: self.visit(n.cond)
        if n.next: self.visit(n.next)
        self.visit(n.stmt)

        self.update_scope()

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
        self.save_id(n.name, n, context='type')
        self.visit(n.type)
        self.visiting_typedef = False
        return n

    # when is this used??
    def visit_TypeDecl(self, n):
        self.save_id(n.declname, n, context='type')
        self.visit(n.type)
        return n

    # similarly, when is this used??
    def visit_Typename(self, n):
        self.save_id(n.name, n, context='type')
        self.visit(n.type)
        return n
