import re

from pycparser import c_ast # type:ignore

class ModifyingVisitor(c_ast.NodeVisitor):

    def generic_visit(self, node):
        c_names = {}
        for c_name, c in node.children():
            ret = self.visit(c)

            m = re.match("([^\\[]*)\\[", c_name)

            # is this a list-based node child (like a Compound's block_items)
            if m:
                name = m.groups()[0]
                if name not in c_names:
                    c_names[name] = []
                c_names[name].append(ret)
            else:
                setattr(node, c_name, ret)
        for name in c_names:
            setattr(node, name, c_names[name])
        return node
