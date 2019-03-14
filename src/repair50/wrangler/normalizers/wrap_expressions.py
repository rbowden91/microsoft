from .ExpressionList import ExpressionList

from .modifying_visitor import ModifyingVisitor

class WrapExpressions(ModifyingVisitor):
    def visit_Compound(self, n):
        if n.block_items is None: return n

        items = []
        expressions = []
        for item in n.block_items:
            item = self.visit(item)
            if item is None: continue
            # break? continue? goto? label?
            if item.__class__.__name__ in ['For', 'If', 'While', 'DoWhile', 'Compound', 'Break', 'Continue', 'Return', 'Goto', 'Label', 'Switch', 'Case', 'Default']:
                items.append(ExpressionList(expressions))
                expressions = []
                items.append(item)
            else:
                expressions.append(item)
        items.append(ExpressionList(expressions))
        n.block_items = items
        return n
