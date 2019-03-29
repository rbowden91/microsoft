from pycparser import c_ast # type:ignore

class ExpressionList(c_ast.Node):
    attr_names = ()
    __slots__ = ('expressions', 'coord', '__weakref__')

    def __init__(self, expressions, coord=None):
        super().__init__()
        self.expressions = expressions
        self.coord = coord

    def children(self):
        return [('expressions[]', e) for e in self.expressions]
