from pycparser import c_ast # type:ignore

class NodeWrapper(c_ast.Node):
    attr_names = ()
    __slots__ = ['old', 'new', 'coord', '__weakref__']

    def __init__(self, old, new, coord=None):
        super().__init__()
        self.old = old
        self.new = new
        self.coord = coord

    def children(self):
        return [] if self.new is None else [('new', self.new)]
