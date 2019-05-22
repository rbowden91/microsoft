from pycparser import c_ast # type:ignore

class BreakPoint(c_ast.Node):
    attr_names = ()
    __slots__ = ['pre', 'post', 'child', 'coord', '__weakref__']

    def __init__(self, pre, child, post, coord=None):
        super().__init__()
        self.pre = pre
        self.post = post
        self.child = child
        self.coord = coord

    def children(self):
        return [] if self.child is None else [('child', self.child)]
