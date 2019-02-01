from ..my_env.abc import ABCMeta, abstractmethod
from ..my_env.typing import List

class PreprocessorException(Exception): pass

class Preprocessor(metaclass=ABCMeta):
    #def __init__(self, args : List[str]) -> None:
    #    self.args = args

    @abstractmethod
    def preprocess(self, code : str) -> str: pass

    @abstractmethod
    def grab_directives(self, code : str) -> str: pass
