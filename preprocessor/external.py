#from ..my_env.typing import List
from ..my_env import os, subprocess

from .preprocessor import Preprocessor, PreprocessorException

class ExternalCPP(Preprocessor):
    def preprocess(self, code) -> None:
    #def preprocess_file(file_, is_code=False):
        # TODO: I give up putting effort into figuring out the right way to use __file__, if at all...
        dir_path = os.path.dirname(os.path.realpath(__file__))
        include_path = os.path.join(dir_path, 'clib/build/include')
        cpp_args = [r'cpp', r'-E', r'-g3', r'-gdwarf-2', r'-nostdinc', r'-I' + include_path,
                    r'-D__attribute__(x)=', r'-D__builtin_va_list=int', r'-D_Noreturn=', r'-Dinline=', r'-D__volatile__=',
                    '-']
        #cpp_args.append(file_ if not is_code else '-')

        # reading from stdin
        # TODO: hmm... should this always be latin-1?
        proc = subprocess.Popen(cpp_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE,
                encoding='latin-1')
        stdout, stderr = proc.communicate(file_ if is_code else None)
        if len(stderr) != 0:
            print('Uh oh! Stderr messages', proc.stderr)
        elif proc.returncode != 0:
            print('Uh oh! Nonzero error code')
        else:
            return stdout
