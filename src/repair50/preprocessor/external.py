import os, subprocess
from typing import Dict, List

from .preprocessor import Preprocessor, PreprocessorException

# TODO: do we want a timeout here?
class ExternalCPP(Preprocessor):
    def preprocess(self, code : str, is_file : bool) -> str:
    #def preprocess_file(file_, is_code=False):
        # TODO: I give up putting effort into figuring out the right way to use __file__, if at all...
        dir_path = os.path.dirname(os.path.realpath(__file__))
        include_path = os.path.join(dir_path, 'headers/fake')
        cpp_args = [r'clang', r'-E', r'-nostdinc', r'-I' + include_path,
                    #   [r'cpp', r'-E', r'-g3', r'-gdwarf-2', r'-nostdinc', r'-I' + include_path,
                    #r'-D__attribute__(x)=', r'-D__builtin_va_list=int', r'-D_Noreturn=', r'-Dinline=', r'-D__volatile__=',
                    '-']
        if is_file:
            with open(code, 'r', encoding='latin-1') as content_file:
                code = content_file.read()

        # reading from stdin
        # TODO: hmm... should this always be latin-1?
        proc = subprocess.Popen(cpp_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE,
                encoding='latin-1')
        stdout, stderr = proc.communicate(code)
        proc.stdin.close()
        if len(stderr) != 0:
            raise PreprocessorException('Uh oh! Stderr messages: {}'.format(proc.stderr))
        elif proc.returncode != 0:
            raise PreprocessorException('Uh oh! Nonzero error code: {}'.format(proc.returncode))
        else:
            return stdout
    def grab_directives(self, code): pass



#
#def grab_directives(string, defines=False):
#    # not perfect...
#    # XXX want to do something with potentially making the #define replacements, since those are what could
#    # be breaking things...
#    if defines:
#        pattern = r"(^\s*#[^\r\n]*[\r\n])"
#    else:
#        pattern = r"(^\s*#\s*define\s[^\r\n]*[\r\n])"
#    regex = re.compile(pattern, re.MULTILINE|re.DOTALL)
#
#    directives = ''.join(regex.findall(string))
#    def _replacer(match):
#        return ""
#    sub = regex.sub(_replacer, string)
#    return directives, sub
