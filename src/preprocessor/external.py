#from ..my_env.typing import List
from ..my_env import os, subprocess
from ..my_env.typing import Dict, List

from .preprocessor import Preprocessor, PreprocessorException

# TODO: do we want a timeout here?
class ExternalCPP(Preprocessor):
    def preprocess(self, code : str) -> str:
    #def preprocess_file(file_, is_code=False):
        # TODO: I give up putting effort into figuring out the right way to use __file__, if at all...
        dir_path = os.path.dirname(os.path.realpath(__file__))
        include_path = os.path.join(dir_path, 'headers/fake')
        cpp_args = [r'clang', r'-E', r'-nostdinc', r'-I' + include_path,
                    #   [r'cpp', r'-E', r'-g3', r'-gdwarf-2', r'-nostdinc', r'-I' + include_path,
                    #r'-D__attribute__(x)=', r'-D__builtin_va_list=int', r'-D_Noreturn=', r'-Dinline=', r'-D__volatile__=',
                    '-']
        #cpp_args.append(file_ if not is_code else '-')

        # reading from stdin
        # TODO: hmm... should this always be latin-1?
        proc = subprocess.Popen(cpp_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE,
                encoding='latin-1')
        stdout, stderr = proc.communicate(code)
        #stdout, stderr = proc.communicate(file_ if is_code else None)
        proc.stdin.close()
        if len(stderr) != 0:
            raise PreprocessorException('Uh oh! Stderr messages: {}'.format(proc.stderr))
        elif proc.returncode != 0:
            raise PreprocessorException('Uh oh! Nonzero error code: {}'.format(proc.returncode))
        else:
            return stdout
    def grab_directives(self, code): pass



# taken from pycparser, but extended to also return stderr
# can also use 'scc' if it exists on thes system, to
#def preprocess_file(filename, path='cpp', args=[]):
#    path_list = [path]
#    if isinstance(args, list):
#        path_list += args
#    elif args != '':
#        path_list += [args]
#    path_list += [filename]
#
#    try:
#        pipe = Popen(path_list,
#                     stdout=PIPE,
#                     stderr=PIPE,
#                     universal_newlines=True)
#        text = pipe.communicate()
#    except OSError as e:
#        raise RuntimeError("Unable to invoke '" + path + "'.  " +
#            'Make sure its path was passed correctly\n' +
#            ('Original error: %s' % e))
#
#    return text
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
