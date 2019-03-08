from typing import NamedTuple, List, Optional, TypeVar, Type
from .preprocessor import Preprocessor
from .external import ExternalCPP

class HeaderConfig(NamedTuple):
    path : str
    cflags : List[str]

fake_headers = HeaderConfig(path='fake', cflags=[])
c9_headers = HeaderConfig(path='c9', cflags=[])
# these flags are needed to let things compiled with musl headers to even parse
musl_headers = HeaderConfig(path='musl', cflags=['-D__attribute__(x)=', '-D__builtin_va_list=int', '-D_Noreturn=',
                                                 '-Dinline=', '-D__volatile__='])
#, '-D__extension__=', '-D__attribute__(x)=', '-D__nonnull(x)=', '-D__restrict=',
#            '-D__THROW=', '-D__volatile__=', '-D__asm__(x)=', '-D__STRING_INLINE=', '-D__inline=']
            #"-D__builtin_va_list=char*"]


#PreprocessorType = TypeVar('PreprocessorType', bound=Preprocessor)
class PreprocessorConfig(NamedTuple):
    header : HeaderConfig
    preprocessor : Type[Preprocessor]
    preprocessor_path : Optional[str]
    on_disk : bool
    cflags : List[str]

default_ppconfig = PreprocessorConfig(fake_headers, ExternalCPP, 'cpp', False, ['-E', '-g3', '-gdwarf-2', '-nostdinc'])

def make_preprocessor(config : PreprocessorConfig) -> Preprocessor:
    return config.preprocessor(PreprocessorConfig)
