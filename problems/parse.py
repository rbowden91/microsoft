import sys, json, os, re
from pprint import pprint
from shutil import copyfile, move
from subprocess import Popen, PIPE
from io import StringIO
from pycparser import c_parser, c_ast, parse_file

sys.path.insert(0,'../../../tree')
import preprocess

output = {
    'preprocessor1': True,
    'preprocessor2': True,
    'preprocessor3': True,
    'preprocessor4': True,
    'parse1': True,
    'parse2': True,
    'parse3': True,
    'compiles': False,
    'passes': False,
    'checks': {}
}

def get_headers():
    with open('../../../headers/' + sys.argv[1] + '.json') as f:
        headers = json.load(f)
    # prepend all headers that appear with at least some frequency (5%)
    # XXX we can also try to look for other headers they have between <>, in case they typo'd #include
    maximum = 0
    for header in headers:
        maximum = max(maximum, headers[header])
    prefix = ""
    for header in headers:
        if headers[header] / maximum > 0.01:
            prefix += '#include <' + header + '>' + '\n'
    return prefix

def replace_file(filename, remove):
    new_cfile = ""
    with open(filename, 'r') as f:
        if remove is not None:
            for line in f.readlines():
                if not re.match(remove, line):
                    new_cfile += line
    with open(filename, "w") as text_file:
        text_file.write(get_headers() + new_cfile)

filename = sys.argv[1]
copyfile(filename, filename + '.orig')

cfile, stderr = preprocess.preprocess_file(filename, path='clang', args=preprocess.cpp_args)

if len(stderr) != 0:
    output['preprocessor1'] = False
    replace_file(filename, '\s*#')
    cfile, stderr = preprocess.preprocess_file(filename, path='clang', args=preprocess.cpp_args)
    if len(stderr) != 0:
        output['preprocessor2'] = False

parser = c_parser.CParser()
try:
    ast = parser.parse(cfile)
except Exception as e:
    output['parse1'] = False
    if not output['preprocessor1']:
        copyfile(filename + '.orig', filename)
    replace_file(filename, '\s*#\s*include')
    cfile, stderr = preprocess.preprocess_file(filename, path='clang', args=preprocess.cpp_args)
    if len(stderr) != 0:
        output['preprocessor3'] = False
    try:
        ast = parser.parse(cfile)
    except Exception as e:
        output['parse2'] = False
        cfile, stderr = preprocess.preprocess_file(filename, path='clang', args=preprocess.cpp_args + ['-I../../../fake_libc_include'])
        if len(stderr) != 0:
            output['preprocessor4'] = False
        try:
            ast = parser.parse(cfile)
        except Exception as e:
            output['parse3'] = False

checks = {
    "caesar": ["cs50/2017/fall/caesar/old", "cs50/2017/fall/caesar/new"],
    "vigenere": ["cs50/2017/fall/vigenere/old", "cs50/2017/fall/vigenere/new"],
    "credit": ["cs50/2017/fall/credit"],
    "greedy": ["cs50/2017/fall/cash"],
    "mario": ["cs50/2017/fall/mario/less", "cs50/2017/fall/mario/more"],
    "recover": ["cs50/2017/fall/recover"],
    "resize": ["cs50/2017/fall/resize/less", "cs50/2017/fall/resize/more"],
    "dictionary": ["cs50/2017/fall/speller"]
}
for check in checks[sys.argv[2]]:
    try:
        pipe = Popen(['check50', '-d', '--offline', check],
                     stdout=PIPE,
                     stderr=PIPE,
                     universal_newlines=True)
        stdout, stderr = pipe.communicate()
    except OSError as e:
        raise RuntimeError("Unable to invoke 'check50'.  " +
            'Make sure its path was passed correctly\n' +
            ('Original error: %s' % e))
    results = json.loads(stdout)
    compiles = None
    passes = True
    for test in results:
        if test['name'] == 'compiles':
            if test['rationale'] == 'timed out while waiting for program to exit':
                print('uh oh, didn\'t compile')
                sys.exit(1)
            compiles = test['status']
        if not test['status']:
            passes = False
    if compiles is None:
        print('uh oh', results, output)
        sys.exit(1)
    output['compiles'] = compiles or output['compiles']
    output['passes'] = passes or output['passes']
    output['checks'][check] = results

with open('output.json', "w") as text_file:
    json.dump(output, text_file)

move(filename + '.orig', filename)
