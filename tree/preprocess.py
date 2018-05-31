import os, sys, glob
import argparse
import dump_ast
import random
import json
import re
from queue import Queue
from threading import Thread, Lock
from pycparser import parse_file, c_parser, c_generator, c_ast, c_lexer
from subprocess import Popen, PIPE

from renamer import IDRenamer
from normalize import RemoveDecls

cpp_args = ['-E', '-P', '-D__extension__=', '-D__attribute__(x)=', '-D__nonnull(x)=', '-D__restrict=',
            '-D__THROW=', '-D__volatile__=', '-D__asm__(x)=', '-D__STRING_INLINE=', '-D__inline=']
            #"-D__builtin_va_list=char*"]

# taken from pycparser, but extended to also return stderr
# can also use 'scc' if it exists on thes system, to
def preprocess_file(filename, path='cpp', args=[]):
    path_list = [path]
    if isinstance(args, list):
        path_list += args
    elif args != '':
        path_list += [args]
    path_list += [filename]

    try:
        pipe = Popen(path_list,
                     stdout=PIPE,
                     stderr=PIPE,
                     universal_newlines=True)
        text = pipe.communicate()
    except OSError as e:
        raise RuntimeError("Unable to invoke '" + path + "'.  " +
            'Make sure its path was passed correctly\n' +
            ('Original error: %s' % e))

    return text

def grab_directives(string, defines=False):
    # not perfect...
    # XXX want to do something with potentially making the #define replacements, since those are what could
    # be breaking things...
    if defines:
        pattern = r"(^\s*#[^\r\n]*[\r\n])"
    else:
        pattern = r"(^\s*#\s*define\s[^\r\n]*[\r\n])"
    regex = re.compile(pattern, re.MULTILINE|re.DOTALL)

    directives = ''.join(regex.findall(string))
    def _replacer(match):
        return ""
    sub = regex.sub(_replacer, string)
    return directives, sub

def lex_code(code):
    lexer = c_lexer.CLexer(
            error_func=lambda x, y, z: True,
            type_lookup_func=lambda x: True,
            on_lbrace_func=lambda: True,
            on_rbrace_func=lambda: True
    )

    lexer.build()
    lexer.input(code)

    tokens = []
    while True:
        token = lexer.token()
        if token is None:
            break
        value = token.value
        tokens.append(value)

    tokens.insert(0,"<sof>")
    tokens.append("<eof>")
    return tokens

# does not handle digraphs/trigraphs
def preprocess_c(filename, options=None, include_dependencies=False):
    #if options['remove_comments']:
    # code, stderr = preprocess_file(filename, path='scc')
    # does the lexer automatically remove comments for us? do we need a separate step for that?
    # we can use the preprocessor to remove comments:
    # gcc -fpreprocessed -dD -E test.c
    # but that will remove any invalid preprocessor lines as well
    #with open(filename, 'r') as content_file:
    #    content = content_file.read()
    #directives, _ = grab_directives(content)

    cfile, stderr = preprocess_file(filename, path='clang', args=cpp_args + [r'-I../fake_libc_include'])
    parser = c_parser.CParser()
    try:
        ast = parser.parse(cfile)
    except Exception as e:
        print('uh oh', e, filename)
        return None
    remove_decls = RemoveDecls()
    ast = remove_decls.visit(ast)

    id_renamer = IDRenamer(False)
    renamed_code = id_renamer.visit(ast)

    # have to make sure we never try to parse something that has had typedefs removed. definitely a better way of doing
    # this
    id_renamer.remove_typedefs = True
    typedef_removed_code = id_renamer.visit(ast)
    linear_tokens = lex_code(typedef_removed_code)

    try:
        ast = parser.parse(renamed_code)
    except Exception as e:
        print(renamed_code)
        print('uh oh2', e, filename)
        return None

    #generator.remove_typedefs = True
    ast_nodes, node_properties = dump_ast.linearize_ast(ast, id_renamer, include_dependencies=include_dependencies)
    return ast_nodes, ast, node_properties, linear_tokens

def tokens_to_ids(tokens, token_to_id, string_check, include_token):
    output = []
    for i in range(len(tokens)):
        token = tokens[i]
        if token in token_to_id:
            id = token_to_id[token]
        elif string_check and token.startswith('"'):
            id = token_to_id['<unk_str>']
        else:
            id = token_to_id['<unk>']
        output.append((id, token) if include_token else id)
    return output

def process_linear(queue, key, lexicon, lock, tokens):
    # XXX if I want to do padding/batching...
    #queue['max_linear_tokens'] = max(queue['max_tokens'], len(tokens))
    with lock:
        if key == 'train':
            for token in set(tokens):
                if token not in lexicon['linear_tokens']:
                    lexicon['linear_tokens'][token] = 0
                lexicon['linear_tokens'][token] += 1
    return { 'label': tokens }

def process_ast(queue, key, lexicon, lock, data):
    label_lexicon = set()
    attr_lexicon = set()
    new_rows = {}
    for i in range(len(data)):
        label = data[i]['label']
        if key == "train" and label not in label_lexicon:
            with lock:
                if label not in lexicon['ast_labels']:
                    lexicon['ast_labels'][label] = 0
                lexicon['ast_labels'][label] += 1
            label_lexicon.add(label)

        # transform attrs to a attr index.
        # XXX strongly assumes everything has at most one attr!
        for (name, val) in data[i]['attrs']:
            if name in ['value', 'op', 'name']:
                data[i]['attr'] = val
                break
        else:
            #print('woo')
            data[i]['attr'] = '<no_attr>'
        if key == 'train' and data[i]['attr'] not in label_lexicon:
            with lock:
                if data[i]['attr'] not in lexicon['ast_attrs']:
                    lexicon['ast_attrs'][data[i]['attr']] = 0
                lexicon['ast_attrs'][data[i]['attr']] += 1
            attr_lexicon.add(data[i]['attr'])
        del(data[i]['attrs'])
    for k in data[0]:
        new_rows[k] = [int(d[k]) if isinstance(d[k], bool) else d[k] for d in data]
    return new_rows



def process_queue(queues, lexicon, lock, args):
    keys = list(queues.keys())
    random.shuffle(keys)
    # relies on GIL?
    for i in range(len(keys)):
        key = keys[i]
        while not queues[key]['queue'].empty():
            filename = queues[key]['queue'].get()
            #print(filename)
            ret = preprocess_c(filename, args)
            if ret is None:
                print('uh oh!** ', filename)
                queues[key]['queue'].task_done()
                continue
            ast_nodes, _, _, linear_tokens = ret
            rows = {}
            rows['linear'] = process_linear(queues[key], key, lexicon, lock, linear_tokens)
            rows['ast'] = process_ast(queues[key], key, lexicon, lock, ast_nodes)
            with lock:
                for j in rows:
                    queues[key][j].append(rows[j])
            queues[key]['queue'].task_done()



if __name__ == "__main__":

    # don't do this above, so that files importing this one don't need to load tensorflow
    import tensorflow as tf

    parser = argparse.ArgumentParser(description='Tokenizer')
    parser.add_argument('filename', help='the name of the data files to process, such as caesar.c')
    parser.add_argument('read_path', help='directory to read from for processing')
    parser.add_argument('store_path', help='directory to store processed data')
    parser.add_argument('-n', '--num_files', help='number of files to parse (default, all files)', type=int)
    parser.add_argument('-t', '--num_threads', help='number of concurrent threads (default 16)', type=int, default=16)
    parser.add_argument('--train_fraction', help='fraction of files for training', type=float, default=.8)
    parser.add_argument('--valid_fraction', help='fraction of files for validating', type=float, default=.1)
    parser.add_argument('--unk_cutoff', help='fraction of files that need to have a token or else it\'s considered unknown', type=float, default=.01)

    #parser.add_argument('-p', '--preserve_preprocesor', help='reinsert things like headers. for now, can\'t preserve #defines')
    #parser.add_argument('-H', '--fake_headers', help='use fake C headers instead of the real ones for preprocessing', action='store_true')
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.read_path,'**',args.filename), recursive=True)
    random.shuffle(files)
    if args.num_files is None:
        args.num_files = len(files)
    else:
        args.num_files = min(args.num_files, len(files))

    queues = {}
    for i in ['train', 'test', 'valid']:
        queues[i] = { 'queue': Queue(maxsize=0), 'ast': [], 'linear': [] }

    for i in range(args.num_files):
        if i < args.train_fraction * args.num_files:
            queues['train']['queue'].put(files[i])
        elif i < (args.train_fraction + args.valid_fraction) * args.num_files:
            queues['valid']['queue'].put(files[i])
        else:
            queues['test']['queue'].put(files[i])

    lexicon = {
        'ast_labels': {},
        'ast_attrs': {},
        'linear_tokens': {}
    }
    lock = Lock()
    for i in range(args.num_threads):
        t = Thread(target=process_queue, args=(queues, lexicon, lock, args))
        t.daemon = True
        t.start()

    for i in queues:
        queues[i]['queue'].join()

    cutoff = args.num_files * args.train_fraction * args.unk_cutoff

    ast_labels = set([label for label in lexicon['ast_labels'] if lexicon['ast_labels'][label] > cutoff])
    ast_attrs = set([attr for attr in lexicon['ast_attrs'] if lexicon['ast_attrs'][attr] > cutoff])
    linear_tokens = set([token for token in lexicon['linear_tokens'] if lexicon['linear_tokens'][token] > cutoff])

    lexicon = {'ast': {}, 'linear': {}}
    lexicon['ast']['label_ids'] = dict(zip(ast_labels, range(1, len(ast_labels) + 1)))
    lexicon['ast']['label_ids']['<nil>'] = 0
    lexicon['ast']['label_ids']['<unk>'] = len(lexicon['ast']['label_ids'])
    lexicon['ast']['attr_ids'] = dict(zip(ast_attrs, range(1, len(ast_attrs) + 1)))
    lexicon['ast']['attr_ids']['<nil>'] = 0
    lexicon['ast']['attr_ids']['<unk>'] = len(lexicon['ast']['attr_ids'])
    lexicon['ast']['attr_ids']['<unk_str>'] = len(lexicon['ast']['attr_ids'])
    lexicon['linear']['label_ids'] = dict(zip(linear_tokens, range(1, len(linear_tokens) + 1)))
    lexicon['linear']['label_ids']['<nil>'] = 0
    lexicon['linear']['label_ids']['<unk>'] = len(lexicon['linear']['label_ids'])
    lexicon['linear']['label_ids']['<unk_str>'] = len(lexicon['linear']['label_ids'])
    lexicon['linear']['attr_ids'] = {'<nil>': 0}

    os.makedirs(args.store_path, exist_ok=True)
    for model in ['ast', 'linear']:
        for k in queues:
            writer = tf.python_io.TFRecordWriter(os.path.join(args.store_path, model + '_' + k + '.tfrecord'))
            for i in range(len(queues[k][model])):
                row = queues[k][model][i]
                if model == 'linear':
                    row['attr'] = ['<nil>'] * len(row['label'])
                    row['left_sibling'] = list(range(len(row['label'])))
                    row['right_sibling'] = list(range(2, len(row['label'])+1)) + [0]
                    for key in queues[k]['ast'][i]:
                        if key not in row:
                            row[key] = [0] * len(row['label'])

                row['mask'] = [1] * len(row['label'])
                row['attr_index'] = tokens_to_ids(row['attr'], lexicon[model]['attr_ids'], True, False)
                row['label_index'] = tokens_to_ids(row['label'], lexicon[model]['label_ids'], model == 'linear', False)
                del(row['attr'])
                del(row['label'])

                features = {}
                for key in row:
                    # prepend a 0 for the nil slot
                    features[key] = tf.train.Feature(int64_list=tf.train.Int64List(value=[0] + row[key]))
                example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(example.SerializeToString())
            writer.close()
        with open(os.path.join(args.store_path, model + '_lexicon.json'), 'w') as f:
            json.dump(lexicon[model], f)
    config = vars(args)
    config['features'] = list(queues['train']['ast'][0].keys())
    with open(os.path.join(args.store_path, 'config.json'), 'w') as f:
        json.dump(config, f)




"""
parser.add_argument('-p', '--preprocessor', help='run the c preprocessor on the code', action='store_true')
parser.add_argument('-i', '--sequence_ids', help='rename ids to ID0, ID1, etc.', action='store_true')
parser.add_argument('-I', '--rename_ids', help='rename all ids to ID (overrides -i)', action='store_true')
parser.add_argument('-S', '--truncate_strings', help='truncate all strings to ""', action='store_true')
parser.add_argument('-u', '--rename_user_functions', help='renames user-defined functions', action='store_true')
parser.add_argument('-f', '--rename_functions', help='renames user and library functions (assumes -f)', action='store_true')
parser.add_argument('-s', '--sequence_renamed_functions', help='renames user and library functions (assumes -f)', action='store_true')

parser.add_argument('-c', '--remove_comments', help='remove comments', action='store_true')
"""
