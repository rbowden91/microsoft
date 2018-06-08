from __future__ import print_function
import sys
import nltk
import collections
import os
import random
import argparse
from nltk.tokenize import RegexpTokenizer
from queue import Queue
from threading import Thread, Lock
import subprocess
import pycparser
import json

sys.path.insert(0,os.path.pardir)
from preprocess import preprocess_c,typedefs


num_threads = 16

config = {
    # remainder are for test
    'train_fraction': .8,
    'valid_fraction': .1,
    # a symbol that appears less frequently than this is just <unk>
    # could also be changed to number of files it appears in
    'unk_cutoff': 0.01,
}


# currently requires the code to parse
def read(filename, token_to_id=None, include_token=False, rename_ids=False, truncate_strings=False):
    try:
        body, id_map, reverse_id_map = preprocess_c(filename)
    except pycparser.plyparser.ParseError:
        return False

    # XXX need to reinsert preprocessor directives, potentially filling back in what used to be #defines?
    # Can follow the example from earlier and make the #define "AUTOSKETCH name def"

    lexer = pycparser.c_lexer.CLexer(
            error_func=lambda x, y, z: True,
            type_lookup_func=lambda x: True,
            on_lbrace_func=lambda: True,
            on_rbrace_func=lambda: True
    )

    lexer.build()
    lexer.input(body)

    tokens = []
    while True:
        token = lexer.token()
        if token is None:
            break
        #print(token.__dict__)
        #if token.value not in typedefs and \
        #   token.type == 'TYPEID' and \
        #   token.value in id_map[token.value]:
        #    value = id_map[token.value]
        #elif token.type == 'STRING_LITERAL':
        #    value = str_map[token.value]
        #elif token.value.startswith("ID"):
        #    value = 'ID'
        #elif token.value.startswith("FUNCTION_ID"):
        #    value = 'FUNCTION_ID'
        #elif token.value.startswith("ARG_ID"):
        #    value = 'ARG_ID'
        #else:
        value = token.value
        tokens.append(value)

    tokens.insert(0,"<sof>")
    tokens.append("<eof>")
    if token_to_id is not None:
        return tokens_to_ids(tokens, token_to_id, include_token)
    else:
        return tokens

def tokens_to_ids(tokens, max_tokens, token_to_id, include_token):
    output = []
    lengths = []
    for i in range(len(tokens)):
        output.append([])
        for j in range(len(tokens[i])):
            token = tokens[i][j]
            if token in token_to_id:
                id = token_to_id[token]
            elif token.startswith('"'):
                id = token_to_id['<unk_str>']
            else:
                id = token_to_id['<unk>']
            output[i].append((id, token) if include_token else id)
        output[i] += [token_to_id['<pad>']] * (max_tokens - len(output[i]))
        lengths.append(len(output))

    return { 'tokens': output, 'lengths': lengths }

def process_queue(queues, lexicon, lex_lock, args):
    keys = list(queues.keys())
    random.shuffle(keys)
    for i in range(len(keys)):
        key = keys[i]
        while not queues[key]['queue'].empty():
            directory = queues[key]['queue'].get()
            print(directory)
            tokens = read(os.path.join(directory, args.filename))
            if (tokens == False):
                print('Invalid file: ' + directory)
                #os.rename(file, 'vigenere/invalid/' + os.path.basename(file))
                continue
            # XXX relies on GIL
            queues[key]['max_tokens'] = max(queues[key]['max_tokens'], len(tokens))
            queues[key]['tokens'].append(tokens)

            # only add to our lexicon things in the training data
            if key == 'train':
                with lex_lock:
                    for token in set(tokens):
                        if token not in lexicon:
                            lexicon[token] = 0
                        lexicon[token] += 1
            queues[key]['queue'].task_done()
        return

def main(args):
    files = [os.path.join(args.read_path, file) for file in os.listdir(args.read_path) if not file.startswith('.')]

    if args.num_files is None:
        args.num_files = len(files)
    else:
        args.num_files = min(args.num_files, len(files))

    queues = {}
    for i in ['train', 'test', 'valid']:
        queues[i] = {
            'queue': Queue(maxsize=0),
            'tokens': [],
            'max_tokens': 0
        }

    for i in range(args.num_files):
        if i < config['train_fraction'] * args.num_files:
            queues['train']['queue'].put(files[i])
        elif i < (config['train_fraction'] + config['valid_fraction']) * args.num_files:
            queues['valid']['queue'].put(files[i])
        else:
            queues['test']['queue'].put(files[i])

    lexicon = {}
    lex_lock = Lock()

    for i in range(num_threads):
        t = Thread(target=process_queue, args=(queues, lexicon, lex_lock, args))
        t.daemon = True
        t.start()

    for i in queues:
        queues[i]['queue'].join()

    cutoff = args.num_files * config['train_fraction'] * config['unk_cutoff']

    # XXX why aren't I just pulling from lexicon??
    tokens = set([token for lst in queues['train']['tokens'] for token in lst if lexicon[token] > cutoff])

    token_to_id = dict(zip(tokens, range(len(tokens))))
    token_to_id['<unk>'] = len(token_to_id)
    token_to_id['<unk_str>'] = len(token_to_id)
    token_to_id['<pad>'] = len(token_to_id)

    os.makedirs(args.store_path, exist_ok=True)
    for dataset in ['train', 'valid', 'test']:
        with open(os.path.join(args.store_path, dataset + '.json'), 'w') as f:
            json.dump(tokens_to_ids(queues[dataset]['tokens'], queues[dataset]['max_tokens'], token_to_id, False), f)
    with open(os.path.join(args.store_path, 'tokens.json'), 'w') as f:
        json.dump(token_to_id, f)

    config['args'] = vars(args)
    with open(os.path.join(args.store_path, 'config.json'), 'w') as f:
        json.dump(config, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tokenizer')
    parser.add_argument('filename', help='the name of the data files to process, such as caesar.c')
    parser.add_argument('read_path', help='directory to read from for processing')
    parser.add_argument('store_path', help='directory to store processed data')
    # XXX rename typedefs?
    # XXX differentiate function ids from variable ids (ID vs FID)
    parser.add_argument('-n', '--num_files', help='number of files to parse (default, all files)', type=int)
    parser.add_argument('-p', '--preprocessor', help='run the c preprocessor on the code', action='store_true')
    parser.add_argument('-i', '--sequence_ids', help='rename ids to ID0, ID1, etc.', action='store_true')
    parser.add_argument('-I', '--rename_ids', help='rename all ids to ID (overrides -i)', action='store_true')
    parser.add_argument('-S', '--truncate_strings', help='truncate all strings to ""', action='store_true')
    parser.add_argument('-H', '--fake_headers', help='use fake C headers instead of the real ones for preprocessing', action='store_true')
    parser.add_argument('-u', '--rename_user_functions', help='renames user-defined functions', action='store_true')
    parser.add_argument('-f', '--rename_functions', help='renames user and library functions (assumes -f)', action='store_true')
    parser.add_argument('-s', '--sequence_renamed_functions', help='renames user and library functions (assumes -f)', action='store_true')

    parser.add_argument('-c', '--remove_comments', help='remove comments', action='store_true')

    args = parser.parse_args()
    main(args)
