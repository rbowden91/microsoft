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

# shouldn't need to import these, but they're doing something to force
# f.read() to not like unicode or something? I dunno
from preprocess_ids import preprocess_c
import pycparser

import json

num_threads = 16

config = {
    'tokenizer': 'pycparser_regex rename_ids',
    'train_fraction': .8,
    'valid_fraction': .1,
    # a symbol that appears less frequently than this is just <unk>
    # could also be changed to number of files it appears in
    'unk_cutoff': 0.01,
    # if None, will be auto-filled with total number of files
    'num_files': 10#None
}

# requires getting preprocessing out of the way...
# are there potential whitespace issues with tokens?
#tknzr = RegexpTokenizer(r'"(?:[^"\\]|\\.)*"|[\d\w]+|\S')
tknzr = RegexpTokenizer(r'[\d\w]+|\S')

def read(filename, token_to_id=None, include_token=False, rename_ids=False, truncate_strings=False):

    try:
        # XXX if both are False, this reduces to just preprocessing, yeah?
        body = preprocess_c(filename, rename_ids=rename_ids, truncate_strings=truncate_strings)
    except pycparser.plyparser.ParseError:
        return False
    #with open(filename) as f:
    #    # should check why some fail with unicode stuff
    #    try:
    #        body = f.read()
    #    except Exception:
    #        return []

    # XXX THIS IS EFFING NONSENSE
    # XXX check error
    #body = subprocess.run('gcc -fpreprocessed -dD -E ' + filename, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #try:
    #    body = body.stdout.decode('ascii')
    #except Exception:
    #    print(body)
    #    return []
    ##print(body.stdout)
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
        tokens.append(token.value)
        # other stuff might be useful here, like position
        # print(token.__dict__)


    # utf8 nonsense
    #try:
    #    tokens = tknzr.tokenize(body)
    #except Exception:
    #    return []


    tokens.append("<eof>")
    tokens.insert(0, "<sof>")
    if token_to_id is not None:
        return tokens_to_ids(tokens, token_to_id, include_token)
    else:
        return tokens

def tokens_to_ids(tokens, token_to_id, include_token):
    #print(tokens)
    output = []
    for i in range(len(tokens)):
        output.append([])
        # XXX XXX XXX Check if this still works in training
        for j in range(len(tokens[i])):
            token = tokens[i][j]
            if token in token_to_id:
                id = token_to_id[token]
            elif token.startswith('"'):
                id = token_to_id['<unk_str>']
            else:
                id = token_to_id['<unk>']
            output[i].append((id, token) if include_token else id)
    print(output)
    return output


def process_queue(queues, lexicon, lex_lock):
    keys = list(queues.keys())
    random.shuffle(keys)
    for i in range(len(keys)):
        key = keys[i]
        while not queues[key]['queue'].empty():
            file = queues[key]['queue'].get()
            print(file)
            tokens = read(file + '/vigenere.c')
            if (tokens == False):
                os.rename(file, 'vigenere/invalid/' + os.path.basename(file))
                continue
            # XXX relies on GIL
            queues[key]['tokens'].append(tokens)

            if key == 'train':
                with lex_lock:
                    for token in set(tokens):
                        if token not in lexicon:
                            lexicon[token] = 0
                        lexicon[token] += 1
            queues[key]['queue'].task_done()

def main(path):
    data_dir = '../vigenere/correct15/'

    # the rest is test data

    def eprint(*args, **kwargs):
        print(*args, file=sys.stderr, **kwargs)

    files = [data_dir + filename for filename in os.listdir(data_dir) if not filename.startswith('.')]

    if config['num_files'] is None:
        config['num_files'] = len(files)
    else:
        config['num_files'] = min(config['num_files'], len(files))

    queues = {}
    for i in ['train', 'test', 'valid']:
        queues[i] = {
            'queue': Queue(maxsize=0),
            'tokens': []
        }

    for i in range(config['num_files']):
        if i < config['train_fraction'] * config['num_files']:
            queues['train']['queue'].put(files[i])
        elif i < (config['train_fraction'] + config['valid_fraction']) * config['num_files']:
            queues['valid']['queue'].put(files[i])
        else:
            queues['test']['queue'].put(files[i])

    lexicon = {}
    lex_lock = Lock()

    for i in range(num_threads):
        t = Thread(target=process_queue, args=(queues, lexicon, lex_lock))
        t.daemon = True
        t.start()

    for i in queues:
        queues[i]['queue'].join()

    cutoff = config['num_files'] * config['train_fraction'] * config['unk_cutoff']
    print(queues)

    # XXX why aren't I just pulling from lexicon??
    tokens = set([token for lst in queues['train']['tokens'] for token in lst if lexicon[token] > cutoff])

    token_to_id = dict(zip(tokens, range(len(tokens))))
    token_to_id['<unk>'] = len(token_to_id)
    token_to_id['<unk_str>'] = len(token_to_id)

    # XXX make this a flag
    with open(path + '/train.json', 'w') as f:
        json.dump(tokens_to_ids(queues['train']['tokens'], token_to_id, False), f)
    with open(path + '/valid.json', 'w') as f:
        json.dump(tokens_to_ids(queues['valid']['tokens'], token_to_id, False), f)
    with open(path + '/test.json', 'w') as f:
        json.dump(tokens_to_ids(queues['test']['tokens'], token_to_id, False), f)
    with open(path + '/tokens.json', 'w') as f:
        json.dump(token_to_id, f)
    with open(path + '/config.json', 'w') as f:
        json.dump(config, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tokenizer')
    parser.add_argument('path', help='data directory to store things')

    args = parser.parse_args()
    main(args.path)
