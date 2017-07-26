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
    'num_files': 100#None
}

def process_queue(queues, lexicon, lock):
    keys = list(queues.keys())
    random.shuffle(keys)
    for i in range(len(keys)):
        key = keys[i]
        while not queues[key]['queue'].empty():
            file = queues[key]['queue'].get()
            print(file)
            with open(file + '/tree_stripped.json') as f:
                data = json.load(f)
            new_row = dict()
            for k in queues[key]:
                # include 0 for the <nil> parent/siblings
                new_row[k] = [0]
            del(new_row['queue'])

            for i in range(len(data)):
                token = data[i]['name']
                with lock:
                    if token not in lexicon:
                        lexicon[token] = len(lexicon)
                data[i]['token'] = lexicon[token]
                for k in new_row:
                    # XXX get rid of true/false from parse.sh, then we don't need "int"
                    new_row[k].append(int(data[i][k]))


            with lock:
                for k in new_row.keys():
                    queues[key][k].append(new_row[k])

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
        # XXX must match data
        queues[i] = {
            'queue': Queue(maxsize=0),
            'token': [],
            'sibling': [],
            'parent': [],
            'leaf_node': [],
            'last_sibling': []
        }

    for i in range(config['num_files']):
        if i < config['train_fraction'] * config['num_files']:
            queues['train']['queue'].put(files[i])
        elif i < (config['train_fraction'] + config['valid_fraction']) * config['num_files']:
            queues['valid']['queue'].put(files[i])
        else:
            queues['test']['queue'].put(files[i])

    lexicon = {'<nil>': 0}
    lock = Lock()

    threads = []
    for i in range(num_threads):
        t = Thread(target=process_queue, args=(queues, lexicon, lock))
        t.daemon = True
        t.start()
        threads.append(t)

    for i in queues:
        queues[i]['queue'].join()

    for t in threads:
        t.join()

    for i in queues:
        del(queues[i]['queue'])

    #cutoff = config['num_files'] * config['train_fraction'] * config['unk_cutoff']
    #tokens = set([token for lst in queues['train']['tokens'] for token in lst if lexicon[token] > cutoff])

    #token_to_id = dict(zip(tokens, range(len(tokens))))
    #token_to_id['<unk>'] = len(token_to_id)
    #token_to_id['<unk_str>'] = len(token_to_id)

    # XXX make this a flag
    with open(path + '/tree_train.json', 'w') as f:
        json.dump(queues['train'], f)
    with open(path + '/tree_valid.json', 'w') as f:
        json.dump(queues['valid'], f)
    with open(path + '/tree_test.json', 'w') as f:
        json.dump(queues['test'], f)
    with open(path + '/tree_tokens.json', 'w') as f:
        json.dump(lexicon, f)
    with open(path + '/tree_config.json', 'w') as f:
        json.dump(config, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tokenizer')
    parser.add_argument('path', help='data directory to store things')

    args = parser.parse_args()
    main(args.path)
