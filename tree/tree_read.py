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

def tokens_to_ids(tokens, token_to_id, include_token):
    output = []
    for i in range(len(tokens)):
        output.append([])
        # XXX XXX XXX Check if this still works in training
        for j in range(len(tokens[i])):
            token = tokens[i][j]
            if token in token_to_id:
                id = token_to_id[token]
            #elif token.startswith('"'):
            #    id = token_to_id['<unk_str>']
            else:
                id = token_to_id['<unk_attr>']
            output[i].append((id, token) if include_token else id)
    return output

def process_queue(queues, lexicon, lock):
    keys = list(queues.keys())
    random.shuffle(keys)
    for i in range(len(keys)):
        key = keys[i]
        while not queues[key]['queue'].empty():
            file = queues[key]['queue'].get()
            print(file)
            with open(os.path.join(file, 'tree_stripped.json')) as f:
                data = json.load(f)
            new_row = dict()

            file_lexicon = set()
            for i in range(len(data)):
                # transform label to label_index
                token = data[i]['label']
                if key == "train":
                    with lock:
                        if token not in lexicon['ast_labels']:
                            lexicon['ast_labels'][token] = len(lexicon['ast_labels'])
                token = token if token in lexicon['ast_labels'] else '<unk_label>'
                data[i]['label_index'] = lexicon['ast_labels'][token]
                del(data[i]['label'])

                # transform attrs to a (XXX single) attr index
                for (name, val) in data[i]['attrs']:
                    if name in ['value', 'op', 'name']:
                        if key == "train":
                            with lock:
                                if val not in lexicon['label_attrs']:
                                    lexicon['label_attrs'][val] = 0
                                # only add once per file
                                if val not in file_lexicon:
                                    lexicon['label_attrs'][val] += 1
                                    file_lexicon.add(val)
                        data[i]['attr'] = val
                        # XXX strongly assumes only one, for now!!!
                        break
                else:
                    data[i]['attr'] = '<no_attr>'
                del(data[i]['attrs'])

                for k in data[i]:
                    if k not in new_row:
                        # include 0 for the <nil> dependencies
                        new_row[k] = [0]
                    new_row[k].append(int(data[i][k]) if isinstance(data[i][k], bool) else data[i][k])


            with lock:
                for k in new_row.keys():
                    if k not in queues[key]:
                        queues[key][k] = []
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
        queues[i] = {
            'queue': Queue(maxsize=0),
        }

    for i in range(config['num_files']):
        if i < config['train_fraction'] * config['num_files']:
            queues['train']['queue'].put(files[i])
        elif i < (config['train_fraction'] + config['valid_fraction']) * config['num_files']:
            queues['valid']['queue'].put(files[i])
        else:
            queues['test']['queue'].put(files[i])

    lexicon = { 'ast_labels': {'<nil>': 0, '<unk_label>': 1}, 'label_attrs': {'<no_attr>': sys.maxsize }}
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

    cutoff = config['num_files'] * config['train_fraction'] * config['unk_cutoff']
    tokens = set([token for token in lexicon['label_attrs'] if lexicon['label_attrs'][token] > cutoff])

    lexicon['label_attrs'] = dict(zip(tokens, range(len(tokens))))
    lexicon['label_attrs']['<unk_attr>'] = len(lexicon['label_attrs'])

    for i in queues:
        queues[i]['attr_index'] = tokens_to_ids(queues[i]['attr'], lexicon['label_attrs'], False)
        del(queues[i]['attr'])

        # don't care
        del(queues[i]['node_number'])

    # XXX make this a flag
    if not os.path.isdir(path):
        os.makedirs(path)
    with open(os.path.join(path, 'tree_train.json'), 'w') as f:
        json.dump(queues['train'], f)
    with open(os.path.join(path, 'tree_valid.json'), 'w') as f:
        json.dump(queues['valid'], f)
    with open(os.path.join(path, 'tree_test.json'), 'w') as f:
        json.dump(queues['test'], f)
    with open(os.path.join(path, 'tree_tokens.json'), 'w') as f:
        json.dump(lexicon, f)
    with open(os.path.join(path, 'tree_config.json'), 'w') as f:
        json.dump(config, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tokenizer')
    parser.add_argument('path', help='data directory to store things')

    args = parser.parse_args()
    main(args.path)
