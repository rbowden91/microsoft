# TODO: make output format customizable (json vs tf.record)

import os
import sys
import glob
import argparse
import random
import json
import tensorflow as tf

from queue import Queue
from threading import Thread, Lock
from subprocess import Popen, PIPE

from typing import Any, List, Dict
from mypy_extensions import TypedDict

from pycparser import c_parser, c_generator, c_lexer # type: ignore

from .normalize import RemoveDecls, RemoveTypedefs, IDRenamer
from .linearize_ast import LinearizeAST

def process_linear(tokens, key=None, lexicon=None, lock=None):
    local_lexicon = set()
    if key == 'train':
        for token in set(tokens):
            if token not in local_lexicon:
                with lock:
                    if token not in lexicon['linear_tokens']:
                        lexicon['linear_tokens'][token] = 0
                    lexicon['linear_tokens'][token] += 1
                    local_lexicon.add(token)
    return { 'forward_label': tokens }

def process_ast(linearizer, key=None, lexicon=None, lock=None):
    local_lexicon = {
        'label': set(),
        'attr': set()
    }
    # skip the nil slot
    linearizer.nodes['reverse'].pop(0)
    nodes = linearizer.nodes['forward']
    nodes.pop(0)
    for i in range(len(nodes)):
        data = nodes[i]
        data['attr'] = data['attr'] if data['attr'] is not None else '<no_attr>'
        data['label'] = data['label'] if data['label'] is not None else '<nil>'
        data['parent_attr'] = data['parent_attr'] if data['parent_attr'] is not None else '<no_attr>'
        data['parent_label'] = data['parent_label'] if data['parent_label'] is not None else '<nil>'

        if lexicon is not None:
            for j in ['label', 'attr']:
                if key == "train" and data[j] not in local_lexicon[j]:
                    with lock:
                        if data[j] not in local_lexicon[j]:
                            lexicon['ast_' + j + 's'][data[j]] = 0
                        lexicon['ast_' + j + 's'][data[j]] += 1
                    local_lexicon[j].add(data[j])

    new_rows = {}
    for k in ['reverse', 'forward']:
        nodes = linearizer.nodes[k]
        for i in nodes[0]:
            if i in ['forward', 'reverse']:
                if i != k: continue
                for j in nodes[0][i]:
                    new_rows[k + '_' + j] = [int(n[i][j]) if isinstance(n[i][j], bool) else n[i][j] for n in nodes]
            else:
                new_rows[k + '_' + i] = [int(n[i]) if isinstance(n[i], bool) else n[i] for n in nodes]
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
            ret = preprocess_c(filename, args, include_dependencies=False)
            if ret is None:
                print('uh oh!** ', filename)
                queues[key]['queue'].task_done()
                continue
            _, linearizer, linear_tokens = ret
            rows = {}
            rows['linear'] = process_linear(linear_tokens, key, lexicon, lock)
            rows['ast'] = process_ast(linearizer, key, lexicon, lock)
            with lock:
                for j in rows:
                    queues[key][j].append(rows[j])
            queues[key]['queue'].task_done()


def main():
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

    td = TypedDict('td', {'queue': Queue, 'ast': List[Any], 'linear': List[Any]})
    queues : Dict[str, td] = {}
    for partition in ['train', 'test', 'valid']:
        queues[partition] = { 'queue': Queue(maxsize=0), 'ast': [], 'linear': [] }

    for i in range(args.num_files):
        if i < args.train_fraction * args.num_files:
            queues['train']['queue'].put(files[i])
        elif i < (args.train_fraction + args.valid_fraction) * args.num_files:
            queues['valid']['queue'].put(files[i])
        else:
            queues['test']['queue'].put(files[i])

    lexicon : Any = {
        'ast_labels': {},
        'ast_attrs': {},
        'linear_tokens': {}
    }
    lock = Lock()
    for i in range(args.num_threads):
        t = Thread(target=process_queue, args=(queues, lexicon, lock, args))
        t.daemon = True
        t.start()

    for partition in queues:
        queues[partition]['queue'].join()

    cutoff = args.num_files * args.train_fraction * args.unk_cutoff

    ast_labels = set([label for label in lexicon['ast_labels'] if lexicon['ast_labels'][label] > cutoff])
    ast_attrs = set([attr for attr in lexicon['ast_attrs'] if lexicon['ast_attrs'][attr] > cutoff])
    linear_tokens = set([token for token in lexicon['linear_tokens'] if lexicon['linear_tokens'][token] > cutoff])

    lexicon = {'ast': {}, 'linear': {}}
    lexicon['ast']['label'] = dict(zip(ast_labels, range(1, len(ast_labels) + 1)))
    lexicon['ast']['label']['<nil>'] = 0
    lexicon['ast']['label']['<unk>'] = len(lexicon['ast']['label'])
    lexicon['ast']['attr'] = dict(zip(ast_attrs, range(1, len(ast_attrs) + 1)))
    lexicon['ast']['attr']['<nil>'] = 0
    lexicon['ast']['attr']['<unk>'] = len(lexicon['ast']['attr'])
    lexicon['ast']['attr']['<unk_str>'] = len(lexicon['ast']['attr'])
    lexicon['linear']['label'] = dict(zip(linear_tokens, range(1, len(linear_tokens) + 1)))
    lexicon['linear']['label']['<nil>'] = 0
    lexicon['linear']['label']['<unk>'] = len(lexicon['linear']['label'])
    lexicon['linear']['label']['<unk_str>'] = len(lexicon['linear']['label'])
    lexicon['linear']['attr'] = {'<nil>': 0}

    os.makedirs(args.store_path, exist_ok=True)
    for model in ['ast', 'linear']:
        for k in queues.keys():
            writer = tf.python_io.TFRecordWriter(os.path.join(args.store_path, model + '_' + k + '.tfrecord'))
            for i in range(len(queues[k][model])): # type: ignore
                row = finish_row(queues[k][model][i], lexicon[model], # type: ignore
                                 queues['train']['ast'][0].keys() if model == 'linear' else None)

                features = {}
                for j in row:
                    # add in 0 for the nil slot
                    features[j] = tf.train.Feature(int64_list=tf.train.Int64List(value=row[j]))
                example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(example.SerializeToString())
            writer.close()
        with open(os.path.join(args.store_path, model + '_lexicon.json'), 'w') as f:
            json.dump(lexicon[model], f)
    config = vars(args)
    config['features'] = list(queues['train']['ast'][0].keys())
    with open(os.path.join(args.store_path, 'config.json'), 'w') as f:
        json.dump(config, f)
