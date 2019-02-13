# TODO: make output format customizable (json vs tf.record)

import os
import sys
import glob
import argparse
import random
import json
import tensorflow as tf
from ..wrangler import wrangle, process_linear, process_ast, finish_row

from queue import Queue
from threading import Thread, Lock
from subprocess import Popen, PIPE

from typing import Any, List, Dict
from mypy_extensions import TypedDict

parser = argparse.ArgumentParser(description='Tokenizer')
parser.add_argument('filename', help='the name of the data files to process, such as caesar.c')
parser.add_argument('read_path', help='directory to read from for processing')
# TODO: make this optional
parser.add_argument('store_path', help='directory to store processed data')
parser.add_argument('-n', '--num_files', help='number of files to parse (default, all files)', type=int)
parser.add_argument('-t', '--num_threads', help='number of concurrent threads (default 16)', type=int, default=16)
parser.add_argument('--train_fraction', help='fraction of files for training', type=float, default=.8)
parser.add_argument('--valid_fraction', help='fraction of files for validating', type=float, default=.1)
parser.add_argument('--unit_tests', help='unit tests for these files', default=None)
parser.add_argument('--unk_cutoff', help='fraction of files that need to have a token or else it\'s considered unknown', type=float, default=.01)

def process_queue(queues, lexicon, lock, args):
    keys = list(queues.keys())
    random.shuffle(keys)
    # relies on GIL?
    for i in range(len(keys)):
        key = keys[i]
        while not queues[key]['queue'].empty():
            filename = queues[key]['queue'].get()
            ast_data, linear_data = wrangle(filename, include_dependencies=False, tests=None)
            rows = {
                'linear': process_linear(linear_data, key, lexicon, lock),
                'ast': process_ast(ast_data, key, lexicon, lock)
            }
            with lock:
                for j in rows:
                    queues[key][j].append(rows[j])
            queues[key]['queue'].task_done()

def main():

    args = parser.parse_args()
    if args.unit_tests is not None:
        with open(args.unit_tests, 'r') as tests:
            args.unit_tests = json.load(tests)

    files = glob.glob(os.path.join(os.getcwd(), args.read_path,'**',args.filename), recursive=True)
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
        'ast': {'label': {}, 'attr': {}},
        'linear': {'label': {}}
    }
    lock = Lock()
    for i in range(args.num_threads):
        t = Thread(target=process_queue, args=(queues, lexicon, lock, args))
        t.daemon = True
        t.start()

    for partition in queues:
        queues[partition]['queue'].join()

    cutoff = args.num_files * args.train_fraction * args.unk_cutoff

    ast_labels = set([label for label in lexicon['ast']['label'] if lexicon['ast']['label'][label] > cutoff])
    ast_attrs = set([attr for attr in lexicon['ast']['attr'] if lexicon['ast']['attr'][attr] > cutoff])
    linear_tokens = set([token for token in lexicon['linear']['label'] if lexicon['linear']['label'][token] > cutoff])

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
                                 queues['train']['ast'][0] if model == 'linear' else None)
                features = {}
                for j in row:
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
