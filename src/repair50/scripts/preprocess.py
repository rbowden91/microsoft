# TODO: make output format customizable (json vs tf.record)

import os
import sys
import glob
import argparse
import random
import json
import functools
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

def process_queue(queues, lexicon, lock, tests):
    keys = list(queues.keys())
    random.shuffle(keys)
    # relies on GIL?
    for i in range(len(keys)):
        key = keys[i]
        while not queues[key]['queue'].empty():
            filename = queues[key]['queue'].get()
            print(filename)
            ast_data, linear_data = wrangle(filename, include_dependencies=False, tests=tests)


            # make sure the file passes all tests
            passed_all = functools.reduce(lambda y, test: test['passed'] and y, ast_data.results, True)
            if not passed_all:
                print(filename + ' failed checks.')
                queues[key]['queue'].task_done()
                continue

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

    tests = [None]
    if args.unit_tests is not None:
        with open(args.unit_tests, 'r') as tests_file:
            args.unit_tests = json.load(tests_file)
        tests.extend([x['name'] for x in args.unit_tests])
    args.test_names = tests

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

    lexicon : Any = {}
    for model in ['ast', 'linear']:
        lexicon[model] = {}
        for test in tests:
            lexicon[model][test] = {}
            for k in ['label', 'attr', 'transitions']:
                lexicon[model][test][k] = {}

    lock = Lock()
    for i in range(args.num_threads):
        t = Thread(target=process_queue, args=(queues, lexicon, lock, args.unit_tests))
        t.daemon = True
        t.start()

    for partition in queues:
        queues[partition]['queue'].join()

    args.num_files = len(queues['train']['ast'])
    cutoff = args.num_files * args.unk_cutoff

    for model in lexicon:
        for test in lexicon[model]:
            trans_percentages = {k : lexicon[model][test]['transitions'][k] / args.num_files for k in lexicon[model][test]['transitions']}
            for k in lexicon[model][test]:
                s = set([label for label in lexicon[model][test][k] if lexicon[model][test][k][label] > cutoff])
                if '<nil>' in s:
                    s.remove('<nil>')
                if '<unk>' in s:
                    s.remove('<unk>')
                lexicon[model][test][k] = dict(zip(s, range(1, len(s) + 1)))
                lexicon[model][test][k]['<nil>'] = 0
                lexicon[model][test][k]['<unk>'] = len(lexicon[model][test][k])
            lexicon[model][test]['transition_percentages'] = trans_percentages

    os.makedirs(args.store_path, exist_ok=True)
    for model in ['ast', 'linear']:
        if model == 'linear':
            continue
        for k in queues.keys():
            writer = tf.python_io.TFRecordWriter(os.path.join(args.store_path, model + '_' + k + '.tfrecord'))
            for i in range(len(queues[k][model])): # type: ignore
                row = finish_row(queues[k][model][i], lexicon[model], # type: ignore
                                 queues['train']['ast'][0][None].keys() if model == 'linear' else None)
                features = {
                    j: tf.train.Feature(int64_list=tf.train.Int64List(value=row[None][j])) for j in row[None]
                }
                example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(example.SerializeToString())
            writer.close()
        with open(os.path.join(args.store_path, model + '_lexicon.json'), 'w') as f:
            json.dump(lexicon[model], f)
    config = vars(args)
    config['features'] = list(queues['train']['ast'][0][None].keys())
    with open(os.path.join(args.store_path, 'config.json'), 'w') as f:
        json.dump(config, f)
