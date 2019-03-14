# TODO: make output format customizable (json vs tf.record)

import os
import sys
import glob
import argparse
import random
import json
import functools
import tensorflow as tf
from ..wrangler.wrangle import wrangle, process_ast, finish_row

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

def process_queue(queues, lexicon, transitions_groups, lock, tests):
    keys = list(queues.keys())
    random.shuffle(keys)
    # relies on GIL?
    for key in keys:
        while not queues[key]['queue'].empty():
            filename = queues[key]['queue'].get()
            print(filename)
            data = wrangle(filename, tests=tests)

            # make sure the file passes all tests
            passed_all = functools.reduce(lambda y, test: test['passed'] and y, data.results, True)
            if not passed_all:
                print(filename + ' failed checks.')
                queues[key]['queue'].task_done()
                continue

            rows = process_ast(data, key, lexicon, transitions_groups, lock)
            with lock:
                for transitions in rows:
                    for test in rows[transitions]:
                        if test not in queues[key]['tests'][transitions]:
                            queues[key]['tests'][transitions][test] = []
                        queues[key]['tests'][transitions][test].append(rows[transitions][test])
            queues[key]['queue'].task_done()

def main():

    args = parser.parse_args()

    tests = ['null']
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

    td = TypedDict('td', {'queue': Queue, 'tests': Dict[Any,Any]})
    queues : Dict[str, td] = {}
    for partition in ['train', 'test', 'valid']:
        queues[partition] = { 'queue': Queue(maxsize=0), 'tests': {True: {}, False: {}} }

    for i in range(args.num_files):
        if i < args.train_fraction * args.num_files:
            queues['train']['queue'].put(files[i])
        elif i < (args.train_fraction + args.valid_fraction) * args.num_files:
            queues['valid']['queue'].put(files[i])
        else:
            queues['test']['queue'].put(files[i])

    lexicon : Any = {True: {}, False: {}}
    for transitions in lexicon:
        for test in tests:
            lexicon[transitions][test] = {}
            for k in ['label', 'attr', 'transitions']:
                lexicon[transitions][test][k] = {}

    transitions_groups = {}
    lock = Lock()
    for i in range(args.num_threads):
        t = Thread(target=process_queue, args=(queues, lexicon, transitions_groups, lock, args.unit_tests))
        t.daemon = True
        t.start()

    for partition in queues:
        queues[partition]['queue'].join()

    args.num_files = len(queues['train']['tests'][True]['null'])
    cutoff = args.num_files * args.unk_cutoff

    config = vars(args)
    config['tests'] = {}
    for transitions in lexicon:
        config['tests'][transitions] = {}
        for test in lexicon[transitions]:
            if transitions and test == 'null': continue
            config['tests'][transitions][test] = conf = {}
            conf['test'] = test
            conf['transitions'] = transitions
            if transitions:
                conf['transitions_groups'] = transitions_groups[test]
            lex = lexicon[transitions][test]
            revlex = {}
            for k in lex:
                s = set([label for label in lex[k] if lex[k][label] > cutoff]) if k == 'attr' else set(lex[k])
                if '<nil>' in s:
                    s.remove('<nil>')
                if '<unk>' in s:
                    s.remove('<unk>')
                lex[k] = dict(zip(s, range(1, len(s) + 1)))
                lex[k]['<nil>'] = 0
                lex[k]['<unk>'] = len(lex[k])
                conf[k + '_size'] = len(lex[k])
                revlex[k] = {lex[k][token]: token for token in lex[k]}

            path = os.path.join(args.store_path, 'tests', 'true' if transitions else 'false', test)
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, 'lexicon.json'), 'w') as f:
                json.dump({'index_to_token': revlex, 'token_to_index': lex}, f)
            for dataset in queues.keys():
                writer = tf.python_io.TFRecordWriter(os.path.join(path, dataset + '_data.tfrecord'))

                for i in range(len(queues[dataset]['tests'][transitions][test])): # type: ignore
                    row = finish_row(queues[dataset]['tests'][transitions][test][i], lex)
                    features = {
                        feature: tf.train.Feature(int64_list=tf.train.Int64List(value=row[feature])) for feature in row
                    }

                    example = tf.train.Example(features=tf.train.Features(feature=features))
                    writer.write(example.SerializeToString())
                writer.close()
            conf['features'] = list(queues['train']['tests'][transitions][test][0].keys())
    with open(os.path.join(args.store_path, 'config.json'), 'w') as f:
        json.dump(config, f)
