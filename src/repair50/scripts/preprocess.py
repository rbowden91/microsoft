# TODO: make output format customizable (json vs tf.record)

import os
import sys
import glob
import argparse
import random
import json
import math
import collections
import functools
import tensorflow as tf #type:ignore
from ..wrangler.wrangle import wrangle, process_ast, finish_row

from queue import Queue
from threading import Thread, Lock
from subprocess import Popen, PIPE

from ..default_dict import data_dict
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

args = parser.parse_args()



def process_queue(queue, lexicon, transitions_groups, lock, tests):
    # relies on GIL?
    while not queue['queue'].empty():
        filename = queue['queue'].get()
        print(filename)
        data = wrangle(filename, tests=tests)

        # make sure the file passes all tests
        passed_all = functools.reduce(lambda y, test: test['passed'] and y, data.results, True)
        if not passed_all:
            print(filename + ' failed checks.')
            queue['queue'].task_done()
            continue

        rows = process_ast(data, lexicon, transitions_groups, lock)
        with lock:
            for test in rows:
                for root_node in rows[test]:
                    for transitions in rows[test][root_node]:
                        if not rows[test][root_node][transitions]: continue
                        root_transitions = data.prop_map[root_node][test][root_node][transitions]['transitions']
                        queue['tests'][test][root_transitions][transitions].append(rows[test][root_node][transitions])
        queue['queue'].task_done()

def generate_lexicon(lex, cutoff_tokens=['attr', 'transitions']):
    cutoff = args.num_files * args.unk_cutoff
    revlex = {}
    for k in lex:
        s = set([label for label in lex[k] if lex[k][label] > cutoff]) if k in cutoff_tokens else set(lex[k])
        if '<nil>' in s:
            s.remove('<nil>')
        if '<unk>' in s:
            s.remove('<unk>')
        lex[k] = dict(zip(s, range(1, len(s) + 1)))
        lex[k]['<nil>'] = 0
        lex[k]['<unk>'] = len(lex[k])
        revlex[k] = {lex[k][token]: token for token in lex[k]}
    return {'index_to_token': revlex, 'token_to_index': lex}

def main():

    if args.unit_tests is not None:
        with open(args.unit_tests, 'r') as tests_file:
            args.unit_tests = json.load(tests_file)

    files = glob.glob(os.path.join(os.getcwd(), args.read_path,'**',args.filename), recursive=True)
    random.shuffle(files)

    if args.num_files is None:
        args.num_files = len(files)
    else:
        args.num_files = min(args.num_files, len(files))

    queue = { 'queue': Queue(maxsize=0), 'tests': data_dict(list) }

    for i in range(args.num_files):
        queue['queue'].put(files[i])

    lex_ctr = lambda: collections.defaultdict(int)
    lexicon = data_dict(lambda: {'label': lex_ctr(), 'attr': lex_ctr(), 'transitions': lex_ctr()})

    transitions_groups = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int))))
    lock = Lock()
    for i in range(args.num_threads):
        t = Thread(target=process_queue, args=(queue, lexicon, transitions_groups, lock, args.unit_tests))
        t.daemon = True
        t.start()

    queue['queue'].join()

    # get the total lexicon
    root_lexicon = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(lex_ctr)))
    for test in lexicon:
        for root_transitions in lexicon[test]:
            for transitions in lexicon[test][root_transitions]:
                lex = lexicon[test][root_transitions][transitions]
                root_lex = root_lexicon[test][transitions]
                for k in lex:
                    for token in lex[k]:
                        root_lex[k][token] += lex[k][token]
        for transitions in [True, False]:
            root_lexicon[test][transitions] = generate_lexicon(root_lexicon[test][transitions])

    config = vars(args)
    config['root_lexicon'] = root_lexicon
    config['tests'] = data_dict(lambda: False)
    for test in lexicon:
        # need to handle '<nil>' root transition first, since all sub-root-transitions need to use
        # the '<nil>' (FileAST's) lexicon
        for root_transitions in lexicon[test]:
            if root_transitions == '<unk>': continue
            for transitions in [False, True]:
                root_lex = root_lexicon[test][transitions]['token_to_index']
                if transitions and test == 'null' or root_transitions not in root_lex['transitions']:
                    continue

                q = queue['tests'][test][root_transitions][transitions]
                datasets = {}
                num_rows = len(q)
                datasets['train'] = q[:math.floor(num_rows * args.train_fraction)]
                q = q[math.floor(num_rows * args.train_fraction):]
                datasets['valid'] = q[:math.floor(num_rows * args.valid_fraction)]
                datasets['test'] = q[math.floor(num_rows * args.valid_fraction):]

                no_data = False
                for k in datasets:
                    if len(datasets[k]) == 0:
                        no_data = True
                        break
                if no_data: continue

                lex = generate_lexicon(lexicon[test][root_transitions][transitions], root_lex)

                root_transitions_idx = str(root_lex['transitions'][root_transitions])
                config['tests'][test][root_transitions_idx][transitions] = True

                conf = {}
                conf['test'] = test
                conf['root_transitions'] = root_transitions
                conf['root_transitions_idx'] = root_transitions_idx
                conf['transitions'] = transitions
                if transitions:
                    conf['transitions_groups'] = transitions_groups[test]
                for k in lex['token_to_index']:
                    conf[k + '_size'] = len(lex['token_to_index'][k])
                conf['lexicon'] = lex
                conf['dataset_sizes'] = {}

                # TODO: something about the FILEAST overall transition???
                path = os.path.join(args.store_path, 'tests', test, root_transitions_idx, 'true' if transitions else 'false')
                os.makedirs(path, exist_ok=True)


                for dataset in datasets:
                    data = datasets[dataset]
                    writer = tf.python_io.TFRecordWriter(os.path.join(path, dataset + '_data.tfrecord'))
                    conf['dataset_sizes'][dataset] = len(data)

                    for i in range(len(data)):
                        row = finish_row(data[i], lex['token_to_index'], root_lex)
                        features = {
                            feature: tf.train.Feature(int64_list=tf.train.Int64List(value=row[feature])) for feature in row
                        }

                        example = tf.train.Example(features=tf.train.Features(feature=features))
                        writer.write(example.SerializeToString())
                        if dataset == 'train' and i == 0:
                            conf['features'] = list(row.keys())
                    writer.close()
                assert 'features' in conf
                with open(os.path.join(path, 'config.json'), 'w') as f:
                    json.dump(conf, f)
    with open(os.path.join(args.store_path, 'config.json'), 'w') as f:
        json.dump(config, f)
