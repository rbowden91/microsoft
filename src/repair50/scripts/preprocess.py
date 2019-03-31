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

from multiprocessing import Process, Queue
import queue

from ..default_dict import data_dict
from typing import Any, List, Dict
from mypy_extensions import TypedDict

parser = argparse.ArgumentParser(description='Tokenizer')
parser.add_argument('filename', help='the name of the data files to process, such as caesar.c')
parser.add_argument('read_path', help='directory to read from for processing')
# TODO: make this optional
parser.add_argument('store_path', help='directory to store processed data')
parser.add_argument('-n', '--num_files', help='number of files to parse (default, all files)', type=int)
parser.add_argument('--num_processes', help='number of concurrent processes (default 16)', type=int, default=16)
parser.add_argument('--train_fraction', help='fraction of files for training', type=float, default=.8)
parser.add_argument('--valid_fraction', help='fraction of files for validating', type=float, default=.1)
parser.add_argument('--unit_tests', help='unit tests for these files', default=None)
parser.add_argument('--unk_cutoff', help='fraction of files that need to have a token or else it\'s considered unknown', type=float, default=.01)

args = parser.parse_args()
lex_ctr = lambda: collections.defaultdict(int)



def process_queue(filequeue, resultsqueue):
    all_rows = data_dict(lambda: collections.defaultdict(list))
    lexicon = data_dict(lambda: {'label': lex_ctr(), 'attr': lex_ctr(), 'transitions': lex_ctr()})
    transitions_groups = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int))))

    while not filequeue.empty():
        filenumber, filename = filequeue.get()
        print("File {} of {}: {}".format(filenumber, args.num_files, filename))
        data = wrangle(filename, tests=args.unit_tests)

        # make sure the file passes all tests
        passed_all = functools.reduce(lambda y, test_group: functools.reduce(lambda y, test: test_group[test]['passed'] and y, test_group, True) and y, data.results, True)
        if not passed_all:
            print(filename + ' failed checks.')
            continue

        rows = process_ast(data, lexicon, transitions_groups)
        for test in rows:
            for root_node in rows[test]:
                for transitions in rows[test][root_node]:
                    if not rows[test][root_node][transitions]: continue
                    root_transitions = data.prop_map[root_node]['props'][test][root_node][transitions]['transitions']
                    for k in rows[test][root_node][transitions]:
                        all_rows[test][root_transitions][transitions][k].append(rows[test][root_node][transitions][k])
    resultsqueue.put(json.loads(json.dumps((all_rows, lexicon, transitions_groups))))

def process_queue2(q):
    while True:
        try:
            # TODO: technically, 2 seconds could not be enough. but realistically...
            datasets, lexicon, root_lex, conf = q.get(True, 2)
        except queue.Empty:
            return

        lex = generate_lexicon(lexicon, root_lex)
        for k in lex['token_to_index']:
            conf[k + '_size'] = len(lex['token_to_index'][k])
        conf['lexicon'] = lex

        path = os.path.join(args.store_path, 'tests', conf['test'], conf['root_transitions_idx'], 'true' if conf['transitions'] else 'false')
        os.makedirs(path, exist_ok=True)

        sizes = conf['dataset_sizes']
        for dataset in datasets:
            writer = tf.python_io.TFRecordWriter(os.path.join(path, dataset + '_data.tfrecord'))
            data = datasets[dataset]
            for i in range(sizes[dataset]):
                print("Writing row {} of {} for dataset {}".format(i+1, sizes[dataset], dataset))
                row = {}
                for k in data:
                    row[k] = data[k][i]

                finished_row = finish_row(row, lex['token_to_index'], root_lex)
                features = {
                    feature: tf.train.Feature(int64_list=tf.train.Int64List(value=finished_row[feature])) for feature in finished_row
                }

                example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(example.SerializeToString())
            writer.close()
        conf['features'] = list(finished_row.keys())
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(conf, f)

def generate_lexicon(lex, root_lex=None):
    revlex = {}
    for k in lex:
        if root_lex:
            s = set([label for label in lex[k] if label in root_lex[k]])
        else:
            # FIXME: this cutoff isn't calculated correctly
            cutoff = args.num_files * args.unk_cutoff
            s = set([label for label in lex[k] if lex[k][label] > cutoff]) if k in ['attr', 'transitions'] else set(lex[k])
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

    files = glob.glob(os.path.join(args.read_path,'**',args.filename), recursive=True)
    random.shuffle(files)

    if args.num_files is None:
        args.num_files = len(files)
    else:
        args.num_files = min(args.num_files, len(files))

    filequeue = Queue(maxsize=0)
    resultsqueue = Queue(maxsize=0)

    for i in range(args.num_files):
        filequeue.put((i+1, files[i]))

    processes = set()
    for i in range(args.num_processes):
        p = Process(target=process_queue, args=(filequeue, resultsqueue))
        p.daemon = True
        p.start()
        processes.add(p)


    all_rows = data_dict(lambda: collections.defaultdict(list))
    lexicon = data_dict(lambda: {'label': lex_ctr(), 'attr': lex_ctr(), 'transitions': lex_ctr()})
    transitions_groups = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int))))

    while len(processes) > 0:
        try:
            a,l,t = resultsqueue.get(True, .01)
            for test in a:
                for root_trans in a[test]:
                    for transitions in a[test][root_trans]:
                        for k in a[test][root_trans][transitions]:
                            all_rows[test][root_trans][transitions == 'true'][k].extend(a[test][root_trans][transitions][k])
                        lex = l[test][root_trans][transitions]
                        for k in lex:
                            for token in lex[k]:
                                lexicon[test][root_trans][transitions == 'true'][k][token] += lex[k][token]
            for t1 in t:
                for t2 in t[t1]:
                    for t3 in t[t1][t2]:
                        for t4 in t[t1][t2][t3]:
                            transitions_groups[t1][t2][t3][t4] += t[t1][t2][t3][t4]
        except queue.Empty:
            for p in set(processes):
                if not p.is_alive():
                    processes.remove(p)

    # get the total lexicon
    root_lexicon = collections.defaultdict(lambda: collections.defaultdict(lex_ctr))
    for test in lexicon:
        for root_transitions in lexicon[test]:
            lex = lexicon[test][root_transitions]
            for k in ['attr', 'label']:
                for token in lex[False][k]:
                    root_lexicon[test][k][token] += lex[False][k][token]
            for token in lex[True]['transitions']:
                root_lexicon[test]['transitions'][token] += lex[True]['transitions'][token]
        root_lexicon[test] = generate_lexicon(root_lexicon[test])

    q = Queue(maxsize=0)
    for i in range(args.num_processes):
        p = Process(target=process_queue2, args=(q,))
        p.daemon = True
        p.start()
        processes.add(p)

    config = vars(args)
    config['root_lexicon'] = root_lexicon
    config['tests'] = data_dict(lambda: False)
    for test in lexicon:
        root_lex = root_lexicon[test]['token_to_index']
        for root_transitions in lexicon[test]:
            if root_transitions == '<unk>': continue
            for transitions in [False, True]:
                if transitions and test == 'null' or root_transitions not in root_lex['transitions']:
                    continue

                rows = all_rows[test][root_transitions][transitions]
                num_rows = len(rows['forward-label'])
                dataset_sizes = {}
                dataset_sizes['train'] = math.floor(num_rows * args.train_fraction)
                dataset_sizes['valid'] = math.floor(num_rows * args.valid_fraction)
                dataset_sizes['test'] = num_rows - dataset_sizes['train'] - dataset_sizes['valid']
                # empty dataset
                if functools.reduce(lambda y, d: dataset_sizes[d] == 0 or y, dataset_sizes, False): continue

                datasets = {}
                for dataset in dataset_sizes:
                    datasets[dataset] = {}
                    for k in rows:
                        datasets[dataset][k] = rows[k][:dataset_sizes[dataset]]
                        rows[k] = rows[k][dataset_sizes[dataset]:]

                root_transitions_idx = str(root_lex['transitions'][root_transitions])
                config['tests'][test][root_transitions_idx][transitions] = True


                conf = {}
                conf['dataset_sizes'] = dataset_sizes
                conf['test'] = test
                conf['root_transitions'] = root_transitions
                conf['root_transitions_idx'] = root_transitions_idx
                conf['transitions'] = transitions
                if transitions:
                    conf['transitions_groups'] = transitions_groups[test]
                q.put(json.loads(json.dumps((datasets, lexicon[test][root_transitions][transitions], root_lex, conf))))

    with open(os.path.join(args.store_path, 'config.json'), 'w') as f:
        json.dump(config, f)

    for p in processes:
        p.join()
