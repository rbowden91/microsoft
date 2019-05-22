# TODO: still need to make sure all input files are different...

import os
import sys
import glob
import argparse
import random
import json
import math
import functools
import queue
import shutil
import collections
from typing import Any, List, Dict
from mypy_extensions import TypedDict
from multiprocessing import Process, Queue, current_process

# suppress tensorflow messages, finally
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow as tf #type:ignore




from ..wrangler.wrangle import wrangle, process_ast, finish_row
from ..default_dict import get_dict_default

parser = argparse.ArgumentParser(description='Tokenizer')
parser.add_argument('filename', help='the name of the data files to process, such as caesar.c')
parser.add_argument('read_path', help='directory to read from for processing')
# TODO: make this optional
parser.add_argument('store_path', help='directory to store processed data')
parser.add_argument('-n', '--num_files', help='number of files to parse (default, all files)', type=int)
parser.add_argument('--num_processes', help='number of concurrent processes (default 16)', type=int, default=16)
parser.add_argument('--train_fraction', help='fraction of files for training', type=float, default=.8)
parser.add_argument('--valid_fraction', help='fraction of files for validating', type=float, default=.1)
parser.add_argument('--force', help='delete any preexisting data', action='store_true')
parser.add_argument('--unit_tests', help='unit tests for these files', default=None)
parser.add_argument('--unk_cutoff', help='fraction of files that need to have a token or else it\'s considered unknown', type=float, default=.01)
# TODO: print statistics

def generate_lexicon(lex, root_lex=None, cutoff=None):
    revlex = {}
    for k in lex:
        if root_lex:
            s = set([label for label in lex[k] if label in root_lex[k]])
        else:
            # FIXME: this cutoff isn't calculated correctly
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

def get_transitions_groups(node, transitions_groups):
    if node.__class__.__name__ == 'NodeWrapper':
        return get_transitions_groups(node.new, transitions_groups) if node.new is not None else None
    props = node.node_properties
    if props['is_root']:
        props = props['test_data']
        for test in props:
            transitions = props[test]['transitions']
            for test2 in props:
                if test2 == 'null' or test == test2: continue
                transitions2 = props[test2]['transitions']
                tg = get_dict_default(transitions_groups, test, transitions, test2, transitions2, 0)
                tg[transitions2] += 1

    children = node.children()
    for i in range(len(children)):
        get_transitions_groups(children[i][1], transitions_groups)

def process_queue(file_queue, results_queue, config_queue, unit_tests):
    while True:
        item = file_queue.get()
        if item is False:
            break
        filenumber, filename = item
        print("Process {} handling file {}: {}".format(current_process().name, filenumber, filename))
        data = wrangle(filename, tests=unit_tests)

        # make sure the file passes all tests
        passed_all = functools.reduce(lambda y, test: data.results[test]['passed'] and y, data.results, True)
        if not passed_all:
            results_queue.put(False)
            print(filename + ' failed checks.')
            continue

        transitions_groups = {}
        get_transitions_groups(data.ast, transitions_groups)

        new_rows = {}
        lexicon = {}
        rows = process_ast(data, lexicon)
        for test in rows:
            for root_node in rows[test]:
                root_transitions = data.prop_map[root_node]['test_data'][test]['transitions']
                for transitions in rows[test][root_node]:
                    for k in rows[test][root_node][transitions]:
                        new_row = get_dict_default(new_rows, test, root_transitions, transitions, k, [])
                        new_row[k].append(rows[test][root_node][transitions][k])

        print("Process {} finished file {}: {}".format(current_process().name, filenumber, filename))
        results_queue.put((new_rows, lexicon, transitions_groups))

    while True:
        item = config_queue.get()
        if item is False:
            break
        datasets, lexicon, root_lex, conf = item

        lex = root_lex
        #lex = generate_lexicon(lexicon, root_lex=root_lex)

        for k in lex['token_to_index']:
            conf[k + '_size'] = len(lex['token_to_index'][k])
        conf['lexicon'] = lex

        path = os.path.join(conf['root_path'], conf['root_transitions_idx'], conf['transitions'])
        print('Writing ' + path)
        os.makedirs(path, exist_ok=True)

        sizes = conf['dataset_sizes']
        for dataset in datasets:
            writer = tf.python_io.TFRecordWriter(os.path.join(path, dataset + '.tfrecord'))
            data = datasets[dataset]
            for i in range(sizes[dataset]):
                #print("Writing row {} of {} for dataset {}".format(i+1, sizes[dataset], dataset))
                row = {}
                for k in data:
                    row[k] = data[k][i]

                #finished_row = finish_row(row, lex['token_to_index'])
                finished_row = finish_row(row, root_lex['token_to_index'])
                features = {
                    feature: tf.train.Feature(int64_list=tf.train.Int64List(value=finished_row[feature])) for feature in finished_row
                }

                example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(example.SerializeToString())
            writer.close()
        conf['features'] = list(finished_row.keys())
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(conf, f)
        print('Written ' + path)

def main():
    args = parser.parse_args()

    if os.path.exists(args.store_path):
        if args.force:
            shutil.rmtree(args.store_path)
        else:
            print('Data exists! Add the --force flag to override it anyway')
            sys.exit(1)
    file_queue = Queue(maxsize=0)
    results_queue = Queue(maxsize=0)
    config_queue = Queue(maxsize=0)


    if args.unit_tests is not None:
        with open(args.unit_tests, 'r') as tests_file:
            unit_tests = json.load(tests_file)
        args.unit_tests = {}
        for i in range(len(unit_tests)):
            for test_name in unit_tests[i]:
                unit_tests[i][test_name]['test_group'] = i
                unit_tests[i][test_name]['name'] = test_name
                args.unit_tests[test_name] = unit_tests[i][test_name]

    processes = set()
    for i in range(args.num_processes):
        p = Process(target=process_queue, args=(file_queue, results_queue, config_queue, args.unit_tests))
        p.daemon = True
        p.start()
        processes.add(p)


    files = glob.glob(os.path.join(args.read_path,'**',args.filename), recursive=True)
    random.shuffle(files)

    if args.num_files is None:
        args.num_files = len(files)
    else:
        args.num_files = min(args.num_files, len(files))

    for i in range(args.num_files):
        file_queue.put((i+1, files[i]))

    for i in range(args.num_processes):
        file_queue.put(False)

    all_rows = {}
    lexicon = {}
    transitions_groups = {}

    # get the total lexicon
    root_lexicon = {}

    duplicate_files = set()
    num_results = 0
    while num_results < args.num_files:
        item = results_queue.get()
        num_results += 1
        if not item:
            continue
        a,l,t = item
        check = tuple(a['null']['<FileAST>']['false']['forward-label'][0] + a['null']['<FileAST>']['false']['forward-attr'][0])
        if check in duplicate_files:
            print('duplicate file')
            continue
        duplicate_files.add(check)

        # transitions_groups
        for t1 in t:
            for t2 in t[t1]:
                for t3 in t[t1][t2]:
                    for t4 in t[t1][t2][t3]:
                        tg = get_dict_default(transitions_groups, t1, t2, t3, t4, 0)
                        tg[t4] += t[t1][t2][t3][t4]
        for test in a:
            for root_trans in a[test]:
                for transitions in a[test][root_trans]:
                    # data rows
                    for k in a[test][root_trans][transitions]:
                        rows = get_dict_default(all_rows, test, root_trans, transitions, k, [])
                        rows[k].extend(a[test][root_trans][transitions][k])
                    # lexicon
                    lex = l[test][root_trans][transitions]
                    for k in lex:
                        for token in lex[k]:
                            l2 = get_dict_default(lexicon, test, root_trans, transitions, k, token, 0)
                            l2[token] += lex[k][token]
                            # root_lexicon
                            if (k == 'transitions') != (transitions == 'false'):
                                rl = get_dict_default(root_lexicon, test, k, token, 0)
                                rl[token] += lex[k][token]


    print('Generating root lexicon')
    root_lexicon['null']['transitions'] = {'<FileAST>': args.num_files}
    for test in lexicon:
        root_lexicon[test] = generate_lexicon(root_lexicon[test], cutoff=args.num_files * args.unk_cutoff)

    # translate transitions to their indices
    for t1 in transitions_groups:
        for t2 in list(transitions_groups[t1].keys()):
            for t3 in transitions_groups[t1][t2]:
                for t4 in list(transitions_groups[t1][t2][t3].keys()):
                    if t4 in root_lexicon[t3]['token_to_index']['transitions']:
                        root_trans_idx = root_lexicon[t3]['token_to_index']['transitions'][t4]
                        transitions_groups[t1][t2][t3][root_trans_idx] = transitions_groups[t1][t2][t3][t4]
                    del(transitions_groups[t1][t2][t3][t4])
            if t2 in root_lexicon[t1]['token_to_index']['transitions']:
                root_trans_idx = root_lexicon[t1]['token_to_index']['transitions'][t2]
                transitions_groups[t1][root_trans_idx] = transitions_groups[t1][t2]
            del(transitions_groups[t1][t2])

    print('Drafting configs')
    for test in all_rows:
        root_lex = root_lexicon[test]['token_to_index']
        for root_transitions in all_rows[test]:
            if root_transitions == '<unk>': continue
            for transitions in all_rows[test][root_transitions]:
                if transitions == 'true' and test == 'null' or root_transitions not in root_lex['transitions']:
                    continue

                rows = all_rows[test][root_transitions][transitions]

                num_rows = len(rows['forward-label'])
                average_row_length = 0
                for row in rows['forward-label']:
                    average_row_length += len(row)
                average_row_length /= num_rows
                dataset_sizes = {}
                dataset_sizes['train'] = math.floor(num_rows * args.train_fraction)
                dataset_sizes['valid'] = math.floor(num_rows * args.valid_fraction)
                dataset_sizes['test'] = num_rows - dataset_sizes['train'] - dataset_sizes['valid']
                # empty dataset
                if functools.reduce(lambda y, d: dataset_sizes[d] == 0 or y, dataset_sizes, False):
                    continue

                datasets = {}
                for dataset in dataset_sizes:
                    datasets[dataset] = {}
                    row_hashes = collections.defaultdict(int)
                    for k in rows:
                        dataset_rows = rows[k][:dataset_sizes[dataset]]
                        rows[k] = rows[k][dataset_sizes[dataset]:]

                        #for row in dataset_rows:
                        #    row_hashes[tuple(row)] += 1
                        #counted_rows = []
                        #for row, count in row_hashes.items():
                        #    counted_rows.append((count, list(row)))

                        #datasets[dataset][k] = counted_rows
                        datasets[dataset][k] = dataset_rows

                root_transitions_idx = str(root_lex['transitions'][root_transitions])

                conf = {}
                conf['root_path'] = os.path.join(args.store_path, test)
                conf['average_row_length'] = average_row_length
                conf['dataset_sizes'] = dataset_sizes
                conf['test'] = test
                conf['root_transitions'] = root_transitions
                conf['root_transitions_idx'] = root_transitions_idx
                conf['transitions'] = transitions
                print('Finished config for {} {} {}'.format(test, root_transitions_idx, transitions))
                config_queue.put((datasets, lexicon[test][root_transitions][transitions], root_lexicon[test], conf))

        test_conf = {
            'root_lex': root_lex,
            'unit_test': args.unit_tests[test] if test != 'null' else None,
            # TODO: use indices instead of raw transitions?
            'transitions_groups': transitions_groups[test] if test != 'null' else None
        }
        test_config_path = os.path.join(args.store_path, test)
        os.makedirs(test_config_path, exist_ok=True)
        with open(os.path.join(test_config_path, 'config.json'), 'w') as f:
            json.dump(test_conf, f)

    # signal to the processes that the q is empty
    for i in range(args.num_processes):
        config_queue.put(False)

    print('Joining subprocesses')
    for p in processes:
        p.join()
        print('Process joined')
