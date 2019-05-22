#!/usr/bin/env python

import sys, json, os
from shutil import move

#from pycparser import c_parser, c_ast, parse_file
#from centipyde.interpreter import run_tests, InterpTooLong
#from robo50.preprocessor.external import ExternalCPP
from robo50.wrangler import wrangle

prefix = 'cs50_data'

num_processes = 64

def process_queue(file_queue, unit_tests):
    while True:
        filename, filepath, correct = file_queue.get()
        if not os.path.isfile(filename):
            continue
        try:
            data = wrangle(filename, tests=unit_test)
        #try:
        #    cfile = ExternalCPP().preprocess(filename, is_file=True)
        except Exception as e:
            print(filename + ' failed1')
            #move(filepath, no_preprocess)
            continue
#            sys.exit(1)
        # make sure the file passes all tests
        passed_all = functools.reduce(lambda y, test: data.results[test]['passed'] and y, data.results, True)
        if not passed_all:
            print(filename + ' failed2')
            continue
        print(filename + ' passed')
        os.move(filepath, correct)

for problem in [('mario', 'mario/less/2017')]:
    unit_tests = os.path.join(prefix, 'unit_tests', problem[1], 'index.json')

    test = os.path.join(unit_tests, f)
    with open(test, "r") as text_file:
        test = json.load(text_file)

    submissions = os.path.join(prefix, 's3_sorted_psets', problem[0])
    move_dir = os.path.join(prefix, 's3_graded_psets', problem[1])
    no_preprocess = os.path.join(move_dir, 'no_preprocess')
    no_parse = os.path.join(move_dir, 'no_parse')
    no_interpret = os.path.join(move_dir, 'no_interpret')
    incorrect = os.path.join(move_dir, 'incorrect')
    correct = os.path.join(move_dir, 'correct')

    os.makedirs(no_preprocess, mode=0o711, exist_ok=True)
    os.makedirs(no_parse, mode=0o711, exist_ok=True)
    os.makedirs(no_interpret, mode=0o711, exist_ok=True)
    os.makedirs(incorrect, mode=0o711, exist_ok=True)
    os.makedirs(correct, mode=0o711, exist_ok=True)

    processes = set()
    for i in range(num_processes):
        p=Process(target=process_queue, args=(file_queue, test))
        p.daemon = True
        p.start()
        processes.add(p)

    # iterate over all submissions
    for f in os.listdir(submissions):
        filepath = os.path.join(submissions, f)
        filename = os.path.join(filepath, problem[0] + '.c')
        file_queue.put((filepath, filename, correct))

        #parser = c_parser.CParser()
        #try:
        #    ast = parser.parse(cfile)
        #except Exception as e:
        #    print('failed to parse')
        #    move(filepath, no_parse)
        #    continue
#       #     sys.exit(2)

        #for key in tests:
        #    try:
        #        # TODO: no dowhile
        #        results = run_tests(ast, tests[key])
        #    except InterpTooLong:
        #        continue


        #    except Exception as e:
        #        print('failed to interpret', e)
        #        move(filepath, no_interpret)
        #        break

        #    #print(results)
        #    #with open('output.json', "w") as text_file:
        #        #json.dump(results, text_file)


        #    for result in results:
        #        if not result['passed']:
        #            print('failed', results)
        #            break
        #    # passed all the tests
        #    else:
        #        move(filepath, os.path.join(move_dir, 'correct', key))
        #        break
        ## did not pass any checks
        #else:
        #    pass
        #    #os.move(filepath, incorrect)
    for p in processes:
        p.join()
    print('Done!')
