#!/usr/bin/env python

import sys, json, os
from shutil import move

from pycparser import c_parser, c_ast, parse_file
from centipyde.interpreter import run_tests, InterpTooLong
from robo50.preprocessor.external import ExternalCPP

prefix = 'cs50_data'

for problem in ['vigenere']:
    unit_tests = os.path.join(prefix, 'unit_tests', problem)

    tests = {}
    for f in os.listdir(unit_tests):
        test = os.path.join(unit_tests, f)
        with open(test, "r") as text_file:
            tests[f] = json.load(text_file)

    submissions = os.path.join(prefix, 's3_sorted_psets', problem)
    move_dir = os.path.join(prefix, 's3_graded_psets', problem)
    no_preprocess = os.path.join(move_dir, 'no_preprocess')
    no_parse = os.path.join(move_dir, 'no_parse')
    no_interpret = os.path.join(move_dir, 'no_interpret')
    incorrect = os.path.join(move_dir, 'incorrect')

    os.makedirs(no_preprocess, mode=0o711, exist_ok=True)
    os.makedirs(no_parse, mode=0o711, exist_ok=True)
    os.makedirs(no_interpret, mode=0o711, exist_ok=True)
    os.makedirs(incorrect, mode=0o711, exist_ok=True)
    for test in tests:
        os.makedirs(os.path.join(move_dir, 'correct', test), mode=0o711, exist_ok=True)

    # iterate over all submissions
    for f in os.listdir(submissions):
        filepath = os.path.join(submissions, f)
        filename = os.path.join(filepath, problem + '.c')
        if not os.path.isfile(filename):
            continue
        print(filename)
        try:
            cfile = ExternalCPP().preprocess(filename, is_file=True)
        except Exception as e:
            print('failed to preprocess')
            move(filepath, no_preprocess)
            continue
#            sys.exit(1)

        parser = c_parser.CParser()
        try:
            ast = parser.parse(cfile)
        except Exception as e:
            print('failed to parse')
            move(filepath, no_parse)
            continue
#            sys.exit(2)

        for key in tests:
            try:
                # TODO: no dowhile
                results = run_tests(ast, tests[key])
            except InterpTooLong:
                continue


            except Exception as e:
                print('failed to interpret', e)
                move(filepath, no_interpret)
                break

            #print(results)
            #with open('output.json', "w") as text_file:
                #json.dump(results, text_file)


            for result in results:
                if not result['passed']:
                    print('failed', results)
                    break
            # passed all the tests
            else:
                move(filepath, os.path.join(move_dir, 'correct', key))
                break
        # did not pass any checks
        else:
            pass
            #os.move(filepath, incorrect)
