#!/usr/bin/python3

import os
import json

import centipyde.interpret

class Interpreter(object):
    def __init__(self, test_name):
        self.test_name = test_name
        dir_path = os.path.dirname(os.path.realpath(__file__))
        include_path = os.path.join(dir_path, 'unit_tests', test_name + '.json')
        with open(include_path) as f:
            self.tests = json.load(f)

    def run_tests(self, ast):
        results = []
        visited = []
        for test in self.tests:
            result = test.copy()
            # can we just instantiate one of these?
            interpreter = centipyde.interpret.Interpreter(ast)
            interpreter.setup_main(test['argv'].split(), test['stdin'])
            interpreter.run()
            try:
                assert len(interpreter.k.passthroughs) == 1
            except:
                #print('Something went wrong interpreting.')
                return None
            # handle lack of "return 0;" in main
            ret = interpreter.k.passthroughs[0][0]
            if ret.type != 'Return':
                value = 0
            else:
                value = ret.value.value

            result['actual_stdout'] = interpreter.stdout
            result['actual_return'] = value
            #result['visited'] = visited
            results.append(result)
            visited.append(interpreter.visited)

        return results, visited
