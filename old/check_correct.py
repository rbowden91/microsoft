#!/usr/bin/python3

import os
import sys
import subprocess
import argparse

tests = [
    {
        'stdin': 'a',
        'argv': 'a',
        'expected': b'a\n'
    },
    {
        'stdin': 'barfoo',
        'argv': 'baz',
        'expected': b'caqgon\n'
    },
    {
        'stdin': 'BaRFoo',
        'argv': 'BaZ',
        'expected': b'CaQGon\n'
    },
    {
        'stdin': 'BARFOO',
        'argv': 'BAZ',
        'expected': b'CAQGON\n'
    },
    {
        'stdin': 'world!\$?',
        'argv': 'baz',
        'expected': b'xoqmd!$?\n'
    },
    {
        'stdin': 'world, say hello!',
        'argv': 'baz',
        'expected': b'xoqmd, rby gflkp!\n'
    },
    {
        'stdin': '',
        'argv': '',
        #'expected': b'xoqmd, rby gflkp!\n'
        'returncode': 1
    },
    {
        'stdin': '',
        'argv': '1 2 3',
        #'expected': b'xoqmd, rby gflkp!\n'
        'returncode': 1
    },
    {
        'stdin': '',
        'argv': 'HaX0r2',
        #'expected': b'xoqmd, rby gflkp!\n'
        'returncode': 1
    }
]


# https://stackoverflow.com/questions/1996518/retrieving-the-output-of-subprocess-call
def run(cmd):
    """
    Execute the external command and get its exitcode, stdout and stderr.
    """

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = proc.communicate()
    exitcode = proc.returncode
    return exitcode, out, err


def check_vigenere(path):
    # XXX how to guide search if some fixes partially fix things?
    #out = subprocess.run('clang ' + args.path + ' -o ' + args.path + '_out cs50.o -lm', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #if out.returncode != 0:
    #    return -1

    exit, _, _ = run('gcc --std=c99 ' + path + ' -o ' + path + '_out cs50.o -lm -I.')
    if exit != 0:
        return -1

    failed = 0
    for i in range(len(tests)):
        exit, out, err = run('echo ' + tests[i]['stdin'] + ' | ./' + path + '_out ' + tests[i]['argv'])
        if 'returncode' in tests[i]:
            #if tests[i]['returncode'] != out.returncode:
            if tests[i]['returncode'] != exit:
                return 1
                #failed += 1
        elif out != tests[i]['expected']:
            return 1
            #failed += 1
    os.unlink(path + '_out')

    return failed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check vigenere program for correctness.')
    parser.add_argument('path', help='.c file to check')

    args = parser.parse_args()
    print(check_vigenere(args.path))
