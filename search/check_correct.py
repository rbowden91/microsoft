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

def check_vigenere(path):
    # XXX what do do about #includes that were stripped by preprocessing?
    out = subprocess.run('clang ' + args.path + ' cs50.o -lm', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if out.returncode != 0:
        return False

    failed = 0
    for i in range(len(tests)):
        out = subprocess.run('echo ' + tests[i]['stdin'] + ' | ./a.out ' + tests[i]['argv'], shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if 'returncode' in tests[i]:
            if tests[i]['returncode'] != out.returncode:
                failed += 1
        elif out.stdout != tests[i]['expected']:
            failed += 1

    return failed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check vigenere program for correctness.')
    parser.add_argument('path', help='.c file to check')

    args = parser.parse_args()
    print(check_vigenere(args.path))
