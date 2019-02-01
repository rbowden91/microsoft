import sys
import os
import json
import argparse
from typing import Dict

from ..server import Server # type: ignore

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='model output directory (default "tmp")',
            default='/home/rbowden/repos/repair50/data/training_data/vig_no_decl10/ast/tmp')
    args = parser.parse_args()

    # TODO: send the possibilities for this to the server?
    server = Server(args.path)
    while True:
        input_ : str = sys.stdin.readline()
        if not input_:
            return
        parsed_input : Dict[str, str] = json.loads(input_)
        # TODO: we can get rid of "model" from this?
        output = server.process_code(parsed_input['code'])
        output['uuid'] = parsed_input['uuid']
        output['success'] = True
        print(json.dumps(output) + '\n\n')
