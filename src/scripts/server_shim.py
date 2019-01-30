import sys
import os
import json
from typing import Dict

from ..server import server # type: ignore

# TODO: send the possibilities for this to the server?
config = server.load_data('../data/vig_no_decl100/ast/tmp')
serv = server.create_server(config)
while True:
    input_ : str = sys.stdin.readline()
    parsed_input : Dict[str, str] = json.loads(input_)
    # TODO: we can get rid of "model" from this?
    output = serv.process_code(parsed_input['code'])
    output['uuid'] = parsed_input['uuid']
    output['success'] = True
    print(json.dumps(output) + '\n\n')
