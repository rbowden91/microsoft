import sys
import os
import json
from typing import Dict

os.chdir('../..')
from repair50.tree import server # type: ignore


serv = server.create_server()
while True:
    input_ : str = sys.stdin.readline()
    parsed_input : Dict[str, str] = json.loads(input_)
    print(serv.process_code(input_))
