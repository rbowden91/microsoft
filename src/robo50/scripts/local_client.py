import sys
import os
import json
import uuid
import socket
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', help='filename', type=str, default="vigenere.c")

#parser.add_argument('-p', '--port', help='port number (default 12347)', type=int, default=12347)
#parser.add_argument('--host', help='host address (default '')', type=str, default='')
#parser.add_argument('-s', '--save_path', help='model output directory (default "tmp")', default='tmp')
#parser.add_argument('-d', '--data_path', help='model output directory (default "tmp")',
#                    default='tmp')
#parser.add_argument('-t', '--subtests', help='which tests to run', type=lambda s: s.split('|'), default=None)
## XXX XXX XXX there was some kind of error loading when this was 32
#parser.add_argument('--num_model_processes', help='number of model processes (default 32)', type=int, default=16)
#parser.add_argument('--num_test_processes', help='number of test processes (default len(test))', type=int, default=None)
#parser.add_argument('--num_socket_processes', help='number of socket processes (technically threads) (default 8)',
#                    type=int, default=1)
#parser.add_argument('--servers', help='servers', type=lambda s: [tuple(x.split(':')) for x in s.split('|')],
#                    default='')
args = parser.parse_args()

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", 12347))
    with open(args.filename, 'r') as content_file:
        content = content_file.read()
    msg = { 'code': content }
    sock.send(json.dumps(msg).encode('latin-1') + b'\n\n')

    #while True:
    #    out = socket_data['pending_output'][0]
    #    if isinstance(out, dict):
    #        out = json.dumps(out).encode('latin-1') + b'\n\n'
    #    while len(out) > 0:
    #        try:
    #            sent = socket_data['fd'].send(out)
    #            out = out[sent:]
    #        except:
    #            break
    #    if len(out) > 0:
    #        socket_data['pending_output'][0] = out
    #        return False
    #    socket_data['pending_output'].pop(0)
    #    if len(socket_data['pending_output']) == 0:
    #        return True

    response = ""
    while True:
        recv_data = sock.recv(4096)
        if not recv_data:
            print("Uh oh! Server socket closed...")
            return 1

        # TODO reject if this gets too big
        response += recv_data

        try:
            response.index(b'\n\n')
        except ValueError:
            continue

        resp = response.split(b'\n\n')
        response = resp.pop()
        for i in range(len(resp)):
            try:
                resp[i] = json.loads(resp[i])
                print(resp[i])
            except:
                print("Uh oh! Server gave invalid response...", resp[i])
                return 2

if __name__=='__main__':
    main()
