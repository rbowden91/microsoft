# https://realpython.com/python-sockets/#multi-connection-client-and-server
import sys
import os
import json
import argparse
import socket
import selectors
import traceback
import time

#from multiprocessing import Process, Queue
from queue import Queue
from threading import Thread
from typing import Dict

# TF_CPP_MIN_LOG_LEVEL = '3'

from ..server import Server # type: ignore

NUM_PROCESSES = 1
HOST = ''

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--port', help='port number (default 12347)', type=int, default=12347)
parser.add_argument('-d', '--datapath', help='model output directory (default "tmp")',
        default='/home/rbowden/repos/repair50/data/training_data/vig_no_decl10/ast/tmp')
parser.add_argument('-t', '--subtests', help='which tests to run', type=lambda s: s.split(), default=None)
args = parser.parse_args()

def send_json(sock, msg):
    output = json.dumps(msg).encode('latin-1') + b'\n\n'
    sleep_error = 0
    while len(output) > 0 and sleep_error <= 10:
        try:
            sent = sock.send(output)
            output = output[sent:]
            sleep_error = 0
        except:
            sleep_error += 1
            time.sleep(1)
    return len(output) == 0



def start_worker(q):
    server = Server(args.datapath, args.subtests)
    while True:
        sock, input_ = q.get()
        print('Handling input')
        try:
            output = server.process_code(input_['code'])
            print(output)
            output['session_id'] = input_['session_id']
            props = output['props']
            del(output['props'])
            output['success'] = True
        except Exception:
            traceback.print_exc()
            props = {}
            output = {'success': False}
        print('Sending code and test results')
        if send_json(sock, output):
            for k in props:
                print('Sending prop ' + str(k))
                output = {
                    'success': True,
                    'session_id': input_['session_id'],
                    'props': {k : props[k]}
                }
                if not send_json(sock, output):
                    break
        print('Done sending output')
        q.task_done()


# TODO: disconnect after timeout, messages too long, etc.
# TODO: kill subprocesses?
def main():
    q = Queue()
    # when you use Process instead of Thread, tensorflow gives an out_of_resources error...
    pool = [Thread(target=start_worker, args=(q,)) for p in range(NUM_PROCESSES)]
    for p in pool:
        p.daemon = True
        p.start()

    sel = selectors.DefaultSelector()
    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lsock.bind((HOST, args.port))
    lsock.listen()
    print('listening on', (HOST, args.port))
    lsock.setblocking(False)
    sel.register(lsock, selectors.EVENT_READ, data=None)
    while True:
        events = sel.select(timeout=None)
        for key, mask in events:
            if key.data is None:
                conn, addr = lsock.accept()  # Should be ready to read
                print('accepted connection from', addr)
                conn.setblocking(False)
                sel.register(conn, selectors.EVENT_READ, data={'input':b''})
            else:
                sock = key.fileobj
                data = key.data
                if mask & selectors.EVENT_READ:
                    try:
                        recv_data = sock.recv(4096)  # Should be ready to read
                    except:
                        recv_data = None

                    if recv_data:
                        data['input'] += recv_data
                        input_ = data['input'].split(b'\n\n')
                        if len(input_) > 1:
                            # TODO: prevent further reading from this socket until the current command is handled
                            data['input'] = b'\n\n'.join(input_[1:])
                            try:
                                input_ = json.loads(input_[0])
                                q.put((sock, input_))
                            except:
                                sel.unregister(sock)
                                sock.close()
                    else:
                        print('Dropped connection')
                        sel.unregister(sock)
                        sock.close()
