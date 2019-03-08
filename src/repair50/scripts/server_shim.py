# https://realpython.com/python-sockets/#multi-connection-client-and-server
import sys
import os
import json
import argparse
import socket
import selectors
import traceback

#from multiprocessing import Process, Queue
from queue import Queue
from threading import Thread
from typing import Dict

# TF_CPP_MIN_LOG_LEVEL = '3'

from ..server import Server # type: ignore

NUM_PROCESSES = 1
HOST = '127.0.0.1'
PORT = 12344

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='model output directory (default "tmp")',
        default='/home/rbowden/repos/repair50/data/training_data/vig_no_decl10/ast/tmp')
args = parser.parse_args()


def start_worker(q):
    server = Server(args.path)
    while True:
        sock, input_ = q.get()
        try:
            output = server.process_code(input_['code'])
            output['success'] = True
        except Exception:
            traceback.print_exc()
            output = {'success': False}
        output = json.dumps(output).encode('latin-1') + b'\n\n'
        while len(output) > 0:
            try:
                sent = sock.send(output)
                output = output[sent:]
            except:
                pass

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
    lsock.bind((HOST, PORT))
    lsock.listen()
    print('listening on', (HOST, PORT))
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
                    recv_data = sock.recv(4096)  # Should be ready to read
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
