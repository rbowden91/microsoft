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
HOST = '127.0.0.1'
PORT = 12344
servers = [('korra.rbowden.com', 12345), ('appa.rbowden.com', 12345)]

sel = selectors.DefaultSelector()

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
    while True:
        sock, input_ = q.get()
        if not send_json(sock, input_):
            # TODO: close socket
            #if sock in reverse_client_map:
            #    del(client_map[reverse_client_map[sock]])
            #sel.unregister(sock)
            #sock.close()
            pass
        q.task_done()


server_map = {} # type:ignore
client_map = {} # type:ignore
reverse_client_map = {} # type:ignore

# TODO: disconnect after timeout, messages too long, etc.
# TODO: kill subprocesses?
def main():
    q = Queue()
    # when you use Process instead of Thread, tensorflow gives an out_of_resources error...
    pool = [Thread(target=start_worker, args=(q,)) for p in range(NUM_PROCESSES)]
    for p in pool:
        p.daemon = True
        p.start()

    #sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lsock.bind((HOST, PORT))
    lsock.listen()
    print('listening on', (HOST, PORT))
    lsock.setblocking(False)

    sel.register(lsock, selectors.EVENT_READ, data={'type': 'listener'})

    # connect to external servers
    for server in servers:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setblocking(False)
        sock.connect_ex(server)
        server_map[server] = sock
        sel.register(sock, events, data={'input':b'', 'type':'server'})

    try:
        while True:
            events = sel.select(timeout=None)
            for key, mask in events:
                sock = key.fileobj
                data = key.data
                if data.type == 'listener':
                    conn, addr = sock.accept()  # Should be ready to read
                    print('accepted connection from', addr)
                    conn.setblocking(False)
                    sel.register(conn, selectors.EVENT_READ, data={'input':b'', 'type':'client'})
                else:
                    if mask & selectors.EVENT_READ:
                        try:
                            recv_data = sock.recv(4096)  # Should be ready to read
                            data['input'] += recv_data
                            input_ = data['input'].split(b'\n\n')
                            while len(input_) > 1:
                                # TODO: prevent further reading from this socket until the current command is handled
                                json_input = input_.pop(0)
                                json_input = json.loads(json_input)
                                assert 'session_id' in json_input

                                if data.type == 'client':
                                    client_map[json_input['session_id']] = sock
                                    for server in server_map:
                                        q.put(send_json(server_map[server]. json_input))
                                else:
                                    assert data.type == 'server'
                                    if json_input['session_id'] not in client_map: continue
                                    client_sock = client_map[json_input['session_id']]
                                    q.put((client_sock, json_input))
                            data['input'] = input_[0]
                        except:
                            # TODO: this is bad if it was to a server
                            sel.unregister(sock)
                            sock.close()
                            continue

    except KeyboardInterrupt:
        print("Caught keyboard interrupt, exiting")
    finally:
        sel.close()
