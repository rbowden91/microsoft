import sys
import os
import json
import selectors
import uuid
import socket
import selectors
import time
import re
import traceback

from collections import namedtuple
from queue import Queue as ThreadQueue
from threading import Lock as ThreadLock, Thread
from typing import Dict, Union
from multiprocessing import Process, Pipe, Lock, Queue

import numpy as np # type:ignore
import tensorflow as tf #type:ignore
import collections

from centipyde.interpreter import run_tests

from ..default_dict import get_dict
from ..model.config import joint_configs, dependency_configs, valid_dependencies #type:ignore
from ..wrangler.wrangle import finish_row, wrangle, process_ast #type:ignore
from ..wrangler.normalizers.BreakPoint import BreakPoint
from .c_generator import CGenerator

# info, warning, error, never_print
LOG_LEVEL = 0

def log_print(string, log_level=0):
    if log_level >= LOG_LEVEL:
        print(string)

class FDMap(object):
    def __init__(self):
        self.server_id = uuid.uuid4().hex
        self.server_sockets = []
        self.map = {}
        self.rev_map = {}
        self.sel = selectors.DefaultSelector()

    def add(self, fd, type_):
        id_ = uuid.uuid4().hex
        self.map[id_] = {
            'fd': fd,
            'id': id_,
            'state': selectors.EVENT_READ,
            'type': type_,
            'requests': {},
            'input': b'',
            'closed': False,
            'output': [],
            'pending_output': None
        }
        data = {'id': id_}
        if type_ == 'server_socket':
            self.server_sockets.append(id_)
            data['serverId'] = self.server_id
        self.sel.register(fd, selectors.EVENT_READ, data=data)
        return id_

    def add_pipe(self, fd1, fd2, type_):
        self.map[fd2] = self.add(fd1, type_)
        return self.map[fd2]

    def get(self, fd_id):
        assert fd_id in self.map
        return self.map[fd_id]

    def unset_state(self, fd_id, mask):
        data = self.map[fd_id]
        data['state'] = data['state'] & ~mask
        if not data['state']:
            self.sel.unregister(data['fd'])
        else:
            self.sel.modify(data['fd'], data['state'], data={'id': data['id']})

        if mask & selectors.EVENT_WRITE:
            assert data['pending_output'] is None
            data['pending_output'] = data['output']
            data['output'] = []

    def update_state(self, request, response, mask):
        if request['type'] == 'request':
            data = self.map[request['fd_id']]
        else:
            data = self.map[request['id']]
        if data['closed']: return

        if request['type'] == 'request':
            if request['id'] not in data['requests']:
                data['requests'][request['id']] = request
            else:
                request = data['requests'][request['id']]

            if not isinstance(response, list):
                response['output'] = [response['output']]

        if data['type'] == 'client_socket' and 'direction' in response and response['direction'] == 'to_servers':
            datas = [self.map[s] for s in self.server_sockets]
        else:

            datas = [data]

        for data in datas:
            if data['closed']: continue

            if 'opaque' in request:
                for out in response['output']:
                    out['opaque'] = request['opaque']

            if 'output' in response:
                data['output'].extend(response['output'])

            new_state = mask | data['state']

            if mask & selectors.EVENT_WRITE:
                data['output'] = data['pending_output'] + data['output']
                if len(data['output']) == 0:
                    new_state = new_state & ~selectors.EVENT_WRITE
                data['pending_output'] = None
            elif len(data['output']) > 0 and data['pending_output'] is None:
                new_state = new_state | selectors.EVENT_WRITE

            if not data['state']:
                self.sel.register(data['fd'], new_state, data={'id': data['id']})
            else:
                self.sel.modify(data['fd'], new_state, data={'id': data['id']})
            data['state'] = new_state


    def close_fd(self, fd_id):
        data = self.map[fd_id]
        if data['closed']: return
        if data['state']:
            self.sel.unregister(data['fd'])
        # TODO: don't close this if ongoing requests?
        data['closed'] = True
        data['fd'].close()

    def shutdown(self):
        for fd_id in self.map:
            self.close_fd(fd_id)
        self.map = {}
        self.sel.close()

# base class
# TODO: if the client crashes, this currently crashes??

class ServerProcess(object):
    def __init__(self, input_queue) -> None:
        self.input_queue = input_queue
        self.loop()

    def loop(self):
        while True:
            handler, *args = self.input_queue.get()
            try:
                getattr(self, handler)(*args)
            except:
                traceback.print_exc()


# TODO: each of these processes can also be multithreaded (instead?)
class ServerModelProcess(ServerProcess):
    def __init__(self, model_configs, num_subprocesses, beam_queues, *args, **kwargs) -> None:
        self.model_configs = model_configs #type:ignore
        self.sessions = [] #type: ignore
        self.beam_queues = beam_queues

        for i in range(num_subprocesses-1):
            try:
                pid = os.fork()
            except OSError:
                print("Failed to fork model process")
                break
            if pid == 0:
                break
        super().__init__(*args, **kwargs)

    def get_session(self, model_conf):
        if 'session' not in model_conf:
            for session in self.sessions:
                try:
                    session['saver'].restore(session['session'], os.path.join(model_conf['best_dir'], 'trimmed-model'))
                except:
                    continue
                session['loaded_conf'] = model_conf
                model_conf['session'] = session
                break
            else:
                with tf.Graph().as_default():
                    saver = tf.train.import_meta_graph(os.path.join(model_conf['best_dir'], "trimmed-model.meta"))
                    session = tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0}))
                self.sessions.append({'session': session, 'saver': saver, 'loaded_conf': None})
                model_conf['session'] = self.sessions[-1]
        if model_conf['session']['loaded_conf'] != model_conf:
            model_conf['session']['saver'].restore(model_conf['session']['session'], os.path.join(model_conf['best_dir'], 'trimmed-model'))
            model_conf['session']['loaded_conf'] = model_conf
        return model_conf['session']['session']

    def handle_input(self, beam_num, root_node_idx, rows, test, root_trans_idx, transitions):
        model_conf = self.model_configs[test][root_trans_idx][transitions]

        log_print('Running model {} {} {}'.format(test, root_trans_idx, transitions), 0)
        lexicon = model_conf['lexicon']
        node_nums = rows['forward-node_num']
        row = finish_row(rows, lexicon['token_to_index'])
        data_dict = {}
        for k in model_conf['features']:
            data_dict[model_conf['features'][k]] = [row[k]]

        session = self.get_session(model_conf)
        session.run(model_conf['tensor_iter'], data_dict)
        vals = session.run(model_conf['fetches'])

        # TODO: send this back to beam process

        codeProps = {}
        for i in range(1, len(row['forward-self'])):
            idx = {'forward': row['forward-self'][i], 'reverse': row['reverse-self'][i]}
            props = self.gather_props(self.model_configs[test][root_trans_idx][transitions], vals, data_dict, idx)
            codeProps[node_nums[i-1]] = props
        self.beam_queues[beam_num].put((test, root_node_idx, root_trans_idx, transitions, codeProps))


    def gather_props(self, model_config, vals, data_dict, diridx):
        tc = model_config
        revlex = tc['lexicon']['index_to_token']
        lex = tc['lexicon']['token_to_index']
        transitions = tc['transitions'] == 'true'
        features = tc['features']

        prop = {dconfig : {cdependency: {} for cdependency in vals[dconfig]} for dconfig in vals}
        for dconfig in vals:
            for cdependency in vals[dconfig]:
                v = vals[dconfig][cdependency]
                direction = 'forward' if dconfig == 'joint_configs' or \
                            dependency_configs[cdependency][-1][0] else 'reverse'
                prop[dconfig][cdependency]['direction'] = direction
                idx = diridx[direction]

                if transitions:
                    token_target = data_dict[features[direction + '-transitions_index']][0][idx]
                else:
                    token_target = data_dict[features[direction + '-label_index']][0][idx]
                    attr_target = data_dict[features[direction + '-attr_index']][0][idx]

                #for dependency in v['cells']:
                #    for k in v['cells'][dependency]:
                #        if node_id not in config['cells'][dconfig][cdependency][dependency][k]:
                #            config['cells'][dconfig][cdependency][dependency][k][node_id] = {}
                #        cells = config['cells'][dconfig][cdependency][dependency][k][node_id]
                #        if transitions:
                #            cells[token_target] = v['cells'][dependency][k][0][idx].tolist()
                #        else:
                #            if token_target not in cells:
                #                cells[token_target] = {}
                #            cells[token_target][attr_target] = v['cells'][dependency][k][0][idx].tolist()

                for k in v['loss']:
                    # attr_index handled within label_index
                    if k == 'attr_index': continue

                    prop[dconfig][cdependency][k] = p = {}

                    probs = v['probabilities'][k][0]
                    if dconfig == 'joint_configs':
                        p['alpha'] = alpha = probs[idx].tolist()
                        sum_probs = 0
                        for jd in range(len(joint_configs[cdependency])):
                            joint_dependency = joint_configs[cdependency][jd]
                            probs = vals['dependency_configs'][joint_dependency][k]['probabilities'][0][idx]
                            sum_probs += alpha[jd] * probs
                        probs = sum_probs
                    else:
                        probs = probs[idx]

                    if k == 'label_index' or k == 'transitions_index':

                        p['actual_probability'] = float(probs[token_target])
                        if not transitions:
                            p['actual_probability'] *= float(v['probabilities']['attr_index'][0][idx][attr_target])

                        if k == 'label_index':
                            new_probs = []
                            for token_idx in range(len(probs)):
                                token = (float(probs[token_idx]), revlex['label'][str(token_idx)])

                                attr_probs = v['attr_all'][0][idx][token_idx]
                                new_probs.extend([(token[0] * float(attr_probs[j]),
                                                    token[1], revlex['attr'][str(j)]) for j in range(len(attr_probs))])
                            probs = new_probs
                            p['actual_attr'] = revlex['attr'][str(attr_target)]
                            p['actual_label'] = revlex['label'][str(token_target)]
                        else:
                            probs = [(float(probs[j]), revlex['transitions'][str(j)]) for j in range(len(probs))]
                            #probs = probs.tolist()
                            p['actual_transitions'] = revlex['transitions'][str(token_target)]

                        probs.sort(key=lambda x: x[0], reverse=True)
                        p['probabilities'] = [x for x in probs if x[0] > .001]
                        expected_probability = float(probs[0][0])
                        p['ratio'] = p['actual_probability'] / expected_probability
                    elif k == 'pointers':
                        p['actual'] = []
                        for q in range(20):
                            target = data_dict[features[direction + '-pointers-mask-' + str(q)]][0][idx]
                            p['actual'].append(target)
                        p['expected'] = probs.tolist()
                    else:
                        target = data_dict[features[direction + '-' + k]][0][idx]
                        p['actual'] = target
                        p['expected'] = probs.tolist()

        return prop

class ServerBeamProcess(ServerProcess):
    def __init__(self, test_configs, model_configs, server_queue, personal_beam_queues, model_queue, *args, **kwargs):
        self.server_queue = server_queue
        self.model_queue = model_queue

        self.personal_beam_queues = personal_beam_queues

        self.test_configs = test_configs
        self.model_configs = model_configs
        self.unit_tests = {}

        for test in test_configs:
            if test != 'null':
                self.unit_tests[test] = self.test_configs[test]['unit_test']

        self.beam_num = len(personal_beam_queues) - 1
        for i in range(self.beam_num):
            try:
                pid = os.fork()
            except OSError:
                print("Failed to fork model process")
                break
            if pid == 0:
                self.beam_num = i
                break

        super().__init__(*args, **kwargs)

    def run_models(self, rows):
        for test in self.test_configs:
            root_lex = self.test_configs[test]['root_lex']['transitions']
            for root_node_idx in rows[test]:
                for transitions in rows[test][root_node_idx]:
                    root_node = ast_data.prop_map[root_node_idx]
                    root_test_data = ast_data.prop_map[root_node_idx]['test_data'][test]
                    root_transitions = root_test_data['transitions']
                    if root_transitions == '<unk>' or root_transitions not in root_lex:
                        root_test_data['unknown_transitions'] = True
                        continue
                    root_trans_idx = str(root_lex[root_transitions])
                    root_test_data['unknown_transitions'] = False
                    root_test_data['root_trans_idx'] = root_trans_idx
                    #if root_trans_idx not in mpm or \
                    #        transitions not in mpm[root_trans_idx]:
                    #    # XXX this one isn't quite "unknown". We just didn't have enough test data???
                    #    root_test_data['unknown_transitions'] = True
                    #    continue
                    self.model_queue.put((
                        'handle_input',
                        self.beam_num,
                        root_node_idx,
                        rows[test][root_node_idx][transitions],
                        test,
                        root_trans_idx,
                        transitions,
                    ))
                    enqueues += 1

        while enqueues > 0:
            test, root_node_idx, root_trans_idx, transitions, codeProps = self.personal_beam_queues[self.beam_num].get()
            print(test, root_node_idx, transitions)
            enqueues -= 1

    def replace_node(self, ast_data, rows, node_idx):
        parent_idx = rows['null'][1]['false']['forward-parent'][node_idx-1]
        parent = ast_data.node_map[parent_idx]
        node = ast_data.node_map[node_idx]
        replacement = BreakPoint(lambda interp: self.show_env(interp),
                    node,
                    lambda interp: self.show_env(interp))
        c_names = {}
        replace_c_name = None
        for c_name, c in parent.children():
            m = re.match("([^\\[]*)\\[", c_name)

            # is this a list-based node child (like a Compound's block_items)
            if m:
                name = m.groups()[0]
                if c == node:
                    replace_c_name = name
                if name not in c_names:
                    c_names[name] = []
                c_names[name].append(c if c != node else replacement)
            else:
                if c == node:
                    setattr(parent, c_name, replacement)
                    return
        assert replace_c_name is not None
        setattr(parent, name, c_names[replace_c_name])



    def generate_tree(self, row, test, root_trans_idx, transitions):
        self.model_queue.put((
            'handle_input',
            self.beam_num,
            row,
            test,
            root_trans_idx,
            transitions
        ))

        interpreter.run(node, starting_state)
        test, root_node_idx, root_trans_idx, transitions, codeProps = self.personal_beam_queues[self.beam_num].get()

    def show_env(self, interp):
        print(interp.scope)

    def handle_input(self, request, code):
        try:
            ast_data = wrangle(code, tests=self.unit_tests, is_file=False)
        except Exception as e:
            return self.server_queue.put((request, {'output': {'error': "Couldn't parse code." + str(e)}}))

        rows = process_ast(ast_data)
        cgen = CGenerator(ast_data)
        self.server_queue.put((request, {'output': { 'codeProps': ast_data.prop_map,
                'testResults': ast_data.results, 'lines': cgen.lines }}))

        # identify the failing test cases
        passed_tests = []
        failed_tests = {}
        for test, results in ast_data.results.items():
            if results['passed']:
                passed_tests.append(test)
            else:
                unknown_transitions = failed_tests[test] = []
                root_lex = self.test_configs[test]['root_lex']['transitions']
                for root_node_idx in rows[test]:
                    root_node = ast_data.prop_map[root_node_idx]
                    root_test_data = root_node['test_data'][test]
                    root_transitions = root_test_data['transitions']
                    if root_transitions == '<unk>' or root_transitions not in root_lex:
                        unknown_transitions.append(root_node_idx)
                        #root_test_data['unknown_transitions'] = True
        # TODO: this should be in preprocessing...
        inv_root_lex = {}
        for test in self.test_configs:
            inv_root_lex[test] = {}
            for transition, idx in self.test_configs[test]['root_lex']['transitions'].items():
                inv_root_lex[test][idx] = transition

        # TODO: order the tests
        replacements = {}
        for passed_test in passed_tests:
            root_lex = self.test_configs[passed_test]['root_lex']['transitions']
            tgroups = self.test_configs[passed_test]['transitions_groups']
            for failed_test, unk_nodes in failed_tests.items():
                for unk_node_idx in unk_nodes:
                    # this node wasn't touched by the passing test
                    unk_node = ast_data.prop_map[unk_node_idx]
                    if passed_test not in unk_node['test_data']: continue
                    root_transitions = unk_node['test_data'][passed_test]['transitions']
                    # the node also has an unknown transition in the passing test case
                    if root_transitions == '<unk>' or root_transitions not in root_lex: continue
                    root_trans_idx = str(root_lex[root_transitions])
                    tg = tgroups[root_trans_idx][failed_test]
                    self.replace_node(ast_data, rows, int(unk_node['node_num']))
                    run_tests(ast_data.ast, {failed_test: self.unit_tests[failed_test]})
                    # TODO: sort by count
                    # TODO: see if we can combine info across tests in any way?
                    for potential_trans_idx in tg:
                        if int(potential_trans_idx) not in inv_root_lex[failed_test]: continue
                        print(inv_root_lex[failed_test][int(potential_trans_idx)])
                        return
                    #    self.generate_tree(rows[failed_test][unk_node_idx]['false'], failed_test,
                    #                potential_trans_idx, 'false')
        # TODO: can also pull from the transitions models for guesses as to the correct transitions

        # TODO: see if plugging in a transitions group causes the program to finish successfully?



class ServerSocketProcess(ServerProcess):
    def __init__(self, beam_queue, beam_response_queue, server_queue, *args, **kwargs):
        self.beam_queue = beam_queue
        self.beam_response_queue = beam_response_queue
        self.server_queue = server_queue
        super().__init__(*args, **kwargs)

    # SENDING RESPONSE METHODS

    def send_msg(self, socket_data):
        while True:
            out = socket_data['pending_output'][0]
            if isinstance(out, dict):
                out = json.dumps(out).encode('latin-1') + b'\n\n'
            while len(out) > 0:
                try:
                    sent = socket_data['fd'].send(out)
                    out = out[sent:]
                except:
                    break
            if len(out) > 0:
                socket_data['pending_output'][0] = out
                return False
            socket_data['pending_output'].pop(0)
            if len(socket_data['pending_output']) == 0:
                return True

    # HANDLING INPUT METHODS
    # accepting input from the origin server (must take self, data, and mask)

    def server_connect(self, server):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setblocking(False)
        sock.connect_ex(server)
        self.server_queue.put(('server_connected', sock))

    def handle_socket(self, socket_data, mask, messageId):
        response = {'output':[]}

        if mask & selectors.EVENT_WRITE:
            assert len(socket_data['pending_output']) > 0
            self.send_msg(socket_data)
            # try again later, though should probably kill it at some point
            self.origin_pipe.put(('done', socket_data, mask))

        elif mask & selectors.EVENT_READ:
            log_print('Handling input')
            while True:
                try:
                    recv_data = socket_data['fd'].recv(4096)
                    if not recv_data:
                        if socket_data['type'] == 'server_socket':
                            # TODO: add in exponential backoff or something
                            #self.queue.put(('server_connect', socket_data['server']))
                            pass
                        return self.server_queue.put(('close', socket_data))

                    # TODO reject if this gets too big
                    socket_data['input'] += recv_data
                except:
                    # this is fine, it just meant we would have blocked
                    break

            try:
                socket_data['input'].index(b'\n\n')
            except ValueError:
                return self.server_queue.put(('done', socket_data, response, mask))

            input_ = socket_data['input'].split(b'\n\n')
            socket_data['input'] = input_.pop()
            for i in range(len(input_)):
                try:
                    input_[i] = json.loads(input_[i])
                except:
                    response['output'].append({'output': { 'error': 'Unable to parse input json'}})
                    input_[i] = None
                    continue

            self.server_queue.put(('done', socket_data, response, mask))

            for i in range(len(input_)):
                if input_[i] is None: continue
                if socket_data['type'] == 'client_socket':
                    input_[i]['opaque'] = {
                            'type': 'request',
                            'id': messageId,
                            'fd_id': socket_data['id'],
                            'opaque': input_[i]['opaque'] if 'opaque' in input_[i] else {}
                    }
                    #self.origin_pipe.send(('done', input_[i]['opaque'], {'direction': 'to_servers', 'output': input_[i]}))
                    self.beam_queue.put(('handle_input', input_[i]['opaque'], input_[i]['code']))

                #else:
                #    request = input_[i]['opaque']
                #    input_[i]['opaque'] = input_[i]['opaque']['opaque']
                #    if socket_data['type'] == 'server_socket':
                #        input_[i]['serverId'] = socket_data['serverId']
                #    data = {'output': input_[i], 'direction': 'to_client'}
                #    self.origin_pipe.put((request, 'done', data, selectors.EVENT_WRITE))

    def handle_beam_response(self):
        request, response = self.beam_response_queue.get()
        response['direction'] = 'to_client'
        response['output']['serverId'] = 0
        self.server_queue.put(('done', request, response, 0))


# TODO: clear out long requests / old fd_map stuff / etc
# does the tensorflow part group over time?
class Server(object):
    def __init__(self, args):
        self.fd_map = FDMap()

        self.model_configs = {}
        self.test_configs = {}
        for test in os.listdir(args.data_path):
            for root_idx in os.listdir(os.path.join(args.data_path, test)):
                rootdir = os.path.join(args.data_path, test, root_idx)
                if root_idx == 'config.json':
                    with open(rootdir, 'r') as f:
                        self.test_configs[test] = json.load(f)
                    continue
                for transitions in os.listdir(rootdir):
                    model_conf_path = os.path.join(rootdir, transitions, args.save_path, 'config.json')
                    if os.path.isfile(model_conf_path):
                        model_conf = get_dict(self.model_configs, test, root_idx)
                        with open(model_conf_path, 'r') as f:
                            model_conf[transitions] = json.load(f)

        personal_beam_queues = [Queue() for i in range(args.num_beam_processes)]
        model_queue = Queue()
        p = Process(target=ServerModelProcess, args=(self.model_configs, args.num_model_processes,
            personal_beam_queues, model_queue))
        #p.daemon = True
        p.start()

        beam_response_queue = Queue()
        self.fd_map.add(beam_response_queue._reader, 'beam_queue')
        beam_queue = Queue()
        p = Process(target=ServerBeamProcess, args=(self.test_configs, self.model_configs, beam_response_queue,
            personal_beam_queues, model_queue, beam_queue))
        #p.daemon = True
        p.start()

        socket_response_queue = Queue()
        self.fd_map.add(socket_response_queue._reader, 'socket_queue')
        self.socket_queue = ThreadQueue()
        for i in range(args.num_socket_processes):
            t = Thread(target=ServerSocketProcess, args=(beam_queue, beam_response_queue, socket_response_queue, self.socket_queue))
            t.start()

        lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        lsock.bind((args.host, args.port))
        lsock.setblocking(False)
        lsock.listen()

        args.servers = []
        for server in args.servers:
            self.socket_queue.put(('server_connect', server))

        self.fd_map.add(lsock, 'listener')
        log_print('listening on {}:{}'.format(args.host, args.port), 1)

        self.loop()


# TODO: disconnect after timeout, messages too long, etc.
# TODO: kill subprocesses?
    def loop(self):
        messageId = 1
        while True:
            events = self.fd_map.sel.select(timeout=None)
            for key, mask in events:
                fd_id = key.data['id']
                data = self.fd_map.get(fd_id)
                if data['type'] == 'listener':
                    conn, addr = key.fileobj.accept()
                    conn.setblocking(False)
                    self.fd_map.add(conn, 'client_socket')
                    log_print('accepted connection from {}'.format(addr, conn))

                elif data['type'] in ['server_socket', 'client_socket']:
                    # TODO: prioritize reading, since it can cancel ongoing requests?
                    # TODO: technically, we only need to unregister from reading, not writing?
                    self.fd_map.unset_state(fd_id, mask)
                    # send the output off to a server socket
                    self.socket_queue.put(('handle_socket', data, mask, messageId))
                    messageId += 1


                elif data['type'] == 'beam_queue':
                    self.socket_queue.put(('handle_beam_response',))

                elif data['type'] == 'socket_queue':
                    # TODO: do we have to worry about this recv at all? slow or failing?
                    # we don't about the ServerSocketProcess's socket / socket_data that came back
                    type_, fd_data, *args = self.socket_queue.get()
                    if type_ == 'server_connected':
                        self.fd_map.add(fd_data, 'server_socket')
                    elif type_ == 'done':
                        self.fd_map.update_state(fd_data, *args)
                    elif type_ == 'close':
                        self.fd_map.close_fd(fd_data['id'])
                    else:
                        print(type_)
                        assert False
                else:
                    print(data['type'])
                    assert False

    def shutdown(self):
        self.fd_map.shutdown()
