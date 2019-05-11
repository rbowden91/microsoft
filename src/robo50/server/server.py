import sys
import os
import json
import selectors
import uuid
import socket
import selectors
import time

from collections import namedtuple
from queue import Queue as ThreadQueue
from threading import Lock as ThreadLock, Thread
from typing import Dict, Union
from multiprocessing import Process, Pipe, Lock, Queue

import numpy as np # type:ignore
import tensorflow as tf #type:ignore
import collections

from ..default_dict import get_dict
from ..model.config import joint_configs, dependency_configs, valid_dependencies #type:ignore
from ..wrangler.wrangle import finish_row, wrangle, process_ast #type:ignore
from .c_generator import CGenerator

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

        request_out = False
        if request['type'] == 'request':
            ctrs = ['model_ctr', 'total_models', 'test_ctr', 'total_tests'] if data['type'] == 'client_socket' else []
            if request['id'] not in data['requests']:
                for k in ctrs:
                    request[k] = 0 if k not in request else request[k]
                data['requests'][request['id']] = request
            else:
                request = data['requests'][request['id']]

            if not isinstance(response, list):
                response['output'] = [response['output']]

            for k in response:
                if k == 'output' or k == 'direction':
                    continue
                request[k] += response[k]

            if data['type'] == 'client_socket':
                if request['model_ctr'] == request['total_models'] and \
                        request['test_ctr'] == self.total_tests:

                    request_out = {'type': 'request', 'status': 'finished',
                                   'totalResponses': request['model_ctr'] + request['test_ctr'] + 1}
                    reqwuest_out['serverId'] = response
            else:
                assert data['type'] == 'server_socket'
                #if request['server_ctr'] == request['total_servers']:
                #    request_out = {'type': 'request', 'status': 'finished',
                #                            'totalServers': request['total_servers']}

        if data['type'] == 'client_socket' and 'direction' in response and response['direction'] == 'to_servers':
            datas = [self.map[s] for s in self.server_sockets]
        else:

            datas = [data]

        for data in datas:
            if data['closed']: continue

            if request_out:
                if 'opaque' in request:
                    request_out['opaque'] = request['opaque']
                data['output'].insert(0, request_out)

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

class ServerProcess(object):
    def __init__(self, input_queue, origin_pipe) -> None:
        self.input_queue = input_queue
        self.origin_pipe = origin_pipe
        self.loop()

    def loop(self):
        while True:
            handler, *args = self.input_queue.get()
            if handler == 'load_config':
                self.load_config(*args)
            elif handler == 'handle_input':
                output = self.handle_input(*args)
                if output is not False:
                    request_data, *args = args
                    self.origin_pipe.send((request_data, *output))
            elif handler == 'server_connect':
                output = self.server_connect(*args)
                self.origin_pipe.send(output)
            else:
                assert False

class ServerModelProcess(ServerProcess):
    def __init__(self, *args, **kwargs) -> None:
        self.test_config = {} #type:ignore
        self.sessions = [] #type: ignore

    def load_config(self, model_path, save_path):
        with open(os.path.join(model_path, 'config.json'), 'r') as f:
            test_conf = json.load(f)
        d = get_dict(self.test_config, test_conf['test'], test_conf['root_transitions_idx'])
        d[test_conf['transitions']] = test_conf
        with open(os.path.join(model_path, save_path, 'config.json'), 'r') as f:
            test_conf.update(json.load(f))

        # fix windows line endings
        test_conf['best_dir'] = test_conf['best_dir'].replace('\\', '/')
        log_print('Loaded model {} {} {}'.format(test_conf['test'], test_conf['root_idx'], test_conf['transitions'], 1)

    def get_session(self, test_conf):
        if 'session' not in test_conf:
            for (session, saver) in self.sessions:
                try:
                    saver.restore(session, os.path.join(test_conf['best_dir'], 'trimmed-model'))
                except:
                    continue
                break
            else:
                with tf.Graph().as_default():
                    saver = tf.train.import_meta_graph(os.path.join(test_conf['best_dir'], "trimmed-model.meta"))
                    session = tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0}))
                self.sessions.append((session, saver))
                test_conf['saver'] = saver
                test_conf['session'] = session
                return session
        else:
            # TODO: this might happen even if it has already been loaded
            test_conf['saver'].restore(session, os.path.join(test_conf['best_dir'], 'trimmed-model'))
            return test_conf['session']
        super().__init__(*args, **kwargs)

    def handle_input(self, request, rows, test,
                     root_node_idx, root_trans_idx, transitions):
        response = {'model_ctr': 1}
        test_conf = self.test_config[test][root_trans_idx][transitions]

        log_print('Running model {} {} {}'.format(test, root_trans_idx, transitions), 0)
        lexicon = test_conf['lexicon']
        node_nums = rows['forward-node_num']
        row = finish_row(rows, lexicon['token_to_index'])
        data_dict = {}
        for k in test_conf['features']:
            data_dict[test_conf['features'][k]] = [row[k]]

        session = self.get_session(test_conf)
        session.run(test_conf['tensor_iter'], data_dict)
        vals = session.run(test_conf['fetches'])

        codeProps = {}
        dependencyConfigs = {cdependency: True for cdependency in vals['dependency_configs']}
        for i in range(1, len(row['forward-self'])):
            idx = {'forward': row['forward-self'][i], 'reverse': row['reverse-self'][i]}
            props = self.gather_props(self.test_config[test][root_trans_idx][transitions], vals, data_dict, idx)
            codeProps[node_nums[i-1]] = {'test_data': { test: { 'model_results': { root_node_idx: { transitions: props } } } } };
        response['output'] = { 'codeProps': codeProps, 'dependencyConfigOptions': dependencyConfigs }
        return (response,)

    #for test in self.test_conf:
    #    for root_idx in rows[test]:
    #        if test == 'null' or len(rows[test][root_idx][True]) == 0: continue

    #        root_node = data.prop_map[root_idx]['props']
    #        root_props = root_node[test][root_idx][True]
    #        if not root_props['unknown_transitions']: continue
    #        root_props['suggested_trans_groups'] = collections.defaultdict(int)
    #        for test2 in data.nodes:
    #            if test2 == 'null' or test == test2: continue
    #            root_props2 = root_node[test2][root_idx][True]
    #            if not root_props2 or 'root_trans_idx' not in root_props2: continue
    #            tg = self.test_conf[test2][root_props2['root_trans_idx']][True]
    #            #if not tg: continue
    #            tg = tg['transitions_groups'][test]
    #            for correct_transitions in tg:
    #                root_props['suggested_trans_groups'][correct_transitions] += tg[correct_transitions]


    def gather_props(self, test_config, vals, data_dict, diridx):
        tc = test_config
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

class ServerTestProcess(ServerProcess):
    def __init__(self, model_processes, c_generator, *args, **kwargs):
        self.test_config = {}
        self.unit_tests = {}
        self.model_processes = model_processes
        self.model_process_map = {}
        self.c_generator = c_generator
        super().__init__(*args, **kwargs)

    def load_config(self, test, path, save_path):
        with open(os.path.join(path, 'config.json'), 'r') as f:
            self.test_config[test] = json.load(f)
        if test != 'null':
            self.unit_tests[test] = self.test_config[test]['unit_test']

        total_models = 0
        for root_idx in os.listdir(path):
            if root_idx == 'config.json': continue
            for transitions in os.listdir(os.path.join(path, root_idx)):
                model_path = os.path.join(path, root_idx, transitions)
                p = self.model_processes[total_models % len(self.model_processes)]
                total_models += 1
                p['queue'].put(('load_config', model_path, save_path))
                get_dict(self.model_process_map, test, root_idx)[transitions] = p

        log_print('Test {} initialized'.format(test), 1)

    def handle_input(self, request, code):
        response = {'total_models': 0, 'test_ctr': 1}
        try:
            ast_data = wrangle(code, tests=self.unit_tests, is_file=False)
        except Exception as e:
            response['output'] = {'error': "Couldn't parse code." + str(e)}
            return (response, )

        rows = process_ast(ast_data)

        # TODO: if test results were successful, no need to run the model?
        for test in self.model_process_map:
            root_lex = self.test_config[test]['root_lex']['transitions']
            mpm = self.model_process_map[test]
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
                    if root_trans_idx not in mpm or \
                            transitions not in mpm[root_trans_idx]:
                        # XXX this one isn't quite "unknown". We just didn't have enough test data???
                        root_test_data['unknown_transitions'] = True
                        continue
                    mpm[root_trans_idx][transitions]['queue'].put((
                        'handle_input',
                        request,
                        rows[test][root_node_idx][transitions],
                        test,
                        root_node_idx,
                        root_trans_idx,
                        transitions,
                        ))
                    response['total_models'] += 1

        response['output'] = { 'codeProps': ast_data.prop_map, 'testResults': ast_data.results }
        if self.c_generator:
            cgen = CGenerator(ast_data)
            response['output']['lines'] = cgen.lines
        return (response,)


class ServerSocketProcess(ServerProcess):
    def __init__(self, test_processes, *args, **kwargs):
        self.test_processes = test_processes
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
        return (sock, 'server_connected')

    def handle_input(self, fileobj_data, mask, messageId):
        fileobj  = fileobj_data['fd']
        if fileobj_data['type'] in [ServerTestProcess, ServerModelProcess]:
            ret = self.handle_server_pipe(fileobj_data, mask)
        else:
            assert fileobj_data['type'] in ('client_socket', 'server_socket')
            ret = self.handle_socket(fileobj_data, mask, messageId)
        if ret:
            return (*ret, mask)
        return False

    def handle_socket(self, socket_data, mask, messageId):
        response = {'output':[]}

        if mask & selectors.EVENT_WRITE:
            assert len(socket_data['pending_output']) > 0
            self.send_msg(socket_data)
            # try again later, though should probably kill it at some point
            return ('done', response)

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
                        return ('close',)

                    # TODO reject if this gets too big
                    socket_data['input'] += recv_data
                except:
                    # this is fine, it just meant we would have blocked
                    break

            try:
                socket_data['input'].index(b'\n\n')
            except ValueError:
                return ('done', response)

            input_ = socket_data['input'].split(b'\n\n')
            socket_data['input'] = input_.pop()
            for i in range(len(input_)):
                try:
                    input_[i] = json.loads(input_[i])
                except:
                    response['output'].append({'output': { 'error': 'Unable to parse input json'}})
                    input_[i] = None
                    continue

            self.origin_pipe.send((socket_data, 'done', response, mask))

            for i in range(len(input_)):
                if input_[i] is None: continue
                if socket_data['type'] == 'client_socket':
                    input_[i]['opaque'] = {
                            'type': 'request',
                            'id': messageId,
                            'fd_id': socket_data['id'],
                            'opaque': input_[i]['opaque'] if 'opaque' in input_[i] else {}
                    }
                    self.origin_pipe.send((input_[i]['opaque'], 'done', {'direction': 'to_servers', 'output': input_[i]}, selectors.EVENT_WRITE))
                    for tp in self.test_processes:
                        tp['queue'].put(('handle_input', input_[i]['opaque'], input_[i]['code']))
                else:
                    request = input_[i]['opaque']
                    input_[i]['opaque'] = input_[i]['opaque']['opaque']
                    if socket_data['type'] == 'server_socket':
                        input_[i]['serverId'] = socket_data['serverId']
                    data = {'output': input_[i], 'direction': 'to_client'}
                    self.origin_pipe.send((request, 'done', data, selectors.EVENT_WRITE))
            return False

    def handle_server_pipe(self, server_pipe_data, mask):
        assert mask & selectors.EVENT_READ
        server_pipe = server_pipe_data['fd']
        try:
            request, response = server_pipe.recv()
        except Exception as e:
            print(e, server_pipe_data['type'])
            assert False
            # TODO: a pipe socket shouldn't ever be closed??
            return ('close',)
        response['direction'] = 'to_client'
        response['output']['serverId'] = 0
        self.origin_pipe.send((request, 'done', response, mask))
        return ('done',{})


# TODO: clear out long requests / old fd_map stuff / etc
# does the tensorflow part group over time?
class Server(object):
    def __init__(self, args):
        self.fd_map = FDMap()

        self.model_processes = self.spawn_processes(ServerModelProcess, False, False, args.num_model_processes)

        total_tests = 0
        self.test_processes = []
        for test in os.listdir(args.data_path):
            if args.subtests is None or test not in args.subtests: continue
            if args.num_test_processes is None or len(self.test_processes) < args.num_test_processes:
                self.test_processes.extend(self.spawn_processes(ServerTestProcess, False, False, 1, self.model_processes, total_tests == 0))
            tp = self.test_processes[total_tests % len(self.test_processes)]
            tp['queue'].put(('load_config', test, os.path.join(args.data_path, test), args.save_path))
            total_tests += 1
        # XXX very hackish...
        self.fd_map.total_tests = total_tests

        self.socket_processes = self.spawn_processes(ServerSocketProcess, True, True, args.num_socket_processes, self.test_processes)

        lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        lsock.bind((args.host, args.port))
        lsock.setblocking(False)
        lsock.listen()

        args.servers = []
        for server in args.servers:
            self.socket_processes[0]['queue'].put(('server_connect', server))

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

                elif data['type'] in ['server_socket', ServerTestProcess, ServerModelProcess, 'client_socket']:
                    # TODO: prioritize reading, since it can cancel ongoing requests?
                    # TODO: technically, we only need to unregister from reading, not writing?
                    self.fd_map.unset_state(fd_id, mask)
                    # send the output off to a server socket
                    self.socket_processes[0]['queue'].put(('handle_input', data, mask, messageId))
                    messageId += 1


                elif data['type'] == ServerSocketProcess:
                    # TODO: do we have to worry about this recv at all? slow or failing?
                    # we don't about the ServerSocketProcess's socket / socket_data that came back
                    fd_data, type_, *args = key.fileobj.recv()
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

    def spawn_processes(self, process_type, is_thread, share_queue, num_processes, *args):
        # also, is there a ThreadPipe of some kind?
        input_q = None
        processes = []
        for i in range(num_processes):
            # TODO: should this pipe be closed???
            parent_pipe, child_pipe = Pipe(False)
            #self.fd_map.add_pipe(parent_pipe, child_pipe, process_type)
            self.fd_map.add(parent_pipe, process_type)
            if is_thread:
                spawn = Thread
                new_q = ThreadQueue #type:ignore
            else:
                spawn = Process
                new_q = Queue
            if input_q is None or not share_queue:
                input_q = new_q()
            p = spawn(target=process_type, args=(*args, input_q, child_pipe))
            p.daemon = True
            p.start()
            processes.append({'process': p, 'queue': input_q, 'pipe': parent_pipe})
        return processes
