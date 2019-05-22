# https://realpython.com/python-sockets/#multi-connection-client-and-server
import sys
import argparse
import signal

# suppress tensorflow messages, finally
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)


# TF_CPP_MIN_LOG_LEVEL = '3'

from ..server import Server # type: ignore

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--port', help='port number (default 12347)', type=int, default=12347)
parser.add_argument('--host', help='host address (default '')', type=str, default='')
parser.add_argument('-s', '--save_path', help='model output directory (default "tmp")', default='tmp')
parser.add_argument('-d', '--data_path', help='model output directory (default "tmp")',
                    default='tmp')
parser.add_argument('-t', '--subtests', help='which tests to run', type=lambda s: s.split('|'), default=None)
# XXX XXX XXX there was some kind of error loading when this was 32
parser.add_argument('--num_model_processes', help='number of model processes (default 32)', type=int, default=16)
parser.add_argument('--num_beam_processes', help='number of beam processes (default len(test))', type=int, default=16)
parser.add_argument('--num_socket_processes', help='number of socket processes (technically threads) (default 8)',
                    type=int, default=8)
parser.add_argument('--servers', help='servers', type=lambda s: [tuple(x.split(':')) for x in s.split('|')],
                    default='')
args = parser.parse_args()

def main():
    server = Server(args)
    def sigint_handler(signum, frame):
        server.shut_down()
        sys.exit()

    signal.signal(signal.SIGINT, sigint_handler)
