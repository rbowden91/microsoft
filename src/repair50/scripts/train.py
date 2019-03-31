# Largely based on Tree-Structured Decoding with Doublyrecurrent Neural Networks
# (https://openreview.net/pdf?id=HkYhZDqxg)
# TODO: validate that the joints are valid given directions

# TODO:
# * Add back in attr to predictions (at least for nodes that have them?)
# TODO (potential):
# * Could eventually do something with attention over the children (for the separate RNN that predicts the parent)
# * Have a decoder for the above RNN, or just give the output directly from this RNN to h_pred?
# * Include placement of node within tree
# * Some way of incorporating the Child RNN into the TreeRNN so that arbitrary children can appear?

import os, sys
import argparse
import time
import json
import math
import collections

from multiprocessing import Process, Queue
import queue

import numpy as np
import tensorflow as tf
from ..model.model import TRNNModel, TRNNJointModel
from ..default_dict import data_dict

#LargeConfig = TrainConfig(
#  init_scale = 0.04,
#  max_grad_norm = 10,
#  num_layers = 2,
#  hidden_size = 1500,
#  epochs = 55,
#  drop_prob = 0.35,
#  batch_size = 20,
#)

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--force', help='blows away the current save directory', action='store_true')
parser.add_argument('-s', '--save_path', help='model output directory (default "tmp")', default='tmp')
parser.add_argument('-p', '--data_path', help='directory to find preprocessed data (default "./data/vig10000")', default='./data/vig10000')
parser.add_argument('-b', '--batch_size', help='batch size', type=int)
parser.add_argument('-d', '--dependency_configs', help='forward/reverse, etc.', type=lambda s: s.split(), default='d2')
parser.add_argument('-t', '--subtests', help='which tests to run', type=lambda s: s.split(), default=None)
parser.add_argument('--num_processes', help='number of concurrent processes (default 16)', type=int, default=16)
parser.add_argument('-j', '--joint_configs', help='both, etc.', type=lambda s: s.split(), default=[])
parser.add_argument('-e', '--epochs', help='how many epochs', type=int)
parser.add_argument('-w', '--summarywriter', help='turns on summary writer', action='store_true')
parser.add_argument('-c', '--cpu', help='don\'t use gpu', action='store_true')
parser.add_argument('--profile', help='enable graph profiling (default false)', action="store_true")
parser.add_argument('--checkpoint', help='attempts to resume training from a checkpoint file (default false)', action="store_true")
parser.add_argument('--init_scale', help='', type=float, default=0.1)
parser.add_argument('--max_grad_norm', help='', type=int, default=5)
parser.add_argument('--num_layers', help='', type=int, default=2)
parser.add_argument('--hidden_size', help='', type=int, default=400)
parser.add_argument('--drop_prob', help='', type=float, default=1.0)
#"learning_rate" : 1.0,
#"max_epoch" : 4,
#"max_max_epoch" : 4,
#"lr_decay" : 0.5,

args = parser.parse_args()

logging = tf.logging

class Trainer(object):

    def __init__(self, config, train_config):
        self.config = config
        self.train_config = train_config
        self.test = config['test']
        self.transitions = config['transitions']
        self.root_idx = config['root_idx']

        self.graph = tf.Graph()
        with self.graph.as_default():
            initializer = tf.random_uniform_initializer(-train_config['init_scale'],
                                                        train_config['init_scale'])
            self.models = {'dependency_configs': {}, 'joint_configs': {}}
            with tf.variable_scope("TRNNModel/{}/{}/{}".format(self.transitions, 'test' if True else self.test, self.root_idx), reuse=None, initializer=initializer):
                rows = None
                for i in train_config['dependency_configs']:
                    self.models['dependency_configs'][i] = TRNNModel(
                        i, config, self.transitions == 'true', rows=rows
                    )
                    if rows is None:
                        rows = self.models['dependency_configs'][i].rows
                        self.iter_model = self.models['dependency_configs'][i]
                for i in train_config['joint_configs']:
                    self.models['joint_configs'][i] = TRNNJointModel(
                        self.models['dependency_configs'],
                        i, config, self.transitions == 'true', rows=rows
                    )


            # save stuff to be used later in inference
            self.saver = tf.train.Saver()

            sess_config = tf.ConfigProto(device_count = {'GPU': (0 if args.cpu else 1)})
            self.session = tf.Session(config=sess_config)

            ckpt = tf.train.get_checkpoint_state(config['checkpoint_dir'])
            if ckpt:
                if not ckpt.model_checkpoint_path:
                    print('invalid checkpoint?')
                    sys.exit(1)
                elif args.checkpoint:
                    self.saver.restore(self.session, ckpt.model_checkpoint_path)
                    print('restoring?')
                elif not args.force:
                    print('It looks like data is already here. Run the command again with either -f or --checkpoint.')
                    sys.exit(1)
                else:
                    self.session.run(tf.global_variables_initializer())
            else:
                self.session.run(tf.global_variables_initializer())


            # need to get global step from models...
            #global_step = session.run(tf.train.get_or_create_global_step())
            #self.epoch = np.asscalar(global_step // self.num_files['train'])
            #self.epoch_step = (global_step - self.epoch * self.num_files['train']) // config['batch_size']
            self.writer = tf.summary.FileWriter(config['log_dir'], graph=tf.get_default_graph()) #\ if self.train_config['summarywriter']
            self.epoch = 0

    def run_epoch(self):
        with self.graph.as_default():
            config = self.config
            train_perplexity = self.run_epoch_mode('train', verbose=True)
            valid_perplexity = self.run_epoch_mode('valid')

            config['train_perplexities'].append(train_perplexity)
            config['valid_perplexities'].append(valid_perplexity)
            if config['best_validation'] is None or valid_perplexity < config['best_validation']:
                self.saver.save(self.session, os.path.join(config['best_dir'], 'model'))#, global_step)
                config['best_validation'] = valid_perplexity
                config['best_validation_epoch'] = self.epoch
                with open(config['model_config_file'], 'w') as f:
                    json.dump(config, f)
            self.epoch += 1

            if valid_perplexity - 1 < .0001: return True

            if config['best_validation'] < valid_perplexity:
                ckpt = tf.train.get_checkpoint_state(config['best_dir'])
                assert ckpt and ckpt.model_checkpoint_path
                self.saver.restore(self.session, ckpt.model_checkpoint_path)
                return True

            return False

    def finish(self):
        with self.graph.as_default():
            config = self.config

            test_perplexity = self.run_epoch_mode('test')
            config['test_perplexity'] = test_perplexity

            # store the placeholder/op/fetch tensor names in our config
            outputs = {}
            for i in self.models:
                outputs[i] = {}
                for j in self.models[i]:
                    outputs[i][j] = self.models[i][j].output_tensor_names()
            config['models'] = outputs

            # we have to find the model that we can feed...
            config['fetches'] = fetches = {} # type: ignore
            config['initials'] = initials = {} # type: ignore
            for d in config['models']:
                fetches[d] = {}
                initials[d] = {}
                for i in config['models'][d]:
                    fetches[d][i] = config['models'][d][i]['fetches']
                    initials[d][i] = config['models'][d][i]['initials']

                    for j in config['models'][d][i]['placeholders']:
                        if 'features' == j:
                            config['features'] = config['models'][d][i]['placeholders'][j]
                            config['tensor_iter'] = config['models'][d][i]['ops']['tensor_iter']

            # TODO: these can be run and saved during training time
            initial_vals = self.session.run(config['initials'])
            config['cells'] = {} # type:ignore
            for dconfig in initial_vals:
                config['cells'][dconfig] = {}
                for cdependency in initial_vals[dconfig]:
                    config['cells'][dconfig][cdependency] = {}
                    for dependency in initial_vals[dconfig][cdependency]:
                        config['cells'][dconfig][cdependency][dependency] = {}
                        for k in initial_vals[dconfig][cdependency][dependency]:
                            config['cells'][dconfig][cdependency][dependency][k] = {
                                    0: { 0: { 0: initial_vals[dconfig][cdependency][dependency][k][0].tolist() } }
                            }
            with open(config['model_config_file'], 'w') as f:
                json.dump(config, f)

            self.session.close()


    #def run_profiler(self):
    #    run_metadata = tf.RunMetadata()
    #    vals = self.session.run(fetches, feed_dict,
    #                    options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
    #                    run_metadata=run_metadata)
    #    writer.add_run_metadata(run_metadata, 'run metadata')
    #    from tensorflow.python.client import timeline
    #    ft = timeline.Timeline(run_metadata.step_stats)
    #    chrome_trace = ft.generate_chrome_trace_format()
    #    with open('timeline_01.json', 'w') as f:
    #        f.write(chrome_trace)
    #    sys.exit(0)

    def run_models(self, is_joint, mode, verbose=False):
        start_time = time.time()

        feed_dict = {}

        c = 'joint_configs' if is_joint else 'dependency_configs'
        models = self.models[c]
        train_ops = []
        fetch_ops = []
        fetches = {}
        for d in models:
            fetches[d] = models[d].fetches['loss']
            if mode == 'train':
                train_ops.append(models[d].ops['train'])
            for k in fetches[d]:
                tf.summary.scalar(c + ' ' + d + ' loss', fetches[d][k])
                fetch_ops.append(fetches[d][k])
        if len(fetch_ops) == 0:
            return 0
        summary_op = tf.summary.merge_all()

        total_loss = {}
        epoch_step = 0
        while epoch_step < self.epoch_size:
            with tf.control_dependencies(fetch_ops):
                summary, vals, _ = self.session.run([summary_op, fetches, train_ops], feed_dict)
                self.writer.add_summary(summary, epoch_step * self.batch_size)

            if self.train_config['profile'] and mode == 'train':
                self.run_profiler()

            if verbose:
                print("\nepoch %d %.02f%%:\tspeed: %.03f fps" %
                    (self.epoch + 1, (epoch_step + 1) * 100.0 / self.epoch_size, (epoch_step + 1) * self.batch_size / (time.time() - start_time)))
            for d in vals:
                if d not in total_loss: total_loss[d] = {}
                for k in vals[d]:
                    if k not in total_loss[d]: total_loss[d][k] = 0
                    loss = np.asscalar(vals[d][k])
                    total_loss[d][k] += loss;
                    if verbose:
                        print("\t%s %s %s %s %s loss: %.3f\tperplexity: %.3f\taverage perplexity: %.3f" %
                            (self.test, self.root_idx, self.transitions, d, k, loss, np.exp(loss), np.exp(total_loss[d][k] / (epoch_step + 1))))
            epoch_step += 1
        return total_loss


    def run_epoch_mode(self, mode, verbose=False):
        self.batch_size = self.train_config['batch_size'] if mode == 'train' else 1
        self.session.run(self.iter_model.ops['batch_size_update'],
                        feed_dict={ self.iter_model.placeholders['new_batch_size'] : self.batch_size })
        total_perplexity = 0
        #if root_idx != '0': continue
        self.epoch_size = self.config['dataset_sizes'][mode] // self.batch_size
        filename = os.path.join(args.data_path, 'tests', self.test, self.root_idx, self.transitions, mode + '_data.tfrecord')
        self.session.run(self.iter_model.ops['file_iter'],
                        feed_dict={ self.iter_model.placeholders['filename'] : filename })

        total_loss = self.run_models(False, mode, verbose)
        # disregard the loss from joints for now for validation
        self.run_models(True, mode, verbose)

        # for now, save after very epoch
        self.saver.save(self.session, os.path.join(self.config['checkpoint_dir'], 'model'))#, global_step)

        # TODO: how do perplexities add?
        for d in total_loss:
            for k in total_loss[d]:
                perplexity = np.exp(total_loss[d][k] / self.epoch_size)
                total_perplexity += np.asscalar(perplexity)
                print("Epoch %d: %s %s %s %s %s %s perplexity: %.3f" %
                        (self.epoch+(1 if mode != 'test' else 0), self.config['test'],
                        self.root_idx, self.transitions, mode, d, k, perplexity))
        return total_perplexity

def process_queue(q):
    while True:
        try:
            conf, config = q.get(True, 5)
        except queue.Empty:
            return

        trainer = Trainer(conf, config)
        for i in range(config['epochs']):
            if trainer.run_epoch():
                break
        trainer.finish()

def main():
    with open(os.path.join(args.data_path, 'config.json')) as f:
        config = json.load(f)
    # Note that at this point, the training, validation, and test data have already been split up.
    # So, we preserve the ratios between them.
    config.update(vars(args))

    save_path = os.path.join(args.data_path, args.save_path)

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(config, f)

    q = Queue(maxsize=0)

    processes = []
    for i in range(args.num_processes):
        p = Process(target=process_queue, args=(q,))
        p.daemon = True
        p.start()
        processes.append(p)

    for test in config['tests']:
        if config['subtests'] is not None and test not in config['subtests']: continue
        for root_idx in config['tests'][test]:
            for transitions in config['tests'][test][root_idx]:
                if config['tests'][test][root_idx][transitions] is False: continue
                path = os.path.join(save_path, 'tests', test, root_idx, transitions)

                best_dir = os.path.join(path, 'best/')
                model_config_file = os.path.join(best_dir, 'config.json')

                if args.checkpoint and os.path.isfile(model_config_file):
                    with open(model_config_file) as f:
                        conf = json.load(f)
                else:
                    # TODO: delete save_path?
                    data_path = os.path.join(args.data_path, 'tests', test, root_idx, transitions)
                    with open(os.path.join(data_path, 'config.json')) as f:
                        conf = json.load(f)

                    for k in ['num_layers', 'max_grad_norm', 'hidden_size']:
                        conf[k] = config[k]

                    checkpoint_dir = os.path.join(path, 'checkpoints/')
                    log_dir = os.path.join(path, 'log/')

                    # TODO: the final train perplexity should go over the epoch again, without backprop
                    conf.update({
                        'train_perplexities': [],
                        'valid_perplexities': [],
                        'best_validation': None,
                        'best_validation_epoch': None,
                        'test': test,
                        'transitions': transitions,
                        'root_idx': root_idx,
                        'checkpoint_dir': checkpoint_dir,
                        'best_dir': best_dir,
                        'log_dir': log_dir,
                        'model_config_file': model_config_file,
                    })

                    os.makedirs(checkpoint_dir, exist_ok=True)
                    os.makedirs(log_dir, exist_ok=True)
                    os.makedirs(best_dir, exist_ok=True)

                    with open(model_config_file, 'w') as f:
                        json.dump(conf, f)
                q.put((conf, config))
    for p in processes:
        p.join()
