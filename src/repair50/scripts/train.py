# Largely based on Tree-Structured Decoding with Doublyrecurrent Neural Networks
# (https://openreview.net/pdf?id=HkYhZDqxg)

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

import numpy as np
import tensorflow as tf
from ..model.model import TRNNModel, TRNNJointModel

from .train_config import *

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--force', help='blows away the current save directory', action='store_true')
parser.add_argument('-m', '--model', help='linear or ast (default "linear")', default='linear')
parser.add_argument('-s', '--save_path', help='model output directory (default "tmp")', default='tmp')
parser.add_argument('-p', '--data_path', help='directory to find preprocessed data (default "./data/vig10000")', default='./data/vig10000')
parser.add_argument('-b', '--batch_size', help='batch size', type=int)
parser.add_argument('-c', '--config', help='test/small/medium/large, (default "small")', default='small')
parser.add_argument('-n', '--num_files', help='number of files to train on (default, all files)', type=int)
parser.add_argument('-d', '--dependency_configs', help='forward/reverse, etc.', type=lambda s: s.split())
parser.add_argument('-j', '--joint_configs', help='both, etc.', type=lambda s: s.split(), default=[])
parser.add_argument('-e', '--epochs', help='how many epochs', type=int)
parser.add_argument('-w', '--summarywriter', help='turns on summary writer', action='store_true')
#parser.add_argument('--profile', help='enable graph profiling (default false)', action="store_true")
parser.add_argument('--checkpoint', help='attempts to resume training from a checkpoint file (default false)', action="store_true")

args = parser.parse_args()

logging = tf.logging

def get_config():
    if args.config == "small":
        return SmallConfig
    elif args.config == "medium":
        return MediumConfig
    elif args.config == "large":
        return LargeConfig
    elif args.config == "test":
        return TestConfig
    else:
        raise ValueError("Invalid size: %s", args.config)


class Trainer(object):

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
        #for c in self.models:
        #    for d in self.models[c]:
        #        feed_dict[self.models[c][d].placeholders['is_inference']] = False

        c = 'joint_configs' if is_joint else 'dependency_configs'
        models = self.models[c]
        train_ops = []
        fetch_ops = []
        fetches = {}
        for d in models:
            fetches[d] = models[d].fetches['loss']
            #if models[d] == self.iter_model:
            #    print('woo')
            #else:
            if mode == 'train':
                train_ops.append(models[d].ops['train'])
            for k in fetches[d]:
                tf.summary.scalar(c + ' ' + d + ' loss', fetches[d][k])
                fetch_ops.append(fetches[d][k])
        if len(fetch_ops) == 0:
            return 0
        summary_op = tf.summary.merge_all()

        total_loss = {}
        self.epoch_size = self.num_files[mode] // self.batch_size
        epoch_step = 0
        #epoch_step = self.epoch_step
        #self.epoch_step = 0
        while epoch_step < self.epoch_size:
            with tf.control_dependencies(fetch_ops):
                #fetches = tf.tuple(fetches, 'barrier', [train_ops])
                summary, vals, _ = self.session.run([summary_op, fetches, train_ops], feed_dict)
                self.writer.add_summary(summary, epoch_step * self.batch_size)

            #if self.config['profile'] and mode == 'train':
            #    self.run_profiler()

            if verbose:
                print("\n%.02f%%:\tspeed: %.03f fps" %
                    ((epoch_step + 1) * 100.0 / self.epoch_size, (epoch_step + 1) * self.batch_size /
                                                            (time.time() - start_time)))
            for d in vals:
                if d not in total_loss: total_loss[d] = {}
                for k in vals[d]:
                    if k not in total_loss[d]: total_loss[d][k] = 0
                    loss = np.asscalar(vals[d][k])
                    total_loss[d][k] += loss;
                    if verbose:
                        print("\t%s %s loss: %.3f\tperplexity: %.3f\taverage perplexity: %.3f" %
                            (d, k, loss, np.exp(loss), np.exp(total_loss[d][k] / (epoch_step + 1))))
            epoch_step += 1
        return total_loss


    def run_epoch(self, mode, verbose=False):
        self.batch_size = self.config['batch_size'] if mode == 'train' else 1
        filename = os.path.join(args.data_path, self.config['model'] + '_' + mode + '.tfrecord')
        self.session.run(self.iter_model.ops['batch_size_update'],
                         feed_dict={ self.iter_model.placeholders['new_batch_size'] : self.batch_size })
        self.session.run(self.iter_model.ops['file_iter'],
                         feed_dict={ self.iter_model.placeholders['filename'] : filename })

        total_loss = self.run_models(False, mode, verbose)
        # disregard the loss from joints for now for validation
        self.run_models(True, mode, verbose)

        # for now, save after very epoch
        self.saver.save(self.session, os.path.join(self.config['checkpoint_dir'], 'model'))#, global_step)

        # TODO: how do perplexities add?
        total_perplexity = 0
        for d in total_loss:
            for k in total_loss[d]:
                perplexity = np.exp(total_loss[d][k] / self.epoch_size)
                total_perplexity += np.asscalar(perplexity)
                print("Epoch %d: %s %s %s perplexity: %.3f" %
                        (self.epoch+(1 if mode != 'test' else 0), mode, d, k, perplexity))
        return total_perplexity

    def __init__(self, config):
        self.config = config

        self.num_files = {
            # we should do this in a better way (by actually calling "skip" on the dataset...)
            'all': config['num_files'],
            'train': math.floor(config['num_files'] * config['train_fraction']),
            'valid': math.floor(config['num_files'] * config['valid_fraction']),
            'test': math.floor(config['num_files'] *
                    (1 - config['valid_fraction'] - config['train_fraction']))
        }

        with tf.Graph().as_default():
            initializer = tf.random_uniform_initializer(-config['init_scale'],
                                                        config['init_scale'])
            self.models = {'dependency_configs': {}, 'joint_configs': {}}
            with tf.variable_scope("TRNNModel", reuse=None, initializer=initializer):
                rows = None
                for i in config['dependency_configs']:
                    self.models['dependency_configs'][i] = TRNNModel(
                        config['model'], i, config['label_size'], config['attr_size'], config['hidden_size'],
                        config['features'], config['num_layers'], config['max_grad_norm'],
                        rows=rows
                    )
                    if rows is None:
                        rows = self.models['dependency_configs'][i].rows
                        self.iter_model = self.models['dependency_configs'][i]
                for i in config['joint_configs']:
                    self.models['joint_configs'][i] = TRNNJointModel(
                        self.models['dependency_configs'],
                        config['model'], i, config['label_size'], config['attr_size'], config['hidden_size'],
                        config['features'], config['num_layers'], config['max_grad_norm'],
                        rows=rows
                    )


            # save stuff to be used later in inference
            self.saver = tf.train.Saver()

            with tf.Session() as session:
                self.session = session


                ckpt = tf.train.get_checkpoint_state(config['checkpoint_dir'])
                if ckpt:
                    if not ckpt.model_checkpoint_path:
                        print('invalid checkpoint?')
                        sys.exit(1)
                    elif args.checkpoint:
                        self.saver.restore(session, ckpt.model_checkpoint_path)
                        print('restoring?')
                    elif not args.force:
                        print('It looks like data is already here. Run the command again with either -f or --checkpoint.')
                        sys.exit(1)
                    else:
                        session.run(tf.global_variables_initializer())
                else:
                    session.run(tf.global_variables_initializer())


                # need to get global step from models...
                #global_step = session.run(tf.train.get_or_create_global_step())
                #self.epoch = np.asscalar(global_step // self.num_files['train'])
                #self.epoch_step = (global_step - self.epoch * self.num_files['train']) // config['batch_size']
                self.writer = tf.summary.FileWriter(self.config['log_dir'], graph=tf.get_default_graph()) #\ if self.config['summarywriter']
                self.epoch = 0
                while self.epoch < config['epochs']:
                    train_perplexity = self.run_epoch('train', verbose=True)
                    config['train_perplexities'].append(train_perplexity)
                    valid_perplexity = self.run_epoch('valid')
                    config['valid_perplexities'].append(valid_perplexity)

                    if config['best_validation'] is None or valid_perplexity <= config['best_validation']:
                        self.saver.save(session, os.path.join(config['best_dir'], 'model'))#, global_step)
                        config['best_validation'] = valid_perplexity
                        config['best_validation_epoch'] = self.epoch
                        with open(config['model_config_file'], 'w') as f:
                            json.dump(config, f)
                    self.epoch += 1

                if config['best_validation'] < valid_perplexity:
                    ckpt = tf.train.get_checkpoint_state(config['best_dir'])
                    assert ckpt and ckpt.model_checkpoint_path
                    self.saver.restore(session, ckpt.model_checkpoint_path)

                test_perplexity = self.run_epoch('test')
                config['test_perplexity'] = test_perplexity

            # store the placeholder/op/fetch tensor names in our config
            outputs = {}
            for i in self.models:
                outputs[i] = {}
                for j in self.models[i]:
                    outputs[i][j] = self.models[i][j].output_tensor_names()
            config['models'] = outputs
            with open(config['model_config_file'], 'w') as f:
                json.dump(config, f)
            session.close()

def main():
    args.data_path = os.path.join(os.getcwd(), args.data_path)

    save_path = os.path.join(args.data_path, args.model, args.save_path)
    checkpoint_dir = os.path.join(save_path, 'checkpoints/')
    best_dir = os.path.join(save_path, 'best/')
    log_dir = os.path.join(save_path, 'log/')
    model_config_file = os.path.join(best_dir, 'config.json')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    if args.checkpoint and os.path.isfile(model_config_file):
        with open(model_config_file) as f:
            config = json.load(f)
    else:
        # we don't want to override the 'config' defaults
        if args.batch_size is None:
            del(args.batch_size)
        if args.epochs is None:
            del(args.epochs)

        if args.dependency_configs is None:
            args.dependency_configs = ['forward', 'reverse'] if args.model == 'linear' else ['d2']

        # TODO: validate that the joints are valid given directions

        # load in all the data
        with open(os.path.join(args.data_path, 'config.json')) as f:
            config = json.load(f)
        with open(os.path.join(args.data_path, args.model + '_lexicon.json')) as f:
            token_ids = json.load(f)

        # Note that at this point, the training, validation, and test data have already been split up.
        # So, we preserve the ratios between them.
        if args.num_files is not None:
            config['num_files'] = min(args.num_files, config['num_files'])
        del(args.num_files)

        # TODO: fix this
        train_config = get_config()
        for i in train_config._fields:
            config[i] = getattr(train_config, i)
        config.update(vars(args))
        config['label_size'] = len(token_ids['label'])
        config['attr_size'] = len(token_ids['attr'])
        # TODO: the final train perplexity should go over the epoch again, without backprop
        config['train_perplexities'] = []
        config['valid_perplexities'] = []
        config['best_validation'] = None
        config['best_validation_epoch'] = None

        with open(model_config_file, 'w') as f:
            json.dump(config, f)

    config.update({
        'save_path': save_path,
        'checkpoint_dir': checkpoint_dir,
        'best_dir': best_dir,
        'log_dir': log_dir,
        'model_config_file': model_config_file
    })

    trainer = Trainer(config)
