# Largely based on Tree-Structured Decoding with Doublyrecurrent Neural Networks
# (https://openreview.net/pdf?id=HkYhZDqxg)

# TODO:
# * Add back in attr to predictions (at least for nodes that have them?)
# TODO (potential):
# * Could eventually do something with attention over the children (for the separate RNN that predicts the parent)
# * Have a decoder for the above RNN, or just give the output directly from this RNN to h_pred?
# * Attention over all the U_dependencies?
# * Include placement of node within tree
# * Some way of incorporating the Child RNN into the TreeRNN so that arbitrary children can appear?

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import inspect
import time
import json
from pprint import pprint
import os, sys
import math

import numpy as np
import tensorflow as tf
from model import TRNNModel

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='linear or ast (default "linear")', default='linear')
parser.add_argument('-s', '--save_path', help='model output directory (default "tmp")')
parser.add_argument('-p', '--data_path', help='directory to find preprocessed data (default "./data/vig10000")', default='./data/vig10000')
parser.add_argument('-b', '--batch_size', help='batch size', type=int)
parser.add_argument('-c', '--config', help='test/small/medium/large, (default "small")', default='small')
parser.add_argument('-n', '--num_files', help='number of files to train on (default, all files)', type=int)
parser.add_argument('-d', '--dependencies', help='forward/reverse | left_sibling/parent/left_prior | '
                                           'children/right_sibling/right_prior',
                                            type=lambda s: s.split())
parser.add_argument('--profile', help='enable graph profiling (default false)', action="store_true")
parser.add_argument('--checkpoint', help='attempts to resume training from a checkpoint file (default false)', action="store_true")

args = parser.parse_args()

logging = tf.logging

SmallConfig = {
  "init_scale" : 0.1,
  "learning_rate" : 1.0,
  "max_grad_norm" : 5,
  "num_layers" : 2,
  "num_steps" : 20, # this isn't used at all in this file, since we aren't doing any truncated backpropagation
  "hidden_size" : 200,
  "embedding_size" : 200,
  "max_epoch" : 4,
  "max_max_epoch" : 6,
  "drop_prob" : 1.0,
  "lr_decay" : 0.5,
  "batch_size" : 20,
}

MediumConfig = {
  "init_scale" : 0.05,
  "learning_rate" : 1.0,
  "max_grad_norm" : 5,
  "num_layers" : 2,
  "num_steps" : 35,
  "hidden_size" : 650,
  "embedding_size" : 650,
  "max_epoch" : 6,
  "max_max_epoch" : 39,
  "drop_prob" : 0.5,
  "lr_decay" : 0.8,
  "batch_size" : 20,
}

LargeConfig = {
  "init_scale" : 0.04,
  "learning_rate" : 1.0,
  "max_grad_norm" : 10,
  "num_layers" : 2,
  "num_steps" : 35,
  "hidden_size" : 1500,
  "embedding_size" : 1500,
  "max_epoch" : 14,
  "max_max_epoch" : 55,
  "drop_prob" : 0.35,
  "lr_decay" : 1 / 1.15,
  "batch_size" : 20,
}

TestConfig = {
  "init_scale" : 0.1,
  "learning_rate" : 1.0,
  "max_grad_norm" : 1,
  "num_layers" : 1,
  "num_steps" : 2,
  "hidden_size" : 4,
  "embedding_size" : 4,
  "max_epoch" : 1,
  "max_max_epoch" : 1,
  "drop_prob" : 1.0,
  "lr_decay" : 0.5,
  "batch_size" : 20,
}

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


class Trainer():


    def run_epoch(self, mode, lr_decay = None, verbose=False):
        model = self.model

        """Runs the model on the given data."""
        start_time = time.time()

        batch_size = model.config['batch_size'] if mode == 'train' else 1
        filename = os.path.join(args.data_path, model.config['model'] + '_' + mode + '.tfrecord')
        self.session.run(model.ops['batch_size_update'], feed_dict={ model.placeholders['new_batch_size'] : batch_size })
        self.session.run(model.ops['file_iter'], feed_dict={ model.placeholders['filename'] : filename })
        if lr_decay is not None:
            self.session.run(model.ops['lr_update'], feed_dict={model.placeholders['new_lr']: lr_decay})

        fetches = {
            "loss": model.fetches['loss'],
        }
        feed_dict = {
            model.placeholders['is_inference']: False,
            #model.placeholders['drop_prob']: model.config['drop_prob'] if mode == 'train' else 0.0
        }
        if mode == 'train':
            fetches["train"] = model.ops['train']

        total_loss = {}

        epoch_size = self.num_files[mode] // batch_size
        epoch_step = self.epoch_step
        self.epoch_step = 0
        while epoch_step < epoch_size:
            if model.config['profile'] and mode == 'train':
                writer = tf.summary.FileWriter('log/')
                run_metadata = tf.RunMetadata()
                vals = self.session.run(fetches, feed_dict,
                                options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                                run_metadata=run_metadata)
                #writer.add_run_metadata(run_metadata, 'run metadata')
                from tensorflow.python.client import timeline
                ft = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = ft.generate_chrome_trace_format()
                with open('timeline_01.json', 'w') as f:
                    f.write(chrome_trace)
                sys.exit(0)

            vals = self.session.run(fetches, feed_dict)

            if verbose:
                print("\n%.02f%%:\tspeed: %.03f fps" %
                    ((epoch_step + 1) * 100.0 / epoch_size, (epoch_step + 1) * batch_size / (time.time() - start_time)))
            for d in vals['loss']:
                if d not in total_loss: total_loss[d] = {}
                for k in vals['loss'][d]:
                    if k not in total_loss[d]: total_loss[d][k] = 0
                    loss = np.asscalar(vals['loss'][d][k]['loss'])
                    total_loss[d][k] += loss;
                    if verbose:
                        print("\t%s %s loss: %.3f\tperplexity: %.3f\taverage perplexity: %.3f" %
                            (d, k, loss, np.exp(loss), np.exp(total_loss[d][k] / (epoch_step + 1))))
            epoch_step += 1

        # for now, save after very epoch
        self.saver.save(self.session, os.path.join(self.config['checkpoint_dir'], 'model'))#, global_step)


        total_perplexity = 0
        for d in total_loss:
            for k in total_loss[d]:
                perplexity = np.exp(total_loss[d][k] / epoch_size)
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
            with tf.variable_scope("TRNNModel", reuse=None, initializer=initializer):
                self.model = TRNNModel(config=config)

            # save stuff to be used later in inference
            self.saver = tf.train.Saver()

            with tf.Session() as session:
                self.session = session

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)

                if args.checkpoint:
                    ckpt = tf.train.get_checkpoint_state(config['checkpoint_dir'])
                    if ckpt and ckpt.model_checkpoint_path:
                        self.saver.restore(session, ckpt.model_checkpoint_path)
                else:
                    session.run(tf.global_variables_initializer())

                global_step = session.run(tf.train.get_or_create_global_step())
                self.epoch = np.asscalar(global_step // self.num_files['train'])
                self.epoch_step = (global_step - self.epoch * self.num_files['train']) // config['batch_size']
                while self.epoch < config['max_max_epoch']:
                    lr_decay = config['lr_decay'] ** max(self.epoch + 1 - config['max_epoch'], 0.0)
                    print("Epoch: %d Learning rate: %.3f" % (self.epoch + 1, lr_decay))

                    train_perplexity = self.run_epoch('train', lr_decay=lr_decay, verbose=True)
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
            coord.request_stop()
            coord.join(threads)

            # store the placeholder/op/fetch tensor names in our config
            self.config.update(self.model.output_tensor_names())
            with open(config['model_config_file'], 'w') as f:
                json.dump(config, f)
            session.close()



def main(_):
    directory = os.path.dirname(os.path.realpath(__file__))
    os.chdir(directory)

    if args.save_path is None:
        args.save_path = 'tmp'

    save_path = os.path.join(args.data_path, args.model, args.save_path)
    checkpoint_dir = os.path.join(save_path, 'checkpoints/')
    best_dir = os.path.join(save_path, 'best/')
    model_config_file = os.path.join(best_dir, 'config.json')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    if args.checkpoint and os.path.isfile(model_config_file):
        with open(model_config_file) as f:
            config = json.load(f)
    else:
        # we don't want to override the 'config' defaults
        if args.batch_size is None:
            del(args.batch_size)

        if args.dependencies is None:
            args.dependencies = ['forward'] if args.model == 'linear' else ['left_sibling']
        if args.model == 'linear':
            for i in range(len(args.dependencies)):
                if args.dependencies[i] == 'forward':
                    args.dependencies[i] = 'left_sibling'
                elif args.dependencies[i] == 'reverse':
                    args.dependencies[i] = 'right_sibling'
                else:
                    print('uh oh. invalid linear dependencies')
                    sys.exit(1)
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

        config.update(get_config())
        config.update(vars(args))
        config['label_size'] = len(token_ids['label_ids'])
        config['attr_size'] = len(token_ids['attr_ids'])
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
        'model_config_file': model_config_file
    })

    trainer = Trainer(config)

if __name__ == "__main__":
    tf.app.run()
