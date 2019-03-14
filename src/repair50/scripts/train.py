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

import numpy as np
import tensorflow as tf
from ..model.model import TRNNModel, TRNNJointModel

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
parser.add_argument('-n', '--num_files', help='number of files to train on (default, all files)', type=int)
parser.add_argument('-d', '--dependency_configs', help='forward/reverse, etc.', type=lambda s: s.split(), default='d2')
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

    def __init__(self, config, test, transitions):
        self.config = config
        self.test = test
        self.transitions = transitions
        self.test_conf = test_conf = config['tests'][transitions][test]

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
            with tf.variable_scope("TRNNModel/{}/{}".format(transitions, test), reuse=None, initializer=initializer):
                rows = None
                for i in config['dependency_configs']:
                    self.models['dependency_configs'][i] = TRNNModel(
                        i, test_conf, transitions == 'true', rows=rows
                    )
                    if rows is None:
                        rows = self.models['dependency_configs'][i].rows
                        self.iter_model = self.models['dependency_configs'][i]
                for i in config['joint_configs']:
                    self.models['joint_configs'][i] = TRNNJointModel(
                        self.models['dependency_configs'],
                        i, test_conf, transitions == 'true', rows=rows
                    )


            # save stuff to be used later in inference
            self.saver = tf.train.Saver()

            sess_config = tf.ConfigProto(device_count = {'GPU': (0 if args.cpu else 1)})
            with tf.Session(config=sess_config) as session:
                self.session = session

                ckpt = tf.train.get_checkpoint_state(test_conf['checkpoint_dir'])
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
                self.writer = tf.summary.FileWriter(test_conf['log_dir'], graph=tf.get_default_graph()) #\ if self.config['summarywriter']
                self.epoch = 0
                while self.epoch < config['epochs']:
                    if test in ['test3', 'null']:
                        train_perplexity = self.run_epoch('train', verbose=True)
                        valid_perplexity = self.run_epoch('valid')
                    else:
                        train_perplexity = 1
                        valid_perplexity = 1

                    test_conf['train_perplexities'].append(train_perplexity)
                    test_conf['valid_perplexities'].append(valid_perplexity)
                    if test_conf['best_validation'] is None or valid_perplexity <= test_conf['best_validation']:
                        self.saver.save(session, os.path.join(test_conf['best_dir'], 'model'))#, global_step)
                        test_conf['best_validation'] = valid_perplexity
                        test_conf['best_validation_epoch'] = self.epoch
                        with open(test_conf['model_config_file'], 'w') as f:
                            json.dump(test_conf, f)
                    self.epoch += 1

                if test_conf['best_validation'] < valid_perplexity:
                    ckpt = tf.train.get_checkpoint_state(test_conf['best_dir'])
                    assert ckpt and ckpt.model_checkpoint_path
                    self.saver.restore(session, ckpt.model_checkpoint_path)

                # TODO: FIXME
                test_perplexity = 1 #self.run_epoch('test')
                test_conf['test_perplexity'] = test_perplexity

            # store the placeholder/op/fetch tensor names in our config
            outputs = {}
            for i in self.models:
                outputs[i] = {}
                for j in self.models[i]:
                    outputs[i][j] = self.models[i][j].output_tensor_names()
            test_conf['models'] = outputs
            with open(test_conf['model_config_file'], 'w') as f:
                json.dump(test_conf, f)
            session.close()


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
        self.epoch_size = self.num_files[mode] // self.batch_size
        epoch_step = 0
        while epoch_step < self.epoch_size:
            with tf.control_dependencies(fetch_ops):
                summary, vals, _ = self.session.run([summary_op, fetches, train_ops], feed_dict)
                self.writer.add_summary(summary, epoch_step * self.batch_size)

            if self.config['profile'] and mode == 'train':
                self.run_profiler()

            if verbose:
                print("\nepoch %d %.02f%%:\tspeed: %.03f fps" %
                    (self.epoch + 1, (epoch_step + 1) * 100.0 / self.epoch_size, (epoch_step + 1) * self.batch_size /
                                                            (time.time() - start_time)))
            for d in vals:
                if d not in total_loss: total_loss[d] = {}
                for k in vals[d]:
                    if k not in total_loss[d]: total_loss[d][k] = 0
                    loss = np.asscalar(vals[d][k])
                    total_loss[d][k] += loss;
                    if verbose:
                        print("\t%s %s %s %s loss: %.3f\tperplexity: %.3f\taverage perplexity: %.3f" %
                            (self.test_conf['test'], self.test_conf['transitions'], d, k, loss, np.exp(loss), np.exp(total_loss[d][k] / (epoch_step + 1))))
            epoch_step += 1
        return total_loss


    def run_epoch(self, mode, verbose=False):
        self.batch_size = self.config['batch_size'] if mode == 'train' else 1
        filename = os.path.join(args.data_path, 'tests', self.transitions, self.test, mode + '_data.tfrecord')
        self.session.run(self.iter_model.ops['batch_size_update'],
                         feed_dict={ self.iter_model.placeholders['new_batch_size'] : self.batch_size })
        self.session.run(self.iter_model.ops['file_iter'],
                         feed_dict={ self.iter_model.placeholders['filename'] : filename })

        total_loss = self.run_models(False, mode, verbose)
        # disregard the loss from joints for now for validation
        self.run_models(True, mode, verbose)

        # for now, save after very epoch
        self.saver.save(self.session, os.path.join(self.test_conf['checkpoint_dir'], 'model'))#, global_step)

        # TODO: how do perplexities add?
        total_perplexity = 0
        for d in total_loss:
            for k in total_loss[d]:
                perplexity = np.exp(total_loss[d][k] / self.epoch_size)
                total_perplexity += np.asscalar(perplexity)
                print("Epoch %d: %s %s %s %s %s perplexity: %.3f" %
                        (self.epoch+(1 if mode != 'test' else 0), self.test_conf['test'],
                         self.test_conf['transitions'], mode, d, k, perplexity))
        return total_perplexity

def main():
    args.data_path = os.path.join(os.getcwd(), args.data_path)
    with open(os.path.join(args.data_path, 'config.json')) as f:
        config = json.load(f)
    # Note that at this point, the training, validation, and test data have already been split up.
    # So, we preserve the ratios between them.
    if args.num_files is not None:
        config['num_files'] = min(args.num_files, config['num_files'])
    del(args.num_files)
    config.update(vars(args))

    save_path = os.path.join(args.data_path, args.save_path)

    for transitions in config['tests']:
        for test in config['tests'][transitions]:
            path = os.path.join(save_path, 'tests', transitions, test)

            checkpoint_dir = os.path.join(path, 'checkpoints/')
            best_dir = os.path.join(path, 'best/')
            log_dir = os.path.join(path, 'log/')

            model_config_file = os.path.join(best_dir, 'config.json')
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(best_dir, exist_ok=True)

            if args.checkpoint and os.path.isfile(model_config_file):
                with open(model_config_file) as f:
                    config['tests'][transitions][test] = json.load(f)
            else:
                conf = config['tests'][transitions][test]

                with open(os.path.join(args.data_path, 'tests', transitions, test, 'lexicon.json'), 'r') as f:
                    lexicon = json.load(f)


                # TODO: the final train perplexity should go over the epoch again, without backprop
                conf.update({
                    'train_perplexities': [],
                    'valid_perplexities': [],
                    'best_validation': None,
                    'best_validation_epoch': None,
                    'checkpoint_dir': checkpoint_dir,
                    'best_dir': best_dir,
                    'log_dir': log_dir,
                    'model_config_file': model_config_file,
                    'lexicon': lexicon
                })

                for k in ['num_layers', 'max_grad_norm', 'hidden_size']:
                    conf[k] = config[k]

                with open(model_config_file, 'w') as f:
                    json.dump(conf, f)

    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(config, f)

    for transitions in config['tests']:
        for test in config['tests'][transitions]:
            trainer = Trainer(config, test, transitions)
