# Example for my blog post at:
# http://danijar.com/introduction-to-recurrent-networks-in-tensorflow/
import functools
import inspect
import time
import json
import read
import os
import random
import numpy as np
import tensorflow as tf

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "test",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_string("data_path", None,
                    "XXX")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS

SmallConfig = {
  "init_scale" : 0.1,
  "learning_rate" : 1.0,
  "max_grad_norm" : 5,
  "num_layers" : 2,
  "num_steps" : 20,
  "hidden_size" : 200,
  "max_epoch" : 4,
  "max_max_epoch" : 10,
  "keep_prob" : 1.0,
  "lr_decay" : 0.5,
  "batch_size" : 40,
  "vocab_size" : 10000
}

MediumConfig = {
  "init_scale" : 0.05,
  "learning_rate" : 1.0,
  "max_grad_norm" : 5,
  "num_layers" : 2,
  "num_steps" : 35,
  "hidden_size" : 650,
  "max_epoch" : 6,
  "max_max_epoch" : 39,
  "keep_prob" : 0.5,
  "lr_decay" : 0.8,
  "batch_size" : 20,
  "vocab_size" : 10000
}

LargeConfig = {
  "init_scale" : 0.04,
  "learning_rate" : 1.0,
  "max_grad_norm" : 10,
  "num_layers" : 2,
  "num_steps" : 35,
  "hidden_size" : 1500,
  "max_epoch" : 14,
  "max_max_epoch" : 55,
  "keep_prob" : 0.35,
  "lr_decay" : 1 / 1.15,
  "batch_size" : 20,
  "vocab_size" : 10000
}

TestConfig = {
  "init_scale" : 0.1,
  "learning_rate" : 1.0,
  "max_grad_norm" : 1,
  "num_layers" : 1,
  "num_steps" : 2,
  "hidden_size" : 2,
  "max_epoch" : 1,
  "max_max_epoch" : 1,
  "keep_prob" : 1.0,
  "lr_decay" : 0.5,
  "batch_size" : 2,
  "vocab_size" : 10000
}

def get_config():
  if FLAGS.model == "small":
    return SmallConfig
  elif FLAGS.model == "medium":
    return MediumConfig
  elif FLAGS.model == "large":
    return LargeConfig
  elif FLAGS.model == "test":
    return TestConfig
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class SequenceLabelling:

    def __init__(self, data, mask, dropout, config):
        self.data = data
        self.dropout = dropout
        self.mask = mask
        self.config = config
        self.vocab_size = config['vocab_size']
        self._num_hidden = config['hidden_size']
        self._num_layers = config['num_layers']
        self.prediction
        self.error
        self.optimize

    @lazy_property
    def prediction(self):
        # Recurrent network.
        network = tf.nn.rnn_cell.GRUCell(self._num_hidden)
        network = tf.nn.rnn_cell.DropoutWrapper(
            network, output_keep_prob=self.dropout)
        network = tf.nn.rnn_cell.MultiRNNCell([network] * self._num_layers)
        data = tf.one_hot(self.data, self.vocab_size, axis=-1)
        output, _ = tf.nn.dynamic_rnn(network, tf.cast(data, tf.float32), dtype=tf.float32)
        # Softmax layer.
        weight, bias = self._weight_and_bias(self._num_hidden, self.vocab_size)
        # Flatten to apply same weights to all time steps.
        output = tf.reshape(output, [-1, self._num_hidden])
        prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
        self.target = tf.slice(self.data, [0,1],[-1,-1])
        # XXX use data in the line above, instead of one_hot again here?
        self.target = tf.one_hot(self.target, self.vocab_size, axis=-1)
        extend = tf.zeros([tf.shape(self.target)[0],1,self.vocab_size], dtype=tf.float32)
        self.target = tf.concat([self.target, extend], 1)
        max_length = int(self.target.get_shape()[1])
        prediction = tf.reshape(prediction, [-1, max_length, self.vocab_size])
        return prediction

    @lazy_property
    def cost(self):
        cross_entropy = -tf.reduce_sum(
            tf.cast(self.target, tf.float32) * tf.log(self.prediction), [1, 2])
        cross_entropy = tf.reduce_mean(cross_entropy)
        cross_entropy = tf.Print(cross_entropy, [tf.shape(self.prediction), tf.shape(self.target)])
        return cross_entropy

    @lazy_property
    def optimize(self):
        learning_rate = 0.003
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 2), tf.argmax(self.prediction, 2))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)


if __name__ == '__main__':
    config = get_config()
    # XXX
    raw_data = dict()
    for dataset in ['train', 'valid', 'test']:
        with open(os.path.join(FLAGS.data_path, 'train.json')) as f:
            raw_data[dataset + '_ids'] = json.load(f)
            #raw_data['train_ids'].reverse()
    with open(os.path.join(FLAGS.data_path, 'tokens.json')) as f:
        raw_data['token_to_id'] = json.load(f)
    config['vocab_size'] = len(raw_data['token_to_id'])

    vocab_size = len(raw_data['token_to_id'])
    example_length = len(raw_data['train_ids']['tokens'][0])
    num_examples = len(raw_data['train_ids'])
    data = tf.placeholder(tf.int32, [None, example_length])
    mask = tf.placeholder(tf.int32, [None])
    dropout = tf.placeholder(tf.float32)
    model = SequenceLabelling(data, mask, dropout, config)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    order = list(range(num_examples))
    for epoch in range(config['max_epoch']):
        random.shuffle(order)
        for i in range(num_examples // config['batch_size']):
            start = i * config['batch_size']
            batch = raw_data['train_ids']['tokens'][start:start + config['batch_size']]
            data_mask = raw_data['train_ids']['lengths'][start:start + config['batch_size']]
            sess.run(model.optimize, { data: batch, mask: data_mask, dropout: 0.2 })
        error = sess.run(model.error, {
            data: raw_data['valid_ids']['tokens'], mask: raw_data['valid_ids']['lengths'], dropout: 1})
        print('Epoch {:2d} error {:3.1f}%'.format(epoch + 1, 100 * error))
    if FLAGS.save_path:
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(FLAGS.save_path, 'model'))
    with open(os.path.join(FLAGS.save_path, 'training_config.json'), 'w') as f:
        json.dump(config, f)

"""

def run_epoch(session, model, data, eval_op=None, verbose=False):
  #Runs the model on the given data.
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
  }

  if eval_op is not None:
    fetches["eval_op"] = eval_op

  epoch_size = ((len(data) // model.input.batch_size) - 1) // model.input.num_steps

  for step in range(epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    final_state = vals["final_state"]
    state = []

    for i in range(len(final_state)):
        state.append(tf.contrib.rnn.LSTMStateTuple(final_state[i][0], final_state[i][1]))
    state = tuple(state)

    costs += cost
    iters += model.input.num_steps

    if verbose and step % (epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters * model.input.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)



def main(_):

  eval_config = config.copy()
  eval_config['batch_size'] = 1
  eval_config['num_steps'] = 1

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config['init_scale'],
                                                config['init_scale'])

    with tf.name_scope("Train"):
      train_data = tf.placeholder(tf.int32, [len(raw_data['train_ids'])])
      train_var = tf.Variable(train_data, trainable=False, collections=[])
      train_input = PTBInput(config=config, data=train_var, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True, config=config, input_=train_input)
      tf.summary.scalar("Training Loss", m.cost)
      tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
      valid_data = tf.placeholder(tf.int32, [len(raw_data['valid_ids'])])
      valid_var = tf.Variable(valid_data, trainable=False, collections=[])
      valid_input = PTBInput(config=config, data=valid_var, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
      tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
      #test_data = tf.placeholder(tf.int32, [len(raw_data['test_ids'])], name="TestPlace")
      #test_var = tf.Variable(test_data, trainable=False, collections=[], name="TestVar")
      test_input = PTBInput(config=eval_config, test=True, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = PTBModel(is_training=False, config=eval_config,
                         input_=test_input)

    coord = tf.train.Coordinator()

    with tf.Session() as session:
      # ugh, Supervisor used to do this
      threads = tf.train.start_queue_runners(sess=session, coord=coord)
      session.run(train_var.initializer, feed_dict={train_data: raw_data['train_ids']})
      session.run(valid_var.initializer, feed_dict={valid_data: raw_data['valid_ids']})
      #session.run(test_var.initializer, feed_dict={test_data: raw_data['test_ids']})

      for i in range(config['max_max_epoch']):
        lr_decay = config['lr_decay'] ** max(i + 1 - config['max_epoch'], 0.0)
        m.assign_lr(session, config['learning_rate'] * lr_decay)

        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        train_perplexity = run_epoch(session, m, raw_data['train_ids'], eval_op=m.train_op,
                                    verbose=True)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        valid_perplexity = run_epoch(session, mvalid, raw_data['valid_ids'])
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

      #test_perplexity = run_epoch(session, mtest, raw_data['test_ids'])
      #print("Test Perplexity: %.3f" % test_perplexity)


if __name__ == "__main__":
  tf.app.run()
"""
