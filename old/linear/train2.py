from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import time
import json
import read

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


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBInput(object):
  """The input data."""

  def __init__(self, config, data=None, test=False, name=None):
    self.data = data
    self.batch_size = batch_size = config['batch_size']
    self.num_steps = num_steps = config['num_steps']
    #self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    if test:
        self.input_data, self.targets = tf.placeholder(tf.int32, [1, 1], name="input"), tf.placeholder(tf.int32, [1, 1], name="target")
    else:
        self.input_data, self.targets = ptb_producer(
            data, batch_size, num_steps, name=name)


class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, input_):
    self._input = input_

    batch_size = input_.batch_size
    num_steps = input_.num_steps
    size = config['hidden_size']
    vocab_size = config['vocab_size']

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def lstm_cell():
      # With the latest TensorFlow source code (as of Mar 27, 2017),
      # the BasicLSTMCell will need a reuse parameter which is unfortunately not
      # defined in TensorFlow 1.0. To maintain backwards compatibility, we add
      # an argument check here:
      if 'reuse' in inspect.getargspec(
          tf.contrib.rnn.BasicLSTMCell.__init__).args:
        return tf.contrib.rnn.BasicLSTMCell(
            size, forget_bias=0.0, state_is_tuple=True,
            reuse=tf.get_variable_scope().reuse)
      else:
        return tf.contrib.rnn.BasicLSTMCell(
            size, forget_bias=0.0, state_is_tuple=True)
    attn_cell = lstm_cell
    if is_training and config['keep_prob'] < 1:
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(), output_keep_prob=config['keep_prob'])
    cell = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(config['num_layers'])], state_is_tuple=True)

    self._initial_state = cell.zero_state(batch_size, data_type())

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

    if is_training and config['keep_prob'] < 1:
      inputs = tf.nn.dropout(inputs, config['keep_prob'])

    # Simplified version of models/tutorials/rnn/rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    #inputs = tf.unstack(inputs, num=num_steps, axis=1)
    #outputs, state = tf.contrib.rnn.static_rnn(
    #    cell, inputs, initial_state=self._initial_state)
    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        print(inputs[:, time_step, :])
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

    output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, size])
    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.matmul(output, softmax_w) + softmax_b
    self._probabilities = tf.nn.softmax(logits, name='Probabilities') #probabilities.append(tf.nn.softmax(logits))
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(input_.targets, [-1])],
        [tf.ones([batch_size * num_steps], dtype=data_type())])
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = tf.identity(state, name="FinalState")

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config['max_grad_norm'])
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def probabilities(self):
    return self._probabilities

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


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
  "batch_size" : 20,
  "vocab_size" : 10000
}

def run_epoch(session, model, data, eval_op=None, verbose=False):
  """Runs the model on the given data."""
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


def main(_):

  raw_data = dict()
  with open(FLAGS.data_path + '/train.json') as f:
    raw_data['train_ids'] = json.load(f)
    #raw_data['train_ids'].reverse()
  with open(FLAGS.data_path + '/valid.json') as f:
    raw_data['valid_ids'] = json.load(f)
    #raw_data['valid_ids'].reverse()
  with open(FLAGS.data_path + '/test.json') as f:
    raw_data['test_ids'] = json.load(f)
    #raw_data['test_ids'].reverse()
  with open(FLAGS.data_path + '/tokens.json') as f:
    raw_data['token_to_id'] = json.load(f)

  config = get_config()
  # XXX
  config['vocab_size'] = len(raw_data['token_to_id'])

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

    saver = tf.train.Saver()
    coord = tf.train.Coordinator()

    with tf.Session() as session:
      # ugh, Supervisor used to do this
      threads = tf.train.start_queue_runners(sess=session, coord=coord)
      session.run(tf.global_variables_initializer())
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

      if FLAGS.save_path:
        saver.save(session, FLAGS.save_path + '/model')
        with open(FLAGS.save_path + '/training_config.json', 'w') as f:
            json.dump(config, f)

    # XXX this isn't fixing the error messages about cancelled enqueue attempts...
    coord.request_stop()
    coord.join(threads)

def ptb_producer(raw_data, batch_size, num_steps, name=None):
  """Iterate on the raw PTB data.

  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).

  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.

  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):

    #raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size

    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])
    y = tf.strided_slice(data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
    y.set_shape([batch_size, num_steps])
    return x, y

if __name__ == "__main__":
  tf.app.run()
