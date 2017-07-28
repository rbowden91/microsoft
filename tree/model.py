# Largely based on Tree-Structured Decoding with Doublyrecurrent Neural Networks
# (https://openreview.net/pdf?id=HkYhZDqxg)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import time
import json
import os

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
  "num_steps" : 20, # this isn't used at all in this file, since we aren't doing any truncated backpropagation
  "hidden_size" : 200,
  "max_epoch" : 4,
  "max_max_epoch" : 10,
  "keep_prob" : 1.0,
  "lr_decay" : 0.5,
  "batch_size" : 40, # currently, this is just 1
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
}

TestConfig = {
  "init_scale" : 0.1,
  "learning_rate" : 1.0,
  "max_grad_norm" : 1,
  "num_layers" : 1,
  "num_steps" : 2,
  "hidden_size" : 4,
  "max_epoch" : 1,
  "max_max_epoch" : 1,
  "keep_prob" : 1.0,
  "lr_decay" : 0.5,
  "batch_size" : 20,
}

def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32

class TRNNModel(object):

  def __init__(self, is_training, config):
    size = config['hidden_size']
    vocab_size = config['vocab_size']
    attr_size = config['attr_vocab_size']

    # declare a bunch of parameters that will be reused later
    with tf.variable_scope('Parameters', reuse=False):
        # the second dimension doesn't have to be "size", but does have to match softmax_w's first dimension
        U_f = tf.get_variable('U_f', [size, size], dtype=tf.float32)
        U_a = tf.get_variable('U_a', [size, size], dtype=tf.float32)
        u_f = tf.get_variable('u_f', [size], dtype=tf.float32)
        u_a = tf.get_variable('u_a', [size], dtype=tf.float32)
        attr_w = tf.get_variable("attr_w", [size, attr_size], dtype=data_type())
        attr_b = tf.get_variable("attr_b", [attr_size], dtype=data_type())
        v_attr = tf.get_variable("v_attr", [vocab_size, attr_size], dtype=data_type())

        softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
        v_a = tf.get_variable("v_a", [vocab_size], dtype=data_type())
        v_f = tf.get_variable("v_f", [vocab_size], dtype=data_type())

    def lstm_cell():
        return tf.contrib.rnn.BasicLSTMCell(
            size, forget_bias=0.0, state_is_tuple=True,
            reuse=tf.get_variable_scope().reuse)
    attn_cell = lstm_cell
    if is_training and config['keep_prob'] < 1:
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(), output_keep_prob=config['keep_prob'])

    self.is_leaf_placeholder = tf.placeholder(
        tf.int32, [None], name='is_leaf_placeholder')
    self.last_sibling_placeholder = tf.placeholder(
        tf.int32, [None], name='last_sibling_placeholder')
    self.node_index_placeholder = tf.placeholder(
        tf.int32, [None], name='node_index_placeholder')
    self.attr_index_placeholder = tf.placeholder(
        tf.int32, [None], name='attr_index_placeholder')

    config['dependencies'] = dependencies = ['sibling', 'parent']

    # XXX XXX XXX better way of doing this? Basically, when doing inference, we want to be able to have different nodes
    # for each dependency, but we only use the Initial States as a place to write, which all are
    # associated with node 0

    self.is_inference = tf.placeholder(tf.bool, [], name='is_inference_placeholder')
    self.inference_placeholders = {}
    for k in config['dependencies']:
        self.inference_placeholders[k] = tf.placeholder(tf.int32, [], name='inference_' + k + '_placeholder')

    # save names of placeholders so we have a handle for them when we load the graph for inference
    # TODO: change most of these is_training's to is_testing? As long as batch size is 1, doesn't matter
    if is_training:
        config['placeholders'] = {
            'is_leaf': self.is_leaf_placeholder.name,
            'last_sibling': self.last_sibling_placeholder.name,
            'node_index': self.node_index_placeholder.name,
            'attr_index': self.attr_index_placeholder.name,
            'is_inference': self.is_inference.name
        }
        for k in config['dependencies']:
            config['placeholders']['inference_' + k] = self.inference_placeholders[k].name


    self.dependency_placeholders = dict()
    self.dependency_initial_states = dict()
    self.dependency_cells = dict()

    # dependency_states can't be a dictionary on sibling/parent, since it needs to be convertible to a tensor (required
    # for the tf.while_loop). The first index in dependency_states is thus aligned with the dependency in the
    # dependencies array
    dependency_states = []
    for i in range(len(dependencies)):
        dependency = dependencies[i]

        with tf.variable_scope("RNN", reuse=None):
            with tf.variable_scope(dependency, reuse=None):
                self.dependency_cells[dependency] = tf.contrib.rnn.MultiRNNCell(
                    [attn_cell() for _ in range(config['num_layers'])], state_is_tuple=True)
                # XXX 1 is the batch_size
                self.dependency_initial_states[dependency] = self.dependency_cells[dependency].zero_state(1, data_type())

        self.dependency_placeholders[dependency] = tf.placeholder(tf.int32, [None], name=(dependency + '_placeholder'))
        if is_training:
            config['placeholders'][dependency] = self.dependency_placeholders[dependency].name

        # need to manually handle LSTM states. This is gross, and happens all over the place
        dependency_states.append([])
        for j, (c, h) in enumerate(self.dependency_initial_states[dependency]):
            dependency_states[i].append([])
            dependency_states[i][-1].append(tf.TensorArray(
                tf.float32,
                size=0,
                dynamic_size=True,
                clear_after_read=False,
                infer_shape=False))
            dependency_states[i][-1].append(tf.TensorArray(
                tf.float32,
                size=0,
                dynamic_size=True,
                clear_after_read=False,
                infer_shape=False))

            # the nil parent/sibling should take on the initial state
            # Just c is in position 0, h is in position 1
            dependency_states[i][-1][0] = dependency_states[i][-1][0].write(0,
                                                        self.dependency_initial_states[dependency][j].c)
            dependency_states[i][-1][1] = dependency_states[i][-1][1].write(0,
                                                        self.dependency_initial_states[dependency][j].h)

            # Record the names of the LSTM states in config, so later when we want to do inference we can use them in
            # the feed_dict
            if is_training:
                if 'initial_states' not in config:
                    config['initial_states'] = dict()
                    config['states'] = dict()
                if dependency not in config['initial_states']:
                    config['initial_states'][dependency] = []
                    config['states'][dependency] = []
                config['initial_states'][dependency].append({
                    'c': self.dependency_initial_states[dependency][j].c.name,
                    'h': self.dependency_initial_states[dependency][j].h.name
                })

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())

    # TODO: dropout??
    #if is_training and config['keep_prob'] < 1:
    #  inputs = tf.nn.dropout(inputs, config['keep_prob'])

    # this returns true as long as the loop counter is less than the length of the example
    loop_cond = lambda loss, ctr, dependency_states, label_prob, attr_prob, lpa, lpf: \
        tf.less(ctr, tf.squeeze(tf.shape(self.is_leaf_placeholder)))

    def loop_body(loss, ctr, dependency_states, label_prob, attr_prob, lpa, lpf):
        is_leaf = tf.cast(tf.gather(self.is_leaf_placeholder, ctr, name="IsLeafGather"), tf.float32)
        last_sibling = tf.cast(tf.gather(self.last_sibling_placeholder, ctr, name="LastSiblingGather"), tf.float32)
        node_index = tf.gather(self.node_index_placeholder, ctr, name="NodeIndexGather")
        attr_index = tf.gather(self.attr_index_placeholder, ctr, name="AttrIndexGather")

        outputs = {}

        # Generate both the sibling and parent RNN states for the current node, based on the previous sibling and parent
        for i in range(len(dependencies)):
            dependency = dependencies[i]
            dependency_node = tf.gather(self.dependency_placeholders[dependency], ctr, name=(dependency+"Gather"))

            # During inference, we want to use the directly-supplied label for the parent, since each node is passed in
            # one-by-one and we won't have access to the parent when the child is passed in. During training, we have
            # all nodes in the example at once, so can directly grab the parent's label
            handle_inference = lambda: [self.inference_placeholders[dependency]]
            handle_training = lambda: tf.gather(self.node_index_placeholder, dependency_node, name=(dependency+"TokenGather"))

            dependency_token = tf.cond(self.is_inference, handle_inference, handle_training)
            dependency_embedding = tf.expand_dims(tf.gather(embedding, dependency_token, name=(dependency+"EmbedGather")), 0)

            with tf.variable_scope("RNN", reuse=None):
                with tf.variable_scope(dependency, reuse=None):
                    # reconstruct the LSTM state of the parent/sibling from the TensorArray
                    state = []
                    for j, (c, h) in enumerate(self.dependency_initial_states[dependency]):
                        c_state = dependency_states[i][j][0].read(dependency_node)
                        h_state = dependency_states[i][j][1].read(dependency_node)
                        # TensorArray returns shape <unknown>, which breaks things when passed to LSTM cell()
                        c_state.set_shape(self.dependency_initial_states[dependency][j].c.shape)
                        h_state.set_shape(self.dependency_initial_states[dependency][j].h.shape)
                        state.append(tf.contrib.rnn.LSTMStateTuple(c_state, h_state))

                    state = tuple(state)

                    # TODO: this technically gets recalculated over and over from the parent for all children, so could
                    # optimize by just doing it once
                    (output, new_state) = self.dependency_cells[dependency](dependency_embedding, state)
                    outputs[dependency] = output

                    # record new LSTM state in the TensorArray
                    for j, (c, h) in enumerate(new_state):
                        dependency_states[i][j][0] = dependency_states[i][j][0].write(ctr, new_state[j].c)
                        dependency_states[i][j][1] = dependency_states[i][j][1].write(ctr, new_state[j].h)

        # grab all of the projection paramaters, now that we have the current node's LSTM state
        with tf.variable_scope('Parameters', reuse=True):
            U_f = tf.get_variable('U_f', [size, size], dtype=tf.float32)
            U_a = tf.get_variable('U_a', [size, size], dtype=tf.float32)
            u_f = tf.get_variable('u_f', [size], dtype=tf.float32)
            u_a = tf.get_variable('u_a', [size], dtype=tf.float32)
            attr_w = tf.get_variable("attr_w", [size, attr_size], dtype=data_type())
            attr_b = tf.get_variable("attr_b", [attr_size], dtype=data_type())
            v_attr = tf.get_variable("v_attr", [vocab_size, attr_size], dtype=data_type())

            softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=data_type())
            softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
            #v_a = tf.get_variable("v_a", [vocab_size], dtype=data_type())
            v_f = tf.get_variable("v_f", [vocab_size], dtype=data_type())

        # this is the vector that combines the sibling and parent hidden states to be directly used in prediction
        # TODO: generalize this in terms of "dependencies"
        h_pred = tf.matmul(outputs['sibling'], U_f) + tf.matmul(outputs['parent'], U_a)

        # predict whether there is a child node (a = ancestor)
        logits_p_a = tf.reduce_sum(tf.multiply(u_a, h_pred))
        predicted_p_a = tf.sigmoid(logits_p_a)
        logits_p_a = tf.expand_dims(logits_p_a, 0)
        actual_p_a = tf.expand_dims(is_leaf, 0)
        # XXX paper uses sigmoid. How does this compare to cross entropy?
        loss_p_a = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_p_a, labels=actual_p_a, name="p_a_loss")

        # predict where there is a sibling node (f = fraternal)
        logits_p_f = tf.reduce_sum(tf.multiply(u_f, h_pred))
        predicted_p_f = tf.sigmoid(logits_p_f)
        logits_p_f = tf.expand_dims(logits_p_f, 0)
        actual_p_f = tf.expand_dims(last_sibling, 0)
        # TODO: paper uses sigmoid. How does this compare to cross entropy?
        loss_p_f = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_p_f, labels=actual_p_f, name="p_f_loss")


        # XXX Testing shouldn't necessarily use is_leaf and last_sibling directly, according to paper?
        # TODO: The paper doesn't seem to have a bias term. Could compare with and without
        #label_logits = tf.matmul(h_pred, softmax_w) + softmax_b + tf.multiply(v_a, is_leaf) + tf.multiply(v_f, last_sibling)
        label_logits = tf.matmul(h_pred, softmax_w) + softmax_b + tf.multiply(v_f, last_sibling)

        # XXX name things...
        label_probabilities = tf.nn.softmax(label_logits)
        actual_label = tf.one_hot(node_index, vocab_size)
        # TODO: switch this to use sparse_softmax?
        label_loss = tf.nn.softmax_cross_entropy_with_logits(logits=label_logits, labels=actual_label, name="label_loss")

        attr_logits = tf.matmul(h_pred, attr_w) + attr_b + tf.matmul(tf.expand_dims(actual_label, 0), v_attr)
        attr_probabilities = tf.nn.softmax(attr_logits)
        actual_attr = tf.one_hot(attr_index, attr_size)
        attr_loss = tf.nn.softmax_cross_entropy_with_logits(logits=attr_logits, labels=actual_attr, name="attr_loss")


        # TODO: could differ the weights for structural predictions vs the label predictions when calculating loss
        #loss = tf.add(loss, tf.reduce_sum(loss_p_a))
        loss = tf.add(loss, tf.reduce_sum(loss_p_f))
        loss = tf.add(loss, tf.reduce_sum(attr_loss))
        loss = tf.add(loss, tf.reduce_sum(label_loss))

        ctr = tf.add(ctr, 1)

        return loss, ctr, dependency_states, label_probabilities, attr_probabilities, predicted_p_a, predicted_p_f

    # start iterating from 1, since 0 is the "empty" parent/sibling
    # The last four arguments we need to "return" from the while loop, so that inference can use them directly
    # XXX the last three args are just there because we need a tensor of the correct size. better way?
    loss, _, dependency_states, label_probabilities, attr_probabilities, predicted_p_a, predicted_p_f = tf.while_loop(loop_cond, loop_body,
        [0.0, # loss
         1, # ctr
         dependency_states, # dependency_states
         tf.zeros([1,vocab_size], tf.float32), # label_probabilities
         tf.zeros([1,attr_size], tf.float32), # attr_probabilities
         0.0, # predicted_p_a
         0.0], # predicted_p_f
        parallel_iterations=1)

    # save some more tensor names for us to use in inference
    if is_training:
        config['fetches'] = {
            'predicted_p_a': predicted_p_a.name,
            'predicted_p_f': predicted_p_f.name,
            'label_probabilities': label_probabilities.name,
            'attr_probabilities': attr_probabilities.name,
            'states': {}
        }
        for i in range(len(dependencies)):
            config['fetches']['states'][dependencies[i]] = []
            for j in range(len(dependency_states[i])):
                # for inference, we only care about the "root" node's state
                config['fetches']['states'][dependencies[i]].append({
                    'c': dependency_states[i][j][0].read(1).name,
                    'h': dependency_states[i][j][1].read(1).name,
                })

    self._cost = cost = tf.reduce_sum(loss)

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


def run_epoch(session, model, data, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0

    fetches = {
        "cost": model.cost,
    }

    if eval_op is not None:
        fetches["eval_op"] = eval_op

    epoch_size = len(data['leaf_node'])

    for step in range(epoch_size):
        feed_dict = {
            model.is_leaf_placeholder: data['leaf_node'][step],
            model.last_sibling_placeholder: data['last_sibling'][step],
            model.node_index_placeholder: data['token'][step],
            model.attr_index_placeholder: data['attr'][step],
            model.dependency_placeholders['sibling']: data['sibling'][step],
            model.dependency_placeholders['parent']: data['parent'][step],

            model.is_inference: False,

            # these aren't used if is_inference is False, but it seems we still need
            # to feed them in
            model.inference_placeholders['parent']: 0,
            model.inference_placeholders['sibling']: 0
        }

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        print(cost)

        costs += cost
        iters += 1

        #if verbose: #and step % (epoch_size // 10) == 10:
        #  print("%.3f perplexity: %.3f speed: %.0f wps" %
        #        (step * 1.0 / epoch_size, np.exp(costs / iters),
        #         #iters * model.input.batch_size / (time.time() - start_time)))
        #         iters / (time.time() - start_time)))

    #return np.exp(costs / iters)
    return costs


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
    # load in all the data
    raw_data = dict()
    with open(os.path.join(FLAGS.data_path, 'tree_train.json')) as f:
        raw_data['train'] = json.load(f)
    with open(os.path.join(FLAGS.data_path,'tree_valid.json')) as f:
        raw_data['valid'] = json.load(f)
    with open(os.path.join(FLAGS.data_path,'tree_test.json')) as f:
        raw_data['test'] = json.load(f)
    with open(os.path.join(FLAGS.data_path, 'tree_tokens.json')) as f:
        token_ids = json.load(f)
        raw_data['token_to_id'] = token_ids['ast_labels']
        raw_data['attr_to_id'] = token_ids['label_attrs']

    config = get_config()
    config['vocab_size'] = len(raw_data['token_to_id'])
    config['attr_vocab_size'] = len(raw_data['attr_to_id'])

    eval_config = config.copy()
    #eval_config['batch_size'] = 1
    #eval_config['num_steps'] = 1

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config['init_scale'],
                                                    config['init_scale'])

        with tf.name_scope("Train"):
            with tf.variable_scope("TRNNModel", reuse=None, initializer=initializer):
                m = TRNNModel(is_training=True, config=config)#, input_=raw_data['train'])
            tf.summary.scalar("Training Loss", m.cost)
            tf.summary.scalar("Learning Rate", m.lr)

        # TODO: Can remove Valid and Test nodes from the saved model
        with tf.name_scope("Valid"):
            with tf.variable_scope("TRNNModel", reuse=True, initializer=initializer):
                mvalid = TRNNModel(is_training=False, config=config)
            tf.summary.scalar("Validation Loss", mvalid.cost)

        with tf.name_scope("Test"):
            with tf.variable_scope("TRNNModel", reuse=True, initializer=initializer):
                mtest = TRNNModel(is_training=False, config=eval_config)

        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            for i in range(config['max_max_epoch']):
                lr_decay = config['lr_decay'] ** max(i + 1 - config['max_epoch'], 0.0)
                m.assign_lr(session, config['learning_rate'] * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, raw_data['train'], eval_op=m.train_op,
                                            verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

                valid_perplexity = run_epoch(session, mvalid, raw_data['valid'])
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            test_perplexity = run_epoch(session, mtest, raw_data['test'])
            print("Test Perplexity: %.3f" % test_perplexity)

            if FLAGS.save_path:
                if not os.path.isdir(FLAGS.save_path):
                    os.makedirs(FLAGS.save_path)
                saver.save(session, os.path.join(FLAGS.save_path, 'tree_model'))
                with open(os.path.join(FLAGS.save_path, 'tree_training_config.json'), 'w') as f:
                    json.dump(config, f)

if __name__ == "__main__":
    tf.app.run()
