from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import time
import json
from pprint import pprint
import os, sys

import numpy as np
import tensorflow as tf
from tensorarray import RNNTensorArray, RNNTensorArrayCell
from tree_lstm import TreeLSTMCell

#possible_configurations = {
#    'top_left',
#    'bottom_right',
#    'bottom_left',
#    'top_right',
#    'top',
#    'bottom',
#    'left',
#    'right'
#}
#
#possible_dependencies = {
#    'parent': 'top',
#    'left_sibling': 'left',
#    'left_prior': 'left',
#    'children': 'bottom',
#    'right_sibling': 'right',
#    'right_prior': 'right'
#}

possible_dependencies = {
    'parent': 'forward',
    'left_sibling': 'forward',
    'left_prior': 'forward',
    'children': 'reverse',
    'right_sibling': 'reverse',
    'right_prior': 'reverse'
}

invert = {
    'right_sibling': 'left_sibling',
    'left_sibling': 'right_sibling',
    'parent': 'left_child',
    'children': 'parent',
    'left_prior': 'right_prior',
    'right_prior': 'left_prior'
}

def data_type():
    return tf.float32

class TRNNModel(object):

    def output_tensor_names(self):
        def extract_names(d):
            if isinstance(d, dict):
                out = {}
                for k in d: out[k] = extract_names(d[k])
                return out
            elif isinstance(d, list):
                x = [extract_names(i) for i in d]
                if len(x) == 1:
                    x = x[0]
                return x
            elif isinstance(d, tf.contrib.rnn.LSTMStateTuple):
                return {
                    'c': d.c.name,
                    'h': d.h.name
                }
            elif isinstance(d, tf.TensorArray):
                return d.stack().name
            else:
                assert(isinstance(d, tf.Tensor) or isinstance(d, tf.Operation))
                return d.name

        return {
            'placeholders': extract_names(self.placeholders),
            'ops': extract_names(self.ops),
            'fetches' : extract_names(self.fetches)
        }

    def collect_directional_logits(self, direction, h_pred, right_hole_h_pred):
        num_rows = tf.shape(self.rows['label_index'])[0]
        size = self.config['hidden_size']
        label_size = self.config['label_size']
        attr_size = self.config['attr_size']

        logits = {}
        labels = {}

        # tiles a parameter matrix/vector for matmul on a whole batch
        def tile(matrix):
            return tf.tile(matrix, [num_rows, 1, 1])

        # grab all of the projection paramaters, now that we have the current node's LSTM state
        with tf.variable_scope("Parameters/{}".format(direction), reuse=True):
            u_end = tf.get_variable('u_end', [size], dtype=data_type())

            attr_w = tf.get_variable("attr_w", [1, size, attr_size], dtype=data_type())
            attr_b = tf.get_variable("attr_b", [1, attr_size], dtype=data_type())
            attr_v = tf.get_variable("v_attr", [1, label_size, attr_size], dtype=data_type())

            label_w = tf.get_variable("label_w", [1, size, label_size], dtype=data_type())
            label_b = tf.get_variable("label_b", [1, label_size], dtype=data_type())

            if direction == 'reverse':
                u_right_hole = tf.get_variable('u_right_hole', [1, size * 2], dtype=data_type())

        # this is the only loss here that linear uses
        logits['label_index'] = tf.matmul(h_pred, tile(label_w)) + label_b
        labels['label_index'] = self.rows['label_index']

        if self.config['model'] == 'ast':
            # loss for whether we are the last sibling on the left or right
            end = 'last_sibling' if direction == 'forward' else 'first_sibling'
            labels[end] = tf.cast(self.rows[end], data_type())
            logits[end] = tf.reduce_sum(tf.multiply(u_end, h_pred), axis=2)

            #right_hole_h_pred = tf.Print(right_hole_h_pred, [right_hole, right_sibling, right_hole_h_pred])
            #if direction == 'reverse':
            #    #right_sibling = self.rows['right_sibling']
            #    #right_hole = self.rows['right_hole']
            #    #logits_right_hole = tf.reduce_sum(tf.multiply(u_end, tf.concat([h_pred, right_hole_h_pred], 0)))
            #    #predicted_right_hole = tf.sigmoid(logits_right_hole)
            #    #actual_right_hole = tf.cast(right_hole == right_sibling, data_type())
            #    ## TODO: paper uses sigmoid. How does this compare to cross entropy?
            #    #loss_right_hole = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_right_hole, labels=actual_right_hole, name="right_hole_loss")
            #    loss_right_hole = tf.constant(0.0)
            #if direction == 'reverse':
            #    loss = tf.cond(tf.cast(right_hole, tf.bool),
            #            lambda: loss + loss_right_hole, lambda: loss)
            #    predicted_right_hole = tf.constant(0.0)
            #else:
            #    predicted_right_hole = tf.constant(0.0)

            actual_label = tf.one_hot(self.rows['label_index'], label_size)
            logits['attr_index'] = tf.matmul(h_pred, tile(attr_w)) + attr_b + tf.matmul(actual_label, tile(attr_v))
            labels['attr_index'] = self.rows['attr_index']
            print(labels['attr_index'], logits['attr_index'])

        return labels, logits

    def collect_joint_logits(self, h_pred_ctr, directional_loss):
        #labels = {'joint':{}}
        #logits = {'joint':{}}

        #for k in loss['forward']:
        #    if k != 'label': continue
        #    label_size = self.config['label_size']
        #    actual_label = tf.one_hot(self.rows['label_index'], label_size, axis=2)
        #    forward = tf.multiply(loss['forward'][k]['probabilities'], actual_label)
        #    forward = tf.reduce_sum(forward, axis=2)
        #    reverse = tf.multiply(loss['reverse'][k]['probabilities'], actual_label)
        #    reverse = tf.reduce_sum(reverse, axis=2)

        #with tf.variable_scope("Parameters/joint", reuse=True):
        #    u_alpha = tf.get_variable("u_alpha", [self.config['hidden_size'] * 2], dtype=data_type())

        #labels['joint']['alpha'] = forward / (forward + reverse)
        #logits['joint']['alpha'] = tf.reduce_sum(tf.multiply(u_alpha, tf.concat([h_pred_ctr['forward'], h_pred_ctr['reverse']], axis=2)), axis=2)

        #logits['joint']['alpha'] = tf.Print(logits['joint']['alpha'], [tf.shape(labels['joint']['alpha']), tf.shape(logits['joint']['alpha'])])

        #return labels, logits
        label_size = self.config['label_size']
        actual_label = tf.one_hot(self.rows['label_index'][0], label_size, axis=1)
        forward = tf.multiply(directional_loss['forward']['label_index']['probabilities'], actual_label)
        forward = tf.reduce_sum(forward, axis=2)

        reverse = tf.multiply(directional_loss['reverse']['label_index']['probabilities'], actual_label)
        reverse = tf.reduce_sum(reverse, axis=2)
        forward_embedding = self.embed(self.rows['left_sibling'][0], 'left_sibling')
        reverse_embedding = self.embed(self.rows['right_sibling'][0], 'right_sibling')
        #reverse = tf.reduce_sum(reverse, axis=1)
        #forward = tf.Print(forward, [forward, reverse])

        with tf.variable_scope("Parameters/joint", reuse=True):
            u_alpha = tf.get_variable("u_alpha", [1, self.config['hidden_size'] * 4, 2], dtype=data_type())
            b_alpha = tf.get_variable("b_alpha", [1, 2], dtype=data_type())

        # something other than transpose?
        labels = tf.transpose([forward / (forward + reverse), reverse / (forward + reverse)])
        combined = tf.concat([h_pred_ctr['forward'], h_pred_ctr['reverse'], forward_embedding, reverse_embedding], axis=2)
        num_rows = tf.shape(self.rows['label_index'])[0]
        logits = tf.matmul(combined, tf.tile(u_alpha, [num_rows, 1, 1])) + b_alpha

        mask = tf.cast(self.rows['mask'][0], tf.float32)
        num_tokens = tf.reduce_sum(mask)
        loss = { 'joint': {'label_index': {}} }
        loss['joint']['label_index']['loss'] = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)  * mask) / num_tokens
        loss['joint']['label_index']['probabilities'] = tf.nn.softmax(logits)

        #reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        #tf.reduce_sum(reg_losses)
        scope = tf.contrib.framework.get_name_scope()
        if scope != '':
            scope += '/'
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "{}Parameters/{}".format(scope, 'joint'))
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss['joint']['label_index']['loss'], tvars), self.config['max_grad_norm'])

        train_vars = zip(grads, tvars)


        return loss, train_vars

    def calculate_loss_and_tvars(self, labels, logits):
        loss = {}
        train_vars = []
        mask = tf.cast(self.rows['mask'], tf.float32)
        num_tokens = tf.reduce_sum(mask)
        for subset in labels:
            loss[subset] = {}
            total_loss = 0.0
            for k in labels[subset]:
                log, lab = logits[subset][k], labels[subset][k]
                cross, predict = (tf.nn.sigmoid_cross_entropy_with_logits, tf.nn.sigmoid) if len(log.shape) == 2 \
                                 else (tf.nn.sparse_softmax_cross_entropy_with_logits, tf.nn.softmax)
                print(subset, k, lab, log)
                loss[subset][k] = {
                    'probabilities': predict(log),
                    'loss': tf.reduce_sum(cross(labels=lab, logits=log) * mask) / num_tokens
                }
                total_loss += loss[subset][k]['loss']
            # grab the current scope, in case the model was instantiated within a scope
            scope = tf.contrib.framework.get_name_scope()
            if scope != '':
                scope += '/'
            # only update the subset of weights relevant to these losses
            tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "{}Parameters/{}".format(scope, subset))
            grads, _ = tf.clip_by_global_norm(tf.gradients(total_loss, tvars), self.config['max_grad_norm'])
            train_vars.extend(zip(grads, tvars))
        return loss, train_vars

    def embed(self, index, dependency):
       #     label_index = tf.zeros([self.batch_size], dtype=tf.int32)
       #     attr_index = tf.zeros([self.batch_size], dtype=tf.int32)
       # else:
       #     #label_index = tf.cond(self.placeholders['is_inference'],
       #     #                      lambda: self.placeholders['inference'][dependency]['label'],
       #     #                      lambda: tf.gather(self.rows['label_index'], index, axis=1,
       #     #                          name="LabelIndexGather"))

       #     #attr_index = tf.cond(self.placeholders['is_inference'],
       #     #                      lambda: self.placeholders['inference'][dependency]['attr'],
       #     #                      lambda: tf.gather(self.rows['attr_index'], index, axis=1,
       #     #                          name="AttrIndexGather"))
       #     index = tf.one_hot(index, tf.shape(self.rows['label_index'])[1], on_value=True, off_value=False)
       #     label_index = tf.boolean_mask(self.rows['label_index'], index)
       #     attr_index = tf.boolean_mask(self.rows['attr_index'], index)

        #if dependency == None:
        #    label_index = [0]
        #    attr_index = [0]
        #else:
            #label_index = tf.cond(self.placeholders['is_inference'],
            #                      lambda: self.placeholders['inference'][dependency]['label'],
            #                      lambda: tf.gather(self.rows['label_index'], index, axis=1,
            #                          name="LabelIndexGather"))

            #attr_index = tf.cond(self.placeholders['is_inference'],
            #                      lambda: self.placeholders['inference'][dependency]['attr'],
            #                      lambda: tf.gather(self.rows['attr_index'], index, axis=1,
            #                          name="AttrIndexGather"))
        label_index = tf.gather(self.rows['label_index'], index, axis=1)
        attr_index = tf.gather(self.rows['attr_index'], index, axis=1)

        #label_index.set_shape([None])
        #attr_index.set_shape([None])
        with tf.variable_scope('Parameters/{}'.format(self.config['dependencies'][dependency]), reuse=True):
            label_embedding = tf.get_variable(
                "label_embedding", [self.config['label_size'], self.config['embedding_size']], dtype=data_type())
            # XXX currently not using this!!
            attr_embedding = tf.get_variable(
                "attr_embedding", [self.config['attr_size'], self.config['embedding_size']], dtype=data_type())
        node_label_embedding = tf.gather(label_embedding, label_index, name="LabelEmbedGather")
        node_attr_embedding = tf.gather(attr_embedding, attr_index, name="AttrEmbedGather")
        node_embedding = node_label_embedding#tf.concat([node_label_embedding, node_attr_embedding], 0)
        return node_embedding

    def declare_params(self):
        # declare a bunch of parameters that will be reused later. doesn't actually do anything
        size = self.config['hidden_size']
        label_size = self.config['label_size']
        attr_size = self.config['attr_size']
        embedding_size = self.config['embedding_size']

        with tf.variable_scope('Parameters'):
            with tf.variable_scope("joint"):
                u_alpha = tf.get_variable("u_alpha", [1, size * 4, 2], regularizer=tf.contrib.layers.l2_regularizer(0.0),
                                                dtype=data_type())
                b_alpha = tf.get_variable("b_alpha", [1, 2], regularizer=tf.contrib.layers.l2_regularizer(0.0),
                                        dtype=data_type())
            for d in set(self.config['dependencies'].values()):
                with tf.variable_scope(d):
                    # the second dimension doesn't have to be "size", but does have to match softmax_w's first dimension
                    for i in self.config['dependencies']:
                        if self.config['dependencies'][i] == d:
                            tf.get_variable('U_' + i, [1, size, size], dtype=data_type())

                    tf.get_variable('u_end', [size], dtype=data_type())

                    tf.get_variable("attr_w", [1, size, attr_size], dtype=data_type())
                    tf.get_variable("attr_b", [1, attr_size], dtype=data_type())
                    tf.get_variable("v_attr", [1, label_size, attr_size], dtype=data_type())

                    tf.get_variable("label_w", [1, size, label_size], dtype=data_type())
                    tf.get_variable("label_b", [1, label_size], dtype=data_type())

                    if d == 'reverse':
                        tf.get_variable('u_right_hole', [1, size * 2], dtype=data_type())

                    # should we have separate parameters for the forward and reverse directions?
                    # currently, we do
                    #with tf.device("/cpu:0"):
                    tf.get_variable("label_embedding", [label_size, embedding_size], dtype=data_type())
                    tf.get_variable("attr_embedding", [attr_size, embedding_size], dtype=data_type())

    # this is the only other method that modifies "self"
    def init_iterator(self):
        feature = {}
        shape = {}
        for k in self.config['features']:
            feature[k] = tf.int32
            shape[k] = tf.TensorShape([None, None])
        iterator = tf.data.Iterator.from_structure(feature, shape)

        features = {}
        shapes = {}
        self.placeholders['features'] = {}
        for k in self.config['features']:
            features[k] = tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
            shapes[k] = [None] # the FixedLenSequence implies a second dimension, and batching gives the third
            self.placeholders['features'][k] = tf.placeholder(tf.int32, shape=[None, None], name=k+'_placeholder')

        # inference/node dataset
        batch_size = tf.cast(self.batch_size, tf.int64)
        dataset = tf.data.Dataset.from_tensor_slices(self.placeholders['features']).padded_batch(batch_size, shapes)
        self.ops['node_iter'] = iterator.make_initializer(dataset)

        # training/file dataset
        def parse_function(example):
            f = tf.parse_single_example(example, features)
            for k in f:
                f[k] = tf.cast(f[k], tf.int32)
            return f
        self.placeholders['filename'] = tf.placeholder(tf.string, [], name='filename_placeholder')
        dataset = tf.data.TFRecordDataset(self.placeholders['filename'])
        dataset = dataset.map(parse_function) \
                         .cache() \
                         .shuffle(buffer_size=100) \
                         .repeat() \
                         .padded_batch(batch_size, shapes)
        self.ops['file_iter'] = iterator.make_initializer(dataset)

        return iterator.get_next()


    def __init__(self, config):
        self.config = config

        self.placeholders = {}
        self.fetches = { 'initial' : {}}
        self.ops = {}

        lr = tf.get_variable('lr', initializer=0.0, trainable=False)
        self.placeholders['new_lr'] = new_lr = tf.placeholder(data_type(), shape=[], name="new_learning_rate")
        self.ops['lr_update'] = tf.assign(lr, new_lr)

        self.batch_size = batch_size = tf.get_variable('batch_size', initializer=1, trainable=False, dtype=tf.int32)
        self.placeholders['new_batch_size'] = new_batch_size = tf.placeholder(tf.int32, shape=[], name='new_batch_size')
        self.ops['batch_size_update'] = tf.assign(batch_size, new_batch_size)

        # map the dependencies to their relevant direction
        self.directions = []
        dependencies = {}
        for d in config['dependencies']:
            dependencies[d] = possible_dependencies[d]
            if possible_dependencies[d] not in self.directions:
                self.directions.append(possible_dependencies[d])
        # TODO: this is the only place we modify "config" in here. We could just use "self", but
        # then we still want to be able to propagate this information.
        config['dependencies'] = dependencies
        config['directions'] = self.directions

        self.rows = rows = self.init_iterator()


        # does not always equal batch size, if we got a small batch
        num_rows = tf.shape(rows['label_index'])[0]

        self.placeholders['drop_prob'] = tf.placeholder_with_default(0.0, shape=(), name='dropout_placeholder')
        self.placeholders['is_inference'] = tf.placeholder(tf.bool, [], name='is_inference_placeholder')

        num_layers = config['num_layers']
        dependencies = config['dependencies']
        embedding_size = config['embedding_size']

        data_length = tf.shape(rows['label_index'])[1]
        hidden_size = config['hidden_size']

        self.declare_params()

        cells = RNNTensorArrayCell(dependencies, data_length, hidden_size, num_rows, num_layers, data_type())
        array = RNNTensorArrayCell.array

        # extra TensorArrays for children, hurray!!
        #self.fetches['children_predictor_states'] = children_predictor_states = array.init_lstm_array()
        #self.fetches['children_tmp_states'] = children_tmp_states = array.init_lstm_array()
        #self.fetches['children_output'] = children_output = array.init_tensor_array()

        #self.fetches['initial']['children_tmp_states'] = initial_children_tmp_states = []

        ## separate RNN just to predict a parent label given children embedding sequence
        #with tf.variable_scope("Parameters/reverse"):
        #    with tf.variable_scope("Children_RNN", reuse=tf.AUTO_REUSE):
        #        children_predictor = tf.contrib.rnn.MultiRNNCell(
        #                [self.rnn_cell(False) for i in range(num_layers)], state_is_tuple=True)

        # XXX Is this right?
        #children_predictor_initial_states = children_predictor.zero_state(1, data_type())
        #array.save_multi_lstm_state(children_predictor_states, children_predictor_initial_states, 0)
        #if 'children' in dependencies:
        #    for j in range(num_layers):
        #        initial_children_tmp_states.append(dependency_cells['children'][j].zero_state(1, data_type()))
        #    (init_output, next_state) = children_predictor(dependency_initial_outputs['children'][num_layers-1],
        #                                children_predictor_initial_states)
        #    initial_children_predictor_output = init_output
        #else:
        #    for j in range(num_layers):
        #        initial_children_tmp_states.append(dependency_cells[dependencies[0]][j].zero_state(1, data_type()))
        #    initial_children_predictor_output = tf.constant(0)



        # this returns true as long as the loop counter is less than the length of the example
        def loop_cond_wrapper(dependency):
            def loop_cond (ctr, cell_data, children_tmp_states, children_output, children_predictor_states):
                if self.config['dependencies'][dependency] == 'forward':
                    return tf.less(ctr, data_length)
                else:
                    return tf.greater(ctr, 0)
            return loop_cond

        def loop_body_wrapper(dependency, layer):
            def loop_body(ctr, cell_data, children_tmp_states, children_output, children_predictor_states):
                cells.data = cell_data


                #def infer(index):
                #    return tf.cond(self.placeholders['is_inference'], lambda: 0, lambda: index)

                # During inference, we want to use the directly-supplied label for the parent, since each node is passed in
                # one-by-one and we won't have access to the parent when the child is passed in. During training, we have
                # all nodes in the example at once, so can directly grab the parent's label
                #dependency_node = infer(tf.gather(rows[dependency], ctr, axis=1,
                #                        name=(dependency+"Gather")) if dependency != 'children' else 0)


                inp = self.embed(ctr, dependency) if layer == 0 \
                      else cells.rnn[dependency][layer-1].get_output(cell_data, ctr)
                state_source = dependency if dependency != 'children' else 'left_child'
                dependency_idx = tf.gather(rows[state_source], ctr, axis=1, name=(i+"Gather"))

                if dependency != 'children':
                    cells.rnn[dependency][layer].step(cell_data, ctr, inp, dependency_idx)
                else:
                    is_leaf = tf.cast(tf.gather(rows['is_leaf'], ctr, axis=1, name="IsLeafGather"), tf.bool)
                    right_sibling = tf.cast(tf.gather(rows['right_sibling'], ctr, axis=1), tf.bool)
                    left_child = tf.cast(tf.gather(rows['left_child'], ctr, axis=1), tf.bool)
                    parent_idx = tf.gather(rows['parent'], ctr, axis=1)
                    parent_inp = self.embed(parent_idx, dependency, dependency) if layer == 0 \
                                 else cells.rnn[dependency][layer-1].get_output(cell_data, parent_idx)
                    inp = tf.stack([inp, parent_embedding])

                    cells.rnn[dependency][layer].step(cell_data, ctr, tree_embedding, left_child, add_state=right_sibling)

                ctr = tf.add(ctr, 1) if self.config['dependencies'][dependency] == 'forward' else tf.subtract(ctr, 1)

                return ctr, cell_data, children_tmp_states, children_output, children_predictor_states
            return loop_body

        h_pred = {}
        for i in dependencies:
            direction = dependencies[i]

            # TODO: don't need to loop over num_layers if we aren't handling children.
            # Can just grab the states and outputs of the final layers
            for layer in range(num_layers):

                # forward starts iterating from 1, since 0 is the "empty" parent/sibling
                ctr = 1 if direction == 'forward' else data_length - 1

                children_tmp_states, children_output, children_predictor_states = 0,0,0
                ctr, cells.data, children_tmp_states, children_output, children_predictor_states = \
                    tf.while_loop(loop_cond_wrapper(i), loop_body_wrapper(i, layer),
                                [ctr,
                                cells.data,
                                children_tmp_states,
                                children_output,
                                children_predictor_states],
                                parallel_iterations=1)

            if i != 'children':
                #predicted_output = cells.rnn[i][num_layers-1].stack_output(cells.data)
                # technically for linear, we just need to shift the outputs to align with the tokens. But that
                # doesn't work in the general case
                predicted_output = cells.rnn[i][num_layers-1].stack_output(cells.data)
                #labels = tf.concat(self.rows[i], axis=0)
                predicted_output = tf.gather(predicted_output, self.rows[i])
            else:
                predicted_output = array.gather(children_output, ctr)

            if config['model'] != 'linear':
                with tf.variable_scope("Parameters/{}".format(dependencies[i]), reuse=True):
                    U_dependency = tf.get_variable('U_' + i, [1, hidden_size, hidden_size], dtype=data_type())
                predicted_output = tf.matmul(predicted_output, U_dependency)
            if direction not in h_pred:
                #h_pred[direction] = tf.zeros([num_rows, data_length, hidden_size])
                h_pred[direction] = tf.zeros([data_length, hidden_size])
            h_pred[direction] += predicted_output
            right_hole_h_pred = tf.zeros([num_rows,hidden_size])

                    #right_hole_h_pred = tf.zeros([1,size])
                        # XXX only do this in reverse direction
                        # if right_hole is non-zero, we still want to read from 0 in the inference case
                        #if 'children' in config['dependencies']:
                        #    h_pred_tmp = lambda: tf.cond(config['placeholders']['is_inference'],
                        #                        lambda: h_pred_infer[i],
                        #                        lambda: h_pred[i].read(right_hole))
                        #    right_hole_h_pred = tf.cond(tf.cast(right_hole, tf.bool),
                        #                                lambda: right_hole_h_pred + h_pred_tmp(),
                        #                                lambda: right_hole_h_pred)

        labels, logits = {}, {}
        for direction in self.directions:
            labels[direction], logits[direction] = self.collect_directional_logits(direction,
                                                        h_pred[direction], right_hole_h_pred)

        loss, train_vars = self.calculate_loss_and_tvars(labels, logits)
        self.fetches['loss'] = loss
        #optimizer = tf.train.GradientDescentOptimizer(lr)
        optimizer = tf.train.AdamOptimizer(1e-2)
        self.ops['train'] = optimizer.apply_gradients(train_vars, global_step=tf.train.get_or_create_global_step())

        if len(self.directions) == 2:
            #labels, logits = self.collect_joint_logits(h_pred, loss)
            #joint_loss, joint_tvars = self.calculate_loss_and_tvars(labels, logits)
            joint_loss, joint_tvars = self.collect_joint_logits(h_pred, loss)
            #loss.update(joint_loss)
            self.fetches['loss'].update(joint_loss)
            #optimizer = tf.train.GradientDescentOptimizer(.0uu01)
            # use a different global step
            self.ops['train_joint'] = optimizer.apply_gradients(joint_tvars, global_step=tf.train.get_or_create_global_step())
