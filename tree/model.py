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
from tensorarray import RNNTensorArray, RNNTensorArrayCell, define_scope

def data_type():
    return tf.float32

valid_dependencies = {
    'children': 'bottom',
    'parent': 'top',
    'left_prior': 'left',
    'left_child': 'left',
    'left_sibling': 'left',
    'right_prior': 'right',
    'right_child': 'right',
    'right_sibling': 'right',
}

# TODO: validate the dependency_configs against valid_dependencies
dependency_configs = {
    'ast': {
        # should never need to have a dependency appear twice in the list
        'left_sibling': [('top', 'left', ['left_sibling'])],
        'children': [('bottom', 'right', ['children'])],
        'parent': [('top', 'left', ['parent'])],
        'd1': [('bottom', 'right', ['children', 'right_sibling'])],
        'd2': [('top', 'left', ['parent', 'left_sibling'])],
        #'d3': [('bottom_right', [
        #{ 'bottom_right': [['children'], ['right_children'],
        #                   ['children', 'initial_right_children']]},
        #{ 'bottom_left': [['left_children'],
        #                   ['children', 'initial_left_children'],
        #                   ['children', 'initial_left_children', 'initial_right_children']]},
        #{ 'top_left': [['parent', 'left_sibling'],
        #               ['parent', 'left_sibling', 'initial_left_children'],
        #               ['parent', 'left_sibling', 'initial_left_children', 'initial_right_children']]},
        #{ 'top_right': [['parent', 'right_sibling'],
        #               ['parent', 'right_sibling', 'initial_right_children'],
        #               ['parent', 'right_sibling', 'initial_left_children', 'initial_right_children']]}
    },
    'linear': {
        'reverse': [('bottom', 'right', ['right_sibling'])],
        'forward': [('top', 'left', ['left_sibling'])],
        'both': [('top', 'left', ['left_sibling']), ('bottom', 'right', ['right_sibling'])]
    }
}

# TODO: similarly, validate the possible_joints are found in dependency_configs
joint_configs = {
    'ast': {
        'j1': ['d1', 'd2']
    },
    'linear': {
        'both': [ 'forward', 'reverse' ]
    }
}


class TRNNBaseModel(object):
    # TODO: cache this result?
    @define_scope
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
                assert(isinstance(d, tf.Tensor) or isinstance(d, tf.Operation) or isinstance(d, tf.SparseTensor)
                       or isinstance(d, tf.Variable))
                return d.name

        return {
            'placeholders': extract_names(self.placeholders),
            'ops': extract_names(self.ops),
            'initials': extract_names(self.initials),
            'fetches' : extract_names(self.fetches)
        }

    @define_scope
    def calculate_loss_and_tvars(self, labels, logits):
        loss = {}
        probabilities = {}
        mask = tf.cast(self.rows['mask'], tf.float32)
        num_tokens = tf.reduce_sum(mask)
        total_loss = 0.0
        for k in labels:
            if self.is_joint:
                cross, predict = (tf.nn.softmax_cross_entropy_with_logits, tf.nn.softmax)
            else:
                cross, predict = (tf.nn.sigmoid_cross_entropy_with_logits, tf.nn.sigmoid) if len(logits[k].shape) == 2 \
                                else (tf.nn.sparse_softmax_cross_entropy_with_logits, tf.nn.softmax)
            loss[k] = tf.reduce_sum(cross(labels=labels[k], logits=logits[k]) * mask) / num_tokens
            total_loss += loss[k]
            probabilities[k] = predict(logits[k])
        # TODO: a tvar can appear multiple times. But we want to clip norms for losses separately???
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.super_scope + self.scope)
        grads, _ = tf.clip_by_global_norm(tf.gradients(total_loss, tvars), self.max_grad_norm)
        return loss, probabilities, zip(grads, tvars)

    @define_scope
    def init_iterator(self):
        placeholders = { 'features': {} }
        iter_ops = {}

        parse_features = {}
        parse_shapes = {}
        iter_features = {}
        iter_shapes = {}
        for k in self.features:
            parse_features[k] = tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
            # the FixedLenSequence implies a second dimension, and batching gives the third
            parse_shapes[k] = [None]
            iter_features[k] = tf.int32
            iter_shapes[k] = tf.TensorShape([None, None])

            #placeholders['features'][k] = tf.sparse_placeholder(tf.int32, shape=[None, None], name=k+'_placeholder')
            placeholders['features'][k] = tf.placeholder(tf.int32, shape=[None, None], name=k+'_placeholder')
        iterator = tf.data.Iterator.from_structure(iter_features, iter_shapes)

        # inference/node dataset
        batch_size = tf.cast(self.batch_size, tf.int64)
        # batch size should always be 1 here anyway
        def densify(example):
            return { k: tf.sparse_tensor_to_dense(example[k]) for k in example }
        dataset = tf.data.Dataset.from_tensor_slices(placeholders['features']) \
                                                    .batch(batch_size)
                                                    #.map(densify) \
        iter_ops['tensor_iter'] = iterator.make_initializer(dataset)

        # training/file dataset
        def parse_function(example):
            f = tf.parse_single_example(example, parse_features)
            for k in f:
                f[k] = tf.cast(f[k], tf.int32)
            return f
        placeholders['filename'] = tf.placeholder(tf.string, [], name='filename_placeholder')
        dataset = tf.data.TFRecordDataset(placeholders['filename'])
        dataset = dataset.map(parse_function) \
                         .shuffle(buffer_size=100) \
                         .repeat() \
                         .padded_batch(batch_size, parse_shapes)
                         #.cache() \
        iter_ops['file_iter'] = iterator.make_initializer(dataset)

        return iterator.get_next(), iter_ops, placeholders


    def __init__(self, is_joint, model, dconfig, label_size, attr_size, hidden_size,
                 features, num_layers, max_grad_norm, rows=None):
        self.is_joint = is_joint
        self.model = model
        self.dconfig = dconfig
        self.label_size = label_size
        self.attr_size = attr_size
        self.hidden_size = hidden_size
        self.features = features
        self.num_layers = num_layers
        self.max_grad_norm = max_grad_norm

        # grab the current scope, in case the model was instantiated within a scope
        # only update the subset of weights relevant to these losses
        self.super_scope = tf.get_variable_scope().name
        if self.super_scope != '':
            self.super_scope += '/'

        self.scope = "{}/{}/{}".format(model, 'Joint' if is_joint else 'Dependency', dconfig)
        depends = dependency_configs if not is_joint else joint_configs
        self.dependencies = depends[model][dconfig]

        # these four fields are meant to be accessed outside the model, and can be exported via
        # model.output_tensor_names()
        self.placeholders = {}
        self.fetches = {}
        self.initials = {}
        self.ops = {}



        with tf.variable_scope(self.scope):
            # so that we can share an iterator between different models, rather than iterate over the same data
            # multiple times
            if rows is None:
                self.batch_size = batch_size = tf.get_variable('batch_size', initializer=1, trainable=False, dtype=tf.int32)
                self.placeholders['drop_prob'] = tf.placeholder_with_default(0.0, shape=(), name='dropout_placeholder')
                self.placeholders['is_inference'] = tf.placeholder(tf.bool, [], name='is_inference_placeholder')
                self.placeholders['new_batch_size'] = new_batch_size = tf.placeholder(tf.int32, shape=[], name='new_batch_size')
                self.ops['batch_size_update'] = tf.assign(batch_size, new_batch_size)
                self.rows, iter_ops, iter_placeholders = self.init_iterator()
                self.placeholders.update(iter_placeholders)
                self.ops.update(iter_ops)
            else:
                self.rows = rows

            # num_rows does not always equal batch size, if we got a small batch
            self.num_rows = tf.shape(self.rows['label_index'])[0]
            self.data_length = tf.shape(self.rows['label_index'])[1]
            self.params = self.declare_params()

            self.global_step = tf.get_variable('global_step', dtype=tf.int32, initializer=0, trainable=False)

            labels, logits = self.collect_logits()

            optimizer = tf.train.AdamOptimizer(1e-2)
            self.fetches['loss'], self.fetches['probabilities'], train_vars = self.calculate_loss_and_tvars(labels, logits)
            self.ops['train'] = optimizer.apply_gradients(train_vars, global_step=self.global_step)

            self.fetches['cells'] = self.cells.get_fetches()
            self.initials = self.cells.initials



################################################################################
################################################################################
################################################################################



class TRNNJointModel(TRNNBaseModel):

    @define_scope
    def collect_logits(self):
        # TODO: expand to more than just label_index?
        actual_label = tf.one_hot(self.rows['label_index'][0], self.label_size, axis=1)

        probs = []
        h_preds = []
        total_prob = None
        for i in self.dependencies:
            h_preds.append(self.joints[i].h_pred)
            prob = self.joints[i].fetches['loss']['label_index']['probabilities']
            prob = tf.multiply(prob, actual_label)
            prob = tf.reduce_sum(prob, axis=2)
            if total_prob is None:
                total_prob = prob
            else:
                total_prob += prob
            probs.append(prob)
        labels = { 'label_index': probs / total_prob }

        #forward_embedding = self.embed(self.rows['left_sibling'][0], 'left_sibling')
        #reverse_embedding = self.embed(self.rows['right_sibling'][0], 'right_sibling')

        # something other than transpose?
        #labels = tf.transpose([forward / (forward + reverse), reverse / (forward + reverse)])
        #combined = tf.concat([h_pred_ctr['forward'], h_pred_ctr['reverse'], forward_embedding, reverse_embedding], axis=2)
        combined = tf.concat(h_preds, axis=2)
        #num_rows = tf.shape(self.rows['label_index'])[0]
        logits = { 'label_index': tf.matmul(combined, tf.tile(self.params['u_alpha'],
                                  [self.num_rows, 1, 1])) + self.params['b_alpha'] }

        labels['label_index'] = tf.transpose(labels['label_index'])
        return labels, logits

    @define_scope
    def declare_params(self):
        size = self.hidden_size
        num_joints = len(self.dependencies)

        params = {}
        #params['u_alpha'] = tf.get_variable("u_alpha", [1, size * (2 * num_joints), num_joints],
        params['u_alpha'] = tf.get_variable("u_alpha", [1, size * num_joints, num_joints],
                regularizer=tf.contrib.layers.l2_regularizer(0.0), dtype=data_type())
        params['b_alpha'] = tf.get_variable("b_alpha", [1, num_joints],
                regularizer=tf.contrib.layers.l2_regularizer(0.0), dtype=data_type())
        return params

    def __init__(self, joints, *args, **kwargs):
        # TODO: enforce that joints has the right things for that dependency
        self.joints = joints
        super().__init__(True, *args, **kwargs)







################################################################################
################################################################################







class TRNNModel(TRNNBaseModel):


    @define_scope
    def collect_logits(self):

        h_pred, left_right = self.calculate_hpred()
        p = self.params

        logits = {}
        labels = {}

        # tiles a parameter matrix/vector for matmul on a whole batch
        def tile(matrix):
            return tf.tile(matrix, [self.num_rows, 1, 1])

        # this is the only directional loss that linear uses
        logits['label_index'] = tf.matmul(h_pred, tile(p['label_w'])) + p['label_b']
        labels['label_index'] = self.rows['label_index']

        if self.model == 'ast':
            # only attrs for constants and IDs?
            attr_w = tf.gather(p['attr_w'], self.rows['label_index'])
            attr_b = tf.gather(p['attr_b'], self.rows['label_index'])
            logits['attr_index'] = tf.squeeze(tf.matmul(tf.expand_dims(h_pred, axis=2), attr_w), axis=2) + attr_b
            labels['attr_index'] = self.rows['attr_index']

            # loss for whether we are the last sibling on the left or right
            end = 'last_sibling' if left_right == 'left' else 'first_sibling'
            u_end = tf.gather_nd(p['u_end'], tf.stack([self.rows['label_index'], self.rows['attr_index']], axis=2))
            # TODO: bias?
            logits[end] = tf.reduce_sum(tf.multiply(u_end, h_pred), axis=2)
            labels[end] = tf.cast(self.rows[end], data_type())


        return labels, logits

    @define_scope
    def embed(self, index):
        label_index = self.gather_dependency('label_index', index)
        attr_index = self.gather_dependency('attr_index', index)

        node_label_embedding = tf.gather(self.params['label_embed'], label_index, name="LabelEmbedGather", axis=0)
        node_attr_embedding = tf.gather(self.params['attr_embed'], attr_index, name="AttrEmbedGather", axis=0)
        node_embedding = tf.concat([node_label_embedding, node_attr_embedding], 1)
        return node_embedding

    @define_scope
    def declare_params(self):
        # declare a bunch of parameters that will be reused later. doesn't actually do anything
        size = self.hidden_size
        label_size = self.label_size
        attr_size = self.attr_size
        embed_size = size / 2

        params = {}

        all_dependencies = set()
        for (_, _, dependencies) in self.dependencies:
            for dependency in dependencies:
                all_dependencies.add(dependency)

        params['u_end'] = tf.get_variable('u_end', [label_size, attr_size, size], dtype=data_type())

        params['attr_w'] = tf.get_variable("attr_w", [label_size, size, attr_size], dtype=data_type())
        params['attr_b'] = tf.get_variable("attr_b", [label_size, attr_size], dtype=data_type())
        #params['attr_v'] = tf.get_variable("attr_v", [label_size, attr_size], dtype=data_type())

        params['label_w'] = tf.get_variable("label_w", [1, size, label_size], dtype=data_type())
        params['label_b'] = tf.get_variable("label_b", [1, label_size], dtype=data_type())

        #if d == 'reverse':
        #    p[d]['u_right_hole'] = tf.get_variable('u_right_hole', [1, size * 2], dtype=data_type())

        #with tf.device("/cpu:0"):
        params['label_embed'] = tf.get_variable("label_embedding", [label_size, embed_size], dtype=data_type())
        params['attr_embed'] = tf.get_variable("attr_embedding", [attr_size, embed_size], dtype=data_type())

        # no need to add in the extra parameters if we aren't combining with any other output
        if len(all_dependencies) > 1:
            params['U'] = {}
            for d in all_dependencies:
                with tf.variable_scope(d):
                    # the second dimension doesn't have to be "size", but does have to match softmax_w's
                    # first dimension
                    params['U'][d] = tf.get_variable('U', [1, size, size], dtype=data_type())

        self.cells = RNNTensorArrayCell(all_dependencies, self.data_length, size, self.num_rows,
                                                    self.num_layers, data_type())

        return params

    def generate_loop_cond(self, left_right):
        def loop_cond (ctr, cell_data):
            if left_right == 'left':
                return tf.less(ctr, self.data_length)
            else:
                return tf.greater(ctr, 0)
        return loop_cond

    def gather_dependency(self, dependency, idx):
        if len(idx.shape) == 0:
            return self.rows[dependency][:,idx]
        assert len(idx.shape) == 1
        r = tf.range(tf.size(idx))
        return tf.gather_nd(self.rows[dependency], tf.stack([r, idx], axis=1))


    def generate_loop_body(self, dependencies, layer):
        def get_input(cell_data, dependency, idx):
            return self.embed(idx) if layer == 0 \
                   else self.cells.rnn[dependency][layer-1].get_output(cell_data, idx)
        def loop_body(ctr, cell_data):
            for dependency in dependencies[2]:
                inp = get_input(cell_data, dependency, ctr)

                if dependency != 'children':
                    dependency_idx = self.gather_dependency(dependency, ctr)
                    add_idx = None
                else:
                    dependency_idx = self.gather_dependency('left_child', ctr)

                    # TODO: this is currently handled
                    parent_idx = self.gather_dependency('parent', ctr)
                    parent_inp = get_input(cell_data, dependency, parent_idx)
                    inp = tf.stack([inp, parent_inp])

                    right_sibling = self.gather_dependency('right_sibling', ctr)
                    add_idx = right_sibling if layer == self.num_layers - 1 else None

                self.cells.rnn[dependency][layer].step(cell_data, ctr, inp, dependency_idx, add_idx=add_idx)

            #ctr = tf.Print(ctr, [ctr, tf.shape(self.rows['label_index'])])
            ctr = tf.add(ctr, 1) if dependencies[1] == 'left' else tf.subtract(ctr, 1)

            return ctr, cell_data
        return loop_body

    @define_scope
    def calculate_hpred(self):
        #h_pred = tf.zeros([num_rows, data_length, hidden_size])
        # TODO: move this to "fetches"?
        self.h_pred = tf.zeros([self.num_rows, self.data_length, self.hidden_size])
        with tf.name_scope('main_loop_body'):
            for i in range(len(self.dependencies)):
                left_right = self.dependencies[i][1]

                # TODO: don't need to loop over num_layers if we aren't handling children.
                # Can just grab the states and outputs of the final layers
                for layer in range(self.num_layers):

                    # forward starts iterating from 1, since 0 is the "empty" dependency
                    ctr = 1 if left_right == 'left' else self.data_length - 1

                    _, self.cells.data = tf.while_loop(self.generate_loop_cond(left_right),
                                                self.generate_loop_body(self.dependencies[i], layer),
                                                [ctr, self.cells.data],
                                                parallel_iterations=1)

                # Technically for left/right siblings, we just need to shift the outputs to align with the tokens.
                # But that doesn't work in the general case.
                for dependency in self.dependencies[i][2]:
                    predicted_output = self.cells.rnn[dependency][self.num_layers-1].stack_output(self.cells.data)
                    #predicted_output = tf.gather(predicted_output, self.rows[dependency if dependency != 'children' else 'left_child'])
                    # predicted_output == [batch_size, data_length, hidden_size]
                    r = tf.range(self.num_rows, dtype=tf.int32)
                    rows = self.rows[dependency if dependency != 'children' else 'left_child']
                    # XXX better way of doing this?
                    predicted_output = tf.map_fn(lambda x: tf.gather(predicted_output[x], rows[x]), r, data_type())

                    if 'U' in self.params:
                        predicted_output = tf.matmul(predicted_output, tf.tile(self.params['U'][dependency], [self.num_rows, 1, 1]))
                    self.h_pred += predicted_output

            # the very last left_right gives us which direction the generation must be going
            return self.h_pred, left_right


    def __init__(self, *args, **kwargs):
        super().__init__(False, *args, **kwargs)
