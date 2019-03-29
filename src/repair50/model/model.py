import tensorflow as tf # type:ignore
import einops as e # type:ignore
from .tensorarray import RNNTensorArray, RNNTensorArrayCell, define_scope

from .config import dependency_configs, joint_configs

def data_type():
    return tf.float32


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
    def reconstruct_example(self, rows, new_rows, keys):
        key = keys.pop(0)
        if len(keys) == 0:
            new_rows[key] = rows
        else:
            # handle the pointer stuff
            if keys[0].isdigit():
                if key not in new_rows:
                    # TODO ROB: this is the hard-coded max pointer memory size
                    new_rows[key] = [0] * self.mem_size
                new_rows[key][int(keys[0])] = rows
                return
            if key not in new_rows:
                new_rows[key] = {}
            self.reconstruct_example(rows, new_rows[key], keys)


    @define_scope
    def calculate_loss_and_tvars(self, labels, logits):
        loss = {}
        probabilities = {}
        # mask is the same regardless of direction
        mask = tf.cast(self.rows['forward']['mask'], tf.float32)
        num_tokens = tf.reduce_sum(mask)
        total_loss = 0.0
        for k in labels:
            if k == 'pointers':
                mask = tf.tile(tf.expand_dims(mask,axis=2), [1,1,self.mem_size])
                mask = mask * tf.cast(self.pointer_mask, tf.float32)
                cross, predict = (tf.nn.sigmoid_cross_entropy_with_logits, tf.nn.sigmoid)
            elif self.is_joint:
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
        dataset = tf.data.Dataset.from_tensor_slices(placeholders['features']) \
                                                    .batch(batch_size)
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
                         .padded_batch(batch_size, parse_shapes)
                         #.repeat() \
                         #.shuffle(buffer_size=100) \
                         #.cache() \
        iter_ops['file_iter'] = iterator.make_initializer(dataset)

        return iterator.get_next(), iter_ops, placeholders


    def __init__(self, is_joint, dconfig, config, transitions, rows=None):
        self.is_joint = is_joint
        self.dconfig = dconfig
        self.transitions = transitions
        for k in ['label_size', 'attr_size', 'transitions_size', 'hidden_size', 'features', 'num_layers', 'max_grad_norm']:
            if k in config:
                setattr(self, k, config[k])
        self.mem_size = 20

        # grab the current scope, in case the model was instantiated within a scope
        # only update the subset of weights relevant to these losses
        self.super_scope = tf.get_variable_scope().name
        if self.super_scope != '':
            self.super_scope += '/'

        self.scope = "{}/{}".format('Joint' if is_joint else 'Dependency', dconfig)
        depends = dependency_configs if not is_joint else joint_configs
        self.dependencies = depends[dconfig]

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
                #self.placeholders['is_inference'] = tf.placeholder(tf.bool, [], name='is_inference_placeholder')
                self.placeholders['new_batch_size'] = new_batch_size = tf.placeholder(tf.int32, shape=[], name='new_batch_size')
                self.ops['batch_size_update'] = tf.assign(batch_size, new_batch_size)
                self.rows, iter_ops, iter_placeholders = self.init_iterator()
                self.placeholders.update(iter_placeholders)
                self.ops.update(iter_ops)
                # this is to handle the fact that we couldn't do a nested dictionary in preprocessing
                rows = {}
                for k in self.rows:
                    self.reconstruct_example(self.rows[k], rows, k.split('-'))
                for i in ['forward', 'reverse']:
                    for j in ['memory', 'mask']:
                        rows[i]['pointers'][j] = e.rearrange(rows[i]['pointers'][j], 'm b n -> b n m');
            self.rows = rows

            # num_rows does not always equal batch size, if we got a small batch
            self.num_rows = tf.shape(self.rows['forward']['label_index'])[0]
            self.data_length = tf.shape(self.rows['forward']['label_index'])[1]
            self.params = self.declare_params()

            self.global_step = tf.get_variable('global_step', dtype=tf.int32, initializer=0, trainable=False)

            labels, logits = self.collect_logits()

            optimizer = tf.train.AdamOptimizer(1e-2)
            self.fetches['loss'], self.fetches['probabilities'], train_vars = self.calculate_loss_and_tvars(labels, logits)
            self.ops['train'] = optimizer.apply_gradients(train_vars, global_step=self.global_step)

            self.fetches['cells'] = {}
            for d in self.cells.fetches:
                self.fetches['cells'][d] = {}
                for k in self.cells.fetches[d]:
                    self.fetches['cells'][d][k] = self.cells.fetches[d][k]()
            self.cells.fetches
            self.initials = self.cells.initials



################################################################################
################################################################################
################################################################################



class TRNNJointModel(TRNNBaseModel):

    @define_scope
    def collect_logits(self):
        # TODO: expand to more than just label_index?
        actual_label = tf.one_hot(self.rows['forward']['label_index'][0], self.label_size, axis=1)

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

        h_pred, direction, forward_array = self.calculate_hpred()
        p = self.params

        logits = {}
        labels = {}

        rows = self.rows[direction]

        # tiles a parameter matrix/vector for matmul on a whole batch
        def tile(matrix):
            return tf.tile(matrix, [self.num_rows, 1, 1])

        end = 'last_sibling' if (direction == 'forward') == forward_array else 'first_sibling'

        if self.transitions:
            logits['transitions_index'] = tf.matmul(h_pred, tile(p['transitions_w'])) + p['transitions_b']
            labels['transitions_index'] = rows['transitions_index']

            #u_end = tf.gather(p['u_end'], rows['transitions_index'])
            #logits[end] = tf.reduce_sum(tf.multiply(u_end, h_pred), axis=1)
            #labels[end] = tf.cast(rows[end], data_type())
        else:
            logits['label_index'] = tf.matmul(h_pred, tile(p['label_w'])) + p['label_b']
            labels['label_index'] = rows['label_index']

            # batch_size x data_length x size x attr_size
            attr_w = tf.gather(p['attr_w'], rows['label_index'])
            # batch_size x data_length x attr_size
            attr_b = tf.gather(p['attr_b'], rows['label_index'])


            logits['attr_index'] = tf.squeeze(tf.matmul(tf.expand_dims(h_pred, axis=2), attr_w), axis=2) + attr_b
            labels['attr_index'] = rows['attr_index']

            self.fetches['attr_all'] = tf.nn.softmax(tf.tensordot(h_pred, p['attr_w'], [[2],[1]]) + p['attr_b'])

            # loss for whether we are the last sibling on the left or right
            u_end = tf.gather_nd(p['u_end'], tf.stack([rows['label_index'], rows['attr_index']], axis=2))
            # TODO: bias?
            logits[end] = tf.reduce_sum(tf.multiply(u_end, h_pred), axis=2)
            labels[end] = tf.cast(rows[end], data_type())

            # hpred: shape = [batch, nodes, hidden_size]
            # self.rows[direction]['pointers']['memory']: shape = [batch, nodes, mem_size]
            # elems: shape = [batch, nodes, (hidden_size + mem_size)]
            elems = tf.concat([h_pred, tf.cast(rows['pointers']['memory'], tf.float32)], axis=2)

            def pointer_map(elems):
                h_pred = elems[:,:self.hidden_size]
                mem = tf.cast(elems[:,self.hidden_size:], tf.int32)
                return tf.gather(h_pred, mem)


            # shape = [batch, nodes, mem, size]
            pointers = tf.map_fn(pointer_map, elems)

            # tensordot generalizes matrix multiplication over batch_size and num_nodes
            w1 = tf.tensordot(pointers, p['pointers_w1'], [[3],[0]])
            w2 = tf.tensordot(h_pred, p['pointers_w2'], [[2],[0]])
            w2 = tf.reshape(w2, [self.num_rows, self.data_length, self.mem_size, self.hidden_size])
            scores = tf.reduce_sum(p['pointers_v'] * tf.tanh(w1 + w2) + p['pointers_b'], axis=3)
            # TODO: this is hackishly inserted
            self.pointer_mask = tf.not_equal(rows['pointers']['memory'], 0)
            logits['pointers'] = scores
            labels['pointers'] = tf.cast(rows['pointers']['mask'], data_type())


        return labels, logits

    @define_scope
    def embed(self, index, direction, parent=False):

        if self.transitions:
            transitions_index = self.rows[direction]['transitions_index' if not parent else 'parent_transitions_index'][:,index]
            return tf.gather(self.params['transitions_embed'], transitions_index, name="TransitionsEmbedGather", axis=0)
        else:
            label_index = self.rows[direction]['label_index' if not parent else 'parent_label_index'][:,index]
            attr_index = self.rows[direction]['attr_index' if not parent else 'parent_attr_index'][:,index]

            label_embedding = tf.gather(self.params['label_embed'], label_index, name="LabelEmbedGather", axis=0)
            attr_embedding = tf.gather(self.params['attr_embed'], attr_index, name="AttrEmbedGather", axis=0)
            embeddings = [label_embedding, attr_embedding]
            return tf.concat(embeddings, 1)

    @define_scope
    def declare_params(self):
        # declare a bunch of parameters that will be reused later. doesn't actually do anything
        size = self.hidden_size
        params = {}

        all_dependencies = set()
        for (_, _, dependencies) in self.dependencies:
            for dependency in dependencies:
                all_dependencies.add(dependency)


        if self.transitions:
            transitions_size = self.transitions_size
            params['u_end'] = tf.get_variable('u_end', [transitions_size, size], dtype=data_type())
            params['transitions_w'] = tf.get_variable("transitions_w", [1, size, transitions_size], dtype=data_type())
            params['transitions_b'] = tf.get_variable("transitions_b", [1, transitions_size], dtype=data_type())
            params['transitions_embed'] = tf.get_variable("transitions_embedding", [transitions_size, size / 2], dtype=data_type())
        else:
            label_size = self.label_size
            attr_size = self.attr_size
            embed_size = size / 2

            params['u_end'] = tf.get_variable('u_end', [label_size, attr_size, size], dtype=data_type())

            params['attr_w'] = tf.get_variable("attr_w", [label_size, size, attr_size], dtype=data_type())
            params['attr_b'] = tf.get_variable("attr_b", [label_size, attr_size], dtype=data_type())

            params['label_w'] = tf.get_variable("label_w", [1, size, label_size], dtype=data_type())
            params['label_b'] = tf.get_variable("label_b", [1, label_size], dtype=data_type())


            params['pointers_w1'] = tf.get_variable("pointers_w1", [size, size], dtype=data_type())
            params['pointers_w2'] = tf.get_variable("pointers_w2", [size, self.mem_size * size], dtype=data_type())
            params['pointers_v'] = tf.get_variable("pointers_v", [1, 1, self.mem_size, size], dtype=data_type())
            params['pointers_b'] = tf.get_variable("pointers_b", [1, 1, self.mem_size, size], dtype=data_type())

            #if d == 'reverse':
            #    p[d]['u_right_hole'] = tf.get_variable('u_right_hole', [1, size * 2], dtype=data_type())
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

    def generate_loop_cond(self, forward_array):
        def loop_cond (ctr, cell_data):
            if forward_array:
                return tf.less(ctr, self.data_length)
            else:
                return tf.greater(ctr, 0)
        return loop_cond

    def generate_loop_body(self, dependencies, direction, forward_array, layer):
        left_to_right = (direction == 'forward') == forward_array
        rows = self.rows[direction]

        def get_input(idx, dependency, cell_data, parent=False):
            if layer == 0:
                if dependency in ['left_hole', 'right_hole']:
                    return self.cells.rnn['children'][self.num_layers-1].get_output(cell_data, idx)
                else:
                    inp = self.embed(idx, direction, parent)
            else:
                inp = self.cells.rnn[dependency][layer-1].get_output(cell_data, idx)
            if dependency == 'children' and not parent:
                parent_inp = get_input(idx, dependency, cell_data, parent=True)
                inp = tf.stack([inp, parent_inp])
            return inp

        def get_dependency_idx(idx, dependency):
            if dependency == 'children':
                return rows['right_child' if left_to_right else 'left_child'][:, idx]
            #elif dependency == 'left_subtree':
            #    return self.rows['reverse']['left_sibling'][:, idx]
            else:
                return rows[dependency][:, idx]

        def loop_body(ctr, cell_data):
            for dependency in dependencies:
                inp = get_input(ctr, dependency, cell_data)
                dependency_idx = get_dependency_idx(ctr, dependency)

                if dependency == 'children' and layer == self.num_layers - 1:
                    # TODO: do we need to make sure that a sibling exists? can we just use "sibling 0" blindly?
                    add_idx = rows['left_sibling' if left_to_right else 'right_sibling'][:, ctr]
                else:
                    add_idx = None

                self.cells.rnn[dependency][layer].step(cell_data, ctr, inp, dependency_idx, add_idx=add_idx)

            ctr = tf.add(ctr, 1) if forward_array else tf.subtract(ctr, 1)

            return ctr, cell_data
        return loop_body

    @define_scope
    def calculate_hpred(self):
        h_pred = tf.zeros([self.num_rows, self.data_length, self.hidden_size])
        with tf.name_scope('main_loop_body'):
            all_dependencies = set()
            end_ctr = None
            for (forward_tree, forward_array, dependencies) in self.dependencies:
                all_dependencies.update(dependencies)
                direction = 'forward' if forward_tree else 'reverse'

                # TODO: don't need to loop over num_layers if we aren't handling children.
                # Can just grab the states and outputs of the final layers
                for layer in range(self.num_layers):

                    # left-to-right starts iterating from 1, since 0 is the "empty" dependency
                    ctr = 1 if forward_array else self.data_length - 1

                    end_ctr, self.cells.data = tf.while_loop(self.generate_loop_cond(forward_array),
                                                self.generate_loop_body(dependencies, direction, forward_array, layer),
                                                [ctr, self.cells.data],
                                                parallel_iterations=1)

            # the last dependency group must be the direction we care about
            (_, _, dependencies) = self.dependencies[-1]
            for dependency in dependencies:
                #predicted_output = tf.gather(predicted_output, self.rows[dependency if dependency != 'children' else 'left_child'])
                # predicted_output == [batch_size, data_length, hidden_size]
                if dependency == 'children':
                    rows = self.rows[direction]['right_child' if (direction == 'forward') == forward_array else 'left_child']
                else:
                    rows = self.rows[direction][dependency]

                predicted_output = self.cells.rnn[dependency][self.num_layers-1].stack_output(self.cells.data)

                # XXX better way of doing this?
                r = tf.range(self.num_rows, dtype=tf.int32)
                predicted_output = tf.map_fn(lambda x: tf.gather(predicted_output[x], rows[x]), r, data_type())

                if 'U' in self.params:
                    predicted_output = tf.matmul(predicted_output, tf.tile(self.params['U'][dependency], [self.num_rows, 1, 1]))
                h_pred += predicted_output

            # the very last traversal gives us the generation direction
            #self.fetches['h_pred'] = h_pred
            return h_pred, direction, forward_array


    def __init__(self, *args, **kwargs):
        super().__init__(False, *args, **kwargs)
