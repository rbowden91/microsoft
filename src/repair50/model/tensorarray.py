# this file is really two separate things: a kludge wrapper to make TensorArrays more tolerable, and a wrapper around
# RNN cells to automatically save their state/outputs int those tensorarrays. Will eventually split it up.
import tensorflow as tf
from .tree_lstm import TreeLSTMCell

def define_scope(function):
    def decorator(self, *args, **kwargs):
        #with tf.name_scope(type(self).__name__,function.__name__):
        with tf.name_scope(function.__name__):
            return function(self, *args, **kwargs)
    return decorator


# we CAN'T use regular objects from this class, since we need to be able to pass the underlying
# TensorArray directly to the tf.while_loop
class RNNTensorArray():

    def __init__(self, data_length, hidden_size, batch_size, data_type):
        self.data_length = data_length
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.data_type = data_type

    @define_scope
    def scatter(self, array, position, data, auto_save=True):
        assert isinstance(position, int) or position.shape==()
        if data.shape[0] == 1:
            data = tf.tile(data, [self.batch_size, 1])
        new_array = array[0].write(position, data)
        if auto_save:
            array[0] = new_array
        else:
            return new_array

    @define_scope
    def gather(self, array, positions):
        if isinstance(positions,int) or positions.shape==():
            data = array[0].gather([positions])
            return data[0]

        data = array[0].gather(positions)
        r = tf.range(self.batch_size, dtype=tf.int32)
        data = tf.gather_nd(data, tf.stack([r, r], axis=1))
        data = tf.reshape(data, [self.batch_size, self.hidden_size])
        return data

    @define_scope
    def cond_scatter(self, array, cond, value, ctr):
        array[0] = tf.cond(cond, lambda: self.scatter(array, ctr, value, auto_save=False),
                                 lambda: array[0])

    @define_scope
    def cond_save(self, array, cond, value, ctr):
        cond_scatter(self, array[0], cond, value[0], ctr)
        cond_scatter(self, array[1], cond, value[1], ctr)


    @define_scope
    def init_tensor_array(self):
        array = tf.TensorArray(
                    self.data_type,
                    size=self.data_length,
                    dynamic_size=False,
                    clear_after_read=False,
                    infer_shape=True,
                    element_shape=[None, self.hidden_size])

        # do this so we can overwrite in place!
        return [array]

    @define_scope
    def stack(self, array):
        return tf.transpose(array[0].stack(), perm=[1,0,2])


class RNNTensorArrayCell():
    class RNNWrapper():
        def __init__(self, dependency, layer, is_last_layer, array):
            # I tried just storing the state/output data directly in here, but the tensorflow while loop wasn't
            # happy. So I now explicitly pass the data back in to all relevant methods
            self.scope = 'RNN_{}_cell_{}'.format(dependency, layer)
            self.layer = layer
            self.dependency = dependency
            self.data_type = array.data_type
            self.array = array

            hidden_size = self.array.hidden_size
            with tf.variable_scope(self.scope):
                self.cell = self.rnn_cell(dependency == 'children')
                if dependency == 'children' and is_last_layer:
                    self.u_child = tf.get_variable('U_child', [hidden_size, hidden_size], dtype=self.data_type)

        @define_scope
        def get_output(self, data, ctr, is_child=False):
            output = 'output' if not is_child else 'child_output'
            return self.array.gather(data[self.dependency][self.layer][output], ctr)

        @define_scope
        def save_output(self, data, ctr, output, is_child=False):
            output_slot = 'output' if not is_child else 'child_output'
            self.array.scatter(data[self.dependency][self.layer][output_slot], ctr, output)

        @define_scope
        def save_lstm_state(self, data, position, state):
            # c is in position 0, h is in position 1
            self.array.scatter(data[self.dependency][self.layer]['state'][0], position, state.c)
            self.array.scatter(data[self.dependency][self.layer]['state'][1], position, state.h)

        # reconstruct the LSTM state from a TensorArray
        @define_scope
        def get_lstm_state(self, data, position, add_state = None):
            c = self.array.gather(data[self.dependency][self.layer]['state'][0], position)
            h = self.array.gather(data[self.dependency][self.layer]['state'][1], position)

            if add_state:
                return tf.contrib.rnn.LSTMStateTuple(c + add_state.c, h + add_state.h)
            else:
                # need to reshape for batch?
                return tf.contrib.rnn.LSTMStateTuple(c, h)


        @define_scope
        def rnn_cell(self, is_tree):
            size = self.array.hidden_size
            def lstm_cell(is_tree):
                if is_tree:
                    return TreeLSTMCell(size, data_type=self.data_type)
                else:
                    return tf.nn.rnn_cell.LSTMCell(
                        self.array.hidden_size, forget_bias=0.0, state_is_tuple=True, name='basic_lstm_cell')
                    #return tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(size, reuse=tf.get_variable_scope().reuse)
            #return tf.contrib.rnn.DropoutWrapper(lstm_cell(is_tree),
            #                                        output_keep_prob=(1 - self.placeholders['drop_prob']))
            return lstm_cell(is_tree)

        @define_scope
        def step(self, data, ctr, inp, dependency_idx, add_idx=None):
            # reconstruct the LSTM state of the dependency from the TensorArray
            old_state = self.get_lstm_state(data, dependency_idx)
            with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
                (output, new_state) = self.cell(inp, old_state)

            if add_idx is not None:
                new_state = self.get_lstm_state(data, add_idx, add_state=new_state)

                # don't add to the initial output if we are the right-most sibling
                new_output = tf.matmul(output, self.u_child)
                new_output = self.get_output(data, add_idx, is_child=True) + new_output
                self.save_output(data, ctr, new_output, is_child=True)

            self.save_lstm_state(data, ctr, new_state)
            self.save_output(data, ctr, output)

        @define_scope
        def stack_output(self, data, is_child=False):
            output = 'output' if self.dependency != 'children' else 'child_output'
            return self.array.stack(data[self.dependency][self.layer][output])
            #return tf.reshape(output, [self.batch_size, self.data_length, self.hidden_size])

        @define_scope
        def stack_state(self, data):
            return {
                'c': self.array.stack(data[self.dependency][self.layer]['state'][0]),
                'h': self.array.stack(data[self.dependency][self.layer]['state'][1]),
            }

    # initialize an array of TensorArrays to store an LSTM
    @define_scope
    def init_lstm_array(self):
        states = []
        # 0 for c, 1 for h
        for k in range(2):
            states.append(self.array.init_tensor_array())
        return states

    def add_dependency(self, dependency):
        assert dependency not in self.data

        self.data[dependency] = data = []
        self.rnn[dependency] = rnn = []
        self.fetches[dependency] = fetches = {}
        self.initials[dependency] = initials = {}
        for layer in range(self.num_layers):
            data.append({
                'state': self.init_lstm_array(),
                'output': self.array.init_tensor_array()
            })
            is_last_layer = layer == self.num_layers - 1
            if dependency == 'children' and is_last_layer:
                data[layer]['child_output'] = self.array.init_tensor_array()
            rnn.append(self.RNNWrapper(dependency, layer, is_last_layer, self.array))
            with tf.variable_scope(rnn[layer].scope):
                # initial state
                initial = { 'c': tf.get_variable('lstm_state_c',
                                        [1, self.array.hidden_size], self.array.data_type),
                            'h': tf.get_variable('lstm_state_h',
                                        [1, self.array.hidden_size], self.array.data_type)}
                rnn[layer].save_lstm_state(self.data, 0, tf.contrib.rnn.LSTMStateTuple(initial['c'],
                                                                                       initial['h']))
                for i in ['c', 'h']:
                    key = 'states-' + str(layer) + '-' + i
                    fetches[key] = lambda: rnn[layer].stack_state(self.data)[i]
                    initials[key] = initial[i]

                # theoretically, we may not need anything other than the last output, but save all layers anyway
                initial_output = tf.get_variable('lstm_initial_output',
                                        [1, self.array.hidden_size], self.array.data_type)

                rnn[layer].save_output(self.data, 0, initial_output)
                key = 'outputs-' + str(layer)
                fetches[key] =  lambda: rnn[layer].stack_output(self.data)
                initials[key] = initial_output
                if dependency == 'children' and is_last_layer:
                    initial_child_output = tf.get_variable('lstm_initial_child_output',
                                            [1, self.array.hidden_size], self.array.data_type)
                    rnn[layer].save_output(self.data, 0, initial_child_output, True)
                    key = 'child-outputs-' + str(layer)
                    fetches[key] = lambda: rnn[layer].stack_output(self.data, True)
                    initials[key] = initial_child_output

    def __init__(self, dependencies, data_length, hidden_size, batch_size, num_layers, data_type):
        self.num_layers = num_layers

        self.array = RNNTensorArray(data_length, hidden_size, batch_size, data_type)

        # we can't store the rnn cells in the same dictionary, because they can't be passed to the tf.while_loop. so
        # this is separate from everything else...
        # all cells pull from the same pool of data
        self.data = {}
        self.rnn = {}
        self.fetches = {}
        self.initials = {}

        for d in dependencies:
            self.add_dependency(d)
