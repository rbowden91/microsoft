# this file is really two separate things: a kludge wrapper to make TensorArrays more tolerable, and a wrapper around
# RNN cells to automatically save their state/outputs int those tensorarrays. Will eventually split it up.
import tensorflow as tf

# we CAN'T use regular objects from this class, since we need to be able to pass the underlying
# TensorArray directly to the tf.while_loop
class RNNTensorArray():

    #def unwrap(self):
    #    if isinstance(self.array, tf.TensorArray):
    #        return self.array
    #    else:
    #        return [unwrap(array) for array in self.array]
    #def wrap(self, array):
    #    if isinstance(self.array, list):
    #        array  = [RNNTensorArray(self.data_length, self.hidden_size, self.batch_size,
    #                                 self.num_layers, i) for i in array]
    #    return RNNTensorArray(self.data_length, self.hidden_size, self.batch_size, self.num_layers, a)

    def __init__(self, data_length, hidden_size, batch_size, data_type):
        self.data_length = data_length
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.data_type = data_type

    def scatter_gather(self, array, positions, data=None, auto_save=True):
        r = tf.range(0, self.batch_size, dtype=tf.int32) * self.data_length
        # XXX verify this is actually happening
        if isinstance(positions,int) or positions.shape==():
            positions = tf.fill([self.batch_size], positions)
        new_pos = tf.cast(tf.add(positions, r), dtype=tf.int32)
        if data is None:
            data = array[0].gather(new_pos)
            return data
            #return tf.reshape(data, [self.batch_size, self.hidden_size])
        else:
            # we need to repeat the data before scattering
            if data.shape[0] == 1:
                data = tf.tile(data, [self.batch_size, 1])
            new_array = array[0].scatter(new_pos, data)
            if auto_save:
                array[0] = new_array
            else:
                return new_array

    def scatter(self, array, positions, data, auto_save=True):
        assert isinstance(positions, int) or positions.shape==()
        return self.scatter_gather(array, positions, data, auto_save)

    def gather(self, array, positions):
        return self.scatter_gather(array, positions)

    def cond_scatter(self, array, cond, value, ctr):
        array[0] = tf.cond(cond, lambda: self.scatter(array, ctr, value, auto_save=False),
                                 lambda: array[0])

    def cond_save(self, array, cond, value, ctr):
        cond_scatter(self, array[0], cond, value[0], ctr)
        cond_scatter(self, array[1], cond, value[1], ctr)

    def init_tensor_array(self):
        array = tf.TensorArray(
                    self.data_type,
                    size=self.data_length * self.batch_size,
                    dynamic_size=False,
                    clear_after_read=False,
                    infer_shape=True,
                    element_shape=[self.hidden_size])

        # do this so we can overwrite in place!
        return [array]

    def stack(self, array):
        return array[0].stack()


class RNNTensorArrayCell():
    # can make this global for the class, because for now all TensorArrays have the same hidden_size, data_length, etc.
    array = None

    class RNNWrapper():
        def __init__(self, dependency, layer, direction, data_type):
            # I tried just storing the state/output data directly in here, but the tensorflow while loop wasn't
            # happy. So I now explicitly pass the data back in to all relevant methods
            self.scope = 'Parameters/{}/RNN_{}_cell_{}'.format(direction, dependency, layer)
            self.layer = layer
            self.dependency = dependency
            self.data_type = data_type
            self.array = RNNTensorArrayCell.array
            self.cell = self.rnn_cell(dependency == 'children')

        def get_output(self, data, ctr):
            return self.array.gather(data[self.dependency][self.layer]['output'], ctr)

        def save_output(self, data, ctr, output):
            self.array.scatter(data[self.dependency][self.layer]['output'], ctr, output)

        def save_lstm_state(self, data, position, state):
            # c is in position 0, h is in position 1
            self.array.scatter(data[self.dependency][self.layer]['state'][0], position, state.c)
            self.array.scatter(data[self.dependency][self.layer]['state'][1], position, state.h)

        # reconstruct the LSTM state from a TensorArray
        def get_lstm_state(self, data, position):
            c = self.array.gather(data[self.dependency][self.layer]['state'][0], position)
            h = self.array.gather(data[self.dependency][self.layer]['state'][1], position)

            # need to reshape for batch?
            return tf.contrib.rnn.LSTMStateTuple(c, h)


        def rnn_cell(self, is_tree):
            size = self.array.hidden_size
            def lstm_cell(is_tree):
                if is_tree:
                    return TreeLSTMCell(size, data_type=self.data_type, forget_bias=0.0,
                                        reuse=tf.get_variable_scope().reuse)
                else:
                    return tf.contrib.rnn.BasicLSTMCell(
                        self.array.hidden_size, forget_bias=0.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
                    #return tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(size, reuse=tf.get_variable_scope().reuse)
            #return tf.contrib.rnn.DropoutWrapper(lstm_cell(is_tree),
            #                                        output_keep_prob=(1 - self.placeholders['drop_prob']))
            return lstm_cell(is_tree)

        def step(self, data, ctr, inp, dependency_idx, add_idx=None):
            # reconstruct the LSTM state of the dependency from the TensorArray
            old_state = self.get_lstm_state(data, dependency_idx)
            with tf.variable_scope(self.scope):
                (new_output, new_state) = self.cell(inp, old_state)

            if add_idx is not None:
                add_state = self.get_lstm_state(data, add_idx)
                new_state.c += add_state.c
                new_state.h += add_state.h

                new_output = tf.matmul(new_output, self.u_child)
                add_output = self.get_output(data, add_idx)
                new_output += add_output

            self.save_lstm_state(data, ctr, new_state)
            self.save_output(data, ctr, new_output)

        def stack_output(self, data):
            return self.array.stack(data[self.dependency][self.layer]['output'])
            #return tf.reshape(output, [self.batch_size, self.data_length, self.hidden_size])

        def stack_state(self, data):
            return {
                'c': self.array.stack(data[self.dependency][self.layer]['state'][0]),
                'h': self.array.stack(data[self.dependency][self.layer]['state'][1]),
            }

    # initialize an array of TensorArrays to store an LSTM
    def init_lstm_array(self):
        states = []
        # 0 for c, 1 for h
        for k in range(2):
            states.append(self.array.init_tensor_array())
        return states


    def __init__(self, dependencies, data_length, hidden_size, batch_size, num_layers, data_type):
        self.num_layers = num_layers
        self.data_type = data_type

        if self.__class__.array is None:
            self.__class__.array = RNNTensorArray(data_length, hidden_size, batch_size, data_type)

        self.array = self.__class__.array

        # we can't store the rnn cells in the same dictionary, because they can't be passed to the tf.while_loop. so
        # this is separate from everything else...
        # all cells pull from the same pool of data
        self.data = {}
        self.rnn = {}

        for i in dependencies:
            self.data[i] = []
            self.rnn[i] = []
            for j in range(self.num_layers):
                self.data[i].append({
                    'state': self.init_lstm_array(),
                    'output': self.array.init_tensor_array()
                })
                self.rnn[i].append(self.RNNWrapper(i, j, dependencies[i], data_type))
                with tf.variable_scope(self.rnn[i][j].scope):
                    initial = tf.get_variable('lstm_initial_output', [1, hidden_size], data_type) \
                              if j == self.num_layers - 1 else tf.zeros([1, hidden_size], data_type)
                    if i == 'children':
                        self.u_child = tf.get_variable('U_child', [1, hidden_size, hidden_size], dtype=data_type())
                    # initial state
                    self.rnn[i][j].save_lstm_state(self.data, 0, tf.contrib.rnn.LSTMStateTuple(
                        tf.get_variable('lstm_state_c', [1, hidden_size], data_type),
                        tf.get_variable('lstm_state_h', [1, hidden_size], data_type)))
                    # not strictly necessary, but we want to be able to .stack() the TensorArray,
                    # so it has to be full. could also do this in the RNNWrapper, but it isn't aware that
                    # it's the last layer currently
                    self.rnn[i][j].save_output(self.data, 0, initial)

        # store the LSTM state in a TensorArray
        #def save_multi_lstm_state(self, array, position, state):
        #    for i in range(self.num_layers):
        #        self.save_lstm_state(array[i], state[i], position)

        #def restore_multi_lstm_state(self, array, position):
        #    state = []
        #    for i in range(self.num_layers):
        #        state.append(self.get_lstm_state(array[i], position))
        #    return tuple(state
