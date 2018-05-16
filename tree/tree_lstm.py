import tensorflow as tf

# we no longer store (c,h), but rather, "c" is the part of the memory cell already passed through the forget gate
class TreeLSTMCell():

    def __init__(self, num_units, data_type, forget_bias=1.0, input_size=None, activation=tf.tanh, reuse=None):
        #super(TreeLSTMCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation
        self.data_type = data_type

        # TODO: get _linear equivalent to work

        # XXX XXX initializers??
        self.W_i = tf.get_variable("W_i", [num_units, num_units], dtype=data_type)
        self.U_i = tf.get_variable("U_i", [num_units, num_units], dtype=data_type)
        self.b_i = tf.get_variable("b_i", [1, num_units], dtype=data_type)

        self.W_f = tf.get_variable("W_f", [num_units, num_units], dtype=data_type)
        self.U_f = tf.get_variable("U_f", [num_units, num_units], dtype=data_type)
        self.b_f = tf.get_variable("b_f", [1, num_units], dtype=data_type)

        self.W_o = tf.get_variable("W_o", [num_units, num_units], dtype=data_type)
        self.U_o = tf.get_variable("U_o", [num_units, num_units], dtype=data_type)
        self.b_o = tf.get_variable("b_o", [1, num_units], dtype=data_type)

        self.W_u = tf.get_variable("W_u", [num_units, num_units], dtype=data_type)
        self.U_u = tf.get_variable("U_u", [num_units, num_units], dtype=data_type)
        self.b_u = tf.get_variable("b_u", [1, num_units], dtype=data_type)

    def zero_state(self, data_type):
        return tf.contrib.rnn.LSTMStateTuple(tf.zeros([batch_size, self._num_units], dtype=data_type),
                tf.zeros([batch_size, self._num_units], dtype=data_type))

    @property
    def state_size(self):
        return tf.contrib.rnn.LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units


    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):

            current_inputs, parent_inputs = tf.unstack(inputs)
            f, h = state

            i = tf.sigmoid(tf.matmul(current_inputs, self.W_i) + tf.matmul(h, self.U_i) + self.b_i)
            o = tf.sigmoid(tf.matmul(current_inputs, self.W_o) + tf.matmul(h, self.U_o) + self.b_o)
            u = self._activation(tf.matmul(current_inputs, self.W_u) + tf.matmul(h, self.U_u) + self.b_u)

            new_c = f + i * u
            new_h = self._activation(new_c) * o

            # XXX double bias?????
            new_f = tf.sigmoid(tf.matmul(parent_inputs, self.W_f) +
                               tf.matmul(new_h, self.U_f) +
                               self.b_f) * new_c

            new_state = tf.contrib.rnn.LSTMStateTuple(new_f, new_h)
            return new_h, new_state
