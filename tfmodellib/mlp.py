# Copyright 2018 Nikolas Hemion. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from tfmodellib import TFModel, TFModelConfig, graph_def, docsig

import tensorflow as tf


@graph_def
@docsig
def build_mlp_graph(input_tensor, out_size, n_hidden, hidden_activation=tf.nn.relu, output_activation=None, use_dropout=False, use_bn=False, bn_is_training=False):
    """
    Defines an MLP graph, with `len(n_hidden)` dense layers, where the `i`-th
    layer has `n_hidden[i]` units, and output of size `out_size`.

    Parameters
    ----------
    input_tensor : Tensor
        The input tensor of size NUM_SAMPLES x INPUT_SIZE.

    out_size : int
        Number of units in the MLP's output layer.

    n_hidden : list
        List of ints, specifying the number of units for each hidden layer.

    hidden_activation : function (optional)
        Activation function for the hidden layers (default: tf.nn.relu).

    output_activation : function (optional)
        Activation function for the output layer (default: linear activation).

    use_dropout : bool (optional)
        Indicates whether or not to use dropout after each hidden layer
        (default: False).
    
    use_bn : bool (optional)
        Indicates whether or not to add a batch norm layer after each hidden
        layer (default: False).

    Returns
    -------
    y_output : Tensor
        The MLP's output.
    """
    # define hidden layers
    current_input = input_tensor

    for ind,n in enumerate(n_hidden):
        current_input = tf.layers.dense(current_input, units=n, activation=hidden_activation, name='dense_layer_{:d}'.format(ind))

        if use_dropout:
            current_input = tf.layers.dropout(current_input, name='dropout_layer_{:d}'.format(ind))

        if use_bn:
            current_input = tf.layers.batch_normalization(current_input, training=bn_is_training, name='batchnorm_layer_{:d}'.format(ind))

    # define output layer
    y_output = tf.layers.dense(current_input, units=out_size, activation=output_activation, name='output_layer')

    return y_output



class MLPConfig(TFModelConfig):

    def init(self):
        self.update(
                in_size=1,
                out_size=1,
                n_hidden=[10,10],
                hidden_activation=tf.nn.relu,
                output_activation=None,
                use_dropout=False,
                use_bn=False)

class MLP(TFModel):

    def build_graph(self):

        # define input and target placeholder
        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.config['in_size']], name='x_input')
        self.y_target = tf.placeholder(dtype=tf.float32, shape=[None, self.config['out_size']], name='y_target')

        # define the base graph
        self.bn_is_training = tf.placeholder(dtype=tf.bool, shape=[], name='bn_is_training')
        self.y_output = build_mlp_graph(input_tensor=self.x_input, bn_is_training=self.bn_is_training, **self.config)

        # define learning rate
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')

        # define loss
        self.loss = tf.losses.mean_squared_error(self.y_target, self.y_output)

        # define optimizer
        with tf.control_dependencies(self.graph.get_collection(tf.GraphKeys.UPDATE_OPS)):
            with tf.variable_scope('optimization'):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='optimizer')
                self.minimize_op = self.optimizer.minimize(self.loss, name='minimize_op')

    def run_update_and_loss(self, batch_inputs, batch_targets, learning_rate):
        loss, _ = self.sess.run([self.loss, self.minimize_op], feed_dict={
                self.x_input: batch_inputs,
                self.y_target: batch_targets,
                self.learning_rate: learning_rate,
                self.bn_is_training: True})
        return loss

    def run_loss(self, batch_inputs, batch_targets, learning_rate):
        loss = self.sess.run(self.loss, feed_dict={
                self.x_input: batch_inputs,
                self.y_target: batch_targets,
                self.bn_is_training: False})
        return loss

    def run_output(self, inputs):
        return self.sess.run(self.y_output, feed_dict={self.x_input: inputs, self.bn_is_training: False})


if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt
    import logging

    # create the model
    conf = MLPConfig(log_level=logging.DEBUG, n_hidden=[25,25], use_bn=True)
    model = MLP(conf)

    # generate some data
    x = np.random.rand(1000).reshape((-1,1))
    y = np.sin(7*x)

    x_train = x[:int(x.shape[0]*0.8)]
    y_train = y[:x_train.shape[0]]
    x_valid = x[x_train.shape[0]:]
    y_valid = y[x_train.shape[0]:]


    # run the training
    for _ in range(1000):
        model.train(
                train_inputs=x_train,
                train_targets=y_train,
                validation_inputs=x_valid,
                validation_targets=y_valid,
                batch_size=100,
                learning_rate=0.001)

    # get estimates
    y_est = model.run_output(inputs=x_valid)

    # plot data and estimates
    plt.plot(x_valid, y_valid, ls='none', marker='.', color='g')
    plt.plot(x_valid, y_est, ls='none', marker='.', color='b')
    plt.show()
