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
def build_linreg_graph(input_tensor, out_size):
    """
    Defines a graph for linear regression.

    Parameters
    ----------
    input_tensor : Tensor
        The input tensor of size NUM_SAMPLES x INPUT_SIZE.

    out_size : int
        Dimensionality of the output space.

    Returns
    -------
    y_output : Tensor
        The network's output.
    """
    
    y_output = tf.layers.dense(input_tensor, units=out_size, activation=None)

    return y_output



class LinRegConfig(TFModelConfig):

    def init(self):
        self.update(
                in_size=1,
                out_size=1)


class LinReg(TFModel):

    def build_graph(self):

        # define input and target placeholder
        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.config['in_size']])
        self.y_target = tf.placeholder(dtype=tf.float32, shape=[None, self.config['out_size']])

        # define the base graph
        self.y_output = build_linreg_graph(input_tensor=self.x_input, **self.config)

        # define learning rate
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[])

        # define loss
        self.loss = tf.losses.mean_squared_error(self.y_target, self.y_output)

        # define optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.minimize_op = self.optimizer.minimize(self.loss)

    def run_update_and_loss(self, batch_inputs, batch_targets, learning_rate):
        loss, _ = self.sess.run([self.loss, self.minimize_op],
                feed_dict={
                        self.x_input: batch_inputs,
                        self.y_target: batch_targets,
                        self.learning_rate: learning_rate})
        return loss

    def run_loss(self, batch_inputs, batch_targets, learning_rate):
        loss = self.sess.run(self.loss,
                feed_dict={
                        self.x_input: batch_inputs,
                        self.y_target: batch_targets})
        return loss

    def run_output(self, inputs):
        return self.sess.run(self.y_output, feed_dict={self.x_input: inputs})


if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt
    import logging

    # create the model
    conf = LinRegConfig(log_level=logging.DEBUG, in_size=1, out_size=1)
    model = LinReg(conf)

    # generate some data
    x = np.random.rand(1000).reshape((-1,1))
    y = 10 * np.random.randn() * x + 10 * np.random.randn() + np.random.randn(x.size).reshape(x.shape)

    x_train = x[:int(x.shape[0]*0.8)]
    y_train = y[:x_train.shape[0]]
    x_valid = x[x_train.shape[0]:]
    y_valid = y[x_train.shape[0]:]


    # run the training
    for _ in range(50):
        model.train(
                train_inputs=x_train,
                train_targets=y_train,
                validation_inputs=x_valid,
                validation_targets=y_valid,
                batch_size=None,
                learning_rate=1.0)

    # get estimates
    y_est = model.run_output(inputs=x_valid)

    # plot data and estimates
    plt.plot(x_valid, y_valid, ls='none', marker='.', color='g')
    plt.plot(x_valid, y_est, ls='none', marker='.', color='b')
    plt.show()
