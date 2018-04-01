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

from tfmodellib import TFModel, TFModelConfig, graph_def, docsig, MLP, build_mlp_graph

import tensorflow as tf


@graph_def
@docsig
def build_autoencoder_graph(input_tensor, latent_size, n_hidden, activation, **kwargs):
    """
    Defines an autoencoder graph, with `2*len(n_hidden)+1` dense layers:
    
    - `len(n_hidden)` layers for the encoder, where the i-th encoder layer has
      `n_hidden[i]` units.

    - The latent (code) layer with `latent_size` units;

    - `len(n_hidden)` layers for the decoder, where the j-th decoder layer has
      `n_hidden[-j-1]` units.
   
    The output layer is of same dimension as the input layer.

    Parameters
    ----------
    input_tensor : Tensor
        The input tensor of size NUM_SAMPLES x INPUT_SIZE.

    latent_size : int
        Number of units in the MLP's output layer.

    n_hidden : list
        List of ints, specifying the number of units for each hidden layer.

    activation : function
        Activation function for the encoder- and decoder-layers.

    For additional *kwargs*, see tfmodels.build_mlp_graph.

    Returns
    -------
    reconstruction : Tensor
        The output (reconstruction of the input).

    See Also
    --------
    tfmodels.build_mlp_graph : Used to construct the encoder and decoder
                              sub-networks of the autoencoder.
    """

    # define encoder
    latent_layer = build_mlp_graph(
            input_tensor=input_tensor,
            out_size=latent_size,
            n_hidden=n_hidden,
            activation=activation)

    # define decoder
    n_hidden.reverse()
    reconstruction = build_mlp_graph(
            input_tensor=latent_layer,
            out_size=input_tensor.shape.as_list()[1],
            n_hidden=n_hidden,
            activation=activation)

    return reconstruction



class AutoEncoderConfig(TFModelConfig):

    def init(self):
        self.update(
                in_size=3,
                latent_size=2,
                n_hidden=[10,10],
                activation=tf.nn.relu)


class AutoEncoder(MLP):

    def build_graph(self):

        # define input and target placeholder
        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.config['in_size']])
        self.y_target = tf.placeholder(dtype=tf.float32, shape=[None, self.config['in_size']])

        # define the base graph
        self.y_output = build_autoencoder_graph(input_tensor=self.x_input, **self.config)

        # define learning rate
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[])

        # define loss
        self.loss = tf.losses.mean_squared_error(self.y_target, self.y_output)

        # define optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.minimize_op = self.optimizer.minimize(self.loss)


if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    import logging

    # create the model
    conf = AutoEncoderConfig(
            log_level=logging.DEBUG,
            n_hidden=[100,50,5],
            latent_size=2)
    model = AutoEncoder(conf)

    # generate some data
    xx,yy = np.random.rand(2*10000).reshape((2,-1))
    zz = 0.1 * (np.sin(10*xx) + np.cos(10*yy)) * np.exp(-((xx-0.5)**2+(yy-0.5)**2)/0.1)
    x = np.vstack((xx.flat,yy.flat,zz.flat)).T

    x_train = x[:int(x.shape[0]*0.8)]
    x_valid = x[x_train.shape[0]:]

    # run the training
    for _ in range(1000):
        model.train(
                train_inputs=x_train,
                train_targets=x_train,
                validation_inputs=x_valid,
                validation_targets=x_valid,
                batch_size=1000,
                learning_rate=0.0001)

    # get estimates
    reconstruction = model.run_output(inputs=x_valid)

    # plot data and estimates
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(*x_valid.T, ls='none', marker='.', mec='none', mfc='g')
    ax.plot(*reconstruction.T, ls='none', marker='.', mec='none', mfc='b')
    plt.show()
