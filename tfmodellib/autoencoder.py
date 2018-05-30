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
def build_autoencoder_graph(
        input_tensor, latent_size, encoder_size, decoder_size=None,
        hidden_activation=tf.nn.relu, latent_activation=tf.nn.relu,
        output_activation=None, use_dropout=False, use_bn=False,
        bn_is_training=False, latent_layer_fun=None, encoder_name='encoder',
        decoder_name='decoder'):
    """
    Defines an autoencoder graph, with `len(encoder_size)+1+len(decoder_size)`
    dense layers:
    
    - `len(encoder_size)` layers for the encoder, where the i-th encoder layer has
      `encoder_size[i]` units.

    - The latent (code) layer with `latent_size` units;

    - `len(decoder_size)` layers for the decoder, where the j-th decoder layer
      has `decoder_size[j]` units.
   
    The output layer is of same dimension as the input layer.

    Parameters
    ----------
    input_tensor : Tensor
        The input tensor of size NUM_SAMPLES x INPUT_SIZE.

    latent_size : int
        Number of units in the autoencoder's latent layer.

    encoder_size : list
        List of ints, specifying the number of units for each hidden layer of
        the encoder.

    decoder_size : list (optional)
        List of ints, specifying the number of units for each hidden layer of
        the decoder. If None (the default), the reverse of encoder_size will be
        used.

    hidden_activation : function (optional)
        Activation function for the encoder- and decoder-layers (default:
        tf.nn.relu).

    latent_activation : function (optional)
        Activation function for the latent (code) layer (default: tf.nn.relu).

    output_activation : function (optional)
        Activation function for the output layer (default: linear activation).

    use_dropout : bool (optional)
        Indicates whether or not to use dropout after each hidden layer
        (default: False).
    
    use_bn : bool (optional)
        Indicates whether or not to add a batch norm layer after each hidden
        layer (default: False).

    bn_is_training : bool (optional)
        Flag for batch norm stage.
    
    latent_layer_fun : function (optional)
        A function, taking a tensor as input and returning another tensor. It
        can be used to modify the output of the encoder, before passing it to
        the decoder. If None, the encoder's output will directly be passed to
        the decoder.

    Returns
    -------
    reconstruction : Tensor
        The output (reconstruction of the input).

    See Also
    --------
    tfmodellib.build_mlp_graph : Used to construct the encoder and decoder
                                 sub-networks of the autoencoder.
    """

    # define encoder
    with tf.variable_scope(encoder_name):
        encoder_out = build_mlp_graph(
                input_tensor=input_tensor,
                out_size=latent_size,
                n_hidden=encoder_size,
                hidden_activation=hidden_activation,
                output_activation=latent_activation,
                use_dropout=use_dropout,
                use_bn=use_bn,
                bn_is_training=bn_is_training)

        if use_bn:
            encoder_out = tf.layers.batch_normalization(encoder_out, training=bn_is_training, name='batchnorm_encoder_out')

    if latent_layer_fun is not None:
        with tf.variable_scope('latent_layers'):
            latent_layer = latent_layer_fun(encoder_out)
    else:
        latent_layer = encoder_out

    # define decoder
    with tf.variable_scope(decoder_name):

        if decoder_size is None:
            decoder_size = encoder_size
            decoder_size.reverse()

        decoder_out = build_mlp_graph(
                input_tensor=latent_layer,
                out_size=decoder_size[-1],
                n_hidden=decoder_size[:-1],
                hidden_activation=hidden_activation,
                output_activation=hidden_activation,
                use_dropout=use_dropout,
                use_bn=use_bn,
                bn_is_training=bn_is_training)

        if use_bn:
            decoder_out = tf.layers.batch_normalization(decoder_out, training=bn_is_training, name='batchnorm_decoder_out')

        reconstruction = tf.layers.dense(
                inputs=decoder_out, units=input_tensor.shape.as_list()[1],
                activation=output_activation)

    return reconstruction


class AutoEncoderConfig(TFModelConfig):

    def init(self):
        self.update(
                in_size=3,
                latent_size=2,
                encoder_size=[10,10],
                decoder_size=[10,10],
                hidden_activation=tf.nn.relu,
                latent_activation=tf.nn.relu,
                output_activation=None,
                use_dropout=False,
                use_bn=False,
                latent_layer_fun=None)
        super(AutoEncoderConfig, self).init()


class AutoEncoder(MLP):

    def build_graph(self):

        # define input and target placeholder
        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.config['in_size']])
        self.y_target = tf.placeholder(dtype=tf.float32, shape=[None, self.config['in_size']])
        self.bn_is_training = tf.placeholder(dtype=tf.bool, shape=[], name='bn_is_training')

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
                in_size=3,
                latent_size=2,
                encoder_size=[10,10])
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
