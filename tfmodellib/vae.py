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


def build_vae_latent_layers(input_tensor, units):

    #
    # define latent mean, sigma_sq, randn
    #

    latent_mean = tf.layers.dense(input_tensor, units=units, activation=None, name='latent_mean')

    # For the computation of the loss, we need:
    #     sigma**2
    #     log(sigma**2)
    # Other implementation of a VAE map the encoder output onto
    # log(sigma**2).  This can however cause numerical problems when
    # computing exp(log(sigma**2)) to obtain sigma**2 (note that
    # float32(exp(88.73))=inf). To avoid this, we instead map the encoder
    # output onto sigma directly, and compute sigma**2 and log(sigma**2)
    # from there (while adding a small constant to sigma**2 before
    # computing the log, in case sigma is exactly 0.0).
    # Furthermore, we use linear activation, which can produce negative
    # values for sigma. We compensate for this by multiplying the random
    # noise with the absolute value of sigma, instead of relying on
    # something like ReLU activation, which could kill off the units.

    latent_sigma = tf.layers.dense(input_tensor, units=units, activation=None, name='latent_sigma_before_abs')
    latent_sigma_sq = tf.square(latent_sigma, name='latent_sigma_sq')
    small_constant_for_numerical_stability = tf.constant(1e-20, dtype=tf.float32, name='small_constant_for_numerical_stability')
    latent_log_sigma_sq = tf.log(latent_sigma_sq + small_constant_for_numerical_stability, name='latent_log_sigma_sq')
    latent_sigma = tf.abs(latent_sigma, name='latent_sigma')
    latent_randn = tf.random_normal(shape=tf.shape(latent_mean), dtype=tf.float32, name='latent_randn')

    # define latent layer
    latent_layer = tf.add(latent_mean, tf.multiply(latent_sigma, latent_randn), name='latent')

    return latent_layer, latent_mean, latent_sigma, latent_sigma_sq, latent_log_sigma_sq


@graph_def
@docsig
def build_vae_graph(input_tensor, latent_size, encoder_size, decoder_size=None, hidden_activation=tf.nn.relu, output_activation=None, use_dropout=False, use_bn=False, bn_is_training=False, latent_layer_build_fun=build_vae_latent_layers):
    """
    Defines a VAE graph, with `len(encoder_size)+1+len(decoder_size)` dense
    layers:
    
    - `len(encoder_size)` layers for the encoder, where the i-th encoder layer has
      `encoder_size[i]` units.

    - The variational latent (code) layer with `latent_size` units;

    - `len(decoder_size)` layers for the decoder, where the j-th decoder layer
      has `decoder_size[j]` units.
   
    The output layer is of same dimension as the input layer.

    Parameters
    ----------
    input_tensor : Tensor
        The input tensor of size NUM_SAMPLES x INPUT_SIZE.

    latent_size : int
        Number of units in the MLP's output layer.

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
    reconstruction : Tensor
        The output (reconstruction of the input).

    latent_layer : Tensor
        The latent code (sampled from noise distribution).

    latent_mean : Tensor
        The mean of the distribution onto which the input is mapped.

    See Also
    --------
    tfmodellib.build_mlp_graph : Used to construct the encoder and decoder
                                 sub-networks of the autoencoder.
    tfmodellib.build_autoencoder_graph : Standard (non-variational)
                                         autoencoder.
    """

    # define encoder
    with tf.variable_scope('encoder'):
        encoder_out = build_mlp_graph(
                input_tensor=input_tensor,
                out_size=encoder_size[-1],
                n_hidden=encoder_size[:-1],
                hidden_activation=hidden_activation,
                output_activation=hidden_activation,
                use_dropout=use_dropout,
                use_bn=use_bn,
                bn_is_training=bn_is_training)

        if use_bn:
            encoder_out = tf.layers.batch_normalization(encoder_out, training=bn_is_training, name='batchnorm_encoder_out')

    with tf.variable_scope('latent_layers'):
        latent_layer, \
        latent_mean, \
        latent_sigma, \
        latent_sigma_sq, \
        latent_log_sigma_sq = build_vae_latent_layers(encoder_out, latent_size)

    # define decoder
    with tf.variable_scope('decoder'):

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

    return reconstruction, latent_layer, latent_mean, latent_sigma, latent_sigma_sq, latent_log_sigma_sq


def variational_loss(latent_mean, latent_sigma_sq, latent_log_sigma_sq):
    return tf.reduce_mean(0.5 * tf.reduce_sum(
            - 1.0
            - latent_log_sigma_sq
            + tf.square(latent_mean)
            + latent_sigma_sq, axis=-1))


class VAEConfig(TFModelConfig):

    def init(self):
        self.update(
                in_size=3,
                latent_size=2,
                encoder_size=[10,10],
                decoder_size=None,
                hidden_activation=tf.nn.relu,
                output_activation=None,
                optimizer=tf.train.AdamOptimizer,
                use_dropout=False,
                use_bn=False,
                reconstruction_loss=tf.losses.mean_squared_error,
                variational_loss=variational_loss,
                build_vae_latent_layers_fun=build_vae_latent_layers)
        super(VAEConfig, self).init()

class VAE(MLP):

    def build_graph(self):

        with tf.variable_scope('placeholders'):
            # define input and target placeholder
            self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.config['in_size']], name='x_input')
            self.y_target = tf.placeholder(dtype=tf.float32, shape=[None, self.config['in_size']], name='y_target')

            # define learning rate
            self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')

            # training flag for batchnorm
            self.bn_is_training = tf.placeholder(dtype=tf.bool, shape=[], name='bn_is_training')

        # define the base graph
        with tf.variable_scope('vae_graph'):
            self.y_output, \
            self.latent_layer, \
            self.latent_mean, \
            self.latent_sigma, \
            self.latent_sigma_sq, \
            self.latent_log_sigma_sq = build_vae_graph(input_tensor=self.x_input, bn_is_training=self.bn_is_training, **self.config)

        # define loss
        with tf.variable_scope('losses'):

            # reconstruction losses for all samples, a vector of shape BATCH_SIZE x 1
            with tf.variable_scope('reconstruction_losses'):
                self.reconstruction_losses = self.config['reconstruction_loss'](self.y_target, self.y_output)

            # we keep beta as a placeholder, to allow adjusting it throughout the training.
            self.beta = tf.placeholder(dtype=tf.float32, shape=[], name='beta')

            # variational (KL) losses for all samples, a vector of shape BATCH_SIZE x 1
            with tf.variable_scope('variational_losses'):
                self.variational_losses = self.config['variational_loss'](
                        self.latent_mean, self.latent_sigma_sq,
                        self.latent_log_sigma_sq)

            # combined loss, scalar
            with tf.variable_scope('loss'):
                self.loss = tf.add(self.reconstruction_losses, self.beta * self.variational_losses)

        # define optimizer
        with tf.control_dependencies(self.graph.get_collection(tf.GraphKeys.UPDATE_OPS)):
            with tf.variable_scope('optimization'):
                self.optimizer = self.config['optimizer'](learning_rate=self.learning_rate)
                self.minimize_op = self.optimizer.minimize(self.loss)

    def run_update_and_loss(self, batch_inputs, batch_targets, learning_rate, beta):
        loss, _ = self.sess.run([self.loss, self.minimize_op], feed_dict={
                self.x_input: batch_inputs,
                self.y_target: batch_targets,
                self.learning_rate: learning_rate,
                self.beta: beta,
                self.bn_is_training: True})
        return loss

    def run_loss(self, batch_inputs, batch_targets, learning_rate, beta):
        loss = self.sess.run(self.loss, feed_dict={
                self.x_input: batch_inputs,
                self.y_target: batch_targets,
                self.beta: beta,
                self.bn_is_training: False})
        return loss

if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d

    # create the model
    conf = VAEConfig(
            in_size=3,
            latent_size=5,
            encoder_size=[150,150],
            hidden_activation=tf.nn.relu,
            output_activation=None,
            reconstruction_loss=tf.losses.mean_squared_error,
            use_bn=True)
    model = VAE(conf)

    # generate some data
    xx,yy = np.random.rand(2*10000).reshape((2,-1))
    z = lambda x, y: (np.sin(10*x) + np.cos(10*y)) * np.exp(-((x-0.5)**2+(y-0.5)**2)/0.1)
    zz = z(xx,yy)
    x = np.vstack((xx.flat,yy.flat,zz.flat)).T

    x -= x.mean(axis=0)
    x /= x.std(axis=0)

    x_train = x[:int(x.shape[0]*0.8)]
    x_valid = x[x_train.shape[0]:]

    # run the training
    for t in range(1000):
        model.train(
                train_inputs=x_train,
                train_targets=x_train,
                validation_inputs=x_valid,
                validation_targets=x_valid,
                batch_size=1000,
                learning_rate=0.01,
                beta=0.01)

    # get estimates
    reconstruction = np.vstack(model.infer(inputs=x_valid, batch_size=None))

    # plot data and estimates
    fig = plt.figure()
    fig_size = fig.get_size_inches()
    fig_size[0] *= 2.5
    fig.set_size_inches(fig_size)

    ax = fig.add_subplot(1,3,1,projection='3d')
    ax.plot(*x_valid.T, ls='none', marker='.', mec='none', mfc='g')
    ax.plot(*reconstruction.T, ls='none', marker='.', mec='none', mfc='b')
    ax.set_title('reconstruction')

    # plot latent mean for the three latent dimensions with the lowest mean
    # standard deviation
    l = np.linspace(0.0, 1.0, 10)
    xx,yy = np.meshgrid(l,l)
    zz = z(xx,yy)
    x = np.vstack((xx.flat,yy.flat,zz.flat)).T

    latent_mean, latent_sigma = model.sess.run([model.latent_mean, model.latent_sigma], feed_dict={model.x_input: x, model.bn_is_training: False})
    latent_sigma_mean = latent_sigma.mean(axis=0)
    dim_inds = np.argsort(latent_sigma_mean)[:3]
    latent_u,latent_v,latent_w = latent_mean[:,dim_inds].reshape((l.size,l.size,3)).transpose((2,0,1))

    ax = fig.add_subplot(1,3,2,projection='3d')
    for ind in range(l.size):
        ax.plot(latent_u[ind], latent_v[ind], latent_w[ind], ls='-', marker='None', color='b')
        ax.plot(latent_u.T[ind], latent_v.T[ind], latent_w.T[ind], ls='-', marker='None', color='b')
    ax.set_title('latent encoding')
    ax.set_xlabel('latent dimension {:d}'.format(dim_inds[0]))
    ax.set_ylabel('latent dimension {:d}'.format(dim_inds[1]))
    ax.set_zlabel('latent dimension {:d}'.format(dim_inds[2]))

    # bar plot mean standard deviation of latent dimensions
    ax = fig.add_subplot(1,3,3)
    ax.bar(range(latent_sigma_mean.size), latent_sigma_mean)
    ax.set_title('mean standard deviations')
    ax.set_xlabel('latent dimension')
    ax.set_ylabel('sigma')

    fig.tight_layout()

    plt.show()
