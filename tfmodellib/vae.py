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

from tfmodellib import TFModel, TFModelConfig, graph_def, docsig, MLP, MLPConfig, build_mlp_graph

import tensorflow as tf


def variational_loss(mean, sigma_sq):
    """
    Computes the variational (KL-divergence) term for a VAE loss.
    """
    return 0.5 * tf.reduce_sum(
            - 1.0
            - tf.log(sigma_sq+1e-15)
            + tf.square(mean)
            + sigma_sq, axis=1)


@graph_def
@docsig
def build_vae_graph(input_tensor, latent_size, n_hidden, activation, **kwargs):
    """
    Defines a VAE graph, with `2*len(n_hidden)+1` dense layers:
    
    - `len(n_hidden)` layers for the encoder, where the i-th encoder layer has
      `n_hidden[i]` units.

    - The variational latent (code) layer with `latent_size` units;

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

    latent_layer : Tensor
        The latent code (sampled from noise distribution).

    latent_mean : Tensor
        The mean of the distribution onto which the input is mapped.

    See Also
    --------
    tfmodels.build_mlp_graph : Used to construct the encoder and decoder
                              sub-networks of the autoencoder.
    tfmodels.build_autoencoder_graph : Standard (non-variational) autoencoder.
    """

    # define encoder
    with tf.variable_scope('encoder'):
        encoder_out = build_mlp_graph(
                input_tensor=input_tensor,
                out_size=n_hidden[-1],
                n_hidden=n_hidden[:-1],
                activation=activation,
                **kwargs)

    # define latent mean, sigma_sq, randn
    latent_mean = tf.layers.dense(encoder_out, units=latent_size, activation=None)
    latent_sigma_sq = tf.layers.dense(encoder_out, units=latent_size, activation=tf.nn.relu)
    latent_randn = tf.random_normal(shape=[latent_size], dtype=tf.float32)
    # latent_randn = tf.random_normal(
    #         shape=(batch_size, latent_size),
    #         mean=0.0, stddev=1.0, dtype=tf.float32)

    # define latent layer
    latent_layer = tf.add(
            latent_mean,
            tf.multiply(
                    tf.sqrt(latent_sigma_sq),
                    latent_randn))

    # define decoder
    n_hidden.reverse()
    with tf.variable_scope('decoder'):
        reconstruction = build_mlp_graph(
                input_tensor=latent_layer,
                out_size=input_tensor.shape.as_list()[1],
                n_hidden=n_hidden,
                activation=activation)

    return reconstruction, latent_layer, latent_mean, latent_sigma_sq



class VAEConfig(MLPConfig):

    def init(self):
        self.update(
                in_size=3,
                latent_size=2,
                n_hidden=[10,10],
                activation=tf.nn.relu)
        super(VAEConfig, self).init()

class VAE(MLP):

    def build_graph(self):

        # define input and target placeholder
        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.config['in_size']])
        self.y_target = tf.placeholder(dtype=tf.float32, shape=[None, self.config['in_size']])

        # define the base graph
        self.y_output, \
        self.latent_layer, \
        self.latent_mean, \
        self.latent_sigma_sq = build_vae_graph(input_tensor=self.x_input, **self.config)

        # define learning rate
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[])

        # define loss
        self.beta = tf.placeholder(dtype=tf.float32, shape=[])

        self.reconstruction_loss = tf.reduce_sum(
                tf.squared_difference(self.y_target, self.y_output), axis=1)

        self.variational_latent_loss = self.beta * variational_loss(
                mean=self.latent_mean,
                sigma_sq=self.latent_sigma_sq)

        self.loss = tf.reduce_mean(tf.add(
                self.reconstruction_loss,
                self.variational_latent_loss), axis=0)

        # define optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.minimize_op = self.optimizer.minimize(self.loss)

    def run_update_and_loss(self, batch_inputs, batch_targets, learning_rate, beta):
        loss, _ = self.sess.run([self.loss, self.minimize_op],
                feed_dict={
                        self.x_input: batch_inputs,
                        self.y_target: batch_targets,
                        self.learning_rate: learning_rate,
                        self.beta: beta})
        return loss

    def run_loss(self, batch_inputs, batch_targets, learning_rate, beta):
        loss = self.sess.run(self.loss,
                feed_dict={
                        self.x_input: batch_inputs,
                        self.y_target: batch_targets,
                        self.beta: beta})
        return loss


if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d

    # create the model
    conf = VAEConfig(
            in_size=3,
            latent_size=2,
            n_hidden=[150,50,10],
            activation=tf.nn.relu)
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
    for t in range(10000):
        model.train(
                train_inputs=x_train,
                train_targets=x_train,
                validation_inputs=x_valid,
                validation_targets=x_valid,
                batch_size=1000,
                learning_rate=0.0001,
                beta=1.0)

    # get estimates
    reconstruction = np.vstack(model.infer(inputs=x_valid, batch_size=None))

    # plot data and estimates
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(*x_valid.T, ls='none', marker='.', mec='none', mfc='g')
    ax.plot(*reconstruction.T, ls='none', marker='.', mec='none', mfc='b')
    ax.set_title('reconstruction')

    # plot latent mean
    l = np.linspace(0.0, 1.0, 10)
    xx,yy = np.meshgrid(l,l)
    zz = z(xx,yy)
    x = np.vstack((xx.flat,yy.flat,zz.flat)).T

    latent_code = model.sess.run(model.latent_mean, feed_dict={model.x_input: x})
    latent_u,latent_v = latent_code.reshape((l.size,l.size,2)).transpose((2,0,1))

    fig = plt.figure()
    ax = fig.gca()
    for ind in range(l.size):
        ax.plot(latent_u[ind], latent_v[ind], ls='-', marker='None', color='b')
        ax.plot(latent_u.T[ind], latent_v.T[ind], ls='-', marker='None', color='b')
    ax.set_title('latent mean')

    plt.show()
