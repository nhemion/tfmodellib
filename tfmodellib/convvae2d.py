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

from tfmodellib import TFModel, TFModelConfig, graph_def, docsig, CAE2d, CAE2dConfig, build_cae_2d_graph, VAE, VAEConfig, build_vae_graph, variational_loss

import tensorflow as tf
import numpy as np
import sys

# def build_vae_graph(input_tensor, latent_size, encoder_size, decoder_size=None, hidden_activation=tf.nn.relu, output_activation=None, use_dropout=False, use_bn=False, bn_is_training=False, latent_layer_build_fun=build_vae_latent_layers):
# def sum_of_squared_differences(a, b):
# def mean_of_squared_differences(a, b):
# def variational_loss(latent_mean, latent_sigma_sq, latent_log_sigma_sq, beta):


class ConvVAE2dConfig(CAE2dConfig, VAEConfig):

    def init(self):
        self = super(ConvVAE2dConfig, self).init()


class ConvVAE2d(TFModel):

    def build_graph(self):

        assert sys.version_info >= (3,0), 'Requires Python 3.0 or newer'

        # define input and target placeholder
        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None]+self.config['in_size'])
        self.y_target = tf.placeholder(dtype=tf.float32, shape=[None]+self.config['in_size'])

        # define the convolutional base graph
        self.y_output = build_cae_2d_graph(
                input_tensor=self.x_input,
                **self.config,
                latent_op=self._get_build_vae_graph_op())

        # define learning rate
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[])

        # define loss
        with tf.variable_scope('losses'):

            # reconstruction losses for all samples, a vector of shape BATCH_SIZE x 1
            with tf.variable_scope('reconstruction_losses'):
                l = self.config['reconstruction_loss'](self.y_target, self.y_output)
                self.reconstruction_losses = tf.reduce_sum(l, axis=list(range(1, len(l.shape))))

            # we keep beta as a placeholder, to allow adjusting it throughout the training.
            self.beta = tf.placeholder(dtype=tf.float32, shape=[], name='beta')

            # variational (KL) losses for all samples, a vector of shape BATCH_SIZE x 1
            with tf.variable_scope('variational_losses'):
                self.variational_losses = self.config['variational_loss'](
                        self.latent_mean, self.latent_sigma_sq,
                        self.latent_log_sigma_sq, self.beta)

            # combined loss, scalar
            with tf.variable_scope('loss'):
                self.loss = tf.reduce_mean(tf.add(self.reconstruction_losses, self.variational_losses), axis=0)

        # define optimizer
        with tf.control_dependencies(self.graph.get_collection(tf.GraphKeys.UPDATE_OPS)):
            with tf.variable_scope('optimization'):
                self.optimizer = self.config['optimizer'](learning_rate=self.learning_rate)
                self.minimize_op = self.optimizer.minimize(self.loss)

    def _get_build_vae_graph_op(self):
    
        def op(input_tensor):

            # flatten tensor
            conv_size = input_tensor.get_shape().as_list()[1:]
            flat_size = np.prod(conv_size)
            self.dense_vae_input_tensor = tf.reshape(input_tensor, [-1, flat_size])

            # build the dense VAE
            self.dense_vae_reconstruction, \
            self.latent_layer, \
            self.latent_mean, \
            self.latent_sigma, \
            self.latent_sigma_sq, \
            self.latent_log_sigma_sq = build_vae_graph(
                    self.dense_vae_input_tensor,
                    latent_layer_build_fun=self.config['build_vae_latent_layers_fun'],
                    **self.config)

            # reshape back to original shape
            self.conv_decoder_input = tf.reshape(self.dense_vae_reconstruction, [-1] + conv_size)

            return self.conv_decoder_input

        return op

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

    def run_output(self, inputs):
        return self.sess.run(self.y_output, feed_dict={self.x_input: inputs})


if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt
    import tempfile
    import logging
    import shutil
    import os

    # fetch MNIST data into a temporary directory
    try:
        tmpdir = tempfile.mkdtemp()
        cwd = os.getcwd()
        os.chdir(tmpdir)

        print('Loading MNIST dataset into {:s}'.format(tmpdir))

        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        os.chdir(cwd)
        train_data = mnist.train.images.reshape((-1,28,28,1))
        eval_data = mnist.test.images.reshape((-1,28,28,1))

        # create the model
        conf = ConvVAE2dConfig(
                log_level=logging.DEBUG,
                in_size=[28,28,1],
                n_filters=[50,50],
                kernel_sizes=[3,3],
                strides=[1,1],
                nonlinearity=tf.nn.relu,
                pooling_sizes=[2,2],
                latent_size=3,
                n_hidden=[100,100],
                activation=tf.nn.relu,
                use_dropout=False,
                use_bn=False)
        model = ConvVAE2d(conf)

        # run the training
        for _ in range(50):
            model.train(
                    train_inputs=train_data,
                    train_targets=train_data,
                    validation_inputs=eval_data,
                    validation_targets=eval_data,
                    batch_size=1000,
                    learning_rate=0.0001,
                    beta=0.001)
    
        # get estimates
        reconstruction = model.infer(inputs=eval_data[:1000], batch_size=None)
    
        # plot data and estimates
        fig = plt.figure()
        for ind,im in enumerate(reconstruction[:9]):
            ax = fig.add_subplot(3,3,ind+1)
            im = im.transpose((2,0,1))[0]
            im = np.uint8(np.minimum(np.maximum(0.0, im), 1.0) * 255)
            ax.imshow(im)

        plt.show()

    except Exception as e:
        print(e)

    finally:
        try:
            print('Deleting MNIST dataset in {:s}'.format(tmpdir))
            shutil.rmtree(tmpdir)
        except FileNotFoundError:
            pass
