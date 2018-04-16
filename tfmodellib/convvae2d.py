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

from tfmodellib import TFModel, TFModelConfig, graph_def, docsig, CAE2dConfig, build_cae_2d_graph, VAEConfig, build_vae_graph
import tensorflow as tf
import numpy as np
import sys


class ConvVAE2dConfig(VAEConfig, CAE2dConfig):

    def init(self):
        super(ConvVAE2dConfig, self).init()
        # a bit of renaming to disambiguate meaning of parameters
        self.update(
                conv_activation=self.pop('nonlinearity'),
                dense_encoder_size=self.pop('encoder_size'),
                dense_decoder_size=self.pop('decoder_size'),
                dense_hidden_activation=self.pop('hidden_activation'),
                dense_output_activation=self.pop('output_activation'),
                conv_layer_activation=None)


class ConvVAE2d(TFModel):

    def build_graph(self):

        # define input and target placeholder
        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None]+self.config['in_size'])
        self.y_target = tf.placeholder(dtype=tf.float32, shape=[None]+self.config['in_size'])

        # define the convolutional base graph
        self.y_output = build_cae_2d_graph(
                input_tensor=self.x_input,
                nonlinearity=self.config['conv_activation'],
                latent_op=self._get_build_vae_graph_op(),
                **self.config)

        # define learning rate
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[])

        # define loss
        with tf.variable_scope('losses'):

            # reconstruction losses for all samples, a vector of shape BATCH_SIZE x 1
            with tf.variable_scope('reconstruction_loss'):
                self.reconstruction_loss = self.config['reconstruction_loss'](self.y_target, self.y_output)

            # we keep beta as a placeholder, to allow adjusting it throughout the training.
            self.beta = tf.placeholder(dtype=tf.float32, shape=[], name='beta')

            # variational (KL) losses for all samples, a vector of shape BATCH_SIZE x 1
            with tf.variable_scope('variational_loss'):
                self.variational_loss = self.config['variational_loss'](
                        self.latent_mean, self.latent_sigma_sq,
                        self.latent_log_sigma_sq)

            # combined loss, scalar
            with tf.variable_scope('loss'):
                self.loss = self.reconstruction_loss + self.beta * self.variational_loss

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
                    encoder_size=self.config['dense_encoder_size'],
                    decoder_size=self.config['dense_decoder_size'],
                    hidden_activation=self.config['dense_hidden_activation'],
                    output_activation=self.config['dense_output_activation'],
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
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import tempfile
    import traceback
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
        test_data = mnist.test.images.reshape((-1,28,28,1))
        test_labels = mnist.test.labels


        # create the model
        conf = ConvVAE2dConfig(
                in_size=[28,28,1],
                n_filters=[50,50],
                kernel_sizes=[3,3],
                strides=[1,1],
                conv_activation=tf.nn.relu,
                pooling_sizes=[2,2],
                latent_size=3,
                dense_encoder_size=[100,100],
                dense_decoder_size=None,
                dense_hidden_activation=tf.nn.relu,
                dense_output_activation=None,
                output_layer_activation=None)

        model = ConvVAE2d(conf)

        # run the training
        for _ in range(50):
            model.train(
                    train_inputs=train_data,
                    train_targets=train_data,
                    validation_inputs=test_data,
                    validation_targets=test_data,
                    batch_size=1000,
                    learning_rate=0.0001,
                    beta=0.001)
    
        # get estimates
        reconstruction = model.infer(inputs=test_data[:1000], batch_size=None)
    
        # plot data and estimates
        fig = plt.figure()
        for ind,im in enumerate(reconstruction[:9]):
            ax = fig.add_subplot(3,3,ind+1)
            im = im.transpose((2,0,1))[0]
            im = np.uint8(np.minimum(np.maximum(0.0, im), 1.0) * 255)
            ax.imshow(im)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        z = model.sess.run(model.latent_mean, feed_dict={model.x_input: test_data})
        n_classes = test_labels.max()+1
        cmap = plt.get_cmap('jet', n_classes)
        for class_id in range(n_classes):
            ax.plot(*z[test_labels==class_id].T, ls='none', marker='.', mec=cmap(class_id), mew=0.0, label='{:d}'.format(class_id))
        ax.legend()

        plt.show()

    except Exception as e:
        print(traceback.format_exc())

    finally:
        try:
            print('Deleting MNIST dataset in {:s}'.format(tmpdir))
            shutil.rmtree(tmpdir)
        except FileNotFoundError:
            pass
