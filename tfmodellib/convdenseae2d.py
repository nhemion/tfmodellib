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

from tfmodellib import CAE2d, CAE2dConfig, build_cae_2d_graph, AutoEncoder, AutoEncoderConfig, build_autoencoder_graph

import tensorflow as tf
import numpy as np


class ConvDenseAE2dConfig(AutoEncoderConfig, CAE2dConfig):

    def init(self):
        super(ConvDenseAE2dConfig, self).init()
        # a bit of renaming to disambiguate meaning of parameters
        hidden_activation = self.pop('hidden_activation')
        latent_activation = self.pop('latent_activation')
        encoder_size      = self.pop('encoder_size')
        decoder_size      = self.pop('decoder_size')
        output_activation = self.pop('output_activation')
        self.update(
                conv_hidden_activation=hidden_activation,
                conv_latent_activation=latent_activation,
                conv_output_activation=output_activation,
                dense_encoder_size=encoder_size,
                dense_decoder_size=decoder_size,
                dense_hidden_activation=hidden_activation,
                dense_output_activation=hidden_activation)

class ConvDenseAE2d(CAE2d):

    def build_graph(self):

        # define input and target placeholder
        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None]+self.config['in_size'])
        self.y_target = tf.placeholder(dtype=tf.float32, shape=[None]+self.config['in_size'])
        self.bn_is_training = tf.placeholder(dtype=tf.bool, shape=[], name='bn_is_training')

        # define the base graph
        self.y_output = build_cae_2d_graph(
                input_tensor=self.x_input,
                hidden_activation=self.config['conv_hidden_activation'],
                latent_activation=self.config['conv_latent_activation'],
                output_activation=self.config['conv_output_activation'],
                encoder_name='conv_encoder',
                decoder_name='conv_decoder',
                latent_op=self._get_build_ae_op(),
                bn_is_training=self.bn_is_training,
                **self.config)

        # define learning rate
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[])

        # define loss
        self.loss = tf.losses.mean_squared_error(self.y_target, self.y_output)

        # define optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.minimize_op = self.optimizer.minimize(self.loss)

    def _get_build_ae_op(self):

        def f(input_tensor):

            # flatten tensor
            conv_size = input_tensor.get_shape().as_list()[1:]
            flat_size = np.prod(conv_size)
            dense_input_tensor = tf.reshape(input_tensor, [-1, flat_size])

            # build the dense AE
            dense_output_tensor = build_autoencoder_graph(
                    dense_input_tensor,
                    encoder_size=self.config['dense_encoder_size'],
                    decoder_size=self.config['dense_decoder_size'],
                    hidden_activation=self.config['dense_hidden_activation'],
                    output_activation=self.config['dense_output_activation'],
                    bn_is_training=self.bn_is_training,
                    latent_layer_fun=self.latent_layer_fun,
                    encoder_name='dense_encoder',
                    decoder_name='dense_decoder',
                    **dict([(k,v) for k,v in self.config.items()   # to avoid double
                            if k is not 'latent_layer_fun']))      # keyword argument

            # reshape back to original shape
            conv_decoder_input = tf.reshape(dense_output_tensor, [-1] + conv_size)

            return conv_decoder_input

        return f

    def latent_layer_fun(self, input_tensor):
        # adding this as a member function allows it to be overwritten by child
        # classes
        return self.config['latent_layer_fun'](input_tensor)


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

        # a simple op to obtain a reference to the latent code tensor
        op_ref = dict()
        def latent_op(inputs):
            op_ref['latent'] = inputs
            return inputs

        # create the model
        conf = ConvDenseAE2dConfig(
                in_size=[28,28,1],
                n_filters=[30,20],
                kernel_sizes=[5,5],
                strides=[1,1],
                pooling_sizes=[2,2],
                conv_hidden_activation=tf.nn.relu,
                conv_latent_activation=tf.nn.relu,
                conv_output_activation=None,
                dense_encoder_size=[10,10],
                dense_decoder_size=[10,10],
                latent_size=3,
                dense_hidden_activation=tf.nn.relu,
                dense_output_activation=tf.nn.relu,
                latent_layer_fun=latent_op)
        model = ConvDenseAE2d(conf)

        # run the training
        for _ in range(50):
            model.train(
                    train_inputs=train_data,
                    train_targets=train_data,
                    validation_inputs=test_data,
                    validation_targets=test_data,
                    batch_size=1000,
                    learning_rate=0.0001)

        # get estimates
        reconstruction = model.infer(inputs=test_data[:9], batch_size=None)
    
        # plot data and estimates
        fig = plt.figure()
        for ind,im in enumerate(reconstruction):
            ax = fig.add_subplot(3,3,ind+1)
            im = im.transpose((2,0,1))[0]
            im = np.uint8(np.minimum(np.maximum(0.0, im), 1.0) * 255)
            ax.imshow(im)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        z = model.sess.run(op_ref['latent'], feed_dict={model.x_input: test_data})
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
