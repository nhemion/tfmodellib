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
def build_cae_2d_graph(
        input_tensor,
        n_filters, kernel_sizes, strides, nonlinearity=tf.nn.relu,
        pooling_sizes=None, pooling_fun=tf.nn.avg_pool,
        unpooling_fun=tf.image.resize_images, use_dropout=False, use_bn=False,
        latent_op=None):
    """
    Defines a convolutional autoencoder graph, with `2*len(n_filters)`
    convolutional layers (`len(n_filters)` layers for both the encoder and the
    decoder).

    The parameters `n_filters`, `kernel_sizes`, and `strides` are lists of
    length equal to the number of layers in both the encoder and decoder.
    Elements of the lists correspond to parameters passed to tf.layers.conv2d
    (`filters`, `kernel_size`, and `stride`, respectively).

    Parameters
    ----------
    input_tensor : Tensor
        The input tensor of size NUM_SAMPLES x HEIGHT x WIDTH x NUM_FILTERS.

    n_filters : list
        List of ints, specifying the number of filters for each convolutional
        layer.

    kernel_sizes : list
        List of ints, or a list of tuples/lists of 2 integers, specifying the
        height and width of the 2D convolution window.

    strides : list
        List of inds, or a list of tuples/lists of 2 integers, specifying the
        stride of the convolution along the HEIGHT and WIDTH dimensions.

    nonlinearity : function
        A nonlinearity to apply to the output of the convolution layer (or
        alternatively to the output of the batch normalization step, if use_bn
        is True). If None, no nonlinearity will be applied.

    pooling_sizes : list (optional)
        List where elements can be either int or None. Specifies for each layer
        if the convolution should be followed by an average pooling step. If a
        list element is an int, it indicates the size of the pooling window
        along the WIDTH and HEIGHT dimension of the input. If a list element is
        None, no pooling is for the corresponding layer. Alternatively, passing
        None as argument switches pooling off for the whole network (the
        default).

    use_dropout : bool (optional)
        Indicates whether or not to use dropout after each convolution layer
        (default: False).
    
    use_bn : bool (optional)
        Indicates whether or not to add a batch norm layer after each
        convolution layer (default: False).

    latent_op : function (optional)
        If not None, this function will be called with the output tensor of the
        CAE's encoder as argument. This allows to transform the latent code
        before it is passed to the decoder. latent_op should return a tensor of
        identical shape and dtype as the one it receives as parameter.

    Returns
    -------
    reconstruction : Tensor
        The output (reconstruction of the input).

    See Also
    --------
    tfmodels.build_autoencoder_graph : Standard (non-convolutional) autoencoder.
    """

    input_n_filters = input_tensor.get_shape().as_list()[-1]
    current_input = input_tensor

    # build the encoder
    map_dimensions = []
    for ind, fs in enumerate(n_filters):

        current_input = tf.layers.conv2d(
                current_input, filters=fs,
                kernel_size=kernel_sizes[ind], strides=strides[ind],
                use_bias=False, padding='same', name='conv{:d}'.format(ind))

        if use_bn:
            current_input = tf.layers.batch_normalization(current_input)

        if nonlinearity is not None:
            current_input = nonlinearity(current_input)

        map_dimensions.append(current_input.get_shape().as_list()[1:3])

        if pooling_sizes is not None and pooling_sizes[ind] is not None:
            current_input = pooling_fun(
                    current_input,
                    ksize=(1, pooling_sizes[ind], pooling_sizes[ind], 1),
                    strides=(1, pooling_sizes[ind], pooling_sizes[ind], 1),
                    padding='SAME')

    # apply latent_op to the latent code
    if latent_op is not None:
        current_input = latent_op(current_input)

    # build decoder

    # for the decoder, the number of output filters for each layer equals the
    # number of input filters of the corresponding layer in the encoder. Thus,
    # we insert the number of filters of the input tensor at the beginning of
    # the n_filters list, and remove the last element of the list.
    n_filters = [input_n_filters] + n_filters[:-1]

    # iterate layers in reverse order
    for ind in reversed(range(len(n_filters))):

        if pooling_sizes is not None and pooling_sizes[ind] is not None:
            # upsampling unpooling
            current_input = unpooling_fun(current_input, map_dimensions[ind])

        # deconvolution
        current_input = tf.layers.conv2d_transpose(
                current_input, filters=n_filters[ind],
                kernel_size=kernel_sizes[ind], strides=strides[ind],
                use_bias=False, padding='same', name='conv{:d}'.format(ind),
                reuse=True)

    reconstruction = current_input

    return reconstruction



class CAE2dConfig(TFModelConfig):

    def init(self):
        self.update(
                in_size=[28,28,3],
                n_filters=[10,10],
                kernel_sizes=[3,3],
                strides=[1,1],
                nonlinearity=tf.nn.relu,
                pooling_sizes=None,
                pooling_fun=tf.nn.avg_pool,
                unpooling_fun=tf.image.resize_images,
                use_dropout=False,
                use_bn=False)
        super(CAE2dConfig, self).init()


class CAE2d(TFModel):

    def build_graph(self):

        # define input and target placeholder
        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None]+self.config['in_size'])
        self.y_target = tf.placeholder(dtype=tf.float32, shape=[None]+self.config['in_size'])

        # define the base graph
        self.y_output = build_cae_2d_graph(input_tensor=self.x_input, **self.config)

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
    import tempfile
    import logging
    import shutil
    import os

    # fetch MNIST data into a temporary directory
    try:
        tmpdir = tempfile.mkdtemp()
        cwd = os.getcwd()
        os.chdir(tmpdir)
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        os.chdir(cwd)
        train_data = mnist.train.images.reshape((-1,28,28,1))
        eval_data = mnist.test.images.reshape((-1,28,28,1))


    except Exception as e:
        print(e)

    else:

        # create the model
        conf = CAE2dConfig(
                log_level=logging.DEBUG,
                in_size=[28,28,1],
                n_filters=[30,20],
                kernel_sizes=[5,5],
                strides=[1,1],
                nonlinearity=tf.nn.relu,
                pooling_sizes=[2,2],
                use_dropout=True,
                use_bn=True)
        model = CAE2d(conf)

        # run the training
        for _ in range(5):
            model.train(
                    train_inputs=train_data,
                    train_targets=train_data,
                    validation_inputs=eval_data,
                    validation_targets=eval_data,
                    batch_size=1000,
                    learning_rate=0.0001)
    
        # get estimates
        reconstruction = model.infer(inputs=eval_data[:9], batch_size=None)
    
        # plot data and estimates
        fig = plt.figure()
        for ind,im in enumerate(reconstruction):
            ax = fig.add_subplot(3,3,ind+1)
            im = im.transpose((2,0,1))[0]
            im = np.uint8(np.minimum(np.maximum(0.0, im), 1.0) * 255)
            ax.imshow(im)

        plt.show()

    finally:
        try:
            shutil.rmtree(tmpdir)
        except FileNotFoundError:
            pass
