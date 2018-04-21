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
def build_conv_encoder_2d_graph(
        input_tensor, n_filters, kernel_sizes, strides,
        hidden_activation=tf.nn.relu, latent_activation=None,
        pooling_sizes=None, use_bias=False, pooling_fun=tf.nn.avg_pool,
        use_dropout=False, use_bn=False):
    """
    Defines a convolutional encoder graph, with `len(n_filters)` convolutional
    layers.

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

    hidden_activation : function (optional)
        Activation function to apply to the output of each convolution layer
        (or alternatively to the output of the batch normalization step, if
        use_bn is True). If None, no nonlinearity will be applied. Default is
        tf.nn.relu.

    latent_activation : function (optional)
        Activation function to apply to the output layer of the encoder (the
        latent code layer). If None, no nonlinearity will be applied (the
        default).

    pooling_sizes : list (optional)
        List where elements can be either int or None. Specifies for each layer
        if the convolution should be followed by an average pooling step. If a
        list element is an int, it indicates the size of the pooling window
        along the WIDTH and HEIGHT dimension of the input. If a list element is
        None, no pooling is for the corresponding layer. Alternatively, passing
        None as argument switches pooling off for the whole network (the
        default).

    pooling_fun : function (optional)
        The pooling function to use (default: tf.nn.avg_pool).

    use_bias : bool (optional)
        Whether or not to use a bias in the convolution layers (default is False).

    use_dropout : bool (optional)
        Indicates whether or not to use dropout after each convolution layer
        (default: False).
    
    use_bn : bool (optional)
        Indicates whether or not to add a batch norm layer after each
        convolution layer (default: False).

    Returns
    -------
    encoder_output : Tensor
        The encoder's output.
    """

    current_input = input_tensor

    # build the encoder
    for ind, fs in enumerate(n_filters):

        current_input = tf.layers.conv2d(
                current_input, filters=fs,
                kernel_size=kernel_sizes[ind], strides=strides[ind],
                use_bias=False, padding='same', name='conv{:d}'.format(ind))

        if use_dropout:
            current_input = tf.layers.dropout(current_input, name='encoder_dropout_layer_{:d}'.format(ind))

        if use_bn:
            current_input = tf.layers.batch_normalization(current_input, name='encoder_batch_normalization_{:d}'.format(ind))

        if ind < len(n_filters)-1:
            nonlinearity = hidden_activation
        else:
            nonlinearity = latent_activation

        if nonlinearity is not None:
            current_input = nonlinearity(current_input)

        if pooling_sizes is not None and pooling_sizes[ind] is not None:
            current_input = pooling_fun(
                    current_input,
                    ksize=(1, pooling_sizes[ind], pooling_sizes[ind], 1),
                    strides=(1, pooling_sizes[ind], pooling_sizes[ind], 1),
                    padding='SAME')

    encoder_output = current_input

    return encoder_output


def params_encoder_to_decoder(input_tensor, n_filters_encoder,
        kernel_sizes_encoder, strides_encoder, pooling_sizes_encoder):

    input_shape = input_tensor.get_shape().as_list()

    # for the decoder, the number of output filters for each layer equals the
    # number of input filters of the corresponding layer in the encoder. Thus,
    # we insert the number of filters of the input tensor at the beginning of
    # the n_filters list, and remove the last element of the list.
    n_filters_decoder    = [n_filters_encoder[ind]    for ind in reversed(range(len(n_filters_encoder)-1))] + [input_shape[-1]]

    # reverse kernel sizes and strides
    kernel_sizes_decoder = [kernel_sizes_encoder[ind] for ind in reversed(range(len(n_filters_encoder)))]
    strides_decoder      = [strides_encoder[ind]      for ind in reversed(range(len(n_filters_encoder)))]

    # reconstruct filter map sizes based on the encoder's pooling sizes
    unpooling_sizes = None
    if pooling_sizes_encoder is not None:
        unpooling_sizes = [input_shape[1:3]]
        for pooling_size in pooling_sizes_encoder[:-1]:
            if pooling_size is not None:
                if isinstance(pooling_size, list) or isinstance(pooling_size, tuple):
                    pooling_size_u, pooling_size_v = pooling_size
                else:
                    pooling_size_u = pooling_size_v = pooling_size
                size_u = 1+(unpooling_sizes[-1][0]-1)//pooling_size_u
                size_v = 1+(unpooling_sizes[-1][1]-1)//pooling_size_v
                unpooling_sizes.append((size_u,size_v))
            else:
                unpooling_sizes.append(None)
        unpooling_sizes.reverse()

    return n_filters_decoder, kernel_sizes_decoder, strides_decoder, unpooling_sizes


@graph_def
@docsig
def build_conv_decoder_2d_graph(
        input_tensor, n_filters, kernel_sizes, strides, 
        hidden_activation=tf.nn.relu, output_activation=None,
        unpooling_sizes=None, unpooling_fun=tf.image.resize_images,
        use_bias=False, reuse=True, use_dropout=False, use_bn=False):
    """
    Defines a convolutional decoder graph, with `len(n_filters)` deconvolution
    layers.

    The parameters `n_filters`, `kernel_sizes`, and `strides` are lists of
    length equal to the number of layers in both the encoder and decoder.
    Elements of the lists correspond to parameters passed to
    tf.layers.conv2d_transpose (`filters`, `kernel_size`, and `stride`,
    respectively).

    Parameters
    ----------
    input_tensor : Tensor
        The input tensor of size NUM_SAMPLES x HEIGHT x WIDTH x NUM_FILTERS.

    n_filters : list
        List of ints, specifying the number of filters for each deconvolution
        layer.

    kernel_sizes : list
        List of ints, or a list of tuples/lists of 2 integers, specifying the
        height and width of the 2D deconvolution window.

    strides : list
        List of inds, or a list of tuples/lists of 2 integers, specifying the
        stride of the deconvolution along the HEIGHT and WIDTH dimensions.

    hidden_activation : function (optional)
        Activation function to apply to the output of each deconvolution layer
        (or alternatively to the output of the batch normalization step, if
        use_bn is True). If None, no nonlinearity will be applied. Default is
        tf.nn.relu.

    output_activation : function (optional)
        Activation function to apply to the output layer. If None, no
        nonlinearity will be applied (the default).


    unpooling_sizes : list (optional)
        List where elements can be either int or None. Specifies for each layer
        if the deconvolution should be followed by an unpooling step. If a
        list element is an int, it indicates the size of the unpooling window
        along the WIDTH and HEIGHT dimension of the input. If a list element is
        None, no unpooling is for the corresponding layer. Alternatively, passing
        None as argument switches unpooling off for the whole network (the
        default).

    unpooling_fun : function (optional)
        The unpooling function to use (default: tf.image.resize_images).

    use_bias : bool (optional)
        Whether or not to use a bias in the deconvolution layers (default is False).

    reuse : bool (optional)
        Whether or not to reuse wights from the encoder. Not compatible with
        use_bias=True. (default is True).

    use_dropout : bool (optional)
        Indicates whether or not to use dropout after each deconvolution layer
        (default: False).
    
    use_bn : bool (optional)
        Indicates whether or not to add a batch norm layer after each
        deconvolution layer (default: False).

    Returns
    -------
    reconstruction : Tensor
        The output (decoded input).
    """

    # build decoder

    current_input = input_tensor

    # iterate layers in reverse order
    for ind, fs in enumerate(n_filters):

        if unpooling_sizes is not None and unpooling_sizes[ind] is not None:
            # upsampling unpooling
            current_input = unpooling_fun(current_input, unpooling_sizes[ind])

        # deconvolution
        if reuse:
            name = 'conv{:d}'
        else:
            name = 'deconv{:d}'
        name = name.format(len(n_filters)-1-ind)

        current_input = tf.layers.conv2d_transpose(
                current_input, filters=fs,
                kernel_size=kernel_sizes[ind], strides=strides[ind],
                use_bias=use_bias, padding='same',
                name=name, reuse=reuse)

        if use_dropout:
            current_input = tf.layers.dropout(current_input, name='decoder_dropout_layer_{:d}'.format(ind))

        if use_bn:
            current_input = tf.layers.batch_normalization(current_input, name='decoder_batch_normalization_{:d}'.format(ind))

        if ind < len(n_filters)-1:
            nonlinearity = hidden_activation
        else:
            nonlinearity = output_activation

        if nonlinearity is not None:
            current_input = nonlinearity(current_input)

    reconstruction = current_input

    return reconstruction


@graph_def
@docsig
def build_cae_2d_graph(
        input_tensor,
        n_filters, kernel_sizes, strides, hidden_activation=tf.nn.relu,
        latent_activation=tf.nn.relu, output_activation=None,
        pooling_sizes=None, pooling_fun=tf.nn.avg_pool,
        unpooling_fun=tf.image.resize_images, use_bias=False, reuse=True,
        use_dropout=False, use_bn=False, latent_op=None):
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

    use_bias : bool (optional)
        Whether or not to use a bias in the (de-)convolution layers (default is
        False).

    reuse : bool (optional)
        Whether or not to reuse wights from the encoder in the decoder. Not compatible with
        use_bias=True. (default is True).

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

    encoder_output = build_conv_encoder_2d_graph(
            input_tensor=input_tensor, n_filters=n_filters,
            kernel_sizes=kernel_sizes, strides=strides,
            hidden_activation=hidden_activation,
            latent_activation=latent_activation, pooling_sizes=pooling_sizes,
            pooling_fun=pooling_fun, use_bis=use_bias, use_dropout=use_dropout,
            use_bn=use_bn)

    # apply latent_op to the latent code
    if latent_op is not None:
        decoder_input = latent_op(encoder_output)
    else:
        decoder_input = encoder_output

    n_filters_decoder, \
    kernel_sizes_decoder, \
    strides_decoder, \
    unpooling_sizes = params_encoder_to_decoder(
            input_tensor, n_filters, kernel_sizes, strides, pooling_sizes)

    reconstruction = build_conv_decoder_2d_graph(
            input_tensor=decoder_input, n_filters=n_filters_decoder,
            kernel_sizes=kernel_sizes_decoder, strides=strides_decoder,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            unpooling_sizes=unpooling_sizes, unpooling_fun=unpooling_fun,
            use_bias=use_bias, reuse=reuse, use_dropout=use_dropout,
            use_bn=use_bn)

    return reconstruction



class CAE2dConfig(TFModelConfig):

    def init(self):
        self.update(
                in_size=[28,28,3],
                n_filters=[10,10],
                kernel_sizes=[3,3],
                strides=[1,1],
                hidden_activation=tf.nn.relu,
                latent_activation=tf.nn.relu,
                output_activation=None,
                pooling_sizes=None,
                pooling_fun=tf.nn.avg_pool,
                unpooling_fun=tf.image.resize_images,
                use_bias=False,
                reuse=True,
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
                in_size=[28,28,1],
                n_filters=[30,20],
                kernel_sizes=[5,5],
                strides=[1,1],
                pooling_sizes=[2,2],
                use_bias=True,
                reuse=False,
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
