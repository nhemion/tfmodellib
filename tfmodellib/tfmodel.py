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

import tensorflow as tf
import numpy as np
import logging
import json
import os
import sys
import inspect
import importlib


def docsig(f):
    """
    A decorator to add the function name and signature to the function's
    docstring.
    """
    if 'signature' in dir(inspect):
        # python3 
        sig = str(inspect.signature(f))
    else:
        # python2
        sig = inspect.formatargspec(*inspect.getargspec(f))
    sigstr = '{:s}{:s}\n\n'.format(f.__code__.co_name, sig)
    f.__doc__ = sigstr + inspect.cleandoc(f.__doc__)
    return f

 
def graph_def(fun):
    """
    A decorator to allow to pass a TFModelConfig directly to a function. Only
    the entries of the TFModelConfig with matching keys to one of the function
    argument names will be passed to the function.
    """
    def select_kwargs(*args, **kwargs):
        fun_varnames = fun.__code__.co_varnames
        selected_kwargs = dict([
                (key,val) for key,val in kwargs.items() if key in fun_varnames])
        return fun(*args, **selected_kwargs)

    select_kwargs.__doc__ = fun.__doc__

    return select_kwargs



def chunklist(values, length):
    """
    Create a generator to split a list into chunks of equal length.

    >>> [l for l in chunklist(range(6),2)]
    [[0, 1], [2, 3], [4, 5]]
    >>> [l for l in chunklist(range(7),3)]
    [[0, 1, 2], [3, 4, 5]]
    """
    for i in range(0, len(values), length):
        out = values[i:i+length]
        if len(out) == length:
            yield out
        else:
            return


def maybe_chunked(values, length, on_chunked, on_not_chunked, *args, **kwargs):
    """
    If the parameter LENGTH is not None, the list provided as VALUES will be
    chunked into sub-lists of length LENGTH, and ON_CHUNKED is called for each
    sub-list. Otherwise, ON_NOT_CHUNKED is called once with the whole list.

    Examples
    --------
    >>> maybe_chunked(range(5), None, np.prod, np.sum)
    (10, False)
    >>> maybe_chunked(range(5), 2, np.prod, np.sum)
    ([0, 6], True)
    """
    was_chunked = None
    if length is None:
        result = on_not_chunked(values, *args, **kwargs)
        was_chunked = False
    else:
        try:
            iter(on_chunked)
        except TypeError:
            result = []
            for v in chunklist(values, length):
                result.append(on_chunked(v, *args, **kwargs))
            was_chunked = True
        else:
            result = []
            for f in on_chunked:
                sub_result = []
                for v in chunklist(values, length):
                    sub_result += f(v, *args, **kwargs)
                result.append(sub_result)
            was_chunked = True

    return result, was_chunked


class TFModelConfigEncoder(json.JSONEncoder):

    FUNCTION_TYPE = 'TFMODELLIB_TFMODEL_JSONENCODER__FUNCTION_TYPE'
    CLASS_TYPE    = 'TFMODELLIB_TFMODEL_JSONENCODER__CLASS_TYPE'

    def default(self, obj):
        if callable(obj):
            if inspect.isclass(obj):
                return TFModelConfigEncoder.CLASS_TYPE, \
                        obj.__name__, \
                        obj.__module__
            else:
                return TFModelConfigEncoder.FUNCTION_TYPE, \
                        obj.__code__.co_name, \
                        obj.__code__.co_filename
        return json.JSONEncoder.default(self, obj)



class TFModelConfig(dict):

    def __init__(self, *args, **kwargs):

        self.defaults = {
            'summaries_root': None,
            'checkpoints_root': None,
            'summaries_interval': 1,
            'saver_interval': None,
            'saver_max_to_keep': 50,
            'scope_name': 'tfmodel',
            'log_level': logging.DEBUG,
            'log_file': None
        }

        self.update(self.defaults)
        self.init()

        return super(TFModelConfig, self).__init__(*args, **kwargs)

    def init(self):
        pass

    def save(self, fname):
        try:
            with open(fname, 'w') as fp:
                json.dump(self, fp, cls=TFModelConfigEncoder)
        except Exception as e:
            print(e)

    def load(self, fname):
        try:
            with open(fname, 'r') as fp:
                conf = json.load(fp)
            for key,val in conf.items():
                if isinstance(val, list) and len(val) > 0:
                    if val[0] == TFModelConfigEncoder.FUNCTION_TYPE:
                        if sys.version_info >= (3,4):
                            spec = importlib.util.spec_from_file_location(val[1], val[2])
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            conf[key] = module.__dict__[val[1]]
                        else:
                            raise NotImplementedError('Restoring function handles from checkpoints only implemented for Python versions 3.4 and newer.')
                    elif val[0] == TFModelConfigEncoder.CLASS_TYPE:
                        conf[key] = importlib.import_module(val[2]).__dict__[val[1]]
            self.update(**conf)
        except Exception as e:
            print(e)



class TFModel(object):

    def __init__(self, config):

        if isinstance(config, TFModelConfig):
            self._init_from_config(config)
        elif isinstance(config, str):
            self._init_from_checkpoint(config)
        else:
            raise ValueError('Expecting a TFModelConfig instance, or path to a config.json file in a checkpoint directory.')

    def _init_from_config(self, config):

        self.config = config

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.build_graph()

        # add common elements to the graph
        with self.graph.as_default():
            with tf.variable_scope(self.config['scope_name']):
                self.global_step = tf.Variable(
                        initial_value=0, trainable=False,
                        dtype=tf.int32, name='global_step')
                self.increment_global_step_op = tf.assign(
                        self.global_step, self.global_step+1)
        
        self.sess = tf.Session(graph=self.graph)
        
        self.init_logger()
        self.init_summaries()
        self.init_saver()
        self.init_variables()

    def _init_from_checkpoint(self, config_json_fname):
        conf = TFModelConfig()
        try:
            conf.load(config_json_fname)
            self._init_from_config(conf)
        except Exception as e:
            self.config = {
                'scope_name': '',
                'log_level': logging.ERROR}
            self.init_logger()
            self.logger.log(logging.ERROR, '\n\nFailed to initialize from config file ({:s}).\n\n'.format(config_json_fname))
            raise e

    def build_graph(self):
        raise NotImplementedError('build_graph must be implemented by child class')

    def run_update_and_loss(self, batch_inputs, batch_targets, *args, **kwargs):
        raise NotImplementedError('run_update_and_loss must be implemented by child class')

    def run_loss(self, batch_inputs, batch_targets, *args, **kwargs):
        raise NotImplementedError('run_loss must be implemented by child class')

    def run_output(self, inputs, *args, **kwargs):
        raise NotImplementedError('run_output must be implemented by child class')

    def init_logger(self):
        self.logger = logging.Logger(
                name=self.config['scope_name'], level=self.config['log_level'])
        self.logger_stderr_handler = logging.StreamHandler(stream=sys.stderr)
        self.logger_stderr_handler.terminator = '\r'
        self.logger.addHandler(self.logger_stderr_handler)
        if self.config['log_file'] is not None:
            self.logger_file_handler = logging.FileHandler(
                    self.config['log_file'], mode='w')
            self.logger.addHandler(self.logger_file_handler)

    def init_summaries(self):
        if self.config['summaries_root'] is not None:
            summaries_root = os.path.realpath(self.config['summaries_root'])
            if not os.path.exists(summaries_root):
                os.makedirs(summaries_root)
                self.summaries_ind = 0
            else:
                self.summaries_ind = np.max([0] + [int(n) for n in os.listdir(summaries_root)]) + 1
            self.summary_fwriter = tf.summary.FileWriter(
                    logdir=os.path.join(summaries_root, str(self.summaries_ind)),
                    graph=self.sess.graph)
            self.train_loss_summary_value = tf.Summary.Value(
                    tag='({:s}) batch training loss'.format(self.config['scope_name']), simple_value=0.0)
            self.valid_loss_summary_value = tf.Summary.Value(
                    tag='({:s}) batch validation loss'.format(self.config['scope_name']), simple_value=0.0)
        else:
            self.summary_fwriter = None

    def write_summary(self):
        
        global_step = self.get_global_step()
        log_message = '{:9d}'.format(global_step)
        log_message += '\ttrain loss: {:4.05f}'.format(self.last_training_loss)
        log_message += '\tvalidate: {:4.05f}  '.format(self.last_validation_loss)
        self.logger.log(logging.INFO, log_message)
        sys.stderr.flush()

        if self.summary_fwriter is not None:
        
            self.train_loss_summary_value.simple_value = self.last_training_loss
            self.summary_fwriter.add_summary(
                    tf.Summary(value=[self.train_loss_summary_value]),
                    global_step=global_step)

            self.valid_loss_summary_value.simple_value = self.last_validation_loss
            self.summary_fwriter.add_summary(
                    tf.Summary(value=[self.valid_loss_summary_value]),
                    global_step=global_step)

    def get_saver_variables(self):
        """ Override this to only save specific variables in checkpoints. """
        return self.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    def init_saver(self):
        with tf.variable_scope(self.config['scope_name']):
            self.saver = tf.train.Saver(
                    var_list=self.get_saver_variables(),
                    max_to_keep=self.config['saver_max_to_keep'])
        if self.config['checkpoints_root'] is not None:
            checkpoints_root = os.path.relpath(self.config['checkpoints_root'])
            self.checkpoint_dir = os.path.join(checkpoints_root, str(self.summaries_ind))
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
        else:
            self.checkpoint_dir = None

    def init_variables(self):
        with self.graph.as_default():
            with tf.variable_scope(self.config['scope_name']):
                initializer = tf.variables_initializer(
                        var_list=self.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES),
                        name='init')
                self.sess.run(initializer)

    def get_global_step(self):
        return self.sess.run(self.global_step)

    def train(self, train_inputs, train_targets, validation_inputs=None, validation_targets=None, batch_size=None, *args, **kwargs):
        n_samples = train_inputs.shape[0]
        inds = np.random.permutation(range(n_samples))
        if validation_inputs is not None:
            # Training is done in mini-batches. We will draw an equal number of
            # validation samples as we have training samples, and will then
            # split both sets into mini-batches.
            validation_inds = np.random.randint(0, validation_inputs.shape[0], n_samples)
            # Stack the training and validation indices into a 2xN_SAMPLES
            # matrix, which will later be divided into mini-batches of size
            # 2xBATCH_SIZE.
            inds = np.vstack((inds, validation_inds)).T
        
        result, was_chunked = maybe_chunked(
                inds, batch_size,
                self.train_step, self.train_step,
                train_inputs, train_targets,
                validation_inputs, validation_targets,
                *args, **kwargs)

        if was_chunked:
            if len(result) > 1:
                result = np.mean(result, axis=0)
            else:
                result = result[0]

        self.last_training_loss, self.last_validation_loss = result

        global_step = self.sess.run(self.increment_global_step_op)

        if self.config['saver_interval'] is not None \
                and global_step % self.config['saver_interval'] == 0:
            self.save()

        if self.config['summaries_interval'] is not None \
                and global_step % self.config['summaries_interval'] == 0:
            self.write_summary()

        return self.last_training_loss, self.last_validation_loss

    def train_step(self, inds, train_inputs, train_targets, validation_inputs, validation_targets, *args, **kwargs):
        """
        The actual training. INDS specifies the indices of the samples to use
        as a mini-batch.
        """

        if validation_inputs is not None:
            training_inds, validation_inds = inds.T
        else:
            training_inds = inds
 
        with self.graph.as_default():
            with tf.control_dependencies(self.graph.get_collection(tf.GraphKeys.UPDATE_OPS)):
                training_loss = self.run_update_and_loss(
                        batch_inputs=train_inputs[training_inds],
                        batch_targets=train_targets[training_inds],
                        *args, **kwargs)

        validation_loss = np.nan
        if validation_inputs is not None:

            validation_loss = self.run_loss(
                    batch_inputs=validation_inputs[validation_inds],
                    batch_targets=validation_targets[validation_inds],
                    *args, **kwargs)

        return training_loss, validation_loss

    def infer(self, inputs, batch_size=None, *args, **kwargs):
        result, was_chunked = maybe_chunked(
                list(range(inputs.shape[0])), batch_size,
                self.infer_step, self.infer_step,
                inputs, *args, **kwargs)
        if was_chunked:
            if len(result) > 1:
                result = np.vstack(result)
            else:
                result = result[0]
        return result

    def infer_step(self, inds, inputs, *args, **kwargs):
        return self.run_output(inputs[inds], *args, **kwargs)

    def save(self):
        if self.checkpoint_dir is not None:
            self.saver.save(
                    sess=self.sess,
                    save_path=os.path.join(self.checkpoint_dir, ''),
                    global_step=self.get_global_step())

            # dump the config dict to a json file
            config_json_fname = os.path.join(self.checkpoint_dir, 'config.json')
            if not os.path.isfile(config_json_fname):
                self.config.save(config_json_fname)
        else:
            self.logger.log(logging.WARNING,
                    '\'checkpoints_root\' was set to None in the model ' \
                    'configuration - saving not possible.\n')

    def restore(self, save_path):
        self.logger.log(logging.DEBUG,
                'restoring from {:s}\n'.format(save_path))
        self.saver.restore(sess=self.sess, save_path=save_path)
