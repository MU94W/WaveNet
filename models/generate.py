import tensorflow as tf
from TFCommon.Layers import EmbeddingLayer
from TFCommon.Model import Model
from tensorflow.python.ops import array_ops

import audio
from hyperparameter import HyperParams


def sample_by_logits(logits, sample_type='prob'):
    if sample_type is 'argmax':
        return tf.cast(tf.argmax(tf.nn.softmax(logits), axis=-1), tf.int32)
    elif sample_type is 'prob':
        batch_size = tf.shape(logits)[0]
        classes = logits.shape[-1].value
        sample_class = tf.multinomial(tf.reshape(logits, (batch_size, classes)), 1)
        return tf.cast(tf.expand_dims(sample_class, axis=-1), tf.int32)
    else:
        raise NotImplementedError


def skip_infer_next(skip_tensor, hyper_params, sample_type='prob'):
    with tf.variable_scope('pred_out'):
        with tf.variable_scope('hid_out_0'):
            hid_out_0 = tf.layers.dense(skip_tensor, units=hyper_params.waveform_categories, activation=tf.nn.relu)
        with tf.variable_scope('hid_out_1'):
            hid_out_1 = tf.layers.dense(hid_out_0, units=hyper_params.waveform_categories, activation=None)
    return sample_by_logits(hid_out_1, sample_type)


def conv_op(x_last, x_cur, hyper_params):
    kernel_shape = (hyper_params.kernel_size, 1) + (hyper_params.residual_channels, 2 * hyper_params.residual_channels)
    with tf.variable_scope('conv2d'):
        kernel = tf.get_variable(name='kernel', shape=kernel_shape, dtype=tf.float32)
    input_x = tf.concat([x_last, x_cur], axis=1)
    only_conv_out = tf.nn.conv2d(input=input_x, filter=kernel, strides=[1]*4, padding='VALID')
    if hyper_params.dilated_causal_use_bias:
        with tf.variable_scope('conv2d'):
            bias_shape = (2 * hyper_params.residual_channels,)
            bias = tf.get_variable(name='bias', shape=bias_shape, dtype=tf.float32)
        conv_hid_0 = tf.nn.bias_add(only_conv_out, bias)
    else:
        conv_hid_0 = only_conv_out
    conv_hid_0_l, conv_hid_0_r = array_ops.split(conv_hid_0, num_or_size_splits=2, axis=-1)
    conv_hid_1 = tf.nn.tanh(conv_hid_0_l) * tf.nn.sigmoid(conv_hid_0_r)
    with tf.variable_scope('residual_out'):
        conv_res_out = tf.identity(x_cur) + tf.layers.dense(conv_hid_1,
                                                            units=hyper_params.residual_channels,
                                                            activation=None, use_bias=hyper_params.residual_use_bias)
    with tf.variable_scope('skip_connection'):
        conv_skip_out = tf.layers.dense(conv_hid_1,
                                        units=hyper_params.skip_dims,
                                        activation=None, use_bias=hyper_params.skip_use_bias)
    return conv_res_out, conv_skip_out


class ResidualBlock(object):
    def __init__(self, hyper_params, scope=None):
        self.hyper_params = hyper_params
        self.scope = type(self).__name__ if scope is None else scope

    def __call__(self, x_cur, receptive_block, reuse=False):
        with tf.variable_scope(self.scope, reuse=reuse):
            skip_out = 0
            for layer_idx, rec_lst in enumerate(receptive_block):
                with tf.variable_scope('residual_layer_{}'.format(layer_idx)):
                    x_last = rec_lst.pop(0)
                    rec_lst.append(x_cur)
                    conv_res_out, conv_skip_out = conv_op(x_last, x_cur, self.hyper_params)
                    x_cur = conv_res_out
                    skip_out += conv_skip_out
        return x_cur, skip_out, receptive_block


class BuildBlocks(object):
    def __init__(self, hyper_params, scope=None):
        self.hyper_params = hyper_params
        self.residual_blocks = [ResidualBlock(hyper_params, 'block_{}'.format(block_idx))
                                for block_idx, block_layers in enumerate(hyper_params.dilation_blocks)]
        self.scope = type(self).__name__ if scope is None else scope

    def __call__(self, x_cur, receptive_blocks, reuse=None):
        with tf.variable_scope(self.scope, reuse=reuse):
            skip_out_sum = 0
            for res_block, rec_block in zip(self.residual_blocks, receptive_blocks):
                res_out, skip_out, rec_block = res_block(x_cur, rec_block)
                x_cur = res_out
                skip_out_sum += skip_out
        return skip_out_sum, receptive_blocks


def init_receptive_blocks(batch_size, hyper_params):
    conv_shape = (batch_size, 1, 1, hyper_params.residual_channels)
    rec_blks = []
    for blk_idx, blk_layers in enumerate(hyper_params.dilation_blocks):
        rec_blks.append([[tf.zeros(shape=conv_shape, dtype=tf.float32)] * (1 << layer_idx)
                         for layer_idx in range(blk_layers)])
    return rec_blks


class WaveNet(Model):
    def __init__(self, waveform_seed, max_infer_samples, hyper_params=None,
                 global_control=None, local_control=None, name='WaveNet'):
        """
        Build the computational graph.
        :param waveform_seed: 16-bit or 8-bit waveform, shape:=(batch_size, 1, 1), dtype:=tf.int32
        :param max_infer_samples: scalar tensor or python int object, dtype={tf.int32, int}
        :param global_control: shape:=(batch_size, feature_dim), dtype:=tf.float32
        :param local_control: shape:=(batch_size, time_steps, feature_dim), dtype:=tf.float32
        :return:
        """
        super(WaveNet, self).__init__(name)
        self.hyper_params = HyperParams() if hyper_params is None else hyper_params

        batch_size = tf.shape(waveform_seed)[0]
        residual_blocks = BuildBlocks(self.hyper_params, 'stacked_conv_blocks')

        # while loop [begin]
        sample_tensor_arr = tf.TensorArray(size=max_infer_samples, dtype=tf.int32)
        receptive_blocks = init_receptive_blocks(batch_size, self.hyper_params)
        time = tf.constant(0, dtype=tf.int32)

        # init_causal_weights
        with tf.variable_scope('init_causal'):
            with tf.variable_scope('conv2d'):
                init_causal_kernel = tf.get_variable(name='kernel', dtype=tf.float32,
                                                     shape=(2, 1, self.hyper_params.waveform_categories, self.hyper_params.residual_channels))
                init_causal_bias = tf.get_variable(name='bias', dtype=tf.float32,
                                                   shape=(self.hyper_params.residual_channels,))
                k_l, k_r = array_ops.split(init_causal_kernel, num_or_size_splits=2, axis=0)
                k_l, k_r = tf.squeeze(k_l), tf.squeeze(k_r)

        def body(this_time, last_sample, cur_sample, out_sample_tensor_arr, old_receptive_blocks):
            x_cur = tf.nn.embedding_lookup(k_l, last_sample) + tf.nn.embedding_lookup(k_r, cur_sample) + init_causal_bias
            skip_out, new_receptive_blocks = residual_blocks(x_cur, old_receptive_blocks)
            pred_sample = skip_infer_next(tf.nn.relu(skip_out), self.hyper_params)
            out_sample_tensor_arr = out_sample_tensor_arr.write(this_time, pred_sample)
            return tf.add(this_time, 1), cur_sample, pred_sample, out_sample_tensor_arr, new_receptive_blocks

        # run loop
        _, _, _, pred_sample_tensor_arr, _ = tf.while_loop(lambda x, *_: tf.less(x, max_infer_samples), body,
                                                           [time, waveform_seed, waveform_seed, sample_tensor_arr, receptive_blocks])
        # while loop [end]
        pred_quantized_miu_wav = tf.transpose(tf.reshape(pred_sample_tensor_arr.stack(),
                                                         shape=(max_infer_samples, batch_size)),
                                              perm=(1, 0))

        self.global_step = tf.Variable(0, name='global_step')

        pred_miu_wav = audio.tf_rev_quantize(pred_quantized_miu_wav, bits=self.hyper_params.waveform_bits)
        self.pred_wav = audio.tf_rev_miu_law(pred_miu_wav, miu=float(hyper_params.waveform_categories - 1))

