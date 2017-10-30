import tensorflow as tf
from tensorflow.python.ops import array_ops
from TFCommon.Model import Model
from TFCommon.Layers import EmbeddingLayer
from WaveNet import audio
from WaveNet.hyperparameter import HyperParams


def sample_by_logits(logits, sample_type='prob'):
    if sample_type is 'argmax':
        return tf.cast(tf.argmax(tf.nn.softmax(logits), axis=-1), tf.int32)
    elif sample_type is 'prob':
        sample_class = tf.multinomial(tf.squeeze(logits), 1)
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
    kernel_shape = (hyper_params.kernel_size, 1) + (hyper_params.dilation_channels, 2 * hyper_params.dilation_channels)
    with tf.variable_scope('conv2d'):
        kernel = tf.get_variable(name='kernel', shape=kernel_shape, dtype=tf.float32)
    input_x = tf.concat([x_last, x_cur], axis=1)
    only_conv_out = tf.nn.conv2d(input=input_x, filter=kernel, strides=[1]*4, padding='VALID')
    if hyper_params.dilated_causal_use_bias:
        with tf.variable_scope('conv2d'):
            bias_shape = (2 * hyper_params.dilation_channels,)
            bias = tf.get_variable(name='bias', shape=bias_shape, dtype=tf.float32)
        conv_hid_0 = tf.nn.bias_add(only_conv_out, bias)
    else:
        conv_hid_0 = only_conv_out
    conv_hid_0_l, conv_hid_0_r = array_ops.split(conv_hid_0, num_or_size_splits=2, axis=-1)
    conv_hid_1 = tf.nn.tanh(conv_hid_0_l) * tf.nn.sigmoid(conv_hid_0_r)
    with tf.variable_scope('residual_out'):
        conv_res_out = tf.identity(x_cur) + tf.layers.dense(conv_hid_1,
                                                            units=hyper_params.dilation_channels,
                                                            activation=None, use_bias=hyper_params.residual_use_bias)
    with tf.variable_scope('skip_connection'):
        conv_skip_out = tf.layers.dense(conv_hid_1,
                                        units=hyper_params.skip_dims,
                                        activation=None, use_bias=hyper_params.skip_use_bias)
    return conv_res_out, conv_skip_out


class ResidualBlock(object):
    def __init__(self, batch_size, block_layers, hyper_params, scope=None):
        conv_shape = (batch_size, 1, 1, hyper_params.dilation_channels)
        self.queue_lst = [[tf.zeros(shape=conv_shape)] * (1 << layer_idx) for layer_idx in range(block_layers)]
        self.scope = type(self).__name__ if scope is None else scope

    def __call__(self, x_cur, reuse=False):
        with tf.variable_scope(self.scope, reuse=reuse):
            skip_out = 0
            for layer_idx, queue in enumerate(self.queue_lst):
                with tf.variable_scope('residual_layer_{}'.format(layer_idx)):
                    x_last = queue.pop(0)
                    queue.append(x_cur)
                    conv_res_out, conv_skip_out = conv_op(x_last, x_cur)
                    x_cur = conv_res_out
                    skip_out += conv_skip_out
        return x_cur, skip_out


class BuildBlocks(object):
    def __init__(self, batch_size, hyper_params, scope=None):
        self.hyper_params = hyper_params
        self.residual_blocks = [ResidualBlock(batch_size, block_layers, 'block_{}'.format(block_idx))
                                for block_idx, block_layers in enumerate(hyper_params.conv_layers)]
        self.scope = type(self).__name__ if scope is None else scope

    def __call__(self, x_cur, reuse=False):
        with tf.variable_scope(self.scope, reuse=reuse):
            skip_out_sum = 0
            for res_block in self.residual_blocks:
                res_out, skip_out = res_block(x_cur)
                x_cur = res_out
                skip_out_sum += skip_out
        return tf.nn.relu(skip_out_sum)


class WaveNet(Model):
    def __init__(self, waveform_seed, max_infer_samples, hyper_params=None, global_control=None, local_control=None,
                 sample_rate=16000, name='WaveNet'):
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
        embedding_layer = EmbeddingLayer(classes=self.hyper_params.waveform_categories,
                                         size=self.hyper_params.dilation_channels)
        residual_blocks = BuildBlocks(batch_size, self.hyper_params, 'stacked_conv_blocks')

        # while loop [begin]
        sample_tensor_arr = tf.TensorArray(size=max_infer_samples, dtype=tf.int32)
        time = tf.constant(0, dtype=tf.int32)

        def body(this_time, waveform_sample, out_sample_tensor_arr):
            x_cur = embedding_layer(waveform_sample, scope='waveform_embedding_lookup')
            skip_out = residual_blocks(x_cur, self.hyper_params)
            pred_sample = skip_infer_next(skip_out)
            out_sample_tensor_arr = out_sample_tensor_arr.write(this_time, pred_sample)
            return tf.add(this_time, 1), pred_sample, out_sample_tensor_arr

        # run loop
        _, _, pred_sample_tensor_arr = tf.while_loop(lambda x, *_: tf.less(x, max_infer_samples),
                                                     body, [time, waveform_seed, sample_tensor_arr])
        # while loop [end]
        pred_quantized_miu_wav = tf.transpose(tf.reshape(pred_sample_tensor_arr.stack(),
                                                         shape=(max_infer_samples, batch_size)),
                                              perm=(1, 0))

        self.summary = []
        pred_miu_wav = audio.tf_rev_quantize(pred_quantized_miu_wav, bits=self.hyper_params.waveform_bits)
        self.pred_wav = audio.tf_rev_miu_law(pred_miu_wav, miu=float(hyper_params.waveform_categories - 1))

