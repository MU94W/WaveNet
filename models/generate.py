import tensorflow as tf
from TFCommon.Model import Model
from .. import hyperparameter as hp
from .. import audio


def sample_by_logits(logits, sample_type='prob'):
    if sample_type is 'argmax':
        return tf.cast(tf.argmax(softmax_score, axis=-1), tf.int32)
    elif sample_type is 'prob':
        sample_class = tf.multinomial(tf.squeeze(logits), 1)
        return tf.cast(tf.expand_dims(sample_class, axis=-1), tf.int32)
    else:
        raise NotImplementedError


def skip_infer_next(skip_tensor):
    with tf.variable_scope('hid_out_0'):
        hid_out_0 = tf.layers.dense(skip_tensor, units=hp.waveform_categories, activation=tf.nn.relu)
    with tf.variable_scope('hid_out_1'):
        hid_out_1 = tf.layers.dense(hid_out_0, units=hp.waveform_categories, activation=None)
    return sample_by_logits(hid_out_1)


def conv_op(x_last, x_cur):
    bias_shape = (2 * hp.dilation_channels,)
    kernel_shape = hp.kernel_size + (hp.dilation_channels, 2 * hp.dilation_channels)
    with tf.variable_scope('conv2d'):
        kernel = tf.get_variable(name='kernel', shape=kernel_shape, dtype=tf.float32)
        if hp.dilated_causal_use_bias:
            bias = tf.get_variable(name='bias', shape=bias_shape, dtype=tf.float32)
    input_x = tf.concat([x_last, x_cur], axis=1)
    if hp.dilated_causal_use_bias:
        conv_hid_0 = tf.nn.bias_add(tf.nn.conv2d(input=input_x, filter=kernel, strides=[1]*4, padding='VALID'),
                                    bias)
    else:
        conv_hid_0 = tf.nn.conv2d(input=input_x, filter=kernel, strides=[1]*4, padding='VALID')
    conv_hid_0_l = conv_hid_0[:, :, :, :hp.dilation_channels]
    conv_hid_0_r = conv_hid_0[:, :, :, hp.dilation_channels:]
    conv_hid_1 = tf.nn.tanh(conv_hid_0_l) * tf.nn.sigmoid(conv_hid_0_r)
    with tf.variable_scope('residual_out'):
        conv_res_out = tf.identity(x_cur) + tf.layers.dense(conv_hid_1,
                                                            units=hp.dilation_channels,
                                                            activation=None, use_bias=hp.residual_use_bias)
    with tf.variable_scope('skip_connection'):
        conv_skip_out = tf.layers.dense(conv_hid_1,
                                        units=hp.skip_dims,
                                        activation=None, use_bias=hp.skip_use_bias)
    return conv_res_out, conv_skip_out


class ResidualBlock(object):
    def __init__(self, conv_shape, dilation_rate_lst):
        self.queue_lst = [[tf.zeros(shape=conv_shape)] * dilation_rate[0] for dilation_rate in dilation_rate_lst]

    def __call__(self, x_cur):
        skip_out = 0
        for layer_idx, queue in enumerate(self.queue_lst):
            with tf.variable_scope('residual_layer_{}'.format(layer_idx)):
                x_last = queue.pop(0)
                queue.append(x_cur)
                conv_res_out, conv_skip_out = conv_op(x_last, x_cur)
                x_cur = conv_res_out
                skip_out += conv_skip_out
        return x_cur, skip_out


class WaveNet(Model):
    def __init__(self, waveform_seed, max_infer_samples, global_control=None, local_control=None, name='WaveNet'):
        """
        Build the computational graph.
        :param waveform_seed: 16-bit or 8-bit waveform, shape:=(batch_size, 1, 1), dtype:=tf.int32
        :param global_control: shape:=(batch_size, feature_dim), dtype:=tf.float32
        :param local_control: shape:=(batch_size, time_steps, feature_dim), dtype:=tf.float32
        :return:
        """
        super(WaveNet, self).__init__(name)
        batch_size = tf.shape(waveform_seed)[0]
        conv_shape = (batch_size, 1, 1, hp.dilation_channels)
        residual_blocks = [ResidualBlock(conv_shape, d_r_l) for d_r_l in hp.dilation_rate_lst_blocks]

        # while loop [begin]
        sample_tensor_arr = tf.TensorArray(size=max_infer_samples, dtype=tf.int32)
        time = tf.constant(0, dtype=tf.int32)

        def cond(this_time, *args):
            return tf.less(this_time, max_infer_samples)

        def body(this_time, waveform_sample, out_sample_tensor_arr):
            with tf.variable_scope('waveform_embedding_lookup'):
                embed_matrix = tf.get_variable(name='embedding', dtype=tf.float32,
                                               shape=(hp.waveform_categories, hp.dilation_channels))
            x_cur = tf.nn.embedding_lookup(embed_matrix, waveform_sample)
            skip_out_sum = 0
            with tf.variable_scope('stacked_conv_blocks'):
                for block_idx, res_block in enumerate(residual_blocks):
                    with tf.variable_scope('block_{}'.format(block_idx)):
                        res_out, skip_out = res_block(x_cur)
                        x_cur = res_out
                        skip_out_sum += skip_out
            pred_sample = skip_infer_next(skip_out_sum)
            out_sample_tensor_arr = out_sample_tensor_arr.write(this_time, pred_sample)
            return tf.add(this_time, 1), pred_sample, out_sample_tensor_arr

        # run loop
        _, _, pred_sample_tensor_arr = tf.while_loop(cond, body, [time, waveform_seed, sample_tensor_arr])
        # while loop [end]
        pred_quantized_miu_wav = tf.transpose(tf.reshape(pred_sample_tensor_arr.stack(),
                                                         shape=(max_infer_samples, batch_size)),
                                              perm=(1, 0))

        self.summary = []
        pred_miu_wav = audio.tf_rev_quantize(pred_quantized_miu_wav, bits=hp.waveform_bits)
        self.pred_wav = audio.tf_rev_miu_law(pred_miu_wav, miu=float(hp.waveform_categories - 1))
        self.summary.append(tf.summary.audio('train/audio', self.pred_wav, sample_rate=hp.sample_rate))

