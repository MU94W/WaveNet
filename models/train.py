import tensorflow as tf
from TFCommon.Model import Model
from TFCommon.Layers import EmbeddingLayer
from tensorflow.python.ops import array_ops
from WaveNet import hyperparameter as hp
from WaveNet import audio
from hyperparameter import HyperParams


def dilated_causal_conv1d(x, dilation_rate):
    left_pad_tensor = tf.zeros(shape=(tf.shape(x)[0], dilation_rate[0], 1, hp.dilation_channels))
    padded_x = tf.concat([left_pad_tensor, x], axis=1)
    conv_hid_0 = tf.layers.conv2d(inputs=padded_x, filters=2 * hp.dilation_channels,
                                  kernel_size=hp.kernel_size, strides=1,
                                  padding='VALID', dilation_rate=dilation_rate, use_bias=hp.dilated_causal_use_bias)
    conv_hid_0_l = conv_hid_0[:, :, :, :hp.dilation_channels]
    conv_hid_0_r = conv_hid_0[:, :, :, hp.dilation_channels:]
    conv_hid_1 = tf.nn.tanh(conv_hid_0_l) * tf.nn.sigmoid(conv_hid_0_r)
    with tf.variable_scope('residual_out'):
        conv_res_out = tf.identity(x) + tf.layers.dense(conv_hid_1,
                                                        units=hp.dilation_channels,
                                                        activation=None, use_bias=hp.residual_use_bias)
    with tf.variable_scope('skip_connection'):
        conv_skip_out = tf.layers.dense(conv_hid_1,
                                        units=hp.skip_dims,
                                        activation=None, use_bias=hp.skip_use_bias)
    return conv_res_out, conv_skip_out


def residual_block(x, dilation_rate_lst):
    last_out = x
    skip_out = 0
    for layer_idx, dilation_rate in enumerate(dilation_rate_lst):
        with tf.variable_scope('residual_layer_{}'.format(layer_idx)):
            conv_res_out, conv_skip_out = dilated_causal_conv1d(last_out, dilation_rate)
            last_out = conv_res_out
            skip_out += conv_skip_out
    return last_out, skip_out


def build_blocks(x, dilation_rate_lst_blocks):
    last_out = x
    skip_out_sum = 0
    for block_idx, dilation_rate_lst in enumerate(dilation_rate_lst_blocks):
        with tf.variable_scope('block_{}'.format(block_idx)):
            resi_out, skip_out = residual_block(last_out, dilation_rate_lst)
            last_out = resi_out
            skip_out_sum += skip_out
    return skip_out_sum


class WaveNet(Model):
    def __init__(self, waveform, waveform_lens, hyper_params=None, global_condition=None, local_condition=None,
                 sample_rate=16000, name='WaveNet'):
        """
        Build the computational graph.
        :param waveform: 16-bit or 8-bit waveform, shape:=(batch_size, time_steps, 1), dtype:=tf.int32
        :param waveform_lens: shape:=(batch_size,), dtype:tf.int32
        :param hyper_params: instance of HyperParams
        :param global_condition: shape:=(batch_size, feature_dim), dtype:=tf.float32
        :param local_condition: shape:=(batch_size, time_steps, feature_dim), dtype:=tf.float32
        :return:
        """
        super(WaveNet, self).__init__(name)
        if hyper_params is None:
            self.hyper_params = HyperParams()
        else:
            self.hyper_params = hyper_params


        left_pad_go = hp.waveform_center_cat + tf.zeros(shape=(tf.shape(waveform)[0], 1, 1), dtype=tf.int32)
        input_waveform = tf.concat([left_pad_go, waveform[:, :-1, :]], axis=1)
        waveform_embed = EmbeddingLayer(classes=hp.waveform_categories,
                                        size=hp.dilation_channels)(input_waveform,
                                                                   scope='waveform_embedding_lookup')
        with tf.variable_scope('stacked_conv_blocks'):
            skip_out = build_blocks(waveform_embed, hp.dilation_rate_lst_blocks)
        with tf.variable_scope('hid_out_0'):
            hid_out_0 = tf.layers.dense(skip_out, units=hp.waveform_categories, activation=tf.nn.relu)
        with tf.variable_scope('hid_out_1'):
            hid_out_1 = tf.layers.dense(hid_out_0, units=hp.waveform_categories, activation=None)
        waveform_mask = tf.expand_dims(array_ops.sequence_mask(waveform_lens,
                                                               tf.shape(waveform)[1],
                                                               dtype=tf.float32), axis=-1)
        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=waveform,
                                                           logits=hid_out_1,
                                                           weights=waveform_mask)
        self.global_step = tf.Variable(0, name='global_step')

        softmax_score = tf.nn.softmax(logits=hid_out_1[:1], dim=-1)
        quantized_miu_wav = tf.argmax(softmax_score, axis=-1)
        miu_wav = audio.tf_rev_quantize(quantized_miu_wav, bits=hp.waveform_bits)
        self.pred_wav = audio.tf_rev_miu_law(miu_wav, miu=float(hp.waveform_categories - 1))
        summary = [tf.summary.scalar('train/loss', self.loss),
                   tf.summary.audio('train/audio', self.pred_wav, sample_rate=sample_rate)]
        self.summary_loss = summary[0]
        self.summary_audio = summary[1]
        self.summary = tf.summary.merge(summary)

