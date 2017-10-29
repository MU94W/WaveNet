import tensorflow as tf
from TFCommon.Model import Model
from TFCommon.Layers import EmbeddingLayer
from .. import hyperparameter as hp
from .. import audio


def dilated_causal_conv1d(x, dilation_rate):
    left_pad_tensor = tf.zeros(shape=(tf.shape(x)[0], dilation_rate[0], 1, hp.dilation_channels))
    padded_x = tf.concat([left_pad_tensor, x], axis=1)
    conv_hid_0 = tf.layers.conv2d(inputs=padded_x, filters=2 * hp.dilation_channels,
                                  kernel_size=hp.kernel_size, strides=1,
                                  padding='VALID', dilation_rate=dilation_rate)
    conv_hid_0_l = conv_hid_0[:, :, :, :hp.dilation_channels]
    conv_hid_0_r = conv_hid_0[:, :, :, hp.dilation_channels:]
    conv_hid_1 = tf.nn.tanh(conv_hid_0_l) * tf.nn.sigmoid(conv_hid_0_r)
    with tf.variable_scope('residual_out'):
        conv_res_out = tf.identity(x) + tf.layers.dense(conv_hid_1,
                                                        units=hp.dilation_channels,
                                                        activation=tf.nn.relu)
    with tf.variable_scope('skip_connection'):
        conv_skip_out = tf.layers.dense(conv_hid_1,
                                        units=hp.skip_dims,
                                        activation=tf.nn.relu)
    return conv_res_out, conv_skip_out


def residual_block(x, dilation_rate_lst):
    last_out = x
    skip_out_lst = []
    for layer_idx, dilation_rate in enumerate(dilation_rate_lst):
        with tf.variable_scope('residual_layer_{}'.format(layer_idx)):
            conv_res_out, conv_skip_out = dilated_causal_conv1d(last_out, dilation_rate)
            last_out = conv_res_out
            skip_out_lst.append(conv_skip_out)
    skip_out = tf.concat(skip_out_lst, axis=-1)
    return last_out, skip_out


def build_blocks(x, dilation_rate_lst_blocks):
    last_out = x
    skip_out_lst = []
    for block_idx, dilation_rate_lst in enumerate(dilation_rate_lst_blocks):
        with tf.variable_scope('block_{}'.format(block_idx)):
            resi_out, skip_out = residual_block(last_out, dilation_rate_lst)
            last_out = resi_out
            skip_out_lst.append(skip_out)
    skip_out = tf.concat(skip_out_lst, axis=-1)
    return skip_out


class WaveNet(Model):
    def __init__(self, waveform, global_control=None, local_control=None, name='WaveNet'):
        """
        Build the computational graph.
        :param waveform: 16-bit or 8-bit waveform, shape:=(batch_size, time_steps, 1), dtype:=tf.int32
        :param global_control: shape:=(batch_size, feature_dim), dtype:=tf.float32
        :param local_control: shape:=(batch_size, time_steps, feature_dim), dtype:=tf.float32
        :return:
        """
        super(WaveNet, self).__init__(name)
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
        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=waveform, logits=hid_out_1)
        self.global_step = tf.Variable(0, name='global_step')

        self.summary = []
        self.summary.append(tf.summary.scalar('train/loss', self.loss))
        softmax_score = tf.nn.softmax(logits=hid_out_1[:1], dim=-1)
        quantized_miu_wav = tf.argmax(softmax_score, axis=-1)
        miu_wav = audio.tf_rev_quantize(quantized_miu_wav, bits=hp.waveform_bits)
        self.pred_wav = audio.tf_rev_miu_law(miu_wav, miu=float(hp.waveform_categories - 1))
        self.summary.append(tf.summary.audio('train/audio', self.pred_wav, sample_rate=hp.sample_rate))
