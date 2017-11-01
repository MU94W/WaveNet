import tensorflow as tf
from tensorflow.python.ops import array_ops
from TFCommon.Model import Model
import audio
from hyperparameter import HyperParams


def causal_conv1d(x, hyper_params):
    left_pad_tensor = tf.zeros(shape=(tf.shape(x)[0], 1, 1, hyper_params.waveform_categories))
    padded_x = tf.concat([left_pad_tensor, x], axis=1)
    conv_out = tf.layers.conv2d(inputs=padded_x, filters=hyper_params.residual_channels,
                                kernel_size=(hyper_params.kernel_size, 1), strides=1,
                                padding='VALID', use_bias=hyper_params.dilated_causal_use_bias)
    return conv_out


def dilated_causal_conv1d(x, dilation_rate, hyper_params):
    left_pad_tensor = tf.zeros(shape=(tf.shape(x)[0], dilation_rate, 1, hyper_params.residual_channels))
    padded_x = tf.concat([left_pad_tensor, x], axis=1)
    conv_hid_0 = tf.layers.conv2d(inputs=padded_x, filters=2 * hyper_params.residual_channels,
                                  kernel_size=(hyper_params.kernel_size, 1), strides=1,
                                  padding='VALID', dilation_rate=(dilation_rate, 1),
                                  use_bias=hyper_params.dilated_causal_use_bias)
    conv_hid_0_l, conv_hid_0_r = array_ops.split(conv_hid_0, num_or_size_splits=2, axis=-1)
    conv_hid_1 = tf.nn.tanh(conv_hid_0_l) * tf.nn.sigmoid(conv_hid_0_r)
    with tf.variable_scope('residual_out'):
        conv_res_out = tf.identity(x) + tf.layers.dense(conv_hid_1,
                                                        units=hyper_params.residual_channels,
                                                        activation=None,
                                                        use_bias=hyper_params.residual_use_bias)
    with tf.variable_scope('skip_connection'):
        conv_skip_out = tf.layers.dense(conv_hid_1,
                                        units=hyper_params.skip_dims,
                                        activation=None,
                                        use_bias=hyper_params.skip_use_bias)
    return conv_res_out, conv_skip_out


def residual_block(x, block_layers, hyper_params):
    last_out = x
    skip_out_lst = []
    for layer_idx in range(block_layers):
        with tf.variable_scope('residual_layer_{}'.format(layer_idx)):
            conv_res_out, conv_skip_out = dilated_causal_conv1d(last_out, 1 << layer_idx, hyper_params)
            last_out = conv_res_out
            skip_out_lst.append(conv_skip_out)
    return last_out, tf.add_n(skip_out_lst)


def build_blocks(x, hyper_params):
    last_out = x
    skip_out_lst = []
    for block_idx, block_layers in enumerate(hyper_params.dilation_blocks):
        with tf.variable_scope('block_{}'.format(block_idx)):
            resi_out, skip_out = residual_block(last_out, block_layers, hyper_params)
            last_out = resi_out
            skip_out_lst.append(skip_out)
    return tf.add_n(skip_out_lst)


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
        self.hyper_params = HyperParams() if hyper_params is None else hyper_params

        with tf.variable_scope('shift_input'):
            waveform_one_hot = tf.one_hot(waveform, depth=self.hyper_params.waveform_categories,
                                          on_value=1., off_value=0., dtype=tf.float32)
            left_pad_go = tf.one_hot(indices=[[[self.hyper_params.waveform_center_cat]]],
                                     depth=self.hyper_params.waveform_categories,
                                     on_value=1., off_value=0., dtype=tf.float32)
            left_pad_go_batch = tf.tile(left_pad_go, [tf.shape(waveform)[0], 1, 1, 1]) #  for <go>
            input_waveform = tf.concat([left_pad_go_batch, waveform_one_hot[:, :-1, :, :]], axis=1)    # only drop for <go>
        with tf.variable_scope('init_causal'):
            feed_for_dilated_blocks = causal_conv1d(input_waveform, hyper_params)
        with tf.variable_scope('stacked_conv_blocks'):
            skip_out = tf.nn.relu(build_blocks(feed_for_dilated_blocks, self.hyper_params))
        with tf.variable_scope('pred_out'):
            with tf.variable_scope('hid_out_0'):
                hid_out_0 = tf.layers.dense(skip_out,
                                            units=self.hyper_params.waveform_categories,
                                            activation=tf.nn.relu)
            with tf.variable_scope('hid_out_1'):
                hid_out_1 = tf.layers.dense(hid_out_0,
                                            units=self.hyper_params.waveform_categories,
                                            activation=None)
        with tf.variable_scope('loss'):
            waveform_mask = tf.expand_dims(array_ops.sequence_mask(waveform_lens,
                                                                   tf.shape(waveform)[1],
                                                                   dtype=tf.float32),
                                           axis=-1)
            self.loss = tf.losses.sparse_softmax_cross_entropy(labels=waveform,
                                                               logits=hid_out_1,
                                                               weights=waveform_mask)
        self.global_step = tf.Variable(0, name='global_step')

        # summary [begin]
        softmax_score = tf.nn.softmax(logits=hid_out_1[:1, :, :, :], dim=-1)
        quantized_miu_wav = tf.argmax(softmax_score, axis=-1)
        miu_wav = audio.tf_rev_quantize(quantized_miu_wav, bits=self.hyper_params.waveform_bits)
        self.pred_wav = audio.tf_rev_miu_law(miu_wav, miu=float(self.hyper_params.waveform_categories - 1))
        summary = [tf.summary.scalar('train/loss', self.loss),
                   tf.summary.audio('train/audio', self.pred_wav, sample_rate=sample_rate)]
        self.summary_loss = summary[0]
        self.summary_audio = summary[1]
        self.summary = tf.summary.merge(summary)
        # summary [end]
