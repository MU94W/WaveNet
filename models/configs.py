import tensorflow as tf
from .layers import dilated_causal_conv1d, dilated_causal_conv1d_no_pad, self_gated_layer
from .audio import mu_law, inv_mu_law


class ConfigV1(object):
    def __init__(self, name="WaveNet"):
        self.name = name
        self.max_global_steps = 200000
        self.blks = 3
        self.layers_per_blk = 10
        self.conv_width = 2
        self.dilation_base = 2
        self.residual_units = 256
        self.skip_units = 256
        self.hidden_out = 256
        self.sample_classes = 256
        self.central_class = self.sample_classes // 2

    @property
    def reception_fields(self):
        return (self.conv_width - 1) * (self.dilation_base ** self.layers_per_blk - 1) * self.blks + 1

    def build(self, inputs, reuse=None):
        with tf.variable_scope(self.name, reuse=reuse):
            wav = inputs["wav"]
            mu_wav = mu_law(wav)
            wav_indices = tf.cast(mu_wav + 128., dtype=tf.int32)

            # 1st. build the initial causal layer.
            with tf.variable_scope("init_causal_layer"):
                # 1st. right shift.
                right_shift_wav_indices = tf.pad(wav_indices[:, :-1], [[0, 0], [1, 0]], constant_values=self.central_class)
                # 2nd. to one-hot representation
                wav_one_hot = tf.one_hot(right_shift_wav_indices, depth=self.sample_classes, dtype=tf.float32)
                # 3rd. calculate
                init_causal_out = dilated_causal_conv1d(inputs=wav_one_hot, dilation_rate=1, filters=self.residual_units, width=self.conv_width)

            # 2nd. build the dilated causal blocks.
            with tf.variable_scope("dilated_causal_blocks"):
                resi_out = init_causal_out
                skip_out = 0.
                for idx in range(self.blks):
                    with tf.variable_scope("blk_{}".format(idx)):
                        for idy in range(self.layers_per_blk):
                            with tf.variable_scope("layer_{}".format(idy)):
                                conv_out = dilated_causal_conv1d(inputs=resi_out, dilation_rate=self.dilation_base ** idy,
                                                                 filters=2*self.residual_units, width=self.conv_width)
                                acti_out = self_gated_layer(inputs=conv_out)
                                resi_out += tf.layers.dense(inputs=acti_out, units=self.residual_units, activation=None, name="residual_out")
                                skip_out += tf.layers.dense(inputs=acti_out, units=self.residual_units, activation=None, name="skip_out")

            # 3rd. calculate probabilities, and get loss.
            with tf.variable_scope("softmax"):
                # 1st. relu
                skip_out = tf.nn.relu(skip_out)
                # 2nd. dense -> relu
                skip_out = tf.layers.dense(inputs=skip_out, units=self.hidden_out, activation=tf.nn.relu)
                # 3rd. dense, get energy(logits).
                logits = tf.layers.dense(inputs=skip_out, units=self.sample_classes, activation=None)
                # 4th. get scores, and modes, then recover to waveform := [-1, 1).
                scores = tf.nn.softmax(logits=logits, dim=-1)
                modes = tf.argmax(scores, axis=-1) - 128
                synthesized_wav = inv_mu_law(modes)
                # 5th. get loss.
                loss = tf.losses.sparse_softmax_cross_entropy(labels=wav_indices, logits=logits)

            return {"origin_wav": wav, "synthesized_wav": synthesized_wav, "loss": loss}


class ConfigV2(object):
    def __init__(self, name="WaveNet"):
        self.name = name
        self.max_global_steps = 200000
        self.blks = 3
        self.layers_per_blk = 10
        self.conv_width = 2
        self.dilation_base = 2
        self.residual_units = 256
        self.skip_units = 256
        self.hidden_out = 256
        self.sample_classes = 256
        self.central_class = self.sample_classes // 2

    @property
    def reception_fields(self):
        return (self.conv_width - 1) * ((self.dilation_base ** self.layers_per_blk - 1) * self.blks + 1) + 1

    def build(self, inputs, reuse=None):
        with tf.variable_scope(self.name, reuse=reuse):
            wav = inputs["wav"]
            mu_wav = mu_law(wav)
            wav_indices = tf.cast(mu_wav + 128., dtype=tf.int32)
            target_indices = tf.slice(wav_indices, [0, self.reception_fields], [-1, -1])

            # 1st. build the initial causal layer.
            with tf.variable_scope("init_causal_layer"):
                # 1st. to one-hot representation
                wav_one_hot = tf.one_hot(wav_indices, depth=self.sample_classes, dtype=tf.float32)
                # 2nd. calculate
                init_causal_out = dilated_causal_conv1d_no_pad(inputs=wav_one_hot, dilation_rate=1, filters=self.residual_units, width=self.conv_width)

            # 2nd. build the dilated causal blocks.
            with tf.variable_scope("dilated_causal_blocks"):
                resi_out = init_causal_out
                skip_out = tf.zeros_like(init_causal_out)
                for idx in range(self.blks):
                    with tf.variable_scope("blk_{}".format(idx)):
                        for idy in range(self.layers_per_blk):
                            with tf.variable_scope("layer_{}".format(idy)):
                                conv_out = dilated_causal_conv1d_no_pad(inputs=resi_out, dilation_rate=self.dilation_base ** idy,
                                                                        filters=2*self.residual_units, width=self.conv_width)
                                acti_out = self_gated_layer(inputs=conv_out)
                                resi_out = tf.slice(resi_out, [0, tf.shape(resi_out)[1] - tf.shape(acti_out)[1], 0], [-1, -1, -1]) \
                                           + tf.layers.dense(inputs=acti_out, units=self.residual_units, activation=None, name="residual_out")
                                skip_out = tf.slice(skip_out, [0, tf.shape(skip_out)[1] - tf.shape(acti_out)[1], 0], [-1, -1, -1]) \
                                           + tf.layers.dense(inputs=acti_out, units=self.residual_units, activation=None, name="skip_out")

            # 3rd. calculate probabilities, and get loss.
            with tf.variable_scope("softmax"):
                # 1st. relu
                skip_out = tf.nn.relu(skip_out)
                # 2nd. dense -> relu
                skip_out = tf.layers.dense(inputs=skip_out, units=self.hidden_out, activation=tf.nn.relu)
                # 3rd. dense, get energy(logits).
                logits = tf.layers.dense(inputs=skip_out, units=self.sample_classes, activation=None)
                # 4th. get scores, and modes, then recover to waveform := [-1, 1).
                scores = tf.nn.softmax(logits=logits, dim=-1)
                modes = tf.argmax(scores, axis=-1) - 128
                synthesized_wav = inv_mu_law(modes)
                # 5th. get loss.
                loss = tf.losses.sparse_softmax_cross_entropy(labels=target_indices, logits=logits)

            return {"origin_wav": wav, "synthesized_wav": synthesized_wav, "loss": loss}

