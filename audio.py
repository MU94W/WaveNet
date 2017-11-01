import numpy as np
import tensorflow as tf


def quantize(wav, bits=8):
    interval = 2. / (1 << bits)
    epsilon = 1E-4
    quantized = np.ceil((np.clip(wav, epsilon-1, 1-epsilon) + 1.) / interval) - 1.
    return quantized.astype(np.int32)


def rev_quantize(wav, bits=8):
    max_amp = 1 << (bits - 1)
    return (wav - max_amp) / max_amp


def miu_law(wav, miu=255.):
    """

    :param wav: -1 < wav < 1
    :param miu:
    :return:
    """
    return np.sign(wav) * np.log(1 + miu*np.abs(wav)) / np.log(1 + miu)


def rev_miu_law(wav, miu=255.):
    """

    :param wav: -1 < wav < 1
    :param miu:
    :return:
    """
    return (np.exp(np.abs(wav) * np.log(1 + miu)) - 1) / miu * np.sign(wav)


def tf_quantize(wav, bits=8):
    interval = 2 / (1 << bits)
    quantized = tf.ceil((wav + 1) / interval) - 1
    return tf.cast(quantized, tf.int32)


def tf_miu_law(wav, miu=255.):
    """

    :param wav: -1 < wav < 1
    :param miu:
    :return:
    """
    return tf.sign(wav) * tf.log(1 + miu*tf.abs(wav)) / tf.log(1 + miu)


tf_rev_quantize = rev_quantize


def tf_rev_miu_law(wav, miu=255.):
    """

    :param wav: -1 < wav < 1
    :param miu:
    :return:
    """
    return tf.cast((tf.exp(tf.abs(wav) * np.log(1 + miu)) - 1) / miu * tf.sign(wav), tf.float32)

