import numpy as np
import tensorflow as tf


def quantize(wav, bits=8):
    max_amp = 1 << (bits - 1)
    return tf.round(wav * max_amp + max_amp)


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

