import tensorflow as tf
import numpy as np


def mu_law(x, mu=255, int8=False):
  """A TF implementation of Mu-Law encoding.
  Args:
    x: The audio samples to encode.
    mu: The Mu to use in our Mu-Law.
    int8: Use int8 encoding.
  Returns:
    out: The Mu-Law encoded int8 data.
  """
  out = tf.sign(x) * tf.log(1 + mu * tf.abs(x)) / np.log(1 + mu)
  out = tf.floor(out * 128)
  if int8:
    out = tf.cast(out, tf.int8)
  return out


def inv_mu_law(x, mu=255):
  """A TF implementation of inverse Mu-Law.
  Args:
    x: The Mu-Law samples to decode.
    mu: The Mu we used to encode these samples.
  Returns:
    out: The decoded data.
  """
  x = tf.cast(x, tf.float32)
  out = (x + 0.5) * 2. / (mu + 1)
  out = tf.sign(out) / mu * ((1 + mu)**tf.abs(out) - 1)
  out = tf.where(tf.equal(x, 0), x, out)
  return out
