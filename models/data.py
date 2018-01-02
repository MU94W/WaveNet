import tensorflow as tf


def parse_single_example(example_proto):
    features = {"sr": tf.FixedLenFeature([], tf.int64),
                "key": tf.FixedLenFeature([], tf.string),
                "wav_raw": tf.FixedLenFeature([], tf.string)}
    parsed = tf.parse_single_example(example_proto, features=features)
    sr = tf.cast(parsed["sr"], tf.int32)
    key = parsed["key"]
    wav = tf.divide(tf.cast(tf.decode_raw(parsed["wav_raw"], tf.int16), dtype=tf.float32), 1 << 15)
    return {"sr": sr, "key": key, "wav": wav}


def crop_wav(crop_length):
    def __crop(inputs):
        wav = tf.random_crop(inputs["wav"], size=[crop_length])
        wav.set_shape([crop_length])
        return {"sr": inputs["sr"], "key": inputs["key"], "wav": wav}
    return __crop


def get_dataset(tfrecord_path, batch_size=16, crop_length=16000):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_single_example)
    dataset = dataset.map(crop_wav(crop_length))
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(batch_size)
    return dataset
