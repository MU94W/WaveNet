import tensorflow as tf


def parse_single_example(example_proto, crop_length):
    features = {"sr": tf.FixedLenFeature([], tf.int64),
                "key": tf.FixedLenFeature([], tf.string),
                "wav_raw": tf.FixedLenFeature([], tf.string)}
    parsed = tf.parse_single_example(example_proto, features=features)
    sr = tf.cast(parsed["sr"], tf.int32)
    key = parsed["key"]
    wav = tf.divide(tf.cast(tf.decode_raw(parsed["wav_raw"], tf.int16), dtype=tf.float32), 1<<15)
    wav = tf.random_crop(wav, size=[crop_length])
    wav.set_shape([crop_length])
    return {"sr": sr, "key": key, "wav": wav}


def get_dataset(tfrecord_path, batch_size=16, crop_length=16000):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_single_example, crop_length)
    dataset = dataset.shuffle(10000)
    dataset = dataset.padded_batch(batch_size, padded_shapes={"sr": (), "key": (), "wav": [None]})
    return dataset