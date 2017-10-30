from TFCommon.DataFeeder import BaseFeeder
from six.moves import xrange
import librosa
import os
import random
import audio
import numpy as np


class Feeder(BaseFeeder):
    def __init__(self, *args, **kwargs):
        super(Feeder, self).__init__(*args, **kwargs)

    def read_by_key(self, key):
        wav, _ = librosa.core.load(os.path.join(self.meta['root'], key), self.meta['sr'])
        return wav

    def pre_process_batch(self, batch):
        len_batch = np.asarray([len(item) for item in batch], dtype=np.int32)
        max_len = np.max(len_batch)
        wav_batch = np.asarray([np.pad(item, (0, max_len - item_len),
                                       mode='constant', constant_values=0.)
                                for item, item_len in zip(batch, len_batch)])
        wav_batch = np.expand_dims(wav_batch, axis=-1)
        wav_batch = audio.quantize(audio.miu_law(wav_batch))
        return wav_batch, len_batch

    def split_strategy(self, many_records):
        sorted_records = sorted(many_records, key=lambda x: len(x), reverse=True)
        sorted_batches = [sorted_records[idx*self.batch_size:(idx+1)*self.batch_size] for idx in xrange(self.split_nums)]
        random.shuffle(sorted_batches)
        for idx in xrange(self.split_nums):
            yield sorted_batches[idx]

