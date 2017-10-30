import json
import codecs
import os


class HyperParams:
    def __init__(self, param_dict=None, param_json_path=None):
        self.waveform_bits = 8
        self.waveform_categories = 1 << self.waveform_bits
        self.waveform_center_cat = 1 << (self.waveform_bits - 1)
        self.conv_layers = [8] * 4
        self.kernel_size = 2
        self.dilation_channels = 32
        self.skip_dims = 512
        self.dilated_causal_use_bias = False
        self.residual_use_bias = False
        self.skip_use_bias = False
        self.learning_rate = 0.001
        self.clip_norm = 1.
        self.batch_size = 1
        self.split_nums = 16
        self.max_global_steps = 1E5
        self.train_meta_path = './train_meta.pkl'
        self.log_dir = './log'
        self.save_path = './save'
        if isinstance(param_dict, dict):
            self._update_from_dict(param_dict)
        elif isinstance(param_json_path, str) and os.path.exists(param_json_path):
            with codecs.open(param_json_path, 'r', encoding='utf-8') as f:
                param_dict = json.load(f)
            self._update_from_dict(param_dict)
        else:
            print('Use default setup.')

    def _update_from_dict(self, param_dict):
        for k, v in param_dict.items():
            assert hasattr(self, k),\
                '[E] param: \"{}\" is not valid.'.format(k)
            assert isinstance(type(v), type(getattr(self, k))),\
                '[E] param: \"{}\" should have type: \"{}\", ' \
                'while got type: \"{}\".'.format(k, type(getattr(self, k)), type(v))
            setattr(self, k, v)
