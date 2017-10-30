import copy
import json
import codecs
import os


global_param_dict = {'waveform_bits': 8,
                     'waveform_categories': 1 << 8,
                     'waveform_center_cat': 1 << 7,
                     'conv_layers': [8] * 2,
                     'kernel_size': (2, 1),
                     'dilation_channels': 32,
                     'skip_dims': 32,
                     'dilated_casual_use_bias': False,
                     'residual_use_bias': False,
                     'skip_use_bias': False,
                     'batch_size': 4,
                     'split_nums': 16,
                     'max_global_steps': 1E5,
                     'train_meta_path': './train_meta.pkl',
                     'log_dir': './log',
                     'save_path': './save'}


class HyperParams:
    def __init__(self, param_dict=None, param_json_path=None):
        if isinstance(param_dict, dict):
            self._init_from_dict(param_dict)
        elif isinstance(param_json_path, str) and os.path.exists(param_json_path):
            with codecs.open(param_json_path, 'r', encoding='utf-8') as f:
                param_dict = json.load(f)
            self._init_from_dict(param_dict)

    def _init_from_dict(self, param_dict):
        valid_keys = global_param_dict.keys()
        tmp_dict = copy.deepcopy(global_param_dict)
        for k, v in param_dict.items():
            if k in valid_keys:
                if isinstance(type(v), type(tmp_dict[k])):
                    tmp_dict[k] = v
        for k, v in tmp_dict.items():
            setattr(self, k, v)
