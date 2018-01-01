import argparse
import os
import time
import numpy as np
import scipy.io.wavfile as siowav
import tensorflow as tf
from models.deprecated_generate import WaveNet
from hyperparameter import HyperParams


def get_arguments():
    parser = argparse.ArgumentParser(description='Train the WaveNet neural vocoder!')
    parser.add_argument('--gen_samples', type=int, default=10,
                        help='randomly generate [gen_samples] waves. Default: {}'.format(10))
    parser.add_argument('--max_infer_samples', type=int, default=5E5)
    parser.add_argument('--hyper_param_path', type=str, default='./hyper_param.json',
                        help='json: hyper_param')
    return parser.parse_args()


def main():
    args = get_arguments()
    if hasattr(args, 'hyper_param_path'):
        hp = HyperParams(param_json_path=args.hyper_param_path)
    else:
        hp = HyperParams()

    with tf.variable_scope('data'):
        waveform_seed_placeholder = tf.placeholder(name='waveform_seed', shape=(None, 1, 1), dtype=tf.int32)
        max_infer_samples_placeholder = tf.placeholder(name='max_infer_samples', shape=(), dtype=tf.int32)

    with tf.variable_scope('model'):
        model = WaveNet(waveform_seed_placeholder, max_infer_samples_placeholder, hyper_params=hp)

    saver = tf.train.Saver()

    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #config.device_count = {'CPU': 24}
    #config.intra_op_parallelism_threads = 0
    #config.inter_op_parallelism_threads = 0
    with tf.Session(config=config) as sess:
        model.sess = sess
        save_path = '/home/tpog/Lab_2017_end/WaveNet/exp/Y2017_M10_D30_h1_m48_s27/save'
        ckpt = tf.train.get_checkpoint_state(save_path)
        assert ckpt, '[E] No trained model found!'
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(save_path, ckpt_name))

        rnd_seed = np.random.randint(low=0, high=hp.waveform_categories, size=(args.gen_samples, 1, 1), dtype=np.int32)
        begin_time = time.time()
        global_step_eval = sess.run(model.global_step)
        pred_wav_eval = sess.run(model.pred_wav, feed_dict={waveform_seed_placeholder: rnd_seed,
                                                            max_infer_samples_placeholder: args.max_infer_samples})
        used_time = time.time() - begin_time
        print(f'Generate {args.gen_samples} waves, per {args.max_infer_samples} samples, use {used_time} seconds.')
        for idx, single_wav in enumerate(pred_wav_eval):
            siowav.write('step_{}_pred_{}.wav'.format(global_step_eval, idx), data=single_wav, rate=hp.sample_rate)


if __name__ == '__main__':
    main()