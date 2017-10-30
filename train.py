import argparse
import os
import pickle as pkl
import tqdm
import tensorflow as tf
from models.train import WaveNet
from util import Feeder
from hyperparameter import HyperParams


def get_arguments():
    parser = argparse.ArgumentParser(description='Train the WaveNet neural vocoder!')
    parser.add_argument('--hyper_param_path', type=str, default='./hyper_param.json',
                        help='json: hyper_param')
    return parser.parse_args()


def main():
    args = get_arguments()
    if hasattr(args, 'hyper_param_path'):
        hp = HyperParams(param_json_path=args.hyper_param_path)
    else:
        hp = HyperParams()

    coord = tf.train.Coordinator()

    with tf.variable_scope('data'):
        audio_placeholder = tf.placeholder(name='audio', shape=(None, None, 1), dtype=tf.int32)
        audio_lens = tf.placeholder(name='audio_lens', shape=(None,), dtype=tf.int32)

    train_meta_path = hp.train_meta_path
    assert os.path.exists(train_meta_path),\
        '[!] Train meta not exists! PATH: {}'.format(train_meta_path)

    with open(train_meta_path, 'rb') as f:
        train_meta = pkl.load(f)

    train_feeder = Feeder(coord, [audio_placeholder, audio_lens], train_meta,
                          batch_size=hp.batch_size, split_nums=hp.split_nums)

    with tf.variable_scope('model'):
        model = WaveNet(*train_feeder.fed_holders, hyper_params=hp, sample_rate=train_meta['sr'])
    with tf.variable_scope('optimizer'):
        opt = tf.train.AdamOptimizer(hp.learning_rate)
        grad, var = zip(*opt.compute_gradients(model.loss))
        clipped_grad, global_norm = tf.clip_by_global_norm(grad, hp.clip_norm)
        upd = opt.apply_gradients(zip(clipped_grad, var), global_step=model.global_step)
        model.summary_norm = tf.summary.scalar('train/norm', global_norm)
        model.summary_scalar = tf.summary.merge([model.summary_loss, model.summary_norm])

    saver = tf.train.Saver()

    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    config.intra_op_parallelism_threads = 0
    config.inter_op_parallelism_threads = 0
    with tf.Session(config=config) as sess:
        model.sess = sess
        train_feeder.start_in_session(sess)

        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        ckpt = tf.train.get_checkpoint_state(hp.save_path)

        if ckpt:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(hp.save_path, ckpt_name))

        writer = tf.summary.FileWriter(hp.log_dir, sess.graph)

        global_step_eval = sess.run(model.global_step)
        pbar = tqdm.tqdm(total=hp.max_global_steps)
        pbar.update(global_step_eval)
        try:
            while global_step_eval < hp.max_global_steps:
                if not coord.should_stop():
                    sess.run(upd)
                global_step_eval += 1
                pbar.update(1)
                if global_step_eval % 10 == 0:
                    summary_eval = sess.run(model.summary_scalar)
                    writer.add_summary(summary_eval, global_step_eval)
                if global_step_eval % 1000 == 0:
                    summary_eval = sess.run(model.summary_audio)
                    writer.add_summary(summary_eval, global_step_eval)
                if global_step_eval % 1000 == 0:
                    model.save(hp.save_path, global_step_eval)
        except Exception as e:
            print('An error occurs.', e)
            coord.request_stop()
        finally:
            print('Training stopped.')
            coord.request_stop()


if __name__ == '__main__':
    main()
