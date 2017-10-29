import argparse
import os
import pickle as pkl
import tqdm
import hyperparameter as hp
import tensorflow as tf
from models.train import WaveNet
from util import Feeder


def get_arguments():
    parser = argparse.ArgumentParser(description='Train the WaveNet neural vocoder!')
    parser.add_argument('--batch_size', type=int, default=hp.batch_size,
                        help='Default: {}'.format(hp.batch_size))
    parser.add_argument('--split_nums', type=int, default=hp.split_nums,
                        help='Default: {}'.format(hp.split_nums))
    parser.add_argument('--max_global_steps', type=int, default=hp.max_global_steps,
                        help='Default: {}'.format(hp.max_global_steps))
    parser.add_argument('--train_meta_path', type=str, default=hp.train_meta_path)
    parser.add_argument('--save_path', type=str, default=hp.save_path,
                        help='Where to store model.')
    parser.add_argument('--log_dir', type=str, default=hp.log_dir,
                        help='Where the log is stored.')
    return parser.parse_args()


def main():
    args = get_arguments()

    coord = tf.train.Coordinator()

    with tf.variable_scope('data'):
        audio_placeholder = tf.placeholder(name='audio', shape=(None, None, 1), dtype=tf.int32)
        audio_lens = tf.placeholder(name='audio_lens', shape=(None,), dtype=tf.int32)

    train_meta_path = args.train_meta_path
    assert os.path.exists(train_meta_path),\
        '[!] Train meta not exists! PATH: {}'.format(train_meta_path)

    with open(train_meta_path, 'rb') as f:
        train_meta = pkl.load(f)

    train_feeder = Feeder(coord, [audio_placeholder, audio_lens], train_meta,
                          batch_size=args.batch_size, split_nums=args.split_nums)

    with tf.variable_scope('model'):
        model = WaveNet(*train_feeder.fed_holders, sample_rate=train_meta['sr'])
        with tf.variable_scope('optimizer'):
            opt = tf.train.AdamOptimizer()
            grads_and_vars = opt.compute_gradients(model.loss)
            upd = opt.apply_gradients(grads_and_vars, global_step=model.global_step)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        model.sess = sess
        train_feeder.start_in_session(sess)

        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        ckpt = tf.train.get_checkpoint_state(args.save_path)

        if ckpt:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(args.save_path, ckpt_name))

        writer = tf.summary.FileWriter(args.log_dir, sess.graph)

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
                    summary_eval = sess.run(model.summary_loss)
                    writer.add_summary(summary_eval, global_step_eval)
                if global_step_eval % 1000 == 0:
                    summary_eval = sess.run(model.summary_audio)
                    writer.add_summary(summary_eval, global_step_eval)
                if global_step_eval % 1000 == 0:
                    model.save(args.save_path, global_step_eval)
        except Exception as e:
            print('An error occurs.', e)
            coord.request_stop()
        finally:
            print('Training stopped.')
            coord.request_stop()


if __name__ == '__main__':
    main()
