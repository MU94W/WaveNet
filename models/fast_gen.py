import argparse
import tensorflow as tf
import os
import tqdm
import numpy as np
import scipy.io.wavfile as siowav
from .configs import FastGen


def get_args():
    parser = argparse.ArgumentParser(description="Train WaveNet!")
    parser.add_argument("--save_path", type=str, default="./save")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--gen_samples", type=int, default=16000)
    parser.add_argument("--gen_path", type=str, default="./fast_gen")
    return parser.parse_args()


def main():
    args = get_args()
    net = FastGen()
    graph = tf.Graph()
    with graph.as_default():
        with tf.variable_scope("data"):
            wav_placeholder = tf.placeholder(shape=(args.batch_size, 1), dtype=tf.int32)
            inputs = {"wav": wav_placeholder}
        # build net.
        net_tensor_dic = net.build(inputs=inputs)
        global_step = tf.Variable(0, dtype=tf.int32, name="global_step")

        # get saver.
        saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:
        # warm-up queues
        sess.run(net_tensor_dic["init_op"])

        # get checkpoint
        ckpt = tf.train.get_checkpoint_state(args.save_path)
        assert ckpt
        saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)

        global_step_eval = sess.run(global_step)
        samples_batch = np.zeros(shape=(args.batch_size, 1), dtype=np.float32)
        samples_batch_time_lst = []
        for _ in tqdm.tqdm(total=args.gen_samples):
            samples_batch = sess.run(net_tensor_dic["synthesized_samples"], feed_dict={wav_placeholder: samples_batch})
            samples_batch_time_lst.append(samples_batch_time_lst)

    # save syn-ed audios
    if not os.path.exists(args.gen_path) or not os.path.isdir(args.gen_path):
        os.makedirs(args.gen_path)
    audio_batch = np.int16(np.concatenate(samples_batch_time_lst, axis=1) * (1 << 15))
    for idx, audio in enumerate(audio_batch):
        siowav.write(os.path.join(args.gen_path, "{}_{}.wav".format(global_step_eval, idx)),
                     data=audio, rate=args.sample_rate)

    print("Congratulations!")


if __name__ == "__main__":
    main()
