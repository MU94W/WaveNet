import argparse
import tensorflow as tf
import os
import tqdm
from .configs import ConfigV1, ConfigV2
from .data import get_dataset


def get_args():
    parser = argparse.ArgumentParser(description="Train WaveNet!")
    parser.add_argument("--data_path", type=str, default="./wave-net.train.tfrecords")
    parser.add_argument("--save_path", type=str, default="./save/")
    parser.add_argument("--log_path", type=str, default="./log")
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--crop_length", type=int, default=8000)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--add_audio_summary_per_steps", type=int, default=1000)
    parser.add_argument("--save_per_steps", type=int, default=5000)
    return parser.parse_args()


def main():
    args = get_args()
    net = ConfigV1()
    graph = tf.Graph()
    with graph.as_default():
        with tf.variable_scope("data"):
            dataset = get_dataset(args.data_path, args.batch_size, args.crop_length)
            dataset.repeat()
            iterator = dataset.make_one_shot_iterator()
            inputs = iterator.get_next()
        # build net.
        net_tensor_dic = net.build(inputs=inputs)

        # get summaries.
        audio_summary = tf.summary.merge([tf.summary.audio("origin", net_tensor_dic["origin_wav"], args.sample_rate),
                                          tf.summary.audio("synthesized", net_tensor_dic["synthesized_wav"], args.sample_rate)])
        loss_summary = tf.summary.scalar("loss", net_tensor_dic["loss"])

        # get optimizer.
        global_step = tf.Variable(0, dtype=tf.int32, name="global_step")
        opt = tf.train.AdamOptimizer(1e-4)
        upd = opt.minimize(net_tensor_dic["loss"], global_step=global_step)

        # get saver.
        saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:
        # get checkpoint
        ckpt = tf.train.get_checkpoint_state(args.save_path)
        if ckpt:
            saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)
        else:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        summary_writer = tf.summary.FileWriter(args.log_path)
        save_path = os.path.join(args.save_path, net.name)

        for idx in tqdm.tqdm(range(args.steps)):
            loss_summary_eval, audio_summary_eval, global_step_eval, _ = sess.run([loss_summary, audio_summary, global_step, upd])
            summary_writer.add_summary(loss_summary_eval, global_step=global_step_eval)
            if global_step_eval % args.add_audio_summary_per_steps == 0:
                summary_writer.add_summary(audio_summary_eval, global_step=global_step_eval)
            if global_step_eval % args.save_per_steps == 0:
                if not os.path.exists(args.save_path) or not os.path.isdir(args.save_path):
                    os.makedirs(args.save_path)
                saver.save(sess=sess, save_path=save_path, global_step=global_step_eval)
        summary_writer.close()

    print("Congratulations!")


if __name__ == "__main__":
    main()
