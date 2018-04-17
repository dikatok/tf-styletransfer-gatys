import argparse
from time import time

import numpy as np
import tensorflow as tf

from core.loss import loss_fun
from core.model import vgg16


def parse_args():
    parser = argparse.ArgumentParser(description='A Neural Algorithm of Artistic Style')
    parser.add_argument('style_image')
    parser.add_argument('content_image')
    parser.add_argument('--out_image', default='output.jpg', help='Path to save the result.', type=str)
    parser.add_argument('--learning_rate', default=1e1, help='Learning rate.', type=int)
    parser.add_argument('--num_iter', default=1000, help='Number of iterations.', type=int)
    parser.add_argument('--log_iter', default=100, help='Log interval.', type=int)
    parser.add_argument('--vgg_path', default='vgg16_weights.npz', help='Path to vgg weights.', type=str)
    parser.add_argument(
        '--content_features',
        default=['conv4_2'],
        help='List of features map to be used as content representation.',
        type=str,
        nargs='+')
    parser.add_argument(
        '--style_features',
        default=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
        help='List of features map to be used as style representation.',
        type=str,
        nargs='+')
    parser.add_argument('--content_weight', default=1, help='Content loss weight.', type=float)
    parser.add_argument('--style_weight', default=1e3, help='Style loss weight.', type=float)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)

    content_image = tf.keras.preprocessing.image.img_to_array(
        tf.keras.preprocessing.image.load_img(args.content_image))
    style_image = tf.keras.preprocessing.image.img_to_array(
        tf.keras.preprocessing.image.load_img(args.style_image))

    vgg_weights = np.load(args.vgg_path)

    tf.reset_default_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        c = sess.run(tf.expand_dims(content_image, axis=0))
        s = sess.run(tf.expand_dims(style_image, axis=0))

        content = tf.placeholder(name="content", dtype=tf.float32, shape=c.shape)
        style = tf.placeholder(name="style", dtype=tf.float32, shape=s.shape)
        transferred = tf.clip_by_value(tf.Variable(initial_value=c, dtype=tf.float32), 0, 255)

        content_net = vgg16(content, vgg_weights)
        style_net = vgg16(style, vgg_weights)
        transferred_net = vgg16(transferred, vgg_weights)

        target_content_features = [content_net[layer] for layer in args.content_features]
        target_style_features = [style_net[layer] for layer in args.style_features]

        transferred_content_features = [transferred_net[layer] for layer in args.content_features]
        transferred_style_features = [transferred_net[layer] for layer in args.style_features]

        loss = loss_fun(
            target_content_features=target_content_features,
            target_style_features=target_style_features,
            transferred_content_features=transferred_content_features,
            transferred_style_features=transferred_style_features,
            content_loss_weight=args.content_weight,
            style_loss_weight=args.style_weight)

        train_op = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss)

        sess.run(tf.global_variables_initializer())

        start = time()
        for i in range(args.num_iter):
            _, loss_val = sess.run([train_op, loss], feed_dict={style: s, content: c})

            if (i + 1) % args.log_iter == 0:
                print(f"Iteration: {i + 1}, loss: {loss_val}")

        end = time()

        result = sess.run(transferred)

    print(f"Finished {args.num_iter} iterations in {end - start} seconds")

    tf.keras.preprocessing.image.array_to_img(result[0]).save(args.out_image)
