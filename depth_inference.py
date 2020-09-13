
from absl import app
from absl import flags
from absl import logging
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
from os import path

from util import load_image
import model

flags.DEFINE_string('depth_image_dir', None,
                    'path to store depth predictions. ')
flags.DEFINE_string('input_image_path', None,
                    'image to evaluate depth on.')
flags.DEFINE_string('checkpoint_dir', None, 'Directory containing checkpoints '
                    'to evaluate.')


FLAGS = flags.FLAGS


def depth_inference():
    model_input_img_size = (416, 128)  # w,h
    img_in = load_image(FLAGS.input_image_path, resize=model_input_img_size)
    inference_model = model.Model(
        is_training=False,
        batch_size=1,
        img_height=model_input_img_size[1],
        img_width=model_input_img_size[0])
    saver = tf.train.Saver()
    depth_img = None
    with tf.Session() as sess:
        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        saver.restore(sess, checkpoint)
        # Each image is a sequence of 3 frames. We use only the first 2.
        depth_imgs = inference_model.inference_depth([img_in], sess)
        depth_img = depth_imgs[0]
    print('depth image of dimension: {}\nsample pixel: {}'.format(
        depth_img.shape, depth_img[0, 0]))
    in_img_name = path.splitext(path.basename(FLAGS.input_image_path))[0]
    depth_map_path = '{}/{}_depth.npy'.format(
        FLAGS.depth_image_dir, in_img_name)
    # cv2.imwrite(depth_image_path, depth_img)
    np.save(depth_map_path, depth_img)
    logging.info('Depth map written to {}'.format(depth_map_path))


def main(_):
    depth_inference()


if __name__ == '__main__':
    app.run(main)
