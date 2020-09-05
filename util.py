import cv2
import tensorflow.compat.v1 as tf
import numpy as np

def load_image(img_file, resize=None, interpolation='linear'):
  """Load image from disk. Output value range: [0,1]."""
  with tf.gfile.Open(img_file, 'rb') as f:
    im_data = np.fromstring(f.read(), np.uint8)
  im = cv2.imdecode(im_data, cv2.IMREAD_COLOR)
  im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  if resize and resize != im.shape[:2]:
    ip = cv2.INTER_LINEAR if interpolation == 'linear' else cv2.INTER_NEAREST
    im = cv2.resize(im, resize, interpolation=ip)
  return im.astype(np.float32) / 255.0
