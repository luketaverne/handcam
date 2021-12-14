import sys
import cv2
import numpy as np
import tensorflow as tf
from random import shuffle
from textwrap import wrap
import re
import itertools
import tfplot
import matplotlib
from sklearn.metrics import confusion_matrix

from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_nn_ops

class _DatasetInitializerHook(tf.train.SessionRunHook):
    def __init__(self, initializer, feed_dict):
        self._initializer = initializer
        self._feed_dict = feed_dict

    def begin(self):
        pass

    def after_create_session(self, session, coord):
        del coord
        session.run(self._initializer, feed_dict=self._feed_dict)


def shuffle_dataset(filenames1, filenames2, labels):
    c = list(zip(filenames1, filenames2, labels))
    shuffle(c)

    return zip(*c)  # rgb_filenames, depth_filenames, labels

def seq_batch_accuracy(batch, seq_lens):
    acc_mask = tf.sequence_mask(lengths=seq_lens, maxlen=tf.reduce_max(seq_lens), dtype=tf.float32)
    masked_batch = 0

def plot_confusion_matrix(correct_labels, predict_labels, labels, title='Confusion matrix', tensor_name = 'MyFigure/image', normalize=False):
    '''
    Parameters:
        correct_labels                  : These are your true classification categories.
        predict_labels                  : These are you predicted classification categories
        labels                          : This is a lit of labels which will be used to display the axix labels
        title='Confusion matrix'        : Title for your matrix
        tensor_name = 'MyFigure/image'  : Name for the output summay tensor

    Returns:
        summary: TensorFlow summary

    Other itema to note:
        - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc.
        - Currently, some of the ticks dont line up due to rotations.

    source: <https://stackoverflow.com/questions/41617463/tensorflow-confusion-matrix-in-tensorboard>
    '''
    # print("correct label shape: " + str(correct_labels.shape))
    # print("predict label shape: " + str(predict_labels.shape))
    # print("correct label dtype: " + str(correct_labels.dtype))
    # print("predict label dtype: " + str(predict_labels.dtype))
    # print(correct_labels)
    # print(predict_labels)
    cm = confusion_matrix(correct_labels, predict_labels, labels=range(len(labels)))
    if normalize:
        cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')

    np.set_printoptions(precision=2)
    ###fig, ax = matplotlib.figure.Figure()

    fig = matplotlib.figure.Figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=4, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=4, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color= "black")
    fig.set_tight_layout(True)
    summary = tfplot.figure.to_summary(fig, tag=tensor_name)
    return summary

def _CheckAtLeast3DImage(image, require_static=True):
  """Assert that we are working with properly shaped image.

  Args:
    image: >= 3-D Tensor of size [*, height, width, depth]
    require_static: If `True`, requires that all dimensions of `image` are
      known and non-zero.

  Raises:
    ValueError: if image.shape is not a [>= 3] vector.

  Returns:
    An empty list, if `image` has fully defined dimensions. Otherwise, a list
    containing an assert op is returned.
  """
  try:
    if image.get_shape().ndims is None:
      image_shape = image.get_shape().with_rank(3)
    else:
      image_shape = image.get_shape().with_rank_at_least(3)
  except ValueError:
    raise ValueError("'image' must be at least three-dimensional.")
  if require_static and not image_shape.is_fully_defined():
    raise ValueError('\'image\' must be fully defined.')
  if any(x == 0 for x in image_shape):
    raise ValueError('all dims of \'image.shape\' must be > 0: %s' %
                     image_shape)
  if not image_shape.is_fully_defined():
    return [check_ops.assert_positive(array_ops.shape(image),
                                      ["all dims of 'image.shape' "
                                       "must be > 0."])]
  else:
    return []

def per_sequence_standardization(image):
  """Linearly scales `image` to have zero mean and unit norm.

  This op computes `(x - mean) / adjusted_stddev`, where `mean` is the average
  of all values in image, and
  `adjusted_stddev = max(stddev, 1.0/sqrt(image.NumElements()))`.

  `stddev` is the standard deviation of all values in `image`. It is capped
  away from zero to protect against division by 0 when handling uniform images.

  Args:
    image: 3-D tensor of shape `[height, width, channels]`.

  Returns:
    The standardized image with same shape as `image`.

  Raises:
    ValueError: if the shape of 'image' is incompatible with this function.
  """
  image = ops.convert_to_tensor(image, name='image')
  image = control_flow_ops.with_dependencies(
      _CheckAtLeast3DImage(image, require_static=False), image)
  num_pixels = math_ops.reduce_prod(array_ops.shape(image))

  image = math_ops.cast(image, dtype=dtypes.float32)
  image_mean = math_ops.reduce_mean(image)

  variance = (math_ops.reduce_mean(math_ops.square(image)) -
              math_ops.square(image_mean))
  variance = gen_nn_ops.relu(variance)
  stddev = math_ops.sqrt(variance)

  # Apply a minimum normalization that protects us against uniform images.
  min_stddev = math_ops.rsqrt(math_ops.cast(num_pixels, dtypes.float32))
  pixel_value_scale = math_ops.maximum(stddev, min_stddev)
  pixel_value_offset = image_mean

  image = math_ops.subtract(image, pixel_value_offset)
  image = math_ops.div(image, pixel_value_scale)
  return image

def per_sequence_standardization_rgbd(image):
  """Linearly scales `image` to have zero mean and unit norm.

  This op computes `(x - mean) / adjusted_stddev`, where `mean` is the average
  of all values in image, and
  `adjusted_stddev = max(stddev, 1.0/sqrt(image.NumElements()))`.

  `stddev` is the standard deviation of all values in `image`. It is capped
  away from zero to protect against division by 0 when handling uniform images.

  Args:
    image: 3-D tensor of shape `[height, width, channels]`.

  Returns:
    The standardized image with same shape as `image`.

  Raises:
    ValueError: if the shape of 'image' is incompatible with this function.
  """
  rgb = image[...,0:3]
  depth = image[...,3:]

  rgb = ops.convert_to_tensor(rgb, name='image_rgb')
  rgb = control_flow_ops.with_dependencies(
      _CheckAtLeast3DImage(rgb, require_static=False), rgb)
  num_pixels = math_ops.reduce_prod(array_ops.shape(rgb))

  rgb = math_ops.cast(rgb, dtype=dtypes.float32)
  rgb_mean = math_ops.reduce_mean(rgb)

  variance = (math_ops.reduce_mean(math_ops.square(rgb)) -
              math_ops.square(rgb_mean))
  variance = gen_nn_ops.relu(variance)
  stddev = math_ops.sqrt(variance)

  # Apply a minimum normalization that protects us against uniform images.
  min_stddev = math_ops.rsqrt(math_ops.cast(num_pixels, dtypes.float32))
  pixel_value_scale = math_ops.maximum(stddev, min_stddev)
  pixel_value_offset = rgb_mean

  rgb = math_ops.subtract(rgb, pixel_value_offset)
  rgb = math_ops.div(rgb, pixel_value_scale)

  depth = ops.convert_to_tensor(depth, name='image_depth')
  depth = control_flow_ops.with_dependencies(
      _CheckAtLeast3DImage(depth, require_static=False), depth)
  num_pixels = math_ops.reduce_prod(array_ops.shape(depth))

  depth = math_ops.cast(depth, dtype=dtypes.float32)
  depth_mean = math_ops.reduce_mean(depth)

  variance = (math_ops.reduce_mean(math_ops.square(depth)) -
              math_ops.square(depth_mean))
  variance = gen_nn_ops.relu(variance)
  stddev = math_ops.sqrt(variance)

  # Apply a minimum normalization that protects us against uniform images.
  min_stddev = math_ops.rsqrt(math_ops.cast(num_pixels, dtypes.float32))
  pixel_value_scale = math_ops.maximum(stddev, min_stddev)
  pixel_value_offset = depth_mean

  depth = math_ops.subtract(depth, pixel_value_offset)
  depth = math_ops.div(depth, pixel_value_scale)

  image = tf.concat([rgb, depth], axis=3)


  return image