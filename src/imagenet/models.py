import os
import sys

import numpy as np
import tensorflow as tf

from src.imagenet.image_ops import conv
from src.imagenet.image_ops import fully_connected
from src.imagenet.image_ops import batch_norm
from src.imagenet.image_ops import relu
from src.imagenet.image_ops import max_pool
from src.imagenet.image_ops import global_avg_pool

from src.utils import count_model_params
from src.utils import get_train_ops
from datasets import dataset_factory
from preprocessing import preprocessing_factory
from imagenet_data import ImagenetData
import image_processing
slim = tf.contrib.slim


class Model(object):
  def __init__(self,
               cutout_size=None,
               batch_size=32,
               eval_batch_size=100,
               clip_mode=None,
               grad_bound=None,
               l2_reg=1e-4,
               lr_init=0.1,
               lr_dec_start=0,
               lr_dec_every=100,
               lr_dec_rate=0.1,
               keep_prob=1.0,
               optim_algo=None,
               sync_replicas=False,
               num_aggregate=None,
               num_replicas=None,
               data_format="NHWC",
               name="generic_model",
               num_readers=4,
               seed=None,
              ):
    """
    Args:
      lr_dec_every: number of epochs to decay
    """
    print "-" * 80
    print "Build model {}".format(name)

    self.cutout_size = cutout_size
    self.batch_size = batch_size
    self.eval_batch_size = eval_batch_size
    self.clip_mode = clip_mode
    self.grad_bound = grad_bound
    self.l2_reg = l2_reg
    self.lr_init = lr_init
    self.lr_dec_start = lr_dec_start
    self.lr_dec_rate = lr_dec_rate
    self.keep_prob = keep_prob
    self.optim_algo = optim_algo
    self.sync_replicas = sync_replicas
    self.num_aggregate = num_aggregate
    self.num_replicas = num_replicas
    self.data_format = data_format
    self.name = name
    self.num_readers = num_readers
    self.labels_offset = 1
    self.train_image_size = 224
    self.seed = seed

    self.global_step = None
    self.valid_acc = None
    self.test_acc = None
    print "Build data ops"
    with tf.device("/cpu:0"):
      #read training data
      train_dataset = dataset_factory.get_dataset("imagenet", "train", "/home/wangshiyao/Documents/data/imagenet/cls_tf/1_10")
      provider = slim.dataset_data_provider.DatasetDataProvider(
          train_dataset,
          num_readers=self.num_readers,
          common_queue_capacity=20 * self.batch_size,
          common_queue_min=10 * self.batch_size)
      [x_train, y_train] = provider.get(['image', 'label'])
      y_train -= self.labels_offset
      y_train = tf.cast(y_train, tf.int32)
      image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        "nasnet_large",
        is_training=True)
      x_train = image_preprocessing_fn(x_train, self.train_image_size, self.train_image_size)

      x_train, y_train = tf.train.shuffle_batch(
          [x_train, y_train],
          batch_size=self.batch_size,
          capacity=20 * self.batch_size,
          min_after_dequeue=0,
          num_threads=16,
          seed=self.seed,
          allow_smaller_final_batch=True,)
      self.num_train_examples = 125113
      self.num_train_batches = (
        self.num_train_examples + self.batch_size - 1) // self.batch_size
      self.lr_dec_every = lr_dec_every * self.num_train_batches
      if self.data_format == "NCHW":
        x_train = tf.transpose(x_train, [0, 3, 1, 2])
      self.x_train = x_train
      self.y_train = y_train

      val_dataset = dataset_factory.get_dataset("imagenet", "validation", "/home/wangshiyao/Documents/data/imagenet/cls_tf/1_10")
      valid_provider = slim.dataset_data_provider.DatasetDataProvider(
          val_dataset,
          num_readers=self.num_readers,
          common_queue_capacity=20 * self.eval_batch_size,
          common_queue_min=10 * self.eval_batch_size)
      [x_valid, y_valid] = valid_provider.get(['image', 'label'])
      y_valid -= self.labels_offset
      y_valid = tf.cast(y_valid, tf.int32)
      image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        "nasnet_large",
        is_training=False)
      x_valid = image_preprocessing_fn(x_valid, self.train_image_size, self.train_image_size)

      x_valid, y_valid = tf.train.batch(
          [x_valid, y_valid],
          batch_size=self.eval_batch_size,
          num_threads=16,
          capacity=5 * self.eval_batch_size)

      self.num_valid_examples = 12511
      self.num_valid_batches = (
        self.num_valid_examples + self.eval_batch_size - 1) // self.eval_batch_size
      if self.data_format == "NCHW":
        x_valid = tf.transpose(x_valid, [0, 3, 1, 2])
      self.x_valid = x_valid
      self.y_valid = y_valid


      test_dataset = dataset_factory.get_dataset("imagenet", "test", "/home/wangshiyao/Documents/data/imagenet/cls_tf/1_10")
      test_provider = slim.dataset_data_provider.DatasetDataProvider(
          test_dataset,
          num_readers=self.num_readers,
          common_queue_capacity=20 * self.eval_batch_size,
          common_queue_min=10 * self.eval_batch_size)
      [x_test, y_test] = test_provider.get(['image', 'label'])
      y_test -= self.labels_offset
      y_test = tf.cast(y_test, tf.int32)
      image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        "nasnet_large",
        is_training=False)
      x_test = image_preprocessing_fn(x_test, self.train_image_size, self.train_image_size)

      x_test, y_test = tf.train.batch(
          [x_test, y_test],
          batch_size=self.eval_batch_size,
          num_threads=16,
          capacity=5 * self.eval_batch_size)

      self.num_test_examples = 12511
      self.num_test_batches = (
        self.num_test_examples + self.eval_batch_size - 1) // self.eval_batch_size
      if self.data_format == "NCHW":
        x_test = tf.transpose(x_test, [0, 3, 1, 2])
      self.x_test = x_test
      self.y_test = y_test

  def eval_once(self, sess, eval_set, feed_dict=None, verbose=False):
    """Expects self.acc and self.global_step to be defined.

    Args:
      sess: tf.Session() or one of its wrap arounds.
      feed_dict: can be used to give more information to sess.run().
      eval_set: "valid" or "test"
    """

    assert self.global_step is not None
    global_step = sess.run(self.global_step)
    print "Eval at {}".format(global_step)

    if eval_set == "valid":
      assert self.x_valid is not None
      assert self.valid_acc is not None
      num_examples = self.num_valid_examples
      num_batches = self.num_valid_batches
      acc_op = self.valid_acc
    elif eval_set == "test":
      assert self.test_acc is not None
      num_examples = self.num_test_examples
      num_batches = self.num_test_batches
      acc_op = self.test_acc
    else:
      raise NotImplementedError("Unknown eval_set '{}'".format(eval_set))

    total_acc = 0
    total_exp = 0
    for batch_id in xrange(num_batches):
      acc = sess.run(acc_op, feed_dict=feed_dict)
      total_acc += acc
      total_exp += self.eval_batch_size
      if verbose:
        sys.stdout.write("\r{:<5d}/{:>5d}".format(total_acc, total_exp))
    if verbose:
      print ""
    print "{}_acc_num/total_num: {:<6.4f}/{:<6.4f}".format(
      eval_set, float(total_acc), float(total_exp))
    print "{}_accuracy: {:<6.4f}".format(
      eval_set, float(total_acc) / total_exp)

  def _build_train(self):
    print "Build train graph"
    logits = self._model(self.x_train, True)
    log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=self.y_train)
    self.loss = tf.reduce_mean(log_probs)

    self.train_preds = tf.argmax(logits, axis=1)
    self.train_preds = tf.to_int32(self.train_preds)
    self.train_acc = tf.equal(self.train_preds, self.y_train)
    self.train_acc = tf.to_int32(self.train_acc)
    self.train_acc = tf.reduce_sum(self.train_acc)

    tf_variables = [var
        for var in tf.trainable_variables() if var.name.startswith(self.name)]
    self.num_vars = count_model_params(tf_variables)
    print "-" * 80
    for var in tf_variables:
      print var

    self.global_step = tf.Variable(
      0, dtype=tf.int32, trainable=False, name="global_step")
    self.train_op, self.lr, self.grad_norm, self.optimizer = get_train_ops(
      self.loss,
      tf_variables,
      self.global_step,
      clip_mode=self.clip_mode,
      grad_bound=self.grad_bound,
      l2_reg=self.l2_reg,
      lr_init=self.lr_init,
      lr_dec_start=self.lr_dec_start,
      lr_dec_every=self.lr_dec_every,
      lr_dec_rate=self.lr_dec_rate,
      optim_algo=self.optim_algo,
      sync_replicas=self.sync_replicas,
      num_aggregate=self.num_aggregate,
      num_replicas=self.num_replicas)

  def _build_valid(self):
    if self.x_valid is not None:
      print "-" * 80
      print "Build valid graph"
      logits = self._model(self.x_valid, False, reuse=True)
      self.valid_preds = tf.argmax(logits, axis=1)
      self.valid_preds = tf.to_int32(self.valid_preds)
      self.valid_acc = tf.equal(self.valid_preds, self.y_valid)
      self.valid_acc = tf.to_int32(self.valid_acc)
      self.valid_acc = tf.reduce_sum(self.valid_acc)

  def _build_test(self):
    print "-" * 80
    print "Build test graph"
    logits = self._model(self.x_test, False, reuse=True)
    self.test_preds = tf.argmax(logits, axis=1)
    self.test_preds = tf.to_int32(self.test_preds)
    self.test_acc = tf.equal(self.test_preds, self.y_test)
    self.test_acc = tf.to_int32(self.test_acc)
    self.test_acc = tf.reduce_sum(self.test_acc)

  def build_valid_rl(self, shuffle=False):
    print "-" * 80
    print "Build valid graph on shuffled data"
    with tf.device("/cpu:0"):
      # shuffled valid data: for choosing validation model
      if not shuffle and self.data_format == "NCHW":
        self.images["valid_original"] = np.transpose(
          self.images["valid_original"], [0, 3, 1, 2])
      x_valid_shuffle, y_valid_shuffle = tf.train.shuffle_batch(
        [self.images["valid_original"], self.labels["valid_original"]],
        batch_size=self.batch_size,
        capacity=25000,
        enqueue_many=True,
        min_after_dequeue=0,
        num_threads=16,
        seed=self.seed,
        allow_smaller_final_batch=True,
      )

      def _pre_process(x):
        x = tf.pad(x, [[4, 4], [4, 4], [0, 0]])
        x = tf.random_crop(x, [32, 32, 3], seed=self.seed)
        x = tf.image.random_flip_left_right(x, seed=self.seed)
        if self.data_format == "NCHW":
          x = tf.transpose(x, [2, 0, 1])

        return x

      if shuffle:
        x_valid_shuffle = tf.map_fn(_pre_process, x_valid_shuffle,
                                    back_prop=False)

    logits = self._model(x_valid_shuffle, False, reuse=True)
    valid_shuffle_preds = tf.argmax(logits, axis=1)
    valid_shuffle_preds = tf.to_int32(valid_shuffle_preds)
    self.valid_shuffle_acc = tf.equal(valid_shuffle_preds, y_valid_shuffle)
    self.valid_shuffle_acc = tf.to_int32(self.valid_shuffle_acc)
    self.valid_shuffle_acc = tf.reduce_sum(self.valid_shuffle_acc)

  def _model(self, images, is_training, reuse=None):
    raise NotImplementedError("Abstract method")
