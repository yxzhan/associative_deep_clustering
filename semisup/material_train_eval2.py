#! /usr/bin/env python
"""
ALV: Dataset of solid waste materials
Paper 1: Learning by association

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from functools import partial

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('emb_size', 128, 'Dimension of embedding space')

flags.DEFINE_integer('sup_per_class', 10,
                     'Number of labeled samples used per class.')

flags.DEFINE_integer('sup_seed', -1,  #-1 -> choose randomly   -2 -> use sup_per_class as seed
                     'Integer random seed used for labeled set selection.')

flags.DEFINE_integer('sup_per_batch', -1,   #-1 -> take all available
                     'Number of labeled samples per class per batch.')

flags.DEFINE_integer('unsup_batch_size', 30,
                     'Number of unlabeled samples per batch.')

flags.DEFINE_integer('eval_interval', 200,
                     'Number of steps between evaluations.')

flags.DEFINE_string('architecture', 'alexnet_model', 'Which network architecture '
                                                           'from architectures.py to use.' )

flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')

flags.DEFINE_float('decay_factor', 0.33, 'Learning rate decay factor.')

flags.DEFINE_float('decay_steps', 400,
                   'Learning rate decay interval in steps.')

flags.DEFINE_float('visit_weight', 1.0, 'Weight for visit loss.')

flags.DEFINE_float('walker_weight', 1.0, 'Weight for walker loss.')
flags.DEFINE_float('logit_weight', 1.0, 'Weight for logits')
flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout factor.')
flags.DEFINE_float('l1_weight', 0.0002, 'Weight for l1 embeddding regularization')

flags.DEFINE_integer('warmup_steps', 0, 'Number of training steps.')
flags.DEFINE_integer('max_steps', 20000, 'Number of training steps.')

flags.DEFINE_string('logdir', './', 'Training log path.')

flags.DEFINE_bool('semisup', False, 'Add unsupervised samples')

flags.DEFINE_bool('augmentation', False,
                  'Apply data augmentation during training.')

flags.DEFINE_float('batch_norm_decay', 0.9,
                   'Batch norm decay factor '
                   '(only used for STL-10 at the moment.')

print(FLAGS.learning_rate, FLAGS.__flags)  # print all flags (useful when logging)

import numpy as np
import semisup
from semisup.tools import material as dataset_tools
from augment import apply_augmentation
Dataset = tf.data.Dataset

NUM_LABELS = dataset_tools.NUM_LABELS
num_labels = NUM_LABELS
IMAGE_SHAPE = dataset_tools.IMAGE_SHAPE
image_shape = IMAGE_SHAPE

def main(_):
    if FLAGS.logdir is not None:
        FLAGS.logdir = FLAGS.logdir + '/t_' + str(random.randint(0,99999))

    # Load image data from npy file
    train_images,test_images, train_labels, test_labels = dataset_tools.get_data(one_hot=False, test_size=0.2)

    unique, counts = np.unique(train_labels, return_counts=True)
    print('train:')
    print(dict(zip(unique, counts)))
    unique, counts = np.unique(test_labels, return_counts=True)
    print('test:')
    print(dict(zip(unique, counts)))

    # Sample labeled training subset.
    if FLAGS.sup_seed >= 0:
      seed = FLAGS.sup_seed
    elif FLAGS.sup_seed == -2:
      seed = FLAGS.sup_per_class
    else:
      seed = np.random.randint(0, 1000)

    # print('Seed:', seed)
    # sup_by_label = semisup.sample_by_label(train_images, train_labels,
    #                                        FLAGS.sup_per_class, NUM_LABELS, seed)
    
    def aug(image, label):
        return apply_augmentation(image, target_shape=image_shape, params=dataset_tools.augmentation_params), label

    graph = tf.Graph()
    with graph.as_default():
        # Apply augmentation
        t_images = tf.placeholder("float", shape=[None] + image_shape)
        t_labels = tf.placeholder(train_labels.dtype, shape=[None])
        dataset = Dataset.from_tensor_slices((t_images, t_labels))
        dataset = dataset.map(aug)
        dataset = dataset.repeat().batch(FLAGS.sup_per_class * NUM_LABELS)
        iterator = dataset.make_initializable_iterator()
        d_sup_images, d_sup_labels = iterator.get_next()

        # Create function that defines the network.
        architecture = getattr(semisup.architectures, FLAGS.architecture)
        model_function = partial(
                architecture,
                new_shape=None,
                img_shape=IMAGE_SHAPE,
                # augmentation_function=augmentation_function,
                batch_norm_decay=FLAGS.batch_norm_decay,
                emb_size=FLAGS.emb_size)

        model = semisup.SemisupModel(model_function, NUM_LABELS,
                                     IMAGE_SHAPE, 
                                     emb_size=FLAGS.emb_size,
                                    #  augmentation_function=augmentation_function,
                                     dropout_keep_prob=FLAGS.dropout_keep_prob)

        # Set up inputs.
        # t_sup_images, t_sup_labels = semisup.create_per_class_inputs(
        #             sup_by_label, FLAGS.sup_per_batch)

        # Compute embeddings and logits.
        t_sup_emb = model.image_to_embedding(d_sup_images)
        t_sup_logit = model.embedding_to_logit(t_sup_emb)

        # Add losses.
        if FLAGS.semisup:
            t_unsup_images, _ = semisup.create_input(train_images, train_labels,
                                                         FLAGS.unsup_batch_size)

            t_unsup_emb = model.image_to_embedding(t_unsup_images)
            model.add_semisup_loss(
                    t_sup_emb, t_unsup_emb, t_sup_labels,
                    walker_weight=FLAGS.walker_weight, visit_weight=FLAGS.visit_weight)

            #model.add_emb_regularization(t_unsup_emb, weight=FLAGS.l1_weight)

        logit_loss = model.add_logit_loss(t_sup_logit, d_sup_labels, weight=FLAGS.logit_weight)

        #model.add_emb_regularization(t_sup_emb, weight=FLAGS.l1_weight)

        t_learning_rate = tf.placeholder("float", shape=[])

        train_op = model.create_train_op(t_learning_rate)

        if FLAGS.logdir is not None:
            summary_writer = tf.summary.FileWriter(FLAGS.logdir, graph)
            saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        sess.run(iterator.initializer, feed_dict={t_images: train_images, t_labels: train_labels})

        learning_rate_ = FLAGS.learning_rate

        for step in range(FLAGS.max_steps):
            lr = learning_rate_
            if step < FLAGS.warmup_steps:
                lr = 1e-6 + semisup.apply_envelope("log", step, FLAGS.learning_rate, FLAGS.warmup_steps, 0)

            _, train_loss = sess.run([train_op, model.train_loss], {
              t_learning_rate: lr
            })
            # sup_images = sess.run(d_sup_images)
            # sup_labels = sess.run(d_sup_labels)
            # sup_emb = sess.run(t_sup_emb)
            # sup_logit = sess.run(t_sup_logit)
            # logit_loss = sess.run(logit_loss)
            # train_loss = sess.run(model.train_loss)
            # _ = sess.run(train_op, {t_learning_rate: lr})

            if (step + 1) % FLAGS.eval_interval == 0 or step == 99 or step == 0:
                print('=======================')
                print('Step: %d' % step)
                test_pred = model.classify(test_images, sess).argmax(-1)
                conf_mtx = semisup.confusion_matrix(test_labels, test_pred, NUM_LABELS)
                test_err = (test_labels != test_pred).mean() * 100
                print(conf_mtx)
                print('Target:', dict(zip(unique, counts)))
                print('Test error: %.2f %%' % test_err)
                print('Learning rate:', lr)
                print('train_loss:', train_loss)
                print('Image shape:', IMAGE_SHAPE)
                print('emb_size: ', FLAGS.emb_size)
                print('sup_per_class: ', FLAGS.sup_per_class)
                print('unsup_batch_size: ', FLAGS.unsup_batch_size)
                print('semisup: ', FLAGS.semisup)
                print('augmentation: ', FLAGS.augmentation)

            if step % FLAGS.decay_steps == 0 and step > 0:
                learning_rate_ = learning_rate_ * FLAGS.decay_factor

if __name__ == '__main__':
    app.run()
