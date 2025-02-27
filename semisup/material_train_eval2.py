#! /usr/bin/env python
"""
ALV: Dataset of solid waste materials
Paper 1: Learning by association

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime

import sys
import time
import random
import tensorflow as tf
import shutil

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from functools import partial

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

IMAGE_SHAPE = [227, 227, 3]

flags.DEFINE_integer('emb_size', 128, 'Dimension of embedding space')

flags.DEFINE_float('test_size', 0.3, 'Test data portion')

flags.DEFINE_integer('sup_per_class', 30,
                     'Number of labeled samples used per class.')

flags.DEFINE_integer('sup_seed', -1,  #-1 -> choose randomly   -2 -> use sup_per_class as seed
                     'Integer random seed used for labeled set selection.')

flags.DEFINE_integer('sup_per_batch', -1,   #-1 -> take all available
                     'Number of labeled samples per class per batch.')

flags.DEFINE_integer('unsup_batch_size', 30,
                     'Number of unlabeled samples per batch.')

flags.DEFINE_integer('eval_interval', 500,
                     'Number of steps between evaluations.')

flags.DEFINE_string('architecture', 'alexnet_model', 'Which network architecture '
                                                           'from architectures.py to use.' )

flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')

flags.DEFINE_float('decay_factor', 0.33, 'Learning rate decay factor.')

flags.DEFINE_integer('decay_steps', 1000,
                   'Learning rate decay interval in steps.')

flags.DEFINE_float('visit_weight', 0.5, 'Weight for visit loss.')

flags.DEFINE_float('walker_weight', 0.5, 'Weight for walker loss.')
flags.DEFINE_float('logit_weight', 1.0, 'Weight for logits')
flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout factor.')
flags.DEFINE_float('l1_weight', 0.0002, 'Weight for l1 embeddding regularization')

flags.DEFINE_integer('warmup_steps', 0, 'Number of training steps.')
flags.DEFINE_integer('max_steps', 10000, 'Number of training steps.')

flags.DEFINE_string('logdir', '../', 'Training log path.')

flags.DEFINE_string('restore_checkpoint', None, 'restore weights from checkpoint, e.g. some autoencoder pretraining')

flags.DEFINE_bool('run_in_background', False, 'run in background')

flags.DEFINE_bool('semisup', True, 'Add unsupervised samples')

flags.DEFINE_bool('augmentation', True,
                  'Apply data augmentation during training.')

flags.DEFINE_float('batch_norm_decay', 0.9, 'Batch norm decay factor ')

print(FLAGS.learning_rate, FLAGS.__flags)  # print all flags (useful when logging)

import numpy as np
import semisup
from semisup.tools import material as dataset_tools
from augment import apply_augmentation
Dataset = tf.data.Dataset

NUM_LABELS = dataset_tools.NUM_LABELS
num_labels = NUM_LABELS
# IMAGE_SHAPE = dataset_tools.IMAGE_SHAPE
image_shape = IMAGE_SHAPE

def main(_):
    if FLAGS.logdir is not None:
        # FLAGS.logdir = FLAGS.logdir + '/t_' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        _unsup_batch_size = FLAGS.unsup_batch_size if FLAGS.semisup else 0
        FLAGS.logdir = "{0}/i{1}_e{2}_s{3}_un{4}_d{5}_{6}_w{7}".format(FLAGS.logdir, image_shape[0], 
                                            FLAGS.emb_size, FLAGS.sup_per_class,
                                            _unsup_batch_size, FLAGS.decay_steps,
                                            int(FLAGS.decay_factor*100), FLAGS.warmup_steps)
        try:
            shutil.rmtree(FLAGS.logdir)
        except OSError as e:
            print ("Error: %s - %s." % (e.filename, e.strerror))


    # Load image data from npy file
    train_images, test_images, train_labels, test_labels = dataset_tools.get_data(one_hot=False, test_size=FLAGS.test_size, image_shape=image_shape)

    unique, counts = np.unique(test_labels, return_counts=True)
    testset_distribution = dict(zip(unique, counts))

    train_X, train_Y = semisup.sample_by_label_v2(train_images, train_labels,
                                           FLAGS.sup_per_class, NUM_LABELS, np.random.randint(0, 100))

    def aug(image, label):
        return apply_augmentation(image, target_shape=image_shape, params=dataset_tools.augmentation_params), label

    def aug_unsup(image):
        return apply_augmentation(image, target_shape=image_shape, params=dataset_tools.augmentation_params)

    graph = tf.Graph()
    with graph.as_default():
        # Create function that defines the network.
        architecture = getattr(semisup.architectures, FLAGS.architecture)
        model_function = partial(
                architecture,
                new_shape=None,
                img_shape=IMAGE_SHAPE,
                batch_norm_decay=FLAGS.batch_norm_decay,
                emb_size=FLAGS.emb_size)

        model = semisup.SemisupModel(model_function, NUM_LABELS,
                                     IMAGE_SHAPE, 
                                     emb_size=FLAGS.emb_size,
                                     dropout_keep_prob=FLAGS.dropout_keep_prob)

        # Set up supervised inputs.
        t_images = tf.placeholder("float", shape=[None] + image_shape)
        t_labels = tf.placeholder(train_labels.dtype, shape=[None])
        dataset = Dataset.from_tensor_slices((t_images, t_labels))
        # Apply augmentation
        if FLAGS.augmentation:
            dataset = dataset.map(aug)
        dataset = dataset.shuffle(buffer_size=FLAGS.sup_per_class * NUM_LABELS)
        dataset = dataset.repeat().batch(FLAGS.sup_per_class * NUM_LABELS)
        iterator = dataset.make_initializable_iterator()
        t_sup_images, t_sup_labels = iterator.get_next()

        # Compute embeddings and logits.
        t_sup_emb = model.image_to_embedding(t_sup_images)
        t_sup_logit = model.embedding_to_logit(t_sup_emb)

        # Add losses.
        if FLAGS.semisup:
            unsup_t_images = tf.placeholder("float", shape=[None] + image_shape)
            unsup_dataset = Dataset.from_tensor_slices(unsup_t_images)
            # Apply augmentation
            if FLAGS.augmentation:
                unsup_dataset = unsup_dataset.map(aug_unsup)
            unsup_dataset = unsup_dataset.shuffle(buffer_size=FLAGS.unsup_batch_size)
            unsup_dataset = unsup_dataset.repeat().batch(FLAGS.unsup_batch_size)
            unsup_iterator = unsup_dataset.make_initializable_iterator()
            t_unsup_images = unsup_iterator.get_next()

            t_unsup_emb = model.image_to_embedding(t_unsup_images)
            model.add_semisup_loss(
                    t_sup_emb, t_unsup_emb, t_sup_labels,
                    walker_weight=FLAGS.walker_weight, visit_weight=FLAGS.visit_weight)

            #model.add_emb_regularization(t_unsup_emb, weight=FLAGS.l1_weight)
        else:
            model.loss_aba = tf.constant(0)
            model.visit_loss = tf.constant(0)

        t_logit_loss = model.add_logit_loss(t_sup_logit, t_sup_labels, weight=FLAGS.logit_weight)
        # t_logit_loss = tf.constant(0)


        #model.add_emb_regularization(t_sup_emb, weight=FLAGS.l1_weight)

        t_learning_rate = tf.placeholder("float", shape=[])

        train_op = model.create_train_op(t_learning_rate)

        summary_op = tf.summary.merge_all()

        if FLAGS.logdir is not None:
            summary_writer = tf.summary.FileWriter(FLAGS.logdir, graph)
            saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:

        sess.run(iterator.initializer, feed_dict={t_images: train_X, t_labels: train_Y})
        if FLAGS.semisup:
            sess.run(unsup_iterator.initializer, feed_dict={unsup_t_images: train_images})

        tf.global_variables_initializer().run()

        if FLAGS.restore_checkpoint is not None:
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='net')
            restorer = tf.train.Saver(var_list=variables)
            restorer.restore(sess, FLAGS.restore_checkpoint)

        learning_rate_ = FLAGS.learning_rate

        for step in range(FLAGS.max_steps):
            lr = learning_rate_
            if step < FLAGS.warmup_steps:
                lr = 1e-6 + semisup.apply_envelope("log", step, FLAGS.learning_rate, FLAGS.warmup_steps, 0)

            step_start_time = time.time()
            _, summaries, train_loss, aba_loss, visit_loss, logit_loss = sess.run([train_op, summary_op, 
                                                    model.train_loss, model.loss_aba, 
                                                    model.visit_loss, t_logit_loss], {
              t_learning_rate: lr
            })

            if not FLAGS.run_in_background:
                sys.stderr.write("\rstep: %d, Step time: %.4f sec" % (step, (time.time() - step_start_time)))
                sys.stdout.flush()
            # sup_images = sess.run(d_sup_image


            if (step + 1) % FLAGS.eval_interval == 0 or step == 99 or step == 0:
                print('\n=======================')
                print('Step: %d' % step)
                test_pred = model.classify(test_images, sess).argmax(-1)
                conf_mtx = semisup.confusion_matrix(test_labels, test_pred, NUM_LABELS)
                test_err = (test_labels != test_pred).mean() * 100
                print(conf_mtx)
                print('Target:', testset_distribution)
                print('Test error: %.2f%%' % test_err)
                print('Learning rate:', lr)
                print('train_loss:', train_loss)
                if FLAGS.semisup:
                    print('walker_loss_aba:', aba_loss)
                    print('visit_loss:', visit_loss)
                print('logit_loss:', logit_loss)
                print('Image shape:', IMAGE_SHAPE)
                print('emb_size: ', FLAGS.emb_size)
                print('sup_per_class: ', FLAGS.sup_per_class)
                if FLAGS.semisup:
                    print('unsup_batch_size: ', FLAGS.unsup_batch_size)
                print('semisup: ', FLAGS.semisup)
                print('augmentation: ', FLAGS.augmentation)
                print('decay_steps: ', FLAGS.decay_steps)
                print('decay_factor: ', FLAGS.decay_factor)
                print('warmup_steps: ', FLAGS.warmup_steps)
                if FLAGS.semisup:
                    print('walker_weight: ', FLAGS.walker_weight)
                    print('visit_weight: ', FLAGS.visit_weight)
                print('logit_weight: ', FLAGS.logit_weight)
                print('=======================\n')


                if FLAGS.logdir is not None:
                    sum_values = {
                        'Test error': test_err
                    }
                    summary_writer.add_summary(summaries, step)
                    for key, value in sum_values.items():
                        summary = tf.Summary(
                                value=[tf.Summary.Value(tag=key, simple_value=value)])
                        summary_writer.add_summary(summary, step)

            if FLAGS.logdir is not None and (step + 1) % 5000 == 0:
                path = saver.save(sess, FLAGS.logdir + '/checkpoint', model.step)
                print('@@model_path:%s' % path)

            if step % FLAGS.decay_steps == 0 and step > 0:
                learning_rate_ = learning_rate_ * FLAGS.decay_factor

if __name__ == '__main__':
    app.run()
