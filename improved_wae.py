# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)

# Various attempts at improving the WAE
import tensorflow as tf
import numpy as np
import wae
from models import encoder, decoder
from datahandler import datashapes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def test(opts):
    checkpoint = opts['checkpoint']
    with tf.Session() as sess:
        with sess.graph.as_default():

            with tf.variable_scope('inference'):
                z = tf.get_variable(
                    "z", [1, opts['zdim']],
                    tf.float32, tf.random_normal_initializer(stddev=1.))
            is_training_ph = tf.placeholder(tf.bool, name='is_training_ph')

            gen, _ = decoder(opts, z, is_training=is_training_ph)
            data_shape = datashapes[opts['dataset']]
            gen.set_shape([1] + data_shape)
            e_gen, _ = encoder(opts, gen, is_training=is_training_ph)
            if opts['e_noise'] == 'gaussian':
                e_gen = e_gen[0]
            ae_gen = decoder(opts, e_gen, reuse=True, is_training=is_training_ph)
            loss = wae.WAE.reconstruction_loss(opts, gen, ae_gen)

            # Now restoring weights from the checkpoint
            # We need to restore all variables except for newly created ones
            all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            for v in all_vars:
                print v.name
            new_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='inference')
            enc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')
            dec_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
            vars_to_restore = enc_vars + dec_vars
            saver = tf.train.Saver(vars_to_restore)
            saver.restore(sess, checkpoint)
            init = tf.variables_initializer(new_vars)
            init.run()
            print 'Success!'

def add_aefixedpoint_cost(opts, wae_model):

    w_aefixedpoint = tf.placeholder(tf.float32, name='w_aefixedpoint')
    wae_model.w_aefixedpoint = w_aefixedpoint

    gen_images = wae_model.decoded
    gen_images.set_shape([opts['batch_size']] + wae_model.data_shape)
    tmp = encoder(opts, reuse=True, inputs=gen_images,
                  is_training=wae_model.is_training)
    tmp_sg = encoder(opts, reuse=True,
                     inputs=tf.stop_gradient(gen_images),
                     is_training=wae_model.is_training)
    encoded_gen_images = tmp[0]
    encoded_gen_images_sg = tmp_sg[0]
    if opts['e_noise'] == 'gaussian':
        # Encoder outputs means and variances of Gaussian
        # Encoding into means
        encoded_gen_images = encoded_gen_images[0]
        encoded_gen_images_sg = encoded_gen_images_sg[0]
    autoencoded_gen_images, _ = decoder(
        opts, reuse=True, noise=encoded_gen_images,
        is_training=wae_model.is_training)
    autoencoded_gen_images_sg, _ = decoder(
        opts, reuse=True, noise=encoded_gen_images_sg,
        is_training=wae_model.is_training)
    a = wae.WAE.reconstruction_loss(opts, gen_images, autoencoded_gen_images)
    b = tf.stop_gradient(a)
    c = wae_model.reconstruction_loss(
            tf.stop_gradient(gen_images),
            autoencoded_gen_images_sg)
    extra_cost = b + a - c
    # Check gradients
    # encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
    # wae_model.grad_extra = tf.gradients(ys=extra_cost, xs=encoder_vars)
    # for idx, el in enumerate(wae_model.grad_extra):
    #    print encoder_vars[idx].name, el

    wae_model.wae_objective += wae_model.w_aefixedpoint * extra_cost

def examples(opts, wae_model):
    # Example code showing how to load the checkpoint with a modified graph
    checkpoint = opts['checkpoint']
    with wae_model.sess.as_default(), wae_model.sess.graph.as_default():
        # Imagine the current graph coincides with the one stored in the
        # checkpoint up to 2 placeholders replaced with variables.
        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        inputs_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='inputs')
        vars_to_restore = [v for v in all_vars if v not in inputs_vars]
        saver = tf.train.Saver(vars_to_restore)
        saver.restore(wae_model.sess, checkpoint)
        init = tf.variables_initializer(inputs_vars)
        init.run()
