# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)

# Various attempts at improving the WAE
import tensorflow as tf
import numpy as np
import wae
import os
import logging
from models import encoder, decoder
from datahandler import datashapes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def improved_sampling(opts):
    MAX_GD_STEPS = 200
    LOSS_EVERY_STEPS = 50
    DEBUG = False
    NUM_POINTS = 10000
    BATCH_SIZE = 100

    checkpoint = opts['checkpoint']

    # Creating a dummy file for later FID evaluations
    dummy_path = os.path.join(opts['work_dir'], 'checkpoints', 'dummy.meta')
    with open(dummy_path, 'w') as f:
        f.write('dummy string')

    with tf.Session() as sess:
        with sess.graph.as_default():

            # Creating the graph

            if opts['pz'] in ('normal', 'sphere'):
                codes = tf.get_variable(
                    "latent_codes", [BATCH_SIZE, opts['zdim']],
                    tf.float32, tf.random_normal_initializer(stddev=1.))
                if opts['pz'] == 'sphere':
                    z = codes / (tf.norm(codes, axis=0) + 1e-8)
                else:
                    z = codes
            elif opts['pz'] == 'uniform':
                codes = tf.get_variable(
                    "latent_codes", [BATCH_SIZE, opts['zdim']],
                    tf.float32, tf.random_uniform_initializer(minval=-1., maxval=1.))
            z = opts['pz_scale'] * z
            is_training_ph = tf.placeholder(tf.bool, name='is_training_ph')
            gen, _ = decoder(opts, z, is_training=is_training_ph)
            data_shape = datashapes[opts['dataset']]
            gen.set_shape([BATCH_SIZE] + data_shape)
            e_gen, _ = encoder(opts, gen, is_training=is_training_ph)
            if opts['e_noise'] == 'gaussian':
                e_gen = e_gen[0]
            ae_gen, _ = decoder(opts, e_gen, reuse=True, is_training=is_training_ph)
            # Cool hack: normalizing by the picture contrast,
            # otherwise SGD manages to decrease the loss by reducing 
            # the contrast
            loss = wae.WAE.reconstruction_loss(
                opts,
                contrast_norm(gen),
                contrast_norm(ae_gen))
            optim = tf.train.AdamOptimizer(opts['lr'], 0.9)
            optim = optim.minimize(loss, var_list=[codes])

            # Now restoring encoder and decoder from the checkpoint

            all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            enc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                         scope='encoder')
            dec_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                         scope='generator')
            new_vars = [v for v in all_vars if \
                        v not in enc_vars and v not in dec_vars]
            vars_to_restore = enc_vars + dec_vars
            saver = tf.train.Saver(vars_to_restore)
            saver.restore(sess, checkpoint)
            logging.error('Restored.')

            init = tf.variables_initializer(new_vars)

            # Finally, start generating the samples

            res_samples = []
            res_codes = []

            for ibatch in range(NUM_POINTS / BATCH_SIZE):

                logging.error('Batch %d of %d' % (ibatch + 1, NUM_POINTS / BATCH_SIZE))
                loss_prev = 1e10
                init.run()
                for step in xrange(MAX_GD_STEPS):

                    # Make a gradient step
                    sess.run(optim, feed_dict={is_training_ph: False})

                    if step == 0 or step % LOSS_EVERY_STEPS == LOSS_EVERY_STEPS - 1:
                        loss_cur, pics, codes = sess.run([loss, gen, z], feed_dict={is_training_ph: False})
                        if DEBUG:
                            if opts['input_normalize_sym']:
                                pics = (pics + 1.) / 2.
                            pic_path = os.path.join(opts['work_dir'],
                                                    'checkpoints',
                                                    'dummy.samples100_%05d' % step)
                            code_path = os.path.join(opts['work_dir'],
                                                     'checkpoints',
                                                     'code%05d' % step)
                            np.save(pic_path, pics)
                            np.save(code_path, codes)
                        rel_imp = abs(loss_cur - loss_prev) / abs(loss_prev)
                        logging.error('-- step %d, loss=%f, rel_imp=%f' % (step, loss_cur, rel_imp))
                        if step > 0 and rel_imp < 0.1:
                            break
                        loss_prev = loss_cur

                res_samples.append(pics)
                res_codes.append(codes)

            samples = np.array(res_samples)
            samples = np.vstack(samples)
            codes = np.array(res_codes)
            codes = np.vstack(codes)
            pic_path = os.path.join(opts['work_dir'], 'checkpoints', 'dummy.samples%d' % (NUM_POINTS))
            code_path = os.path.join(opts['work_dir'], 'checkpoints', 'codes%d' % (NUM_POINTS))
            np.save(pic_path, samples)
            np.save(code_path, codes)

def contrast_norm(pics):
    # pics is a [N, H, W, C] tensor
    mean, var = tf.nn.moments(pics, axes=[1, 2, 3], keep_dims=True)
    return pics / tf.sqrt(var + 1e-08)

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
