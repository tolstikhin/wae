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

def block_diagonal(matrices, dtype=tf.float32):
    """Constructs block-diagonal matrices from a list of batched 2D tensors.
    Taken from: https://stackoverflow.com/questions/42157781/block-diagonal-matrices-in-tensorflow

    Args:
    matrices: A list of Tensors with shape [..., N_i, M_i] (i.e. a list of
      matrices with the same batch dimension).
    dtype: Data type to use. The Tensors in `matrices` must match this dtype.
    Returns:
    A matrix with the input matrices stacked along its main diagonal, having
    shape [..., \sum_i N_i, \sum_i M_i].

    """
    matrices = [tf.convert_to_tensor(matrix, dtype=dtype) for matrix in matrices]
    blocked_rows = tf.Dimension(0)
    blocked_cols = tf.Dimension(0)
    batch_shape = tf.TensorShape(None)
    for matrix in matrices:
        full_matrix_shape = matrix.get_shape().with_rank_at_least(2)
        batch_shape = batch_shape.merge_with(full_matrix_shape[:-2])
        blocked_rows += full_matrix_shape[-2]
        blocked_cols += full_matrix_shape[-1]
    ret_columns_list = []
    for matrix in matrices:
        matrix_shape = tf.shape(matrix)
        ret_columns_list.append(matrix_shape[-1])
    ret_columns = tf.add_n(ret_columns_list)
    row_blocks = []
    current_column = 0
    for matrix in matrices:
        matrix_shape = tf.shape(matrix)
        row_before_length = current_column
        current_column += matrix_shape[-1]
        row_after_length = ret_columns - current_column
        row_blocks.append(tf.pad(
            tensor=matrix,
            paddings=tf.concat(
                [tf.zeros([tf.rank(matrix) - 1, 2], dtype=tf.int32),
                 [(row_before_length, row_after_length)]],
                axis=0)))
    blocked = tf.concat(row_blocks, -2)
    blocked.set_shape(batch_shape.concatenate((blocked_rows, blocked_cols)))
    return blocked

def sq_distances(points):
    sq_norms = tf.reduce_sum(tf.square(points), axis=1, keep_dims=True)
    dotprods = tf.matmul(points, points, transpose_b=True)
    return sq_norms, sq_norms + tf.transpose(sq_norms) - 2. * dotprods

def mmdpp_penalty(opts, wae_model, sample_pz):
    """ Paul's MMD++ penalty
        For now assuming it works only with Gaussian encoders

        Assuming
            N is dataset size
            n is the picture minibatch size
            k is number of random points per Q(Z|Xi)
            zi are iid samples from Pz
            z^i_m is m-th sample from Q(Z|Xi)

        Unbiased statistic is:
            (1) sum_{i neq j} k(zi, zj) / n / (n-1) -
            (2) 2 \sum_{i, j} \sum_m k(z^i_m, zj) / k / n / n +
            (3) (N - 1) \sum_{i neq j} \sum_{m1, m2} k(z^i_m1, z^j_m2) / n / (n - 1) / k / k / N +
            (4) \sum_i \sum_{m1 neq m2} k(z^i_m1, z^i_m2) / n / k / (k - 1) / N
    """
    assert opts["e_noise"] in ('gaussian'), \
        'MMD++ works only with Gaussian encoders!'

    # Number of codes per the same pic
    NUMCODES = 10
    n = opts['batch_size']
    N = wae_model.train_size
    kernel = opts['mmd_kernel']
    sigma2_p = opts['pz_scale'] ** 2

    # First we need to sample multiple codes per minibatch picture:
    # Qhat sample = Zi1, ..., ZiK from Q(Z|Xi) for i = 1 ... batch_size
    # For that it is enough to sample batch_size * K standard normal vectors
    # rescale those and then add encoder means
    eps = tf.random_normal((n * NUMCODES, opts['zdim']),
                           0., 1., dtype=tf.float32)
    sigmas_q = wae_model.enc_sigmas
    block_var = tf.reshape(tf.tile(sigmas_q, [1, NUMCODES]), [-1, opts['zdim']])
    eps_q = tf.multiply(eps, tf.sqrt(1e-8 + tf.exp(block_var)))
    means_q = wae_model.enc_mean
    block_means = tf.reshape(tf.tile(means_q, [1, NUMCODES]),
                             [-1, opts['zdim']])
    sample_qhat = block_means + eps_q
    # sample_qhat = tf.random_normal((n * NUMCODES, opts['zdim']),
    #                        0., 1., dtype=tf.float32)

    sq_norms_pz, dist_pz = sq_distances(sample_pz)
    sq_norms_qhat, dist_qhat = sq_distances(sample_qhat)
    dotprods_pz_qhat = tf.matmul(sample_pz, sample_qhat, transpose_b=True)
    dist_pz_qhat = sq_norms_pz + tf.transpose(sq_norms_qhat) \
                   - 2. * dotprods_pz_qhat

    mask = block_diagonal(
        [np.ones((NUMCODES, NUMCODES), dtype=np.float32) for i in range(n)],
        tf.float32)

    if kernel == 'RBF':
        # Median heuristic for the sigma^2 of Gaussian kernel
        sigma2_k = tf.nn.top_k(
            tf.reshape(dist_pz_qhat, [-1]), n / 2).values[n / 2 - 1]
        sigma2_k += tf.nn.top_k(
            tf.reshape(dist_qhat, [-1]), n / 2).values[n / 2 - 1]

        if opts['verbose']:
            sigma2_k = tf.Print(sigma2_k, [sigma2_k], 'Kernel width:')
        # Part (1)
        res1 = tf.exp( - dist_pz / 2. / sigma2_k)
        res1 = tf.multiply(res1, 1. - tf.eye(n))
        res1 = tf.reduce_sum(res1) / (n * n - n)
        # Part (2)
        res2 = tf.exp( - dist_pz_qhat / 2. / sigma2_k)
        res2 = tf.reduce_sum(res2) / (n * n) / NUMCODES
        # Part (3)
        res3 = tf.exp( - dist_qhat / 2. / sigma2_k)
        res3 = tf.multiply(res3, 1. - mask)
        res3 = tf.reduce_sum(res3) * (N - 1) / N / n / (n - 1) / (NUMCODES ** 2)
        res3 = tf.Print(res3, [res3], 'Qhat vs Qhat off diag:')
        # Part (4) 
        res4 = tf.exp( - dist_qhat / 2. / sigma2_k)
        res4 = tf.multiply(res4, mask - tf.eye(n * NUMCODES))
        res4 = tf.reduce_sum(res4) / n / NUMCODES / (NUMCODES - 1) / N
        res4 = tf.Print(res4, [res4], 'Qhat vs Qhat diag:')
        stat = res1 - 2 * res2 + res3 + res4

    elif kernel == 'IMQ':
        if opts['pz'] == 'normal':
            Cbase = 2. * opts['zdim'] * sigma2_p
        elif opts['pz'] == 'sphere':
            Cbase = 2.
        elif opts['pz'] == 'uniform':
            # E ||x - y||^2 = E[sum (xi - yi)^2]
            #               = zdim E[(xi - yi)^2]
            #               = const * zdim
            Cbase = opts['zdim']
        stat = 0.
        # scales = [.1, .2, .5, 1., 2., 5., 10.]
        scales = [(1., 1.), (1./N, 1)]
        # scales = [(1., 1.)]
        for scale, weight in scales:
            C = Cbase * scale
            # Part (1)
            res1 = C / (C + dist_pz)
            res1 = tf.multiply(res1, 1. - tf.eye(n))
            res1 = tf.reduce_sum(res1) / (n * n - n)
            # res1 = tf.Print(res1, [res1], 'Pz vs Pz:')
            # Part (2)
            res2 = C / (C + dist_pz_qhat)
            res2 = tf.reduce_sum(res2) / (n * n) / NUMCODES
            # res2 = tf.Print(res2, [res2], 'Pz vs Qhat:')
            # Part (3)
            res3 = C / (C + dist_qhat)
            res3 = tf.multiply(res3, 1. - mask)
            res3 = tf.reduce_sum(res3) * (N - 1) / N / n / (n - 1) / (NUMCODES ** 2)
            res3 = tf.Print(res3, [res3], 'Qhat vs Qhat off diag [%f]:' % weight)
            # Part (4) 
            res4 = C / (C + dist_qhat)
            res4 = tf.multiply(res4, mask - tf.eye(n * NUMCODES))
            res4 = tf.reduce_sum(res4) / n / NUMCODES / (NUMCODES - 1) / N
            res4 = tf.Print(res4, [res4], 'Qhat vs Qhat diag [%f]:' % weight)
            stat += weight * (res1 - 2 * res2 + res3 + res4)
    return stat

def sq_distances_1d(points):
    """
        points is a (N, d) tensor
        we want to return (N,d,N) tensor M, where
        M(ijk) = (points[i,j] - points[k,j])^2
    """
    a = tf.expand_dims(points, 2)
    b = tf.transpose(a, [2, 1, 0])
    return tf.multiply(a, a) + tf.multiply(b, b) \
            - 2. * tf.multiply(a, b)

def diag_3d(n, zdim):
    return tf.tile(tf.transpose(tf.expand_dims(tf.eye(n), 2), [0, 2, 1]),
                   [1, zdim, 1])

def mmdpp_1d_penalty(opts, wae_model, sample_pz):
    """ Paul's MMD++ penalty for all the individual coordinates
    """
    assert opts["e_noise"] in ('gaussian'), \
        '1d MMD++ works only with Gaussian encoders!'

    # Number of codes per the same pic
    NUMCODES = 10
    n = opts['batch_size']
    N = wae_model.train_size
    kernel = opts['mmd_kernel']
    sigma2_p = opts['pz_scale'] ** 2

    # First we need to sample multiple codes per minibatch picture:
    # Qhat sample = Zi1, ..., ZiK from Q(Z|Xi) for i = 1 ... batch_size
    # For that it is enough to sample batch_size * K standard normal vectors
    # rescale those and then add encoder means
    eps = tf.random_normal((n * NUMCODES, opts['zdim']),
                           0., 1., dtype=tf.float32)
    sigmas_q = wae_model.enc_sigmas
    block_var = tf.reshape(tf.tile(sigmas_q, [1, NUMCODES]), [-1, opts['zdim']])
    eps_q = tf.multiply(eps, tf.sqrt(1e-8 + tf.exp(block_var)))
    means_q = wae_model.enc_mean
    block_means = tf.reshape(tf.tile(means_q, [1, NUMCODES]),
                             [-1, opts['zdim']])
    sample_qhat = block_means + eps_q
    # sample_qhat = tf.random_normal((n * NUMCODES, opts['zdim']),
    #                        0., 1., dtype=tf.float32)

    dist_pz = sq_distances_1d(sample_pz)
    dist_qhat = sq_distances_1d(sample_qhat)
    temp_pz = tf.expand_dims(sample_pz, 2)
    temp_qhat = tf.expand_dims(sample_qhat, 2)
    temp_qhat_t = tf.transpose(temp_qhat, [2, 1, 0])
    dist_pz_qhat = tf.multiply(temp_pz, temp_pz) \
                   + tf.multiply(temp_qhat_t, temp_qhat_t) \
                   - 2. * tf.multiply(temp_pz, temp_qhat_t)
    mask = block_diagonal(
        [np.ones((NUMCODES, NUMCODES), dtype=np.float32) for i in range(n)],
        tf.float32)
    mask = tf.expand_dims(mask, 2)
    mask = tf.transpose(mask, [0, 2, 1])
    mask = tf.tile(mask, [1, opts['zdim'], 1])
    diag_pz = diag_3d(n, opts['zdim'])
    diag_qhat = diag_3d(n * NUMCODES, opts['zdim'])

    if kernel == 'RBF':
        # Median heuristic for the sigma^2 of Gaussian kernel
        sigma2_k = tf.nn.top_k(
            tf.reshape(dist_pz_qhat, [-1]), n / 2).values[n / 2 - 1]
        sigma2_k += tf.nn.top_k(
            tf.reshape(dist_qhat, [-1]), n / 2).values[n / 2 - 1]

        if opts['verbose']:
            sigma2_k = tf.Print(sigma2_k, [sigma2_k], 'Kernel width:')

        # Part (1)
        res1 = tf.exp( - dist_pz / 2. / sigma2_k)
        res1 = tf.multiply(res1, 1. - diag_pz)
        res1 = tf.reduce_sum(res1) / (n * n - n)
        # Part (2)
        res2 = tf.exp( - dist_pz_qhat / 2. / sigma2_k)
        res2 = tf.reduce_sum(res2) / (n * n) / NUMCODES
        # Part (3)
        res3 = tf.exp( - dist_qhat / 2. / sigma2_k)
        res3 = tf.multiply(res3, 1. - mask)
        res3 = tf.reduce_sum(res3) * (N - 1) / N / n / (n - 1) / (NUMCODES ** 2)
        res3 = tf.Print(res3, [res3], 'Qhat vs Qhat off diag:')
        # Part (4) 
        res4 = tf.exp( - dist_qhat / 2. / sigma2_k)
        res4 = tf.multiply(res4, mask - diag_qhat)
        res4 = tf.reduce_sum(res4) / n / NUMCODES / (NUMCODES - 1) / N
        res4 = tf.Print(res4, [res4], 'Qhat vs Qhat diag:')
        stat = res1 - 2 * res2 + res3 + res4

    elif kernel == 'IMQ':
        if opts['pz'] == 'normal':
            Cbase = 2. * opts['zdim'] * sigma2_p
        elif opts['pz'] == 'sphere':
            Cbase = 2.
        elif opts['pz'] == 'uniform':
            # E ||x - y||^2 = E[sum (xi - yi)^2]
            #               = zdim E[(xi - yi)^2]
            #               = const * zdim
            Cbase = opts['zdim']
        stat = 0.
        # scales = [.1, .2, .5, 1., 2., 5., 10.]
        scales = [(1., 1.), (1./N, N)]
        for scale, weight in scales:
            C = Cbase * scale
            # Part (1)
            res1 = C / (C + dist_pz)
            res1 = tf.multiply(res1, 1. - diag_pz)
            res1 = tf.reduce_sum(res1) / (n * n - n)
            # res1 = tf.Print(res1, [res1], 'Pz vs Pz:')
            # Part (2)
            res2 = C / (C + dist_pz_qhat)
            res2 = tf.reduce_sum(res2) / (n * n) / NUMCODES
            # res2 = tf.Print(res2, [res2], 'Pz vs Qhat:')
            # Part (3)
            res3 = C / (C + dist_qhat)
            res3 = tf.multiply(res3, 1. - mask)
            res3 = tf.reduce_sum(res3) * (N - 1) / N / n / (n - 1) / (NUMCODES ** 2)
            res3 = tf.Print(res3, [res3], 'Qhat vs Qhat off diag:')
            # Part (4) 
            res4 = C / (C + dist_qhat)
            res4 = tf.multiply(res4, mask - diag_qhat)
            res4 = tf.reduce_sum(res4) / n / NUMCODES / (NUMCODES - 1) / N
            res4 = tf.Print(res4, [res4], 'Qhat vs Qhat diag:')
            stat += weight * (res1 - 2 * res2 + res3 + res4)

    return stat
