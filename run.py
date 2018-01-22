import os
import logging
import configs
from wae import WAE
from datahandler import DataHandler
import utils
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("zdim", 8, "Dimensionality of the latent space")
flags.DEFINE_float("wae_lambda", 10., "POT regularization")
flags.DEFINE_string("work_dir", 'results_mnist', "Working directory ['results']")
flags.DEFINE_string("dataset", 'mnist', "mnist, celebA, ...")
FLAGS = flags.FLAGS

def main():

    if FLAGS.dataset == 'celebA':
        opts = configs.config_celebA
    elif FLAGS.dataset == 'mnist':
        opts = configs.config_mnist
    elif FLAGS.dataset == 'dsprites':
        opts = configs.config_dsprites
    else:
        assert False, 'Unknown experiment configuration'

    opts['zdim'] = FLAGS.zdim
    opts['work_dir'] = FLAGS.work_dir
    opts['lambda'] = FLAGS.wae_lambda

    if opts['verbose']:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    utils.create_dir(opts['work_dir'])
    utils.create_dir(os.path.join(opts['work_dir'],
                     'checkpoints'))
    # Dumping all the configs to the text file
    with utils.o_gfile((opts['work_dir'], 'params.txt'), 'w') as text:
        text.write('Parameters:\n')
        for key in opts:
            text.write('%s : %s\n' % (key, opts[key]))

    # Loading the dataset

    data = DataHandler(opts)
    assert data.num_points >= opts['batch_size'], 'Training set too small'

    # Training WAE

    wae = WAE(opts)
    wae.train(data)

main()
