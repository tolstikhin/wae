import os
import sys
import logging
import argparse
import configs
from wae import WAE
from datahandler import DataHandler
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='mnist_small',
                    help='dataset [mnist/celebA/dsprites]')
parser.add_argument("--zdim",
                    help='dimensionality of the latent space',
                    type=int)
parser.add_argument("--z_test",
                    help='method of choice for verifying Pz=Qz [mmd/gan]')
parser.add_argument("--wae_lambda", help='WAE regularizer', type=int)
parser.add_argument("--work_dir")
parser.add_argument("--enc_noise",
                    help="type of encoder noise:"\
                         " 'deterministic': no noise whatsoever,"\
                         " 'random': gaussian encoder,"\
                         " 'add_noise': add noise before feeding "\
                         "to deterministic encoder")

FLAGS = parser.parse_args()

def main():

    if FLAGS.dataset == 'celebA':
        opts = configs.config_celebA
    elif FLAGS.dataset == 'celebA_small':
        opts = configs.config_celebA_small
    elif FLAGS.dataset == 'mnist':
        opts = configs.config_mnist
    elif FLAGS.dataset == 'mnist_small':
        opts = configs.config_mnist_small
    elif FLAGS.dataset == 'dsprites':
        opts = configs.config_dsprites
    else:
        assert False, 'Unknown experiment configuration'

    if FLAGS.zdim:
        opts['zdim'] = FLAGS.zdim
    if FLAGS.z_test:
        opts['z_test'] = FLAGS.z_test
    if FLAGS.work_dir:
        opts['work_dir'] = FLAGS.work_dir
    if FLAGS.wae_lambda:
        opts['lambda'] = FLAGS.wae_lambda
    if FLAGS.enc_noise:
        if FLAGS.enc_noise == 'deterministic':
            opts['e_is_random'] = False
            opts['e_add_noise'] = False
        elif FLAGS.enc_noise == 'random':
            opts['e_is_random'] = True
            opts['e_add_noise'] = False
        elif FLAGS.enc_noise == 'add_noise':
            opts['e_is_random'] = False
            opts['e_add_noise'] = True
    if opts['e_is_random'] and opts['e_add_noise']:
        assert False, 'can not combine random encoder with additive noise'

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
