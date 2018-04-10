import os
import sys
import logging
import argparse
import configs
from wae import WAE
import improved_wae
from datahandler import DataHandler
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--exp", default='mnist_small',
                    help='dataset [mnist/celebA/dsprites]')
parser.add_argument("--zdim",
                    help='dimensionality of the latent space',
                    type=int)
parser.add_argument("--lr",
                    help='ae learning rate',
                    type=float)
parser.add_argument("--w_aef",
                    help='weight of ae fixedpoint cost',
                    type=float)
parser.add_argument("--z_test",
                    help='method of choice for verifying Pz=Qz [mmd/gan]')
parser.add_argument("--pz",
                    help='Prior latent distribution [normal/sphere/uniform]')
parser.add_argument("--wae_lambda", help='WAE regularizer', type=int)
parser.add_argument("--work_dir")
parser.add_argument("--lambda_schedule",
                    help='constant or adaptive')
parser.add_argument("--enc_noise",
                    help="type of encoder noise:"\
                         " 'deterministic': no noise whatsoever,"\
                         " 'gaussian': gaussian encoder,"\
                         " 'implicit': implicit encoder,"\
                         " 'add_noise': add noise before feeding "\
                         "to deterministic encoder")
parser.add_argument("--mode", default='train',
                    help='train or test')
parser.add_argument("--checkpoint",
                    help='full path to the checkpoint file without extension')

FLAGS = parser.parse_args()

def main():

    if FLAGS.exp == 'celebA':
        opts = configs.config_celebA
    elif FLAGS.exp == 'celebA_small':
        opts = configs.config_celebA_small
    elif FLAGS.exp == 'mnist':
        opts = configs.config_mnist
    elif FLAGS.exp == 'mnist_small':
        opts = configs.config_mnist_small
    elif FLAGS.exp == 'dsprites':
        opts = configs.config_dsprites
    elif FLAGS.exp == 'grassli':
        opts = configs.config_grassli
    elif FLAGS.exp == 'grassli_small':
        opts = configs.config_grassli_small
    else:
        assert False, 'Unknown experiment configuration'

    opts['mode'] = FLAGS.mode
    if opts['mode'] == 'test':
        assert FLAGS.checkpoint is not None, 'Checkpoint must be provided'
        opts['checkpoint'] = FLAGS.checkpoint

    if FLAGS.zdim is not None:
        opts['zdim'] = FLAGS.zdim
    if FLAGS.pz is not None:
        opts['pz'] = FLAGS.pz
    if FLAGS.lr is not None:
        opts['lr'] = FLAGS.lr
    if FLAGS.w_aef is not None:
        opts['w_aef'] = FLAGS.w_aef
    if FLAGS.z_test is not None:
        opts['z_test'] = FLAGS.z_test
    if FLAGS.lambda_schedule is not None:
        opts['lambda_schedule'] = FLAGS.lambda_schedule
    if FLAGS.work_dir is not None:
        opts['work_dir'] = FLAGS.work_dir
    if FLAGS.wae_lambda is not None:
        opts['lambda'] = FLAGS.wae_lambda
    if FLAGS.enc_noise is not None:
        opts['e_noise'] = FLAGS.enc_noise

    if opts['verbose']:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    utils.create_dir(opts['work_dir'])
    utils.create_dir(os.path.join(opts['work_dir'],
                     'checkpoints'))

    if opts['e_noise'] == 'gaussian' and opts['pz'] != 'normal':
        assert False, 'Gaussian encoders compatible only with Gaussian prior'
        return

    # Dumping all the configs to the text file
    with utils.o_gfile((opts['work_dir'], 'params.txt'), 'w') as text:
        text.write('Parameters:\n')
        for key in opts:
            text.write('%s : %s\n' % (key, opts[key]))

    # Loading the dataset
    data = DataHandler(opts)
    assert data.num_points >= opts['batch_size'], 'Training set too small'

    if opts['mode'] == 'train':

        # Creating WAE model
        wae = WAE(opts)

        # Training WAE
        wae.train(data)

    elif opts['mode'] == 'test':

        # Do something else
        improved_wae.test(opts)

main()
