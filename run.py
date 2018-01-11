import os
import logging
import configs
from wae import WAE
from datahandler import DataHandler
import utils

def main():

    opts = configs.config_celebA_small

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
