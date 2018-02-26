# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)
"""This class helps to handle the data.

"""

import os
import random
import logging
import tensorflow as tf
import numpy as np
from six.moves import cPickle
import utils
import PIL
from utils import ArraySaver
from PIL import Image
import sys

datashapes = {}
datashapes['mnist'] = [28, 28, 1]
datashapes['cifar10'] = [32, 32, 3]
datashapes['celebA'] = [64, 64, 3]
datashapes['grassli'] = [64, 64, 3]
datashapes['dsprites'] = [64, 64, 1]

def _data_dir(opts):
    if opts['data_dir'].startswith("/"):
        return opts['data_dir']
    else:
        return os.path.join('./', opts['data_dir'])

def load_cifar_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.

    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.

    # Returns
        A tuple `(data, labels)`.
    """
    f = utils.o_gfile(fpath, 'rb')
    if sys.version_info < (3,):
        d = cPickle.load(f)
    else:
        d = cPickle.load(f, encoding='bytes')
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf8')] = v
        d = d_decoded
    f.close()
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels

def transform_mnist(pic, mode='n'):
    """Take an MNIST picture normalized into [0, 1] and transform
        it according to the mode:
        n   -   noise
        i   -   colour invert
        s*  -   shift
    """
    pic = np.copy(pic)
    if mode == 'n':
        noise = np.random.randn(28, 28, 1)
        return np.clip(pic + 0.25 * noise, 0, 1)
    elif mode == 'i':
        return 1. - pic
    pixels = 3 + np.random.randint(5)
    if mode == 'sl':
        pic[:, :-pixels] = pic[:, pixels:] + 0.0
        pic[:, -pixels:] = 0.
    elif mode == 'sr':
        pic[:, pixels:] = pic[:, :-pixels] + 0.0
        pic[:, :pixels] = 0.
    elif mode == 'sd':
        pic[pixels:, :] = pic[:-pixels, :] + 0.0
        pic[:pixels, :] = 0.
    elif mode == 'su':
        pic[:-pixels, :] = pic[pixels:, :] + 0.0
        pic[-pixels:, :] = 0.
    return pic


class Data(object):
    """
    If the dataset can be quickly loaded to memory self.X will contain np.ndarray
    Otherwise we will be reading files as we train. In this case self.X is a structure:
        self.X.paths        list of paths to the files containing pictures
        self.X.dict_loaded  dictionary of (key, val), where key is the index of the
                            already loaded datapoint and val is the corresponding index
                            in self.X.loaded
        self.X.loaded       list containing already loaded pictures
    """
    def __init__(self, opts, X, paths=None, dict_loaded=None, loaded=None):
        """
        X is either np.ndarray or paths
        """
        data_dir = _data_dir(opts)
        self.X = None
        self.normalize = opts['input_normalize_sym']
        self.paths = None
        self.dict_loaded = None
        self.loaded = None
        if isinstance(X, np.ndarray):
            self.X = X
            self.shape = X.shape
        else:
            assert isinstance(data_dir, str), 'Data directory not provided'
            assert paths is not None and len(paths) > 0, 'No paths provided for the data'
            self.data_dir = data_dir
            self.paths = paths[:]
            self.dict_loaded = {} if dict_loaded is None else dict_loaded
            self.loaded = [] if loaded is None else loaded
            self.crop_style = opts['celebA_crop']
            self.dataset_name = opts['dataset']
            self.shape = (len(self.paths), None, None, None)

    def __len__(self):
        if isinstance(self.X, np.ndarray):
            return len(self.X)
        else:
            # Our dataset was too large to fit in the memory
            return len(self.paths)

    def drop_loaded(self):
        if not isinstance(self.X, np.ndarray):
            self.dict_loaded = {}
            self.loaded = []

    def __getitem__(self, key):
        if isinstance(self.X, np.ndarray):
            return self.X[key]
        else:
            # Our dataset was too large to fit in the memory
            if isinstance(key, int):
                keys = [key]
            elif isinstance(key, list):
                keys = key
            elif isinstance(key, np.ndarray):
                keys = list(key)
            elif isinstance(key, slice):
                start = key.start
                stop = key.stop
                step = key.step
                start = start if start is not None else 0
                if start < 0:
                    start += len(self.paths)
                stop = stop if stop is not None else len(self.paths) - 1
                if stop < 0:
                    stop += len(self.paths)
                step = step if step is not None else 1
                keys = range(start, stop, step)
            else:
                print type(key)
                raise Exception('This type of indexing yet not supported for the dataset')
            res = []
            new_keys = []
            new_points = []
            for key in keys:
                if key in self.dict_loaded:
                    idx = self.dict_loaded[key]
                    res.append(self.loaded[idx])
                else:
                    if self.dataset_name == 'celebA':
                        point = self._read_celeba_image(self.data_dir, self.paths[key])
                    else:
                        raise Exception('Disc read for this dataset not implemented yet...')
                    if self.normalize:
                        point = (point - 0.5) * 2.
                    res.append(point)
                    new_points.append(point)
                    new_keys.append(key)
            n = len(self.loaded)
            cnt = 0
            for key in new_keys:
                self.dict_loaded[key] = n + cnt
                cnt += 1
            self.loaded.extend(new_points)
            return np.array(res)

    def _read_celeba_image(self, data_dir, filename):
        width = 178
        height = 218
        new_width = 140
        new_height = 140
        im = Image.open(utils.o_gfile((data_dir, filename), 'rb'))
        if self.crop_style == 'closecrop':
            # This method was used in DCGAN, pytorch-gan-collection, AVB, ...
            left = (width - new_width) / 2
            top = (height - new_height) / 2
            right = (width + new_width) / 2
            bottom = (height + new_height)/2
            im = im.crop((left, top, right, bottom))
            im = im.resize((64, 64), PIL.Image.ANTIALIAS)
        elif self.crop_style == 'resizecrop':
            # This method was used in ALI, AGE, ...
            im = im.resize((64, 78), PIL.Image.ANTIALIAS)
            im = im.crop((0, 7, 64, 64 + 7))
        else:
            raise Exception('Unknown crop style specified')
        return np.array(im).reshape(64, 64, 3) / 255.

class DataHandler(object):
    """A class storing and manipulating the dataset.

    In this code we asume a data point is a 3-dimensional array, for
    instance a 28*28 grayscale picture would correspond to (28,28,1),
    a 16*16 picture of 3 channels corresponds to (16,16,3) and a 2d point
    corresponds to (2,1,1). The shape is contained in self.data_shape
    """


    def __init__(self, opts):
        self.data_shape = None
        self.num_points = None
        self.data = None
        self.test_data = None
        self.labels = None
        self.test_labels = None
        self._load_data(opts)

    def _load_data(self, opts):
        """Load a dataset and fill all the necessary variables.

        """
        if opts['dataset'] == 'mnist':
            self._load_mnist(opts)
        elif opts['dataset'] == 'dsprites':
            self._load_dsprites(opts)
        elif opts['dataset'] == 'mnist_mod':
            self._load_mnist(opts, modified=True)
        elif opts['dataset'] == 'zalando':
            self._load_mnist(opts, zalando=True)
        elif opts['dataset'] == 'mnist3':
            self._load_mnist3(opts)
        elif opts['dataset'] == 'gmm':
            self._load_gmm(opts)
        elif opts['dataset'] == 'circle_gmm':
            self._load_mog(opts)
        elif opts['dataset'] == 'guitars':
            self._load_guitars(opts)
        elif opts['dataset'] == 'cifar10':
            self._load_cifar(opts)
        elif opts['dataset'] == 'celebA':
            self._load_celebA(opts)
        elif opts['dataset'] == 'grassli':
            self._load_grassli(opts)
        else:
            raise ValueError('Unknown %s' % opts['dataset'])

        sym_applicable = ['mnist',
                          'dsprites',
                          'mnist3',
                          'guitars',
                          'cifar10',
                          'celebA',
                          'grassli']

        if opts['input_normalize_sym'] and opts['dataset'] not in sym_applicable:
            raise Exception('Can not normalyze this dataset')

        if opts['input_normalize_sym'] and opts['dataset'] in sym_applicable:
            # Normalize data to [-1, 1]
            if isinstance(self.data.X, np.ndarray):
                self.data.X = (self.data.X - 0.5) * 2.
                self.test_data.X = (self.test_data.X - 0.5) * 2.
            # Else we will normalyze while reading from disk


    def _load_mog(self, opts):
        """Sample data from the mixture of Gaussians on circle.

        """

        # Only use this setting in dimension 2
        assert opts['toy_dataset_dim'] == 2

        # First we choose parameters of gmm and thus seed
        radius = opts['gmm_max_val']
        modes_num = opts["gmm_modes_num"]
        np.random.seed(opts["random_seed"])

        thetas = np.linspace(0, 2 * np.pi, modes_num)
        mixture_means = np.stack((radius * np.sin(thetas), radius * np.cos(thetas)), axis=1)
        mixture_variance = 0.01

        # Now we sample points, for that we unseed
        np.random.seed()
        num = opts['toy_dataset_size']
        X = np.zeros((num, opts['toy_dataset_dim'], 1, 1))
        for idx in xrange(num):
            comp_id = np.random.randint(modes_num)
            mean = mixture_means[comp_id]
            cov = mixture_variance * np.identity(opts["toy_dataset_dim"])
            X[idx, :, 0, 0] = np.random.multivariate_normal(mean, cov, 1)

        self.data_shape = (opts['toy_dataset_dim'], 1, 1)
        self.data = Data(opts, X)
        self.num_points = len(X)

    def _load_gmm(self, opts):
        """Sample data from the mixture of Gaussians.

        """

        logging.debug('Loading GMM dataset...')
        # First we choose parameters of gmm and thus seed
        modes_num = opts["gmm_modes_num"]
        np.random.seed(opts["random_seed"])
        max_val = opts['gmm_max_val']
        mixture_means = np.random.uniform(
            low=-max_val, high=max_val,
            size=(modes_num, opts['toy_dataset_dim']))

        def variance_factor(num, dim):
            if num == 1: return 3 ** (2. / dim)
            if num == 2: return 3 ** (2. / dim)
            if num == 3: return 8 ** (2. / dim)
            if num == 4: return 20 ** (2. / dim)
            if num == 5: return 10 ** (2. / dim)
            return num ** 2.0 * 3

        mixture_variance = \
                max_val / variance_factor(modes_num, opts['toy_dataset_dim'])

        # Now we sample points, for that we unseed
        np.random.seed()
        num = opts['toy_dataset_size']
        X = np.zeros((num, opts['toy_dataset_dim'], 1, 1))
        for idx in xrange(num):
            comp_id = np.random.randint(modes_num)
            mean = mixture_means[comp_id]
            cov = mixture_variance * np.identity(opts["toy_dataset_dim"])
            X[idx, :, 0, 0] = np.random.multivariate_normal(mean, cov, 1)

        self.data_shape = (opts['toy_dataset_dim'], 1, 1)
        self.data = Data(opts, X)
        self.num_points = len(X)

        logging.debug('Loading GMM dataset done!')

    def _load_guitars(self, opts):
        """Load data from Thomann files.

        """
        logging.debug('Loading Guitars dataset')
        data_dir = os.path.join('./', 'thomann')
        X = None
        files = utils.listdir(data_dir)
        pics = []
        for f in sorted(files):
            if '.jpg' in f and f[0] != '.':
                im = Image.open(utils.o_gfile((data_dir, f), 'rb'))
                res = np.array(im.getdata()).reshape(128, 128, 3)
                pics.append(res)
        X = np.array(pics)

        seed = 123
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed()

        self.data_shape = (128, 128, 3)
        self.data = Data(opts, X/255.)
        self.num_points = len(X)

        logging.debug('Loading Done.')

    def _load_dsprites(self, opts):
        """Load data from dsprites dataset

        """
        logging.debug('Loading dsprites')
        data_dir = _data_dir(opts)
        data_file = os.path.join(data_dir, 'dsprites.npz')
        X = np.load(data_file)['imgs']
        X = X[:, :, :, None]

        seed = 123
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed()

        self.data_shape = (64, 64, 1)
        test_size = 10000

        self.data = Data(opts, X[:-test_size])
        self.test_data = Data(opts, X[-test_size:])
        self.num_points = len(self.data)

        logging.debug('Loading Done.')

    def _load_mnist(self, opts, zalando=False, modified=False):
        """Load data from MNIST or ZALANDO files.

        """
        if zalando:
            logging.debug('Loading Fashion MNIST')
        elif modified:
            logging.debug('Loading modified MNIST')
        else:
            logging.debug('Loading MNIST')
        data_dir = _data_dir(opts)
        # pylint: disable=invalid-name
        # Let us use all the bad variable names!
        tr_X = None
        tr_Y = None
        te_X = None
        te_Y = None

        with utils.o_gfile((data_dir, 'train-images-idx3-ubyte'), 'rb') as fd:
            loaded = np.frombuffer(fd.read(), dtype=np.uint8)
            tr_X = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

        with utils.o_gfile((data_dir, 'train-labels-idx1-ubyte'), 'rb') as fd:
            loaded = np.frombuffer(fd.read(), dtype=np.uint8)
            tr_Y = loaded[8:].reshape((60000)).astype(np.int)

        with utils.o_gfile((data_dir, 't10k-images-idx3-ubyte'), 'rb') as fd:
            loaded = np.frombuffer(fd.read(), dtype=np.uint8)
            te_X = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        with utils.o_gfile((data_dir, 't10k-labels-idx1-ubyte'), 'rb') as fd:
            loaded = np.frombuffer(fd.read(), dtype=np.uint8)
            te_Y = loaded[8:].reshape((10000)).astype(np.int)

        tr_Y = np.asarray(tr_Y)
        te_Y = np.asarray(te_Y)

        X = np.concatenate((tr_X, te_X), axis=0)
        y = np.concatenate((tr_Y, te_Y), axis=0)
        X = X / 255.

        seed = 123
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)
        np.random.seed()

        self.data_shape = (28, 28, 1)
        test_size = 10000

        if modified:
            self.original_mnist = X
            n = opts['toy_dataset_size']
            n += test_size
            points = []
            labels = []
            for _ in xrange(n):
                idx = np.random.randint(len(X))
                point = X[idx]
                modes = ['n', 'i', 'sl', 'sr', 'su', 'sd']
                mode = modes[np.random.randint(len(modes))]
                point = transform_mnist(point, mode)
                points.append(point)
                labels.append(y[idx])
            X = np.array(points)
            y = np.array(y)
        self.data = Data(opts, X[:-test_size])
        self.test_data = Data(opts, X[-test_size:])
        self.labels = y[:-test_size]
        self.test_labels = y[-test_size:]
        self.num_points = len(self.data)

        logging.debug('Loading Done.')

    def _load_mnist3(self, opts):
        """Load data from MNIST files.

        """
        logging.debug('Loading 3-digit MNIST')
        data_dir = _data_dir(opts)
        # pylint: disable=invalid-name
        # Let us use all the bad variable names!
        tr_X = None
        tr_Y = None
        te_X = None
        te_Y = None

        with utils.o_gfile((data_dir, 'train-images-idx3-ubyte'), 'rb') as fd:
            loaded = np.frombuffer(fd.read(), dtype=np.uint8)
            tr_X = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

        with utils.o_gfile((data_dir, 'train-labels-idx1-ubyte'), 'rb') as fd:
            loaded = np.frombuffer(fd.read(), dtype=np.uint8)
            tr_Y = loaded[8:].reshape((60000)).astype(np.int)

        with utils.o_gfile((data_dir, 't10k-images-idx3-ubyte'), 'rb') as fd:
            loaded = np.frombuffer(fd.read(), dtype=np.uint8)
            te_X = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        with utils.o_gfile((data_dir, 't10k-labels-idx1-ubyte'), 'rb') as fd:
            loaded = np.frombuffer(fd.read(), dtype=np.uint8)
            te_Y = loaded[8:].reshape((10000)).astype(np.int)

        tr_Y = np.asarray(tr_Y)
        te_Y = np.asarray(te_Y)

        X = np.concatenate((tr_X, te_X), axis=0)
        y = np.concatenate((tr_Y, te_Y), axis=0)

        num = opts['mnist3_dataset_size']
        ids = np.random.choice(len(X), (num, 3), replace=True)
        if opts['mnist3_to_channels']:
            # Concatenate 3 digits ito 3 channels
            X3 = np.zeros((num, 28, 28, 3))
            y3 = np.zeros(num)
            for idx, _id in enumerate(ids):
                X3[idx, :, :, 0] = np.squeeze(X[_id[0]], axis=2)
                X3[idx, :, :, 1] = np.squeeze(X[_id[1]], axis=2)
                X3[idx, :, :, 2] = np.squeeze(X[_id[2]], axis=2)
                y3[idx] = y[_id[0]] * 100 + y[_id[1]] * 10 + y[_id[2]]
            self.data_shape = (28, 28, 3)
        else:
            # Concatenate 3 digits in width
            X3 = np.zeros((num, 28, 3 * 28, 1))
            y3 = np.zeros(num)
            for idx, _id in enumerate(ids):
                X3[idx, :, 0:28, 0] = np.squeeze(X[_id[0]], axis=2)
                X3[idx, :, 28:56, 0] = np.squeeze(X[_id[1]], axis=2)
                X3[idx, :, 56:84, 0] = np.squeeze(X[_id[2]], axis=2)
                y3[idx] = y[_id[0]] * 100 + y[_id[1]] * 10 + y[_id[2]]
            self.data_shape = (28, 28 * 3, 1)

        self.data = Data(opts, X3/255.)
        y3 = y3.astype(int)
        self.labels = y3
        self.num_points = num

        logging.debug('Training set JS=%.4f' % utils.js_div_uniform(y3))
        logging.debug('Loading Done.')

    def _load_cifar(self, opts):
        """Load CIFAR10

        """
        logging.debug('Loading CIFAR10 dataset')

        num_train_samples = 50000
        data_dir = _data_dir(opts)
        x_train = np.zeros((num_train_samples, 3, 32, 32), dtype='uint8')
        y_train = np.zeros((num_train_samples,), dtype='uint8')

        for i in range(1, 6):
            fpath = os.path.join(data_dir, 'data_batch_' + str(i))
            data, labels = load_cifar_batch(fpath)
            x_train[(i - 1) * 10000: i * 10000, :, :, :] = data
            y_train[(i - 1) * 10000: i * 10000] = labels

        fpath = os.path.join(data_dir, 'test_batch')
        x_test, y_test = load_cifar_batch(fpath)

        y_train = np.reshape(y_train, (len(y_train), 1))
        y_test = np.reshape(y_test, (len(y_test), 1))
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

        X = np.vstack([x_train, x_test])
        X = X/255.
        y = np.vstack([y_train, y_test])

        seed = 123
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)
        np.random.seed()

        self.data_shape = (32, 32, 3)

        self.data = Data(opts, X[:-1000])
        self.test_data = Data(opts, X[-1000:])
        self.labels = y[:-1000]
        self.test_labels = y[-1000:]
        self.num_points = len(self.data)

        logging.debug('Loading Done.')

    def _load_celebA(self, opts):
        """Load CelebA
        """
        logging.debug('Loading CelebA dataset')

        num_samples = 202599

        datapoint_ids = range(1, num_samples + 1)
        paths = ['%.6d.jpg' % i for i in xrange(1, num_samples + 1)]
        seed = 123
        random.seed(seed)
        random.shuffle(paths)
        random.shuffle(datapoint_ids)
        random.seed()

        saver = ArraySaver('disk', workdir=opts['work_dir'])
        saver.save('shuffled_training_ids', datapoint_ids)

        self.data_shape = (64, 64, 3)
        test_size = 512
        self.data = Data(opts, None, paths[:-test_size])
        self.test_data = Data(opts, None, paths[-test_size:])
        self.num_points = num_samples - test_size
        self.labels = np.array(self.num_points * [0])
        self.test_labels = np.array(test_size * [0])

        logging.debug('Loading Done.')

    def _load_grassli(self, opts):
        """Load grassli

        """
        logging.debug('Loading grassli dataset')

        data_dir = _data_dir(opts)
        X = np.load(utils.o_gfile((data_dir, 'grassli.npy'), 'rb')) / 255.

        seed = 123
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.seed()

        self.data_shape = (64, 64, 3)
        test_size = 5000

        self.data = Data(opts, X[:-test_size])
        self.test_data = Data(opts, X[-test_size:])
        self.num_points = len(self.data)

        logging.debug('Loading Done.')
