import copy

# CelebA config from ICLR paper

config_celebA = {}
config_celebA['dataset'] = 'celebA'
config_celebA['verbose'] = True
config_celebA['save_every_epoch'] = 20
config_celebA['print_every'] = 500
config_celebA['work_dir'] = 'results_celeba'
config_celebA['plot_num_pics'] = 30
config_celebA['plot_num_cols'] = 5

config_celebA['input_normalize_sym'] = True
config_celebA['data_dir'] = 'celebA/datasets/celeba/img_align_celeba'
config_celebA['celebA_crop'] = 'closecrop' # closecrop, resizecrop

config_celebA['optimizer'] = 'adam' # adam, sgd
config_celebA['adam_beta1'] = 0.5
config_celebA['lr'] = 0.0005 #0.001 for WAE-MMD and 0.0003 for WAE-GAN
config_celebA['lr_adv'] = 0.001
config_celebA['lr_schedule'] = 'plateau' #manual, plateau, or a number
config_celebA['batch_size'] = 100
config_celebA['epoch_num'] = 100
config_celebA['init_std'] = 0.0099999
config_celebA['init_bias'] = 0.0
config_celebA['batch_norm'] = True
config_celebA['batch_norm_eps'] = 1e-05
config_celebA['batch_norm_decay'] = 0.9
config_celebA['conv_filters_dim'] = 5

config_celebA['e_pretrain'] = True
config_celebA['e_pretrain_sample_size'] = 256
config_celebA['e_noise'] = 'add_noise'
config_celebA['e_num_filters'] = 1024
config_celebA['e_num_layers'] = 4
config_celebA['e_arch'] = 'dcgan' # mlp, dcgan, ali

config_celebA['g_num_filters'] = 1024
config_celebA['g_num_layers'] = 4
config_celebA['g_arch'] = 'dcgan_mod' # mlp, dcgan, dcgan_mod, ali

config_celebA['gan_p_trick'] = True
config_celebA['d_num_layers'] = 4
config_celebA['d_num_filters'] = 1024

config_celebA['zdim'] = 64
config_celebA['pz'] = 'normal' # uniform, normal, sphere
config_celebA['cost'] = 'l2sq' #l2, l2sq, l1
config_celebA['pz_scale'] = 1.
config_celebA['z_test'] = 'mmd'
config_celebA['mmd_kernel'] = 'IMQ' # RBF, IMQ
config_celebA['lambda'] = 100.
config_celebA['lambda_schedule'] = 'constant'

# MNIST config from ICLR paper

config_mnist = {}
config_mnist['dataset'] = 'mnist'
config_mnist['verbose'] = True
config_mnist['save_every_epoch'] = 20
config_mnist['print_every'] = 50
config_mnist['work_dir'] = 'results_mnist'
config_mnist['plot_num_pics'] = 400
config_mnist['plot_num_cols'] = 20

config_mnist['input_normalize_sym'] = False
config_mnist['data_dir'] = 'mnist'

config_mnist['optimizer'] = 'adam' # adam, sgd
config_mnist['adam_beta1'] = 0.5
config_mnist['lr'] = 0.001
config_mnist['lr_adv'] = 0.0005
config_mnist['lr_schedule'] = 'manual' #manual, plateau, or a number
config_mnist['batch_size'] = 100
config_mnist['epoch_num'] = 100
config_mnist['init_std'] = 0.0099999
config_mnist['init_bias'] = 0.0
config_mnist['batch_norm'] = True
config_mnist['batch_norm_eps'] = 1e-05
config_mnist['batch_norm_decay'] = 0.9
config_mnist['conv_filters_dim'] = 4

config_mnist['e_pretrain'] = True
config_mnist['e_pretrain_sample_size'] = 1000
config_mnist['e_noise'] = 'add_noise'
config_mnist['e_num_filters'] = 1024
config_mnist['e_num_layers'] = 4
config_mnist['e_arch'] = 'dcgan' # mlp, dcgan, ali

config_mnist['g_num_filters'] = 1024
config_mnist['g_num_layers'] = 3
config_mnist['g_arch'] = 'dcgan_mod' # mlp, dcgan, dcgan_mod, ali

config_mnist['gan_p_trick'] = False
config_mnist['d_num_filters'] = 512
config_mnist['d_num_layers'] = 4

config_mnist['zdim'] = 8
config_mnist['pz'] = 'normal' # uniform, normal, sphere
config_mnist['cost'] = 'l2sq' #l2, l2sq, l1
config_mnist['pz_scale'] = 1.
config_mnist['z_test'] = 'mmd'
config_mnist['mmd_kernel'] = 'IMQ' # RBF, IMQ
config_mnist['lambda'] = 10.
config_mnist['lambda_schedule'] = 'constant'

# Toy MNIST experiment
config_mnist_small = copy.deepcopy(config_mnist)
config_mnist_small['z_test'] = 'mmd'
config_mnist_small['g_arch'] = 'mlp'
config_mnist_small['e_arch'] = 'mlp'
config_mnist_small['g_num_layers'] = 3
config_mnist_small['g_num_filters'] = 256
config_mnist_small['e_num_layers'] = 3
config_mnist_small['e_num_filters'] = 256
config_mnist_small['print_every'] = 500
config_mnist_small['save_every_epoch'] = 1
config_mnist_small['lr_schedule'] = 'plateau'
config_mnist_small['epoch_num'] = 5

# Toy celebA experiment
config_celebA_small = copy.deepcopy(config_celebA)
config_celebA_small['zdim'] = 2
config_celebA_small['g_arch'] = 'mlp'
config_celebA_small['e_arch'] = 'mlp'
config_celebA_small['g_num_layers'] = 3
config_celebA_small['g_num_filters'] = 256
config_celebA_small['e_num_layers'] = 3
config_celebA_small['e_num_filters'] = 256
config_celebA_small['print_every'] = 50
config_celebA_small['lr_schedule'] = 'plateau'


# dsprites config 
config_dsprites = {}
config_dsprites['dataset'] = 'dsprites'
config_dsprites['verbose'] = True
config_dsprites['save_every_epoch'] = 20
config_dsprites['print_every'] = 500
config_dsprites['work_dir'] = 'results_dsprites'
config_dsprites['plot_num_pics'] = 400
config_dsprites['plot_num_cols'] = 20

config_dsprites['input_normalize_sym'] = False
config_dsprites['data_dir'] = 'dsprites'

config_dsprites['optimizer'] = 'adam' # adam, sgd
config_dsprites['adam_beta1'] = 0.5
config_dsprites['lr'] = 0.001
config_dsprites['lr_adv'] = 0.0005
config_dsprites['lr_schedule'] = 'plateau' #manual, plateau, or a number
config_dsprites['batch_size'] = 100
config_dsprites['epoch_num'] = 100
config_dsprites['init_std'] = 0.0099999
config_dsprites['init_bias'] = 0.0
config_dsprites['batch_norm'] = True
config_dsprites['batch_norm_eps'] = 1e-05
config_dsprites['batch_norm_decay'] = 0.9
config_dsprites['conv_filters_dim'] = 4

config_dsprites['e_pretrain'] = True
config_dsprites['e_pretrain_sample_size'] = 1000
config_dsprites['e_noise'] = 'add_noise'
config_dsprites['e_num_filters'] = 256
config_dsprites['e_num_layers'] = 4
config_dsprites['e_arch'] = 'dcgan' # mlp, dcgan, ali

config_dsprites['g_num_filters'] = 256
config_dsprites['g_num_layers'] = 4
config_dsprites['g_arch'] = 'dcgan_mod' # mlp, dcgan, dcgan_mod, ali

config_dsprites['gan_p_trick'] = False
config_dsprites['d_num_filters'] = 256
config_dsprites['d_num_layers'] = 4

config_dsprites['zdim'] = 12
config_dsprites['pz'] = 'normal' # uniform, normal, sphere
config_dsprites['cost'] = 'l2sq' #l2, l2sq, l1
config_dsprites['pz_scale'] = 1.
config_dsprites['z_test'] = 'mmd'
config_dsprites['mmd_kernel'] = 'IMQ' # RBF, IMQ
config_dsprites['lambda'] = 10.
config_dsprites['lambda_schedule'] = 'constant'

# grassli config 

config_grassli = {}
config_grassli['dataset'] = 'grassli'
config_grassli['verbose'] = True
config_grassli['save_every_epoch'] = 20
config_grassli['print_every'] = 500
config_grassli['work_dir'] = 'results_grassli'
config_grassli['plot_num_pics'] = 30
config_grassli['plot_num_cols'] = 5

config_grassli['input_normalize_sym'] = True
config_grassli['data_dir'] = 'grassli'

config_grassli['optimizer'] = 'adam' # adam, sgd
config_grassli['adam_beta1'] = 0.5
config_grassli['lr'] = 0.0005 #0.001 for WAE-MMD and 0.0003 for WAE-GAN
config_grassli['lr_adv'] = 0.001
config_grassli['lr_schedule'] = 'manual' #manual, plateau, or a number
config_grassli['batch_size'] = 100
config_grassli['epoch_num'] = 100
config_grassli['init_std'] = 0.0099999
config_grassli['init_bias'] = 0.0
config_grassli['batch_norm'] = True
config_grassli['batch_norm_eps'] = 1e-05
config_grassli['batch_norm_decay'] = 0.9
config_grassli['conv_filters_dim'] = 5

config_grassli['e_pretrain'] = True
config_grassli['e_pretrain_sample_size'] = 256
config_grassli['e_noise'] = 'implicit'
config_grassli['e_num_filters'] = 1024
config_grassli['e_num_layers'] = 4
config_grassli['e_arch'] = 'dcgan' # mlp, dcgan, ali

config_grassli['g_num_filters'] = 1024
config_grassli['g_num_layers'] = 4
config_grassli['g_arch'] = 'dcgan_mod' # mlp, dcgan, dcgan_mod, ali

config_grassli['gan_p_trick'] = True
config_grassli['d_num_layers'] = 4
config_grassli['d_num_filters'] = 1024

config_grassli['zdim'] = 30
config_grassli['pz'] = 'normal' # uniform, normal, sphere
config_grassli['cost'] = 'l2sq' #l2, l2sq, l1
config_grassli['pz_scale'] = 1.
config_grassli['z_test'] = 'mmd'
config_grassli['mmd_kernel'] = 'IMQ' # RBF, IMQ
config_grassli['lambda'] = 100.
config_grassli['lambda_schedule'] = 'constant'

# Toy grassli experiment
config_grassli_small = copy.deepcopy(config_grassli)
config_grassli_small['zdim'] = 2
# config_grassli_small['g_arch'] = 'mlp'
# config_grassli_small['e_arch'] = 'mlp'
config_grassli_small['g_num_layers'] = 2
config_grassli_small['g_num_filters'] = 64
config_grassli_small['e_num_layers'] = 2
config_grassli_small['e_num_filters'] = 64
config_grassli_small['print_every'] = 50
config_grassli_small['lr_schedule'] = 'plateau'
