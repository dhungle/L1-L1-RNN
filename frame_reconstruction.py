from data_loader.data_loader import Moving_MNIST_Loader
from unfolded_rnn import Unfolded_RNN
from generic_rnn import Simple_RNN
import numpy as np
from scipy.io import loadmat
import yaml
import argparse
from datetime import datetime
import os
from os.path import join

if __name__ == '__main__':
    CONFIG_PATH = 'configs/frame_reconstruction_configs.yaml'
    with open(CONFIG_PATH, 'r') as stream:
        config = yaml.load(stream)
        for key, val in config.items():
            try:
                val = int(val)
            except:
                pass

    parser = argparse.ArgumentParser(description='Multiple L1 for frame reconstruction')
    parser.add_argument('-m', '--model')
    parser.add_argument('-ld0', '--lambda0')
    parser.add_argument('-ld1', '--lambda1')
    parser.add_argument('-ld2', '--lambda2')
    parser.add_argument('-learn_ld0', '--learn_lambda0')
    parser.add_argument('-learn_ld1', '--learn_lambda1')
    parser.add_argument('-learn_ld2', '--learn_lambda2')

    parser.add_argument('-wd', '--weight_decay')
    parser.add_argument('-lr', '--lr')
    parser.add_argument('-dr', '--lr_decay_rate')
    parser.add_argument('-da', '--lr_decay_after')
    parser.add_argument('-cf', '--compression_factor')
    parser.add_argument('-iter', '--num_iterations')

    args = vars(parser.parse_args())
    if args['model']:
        config['model'] = args['model']
    if args['lambda0']:
        config['lambda0'] = float(args['lambda0'])
    if args['lambda1']:
        config['lambda1'] = float(args['lambda1'])
    if args['lambda2']:
        config['lambda2'] = float(args['lambda2'])
    if args['learn_lambda0']:
        config['learn_lambda0'] = int(args['learn_lambda0'])
    if args['learn_lambda1']:
        config['learn_lambda1'] = int(args['learn_lambda1'])
    if args['learn_lambda2']:
        config['learn_lambda2'] = int(args['learn_lambda2'])

    if args['compression_factor']:
        config['compression_factor'] = int(args['compression_factor'])
    if args['weight_decay']:
        config['weight_decay'] = float(args['weight_decay'])
    if args['lr']:
        config['lr'] = float(args['lr'])
    if args['lr_decay_rate']:
        config['lr_decay_rate'] = float(args['lr_decay_rate'])
    if args['lr_decay_after']:
        config['lr_decay_after'] = float(args['lr_decay_after'])
    if args['num_iterations']:
        config['n_iter'] = int(args['num_iterations'])
    
    config['log_folder'] = join(config['log_path'], str(datetime.now()))
    os.makedirs(config['log_folder'])

    # save configs to file
    with open(join(config['log_folder'], 'configs.yml'), 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style = False)
    
    data_loader = Moving_MNIST_Loader(config['moving_mnist_path'], time_steps = config['time_steps'], load_only=-1,
                                      flatten=True, scale=False)
    D_init = loadmat(config['D_init_file_path'])['dict'].astype('float32')
    n_input = int(config['n_features']/config['compression_factor'])
    rng = np.random.RandomState(seed=2018)
    A_init = np.asarray(
        rng.uniform(
            low=-np.sqrt(6.0 / (n_input + config['n_features'])),
            high=np.sqrt(6.0 / (n_input + config['n_features'])),
            size=(n_input, config['n_features'])
        ) / 2.0, dtype=np.float32)

    if config['model'].lower() == 'generic':
        model = Simple_RNN(config)
    elif config['model'].lower() == 'sista':
        model = Unfolded_RNN(A_init, D_init, config)
    else:
        print('Model has to be either Sista or generic!')
        exit()
    model.train(data_loader)


