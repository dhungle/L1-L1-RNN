import torch
from torch import nn
from torch import functional as F
from tensorflow import keras
import numpy as np
from data_loader.data_loader import Moving_MNIST_Loader
import yaml

class Simple_RNN():
    def __init__(self, A_init):
        super(Simple_RNN, self).__init__()
        self.config = config
        self.n_input = 16 * 16/4
        # self.dtype = torch.float32
        # self.device = torch.device('cuda')
        self.batch_size = 32
        self.A = A_init
        self.model = keras.Sequential()
        self.model.add(keras.layers.LSTM(16 * 16, input_shape=(20, 16 * 4)))
        self.model.compile(loss='mse', optimizer = 'adam', metrics = ['mse'])

    def forward(self, input):
        np.swapaxes(input, 0, 1)
        raw_input_reshape = input.reshape([-1, 16*16])
        compressed_input_reshape = raw_input_reshape.dot(self.A.T)
        self.compressed_input = compressed_input_reshape.reshape([-1, 20, 16 * 4])
        print(self.compressed_input.shape)
        self.model.fit(self.compressed_input, input, batch_size=self.batch_size)

rng = np.random.RandomState(seed=2018)
A_init = np.asarray(
    rng.uniform(
        low=-np.sqrt(6.0 / (16 * 4 + 16 * 16)),
        high=np.sqrt(6.0 / (16 * 4 + 16 * 16)),
        size=(16 * 4, 16 * 16)
    ) / 2.0, dtype=np.float32)

CONFIG_PATH = 'configs/frame_reconstruction_configs.yaml'
with open(CONFIG_PATH, 'r') as stream:
    config = yaml.load(stream)
    for key, val in config.items():
        try:
            val = int(val)
        except:
            pass


data_loader = Moving_MNIST_Loader(config['moving_mnist_path'], time_steps=config['time_steps'], load_only=-1,
                                  flatten=True, scale=False)

rnn = Simple_RNN(A_init)
rnn.forward(data_loader.eval)

