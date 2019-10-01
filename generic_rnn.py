from utils import psnr
import torch
from torch import nn
from os.path import join
from datetime import datetime
import numpy as np
import os


class Simple_RNN(nn.Module):
    def __init__(self, config):
        super(Simple_RNN, self).__init__()
        self.config = config
        self.n_input = int(self.config['n_features'] / self.config['compression_factor'])
        self.batch_size = self.config['batch_size'] * self.config['scale']
        self.dtype = torch.float32
        self.device = torch.device('cuda')
        self.writer = open(join(config['log_folder'], 'output.txt'), 'w')
        self.compression = nn.Linear(self.config['n_features'], self.n_input).to(device=self.device)
        self.rnn = nn.GRU(input_size=self.n_input, hidden_size=self.config['n_hidden'], num_layers=3,
                          batch_first=False).to(device=self.device)
        self.reconstruction = nn.Linear(self.config['n_hidden'], self.config['n_features']).to(device=self.device)

    def forward(self, input):
        compressed = self.compression(input)
        hidden, _ = self.rnn(compressed, None)
        out = self.reconstruction(hidden)
        return out

    @staticmethod
    def psnr(ref, reconstructed):
        reconstructed[reconstructed < 0] = 0
        reconstructed[reconstructed > 255] = 255
        reconstructed = reconstructed.int().float()
        ref = ref.int().float()
        mse = torch.mean((ref - reconstructed) ** 2)
        if mse == 0:
            return 100
        return 20 * torch.log10(255 / torch.sqrt(mse))

    def train(self, data_loader):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        for iter in range(1, self.config['n_iter'] + 1):
            batch = data_loader.load_batch_train(self.batch_size)
            input = torch.tensor(batch, dtype=self.dtype, device=self.device)
            output = self.forward(input)
            loss = torch.mean((output - input) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iter % self.config['display_each'] == 0:
                to_print = 'iter {}, train_loss {}'.format(iter, loss)
                print(to_print)
                self.writer.write(to_print + '\n')
                with torch.no_grad():
                    eval_psnr = []
                    eval_loss = []
                    while True:
                        eval_batch = data_loader.load_batch_eval(self.batch_size)
                        if len(eval_batch) == 0:
                            break
                        eval_input = torch.tensor(eval_batch, dtype=self.dtype, device=self.device)
                        z_hat = self.forward(eval_input)
                        eval_loss.append(torch.mean((eval_input - z_hat) ** 2))
                        eval_psnr.append(psnr(eval_input, z_hat))
                    to_print = 'eval_loss: {} psnr: {}'.format(sum(eval_loss) / len(eval_loss),
                                                               sum(eval_psnr) / len(eval_psnr))
                    print(to_print)
                    self.writer.write(to_print + '\n')
        # finish training, now testing
        with torch.no_grad():
            test_loss = []
            test_psnr = []
            while True:
                test_batch = data_loader.load_batch_test(self.batch_size)
                if len(test_batch) == 0:
                    break
                test_input = torch.tensor(test_batch, dtype=self.dtype, device=self.device)
                test_z_hat = self.forward(test_input)
                test_loss.append(torch.mean((test_input - test_z_hat) ** 2))
                test_psnr.append(psnr(test_input, test_z_hat))
            to_print = 'test_loss: {}, psnr: {}'.format(sum(test_loss) / len(test_loss),
                                                        sum(test_psnr) / len(test_psnr))
            print(to_print)
            self.writer.write(to_print + '\n')
            data_dir = join(self.config['result_path'], str(datetime.now()))
            os.makedirs(data_dir)
            file_name = join(data_dir, 'final.npy')
            save_npy = test_z_hat.data.cpu().reshape(
                [test_z_hat.size()[0], test_z_hat.size()[1], self.config['width'], self.config['height']])
            np.save(file_name, save_npy)
            self.writer.close()
