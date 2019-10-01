import torch
from utils import psnr
import torch.nn.functional as functional
from os.path import join, isfile
from datetime import datetime
import os
import numpy as np


class Unfolded_RNN(torch.nn.Module):
    def __init__(self, A_initializer, D_initializer, config):
        super(Unfolded_RNN, self).__init__()
        self.config = config
        self.n_input = int(self.config['n_features'] / self.config['compression_factor'])
        self.dtype = torch.float32
        self.device = torch.device('cuda')
        self.batch_size = self.config['batch_size'] * self.config['scale']
        self.writer = open(join(config['log_folder'], 'output.txt'), 'w')
        self.A_range = [A_initializer.min(), A_initializer.max()]
        # model parameters
        self.D = torch.tensor(D_initializer, device=self.device, dtype=self.dtype, requires_grad=True)
        self.A = torch.tensor(A_initializer, device=self.device, dtype=self.dtype, requires_grad=True)
        self.alpha = torch.tensor(self.config['alpha'], device=self.device, dtype=self.dtype, requires_grad=True)
        self.lambda0 = torch.tensor(self.config['lambda0'], device=self.device, dtype=self.dtype, requires_grad=True)
        self.lambda1 = torch.tensor(self.config['lambda1'], device=self.device, dtype=self.dtype, requires_grad=True)
        self.lambda2 = torch.tensor(self.config['lambda2'], device=self.device, dtype=self.dtype, requires_grad=True)  # for l2 term
        self.h_0 = torch.zeros((self.batch_size, self.config['n_hidden']), device=self.device, dtype=self.dtype,
                               requires_grad=True)
        self.optimized_params = [self.D, self.A, self.alpha, self.h_0]
        if self.config['learn_lambda0']:
            self.optimized_params.append(self.lambda0)
        if self.config['learn_lambda1']:
            self.G = torch.eye(self.config['n_hidden'], device=self.device, dtype=self.dtype, requires_grad=True)
            self.optimized_params += [self.lambda1, self.G]
        if self.config['learn_lambda2']:
            self.F = torch.eye(self.config['n_features'], device=self.device, dtype=self.dtype, requires_grad=True)
            self.optimized_params += [self.F, self.lambda2]

    def soft_l1(self, x, b):
        out = torch.sign(x) * functional.relu(torch.abs(x) - b)
        return out

    def soft_l1_l1(self, z, w0, w1, alpha1):
        alpha0 = torch.zeros(alpha1.size(), device=self.device, dtype=self.dtype)
        condition = alpha0 <= alpha1
        alpha0_sorted = torch.where(condition, alpha0, alpha1)
        alpha1_sorted = torch.where(condition, alpha1, alpha0)
        w0_sorted = torch.where(condition, w0, w1)
        w1_sorted = torch.where(condition, w1, w0)

        cond1 = z >= alpha1_sorted + w0_sorted + w1_sorted
        cond2 = z >= alpha1_sorted + w0_sorted - w1_sorted
        cond3 = z >= alpha0_sorted + w0_sorted - w1_sorted
        cond4 = z >= alpha0_sorted - w0_sorted - w1_sorted

        res1 = z - w0_sorted - w1_sorted
        res2 = alpha1_sorted
        res3 = z - w0_sorted + w1_sorted
        res4 = alpha0_sorted
        res5 = z + w0_sorted + w1_sorted
        return torch.where(cond1, res1,
                           torch.where(cond2, res2, torch.where(cond3, res3, torch.where(cond4, res4, res5))))

    def normalize_compression_matrix(self):
        old_range = self.A.data.max() - self.A.data.min() + 1e-6
        new_range = self.A_range[1] - self.A_range[0]
        self.A.data -= self.A.data.min()
        self.A.data *= new_range/old_range
        self.A.data += self.A_range[0]

    def build_graph_l1_l1(self, input):
        At = self.A.t()
        Dt = self.D.t()
        AtA = torch.mm(At, self.A)

        # initialize V
        V = 1.0 / self.alpha * torch.mm(Dt, At)

        # initialize S
        temp = 1. / self.alpha * torch.mm(torch.mm(Dt, AtA), self.D)
        S = torch.eye(self.config['n_hidden'], device=self.device, dtype=self.dtype) - temp

        # initialize W
        W_1 = self.G - torch.mm(temp, self.G)
        # W_k = torch.zeros([self.config['n_hidden'], self.config['n_hidden']], dtype=self.dtype, device=self.device)

        # Hidden layers
        h = []
        h_t_kth_layer = self.h_0

        for t in range(self.config['time_steps']):
            h_t_last_layer = h_t_kth_layer
            # first ISTA step
            h_t_kth_layer = self.soft_l1_l1(torch.mm(h_t_last_layer, W_1.t()) + torch.mm(
                input[t + 1], V.t()), self.lambda0 / self.alpha, self.lambda1 / self.alpha,
                                            torch.mm(h_t_last_layer, self.G))

            # 2-k ISTA steps
            for k in range(1, self.config['K']):
                h_t_kth_layer = self.soft_l1_l1(torch.mm(input[t + 1], V.t()) + \
                                                torch.mm(h_t_kth_layer, S.t()),
                                                self.lambda0 / self.alpha,
                                                self.lambda1 / self.alpha, torch.mm(h_t_last_layer, self.G))
            h.append(h_t_kth_layer)

        self.sparse_code = torch.stack(h)

    def build_graph_l1_l2(self, input):
        At = self.A.t()
        Dt = self.D.t()
        AtA = torch.mm(At, self.A)
        I = torch.eye(self.config['n_features'], device=self.device, dtype=self.dtype)
        P = torch.mm(torch.mm(Dt, self.F), self.D)

        # initialize V
        V = 1.0 / self.alpha * torch.mm(Dt, At)

        # initialize S
        temp = 1. / self.alpha * torch.mm(torch.mm(Dt, AtA + self.lambda2 * I), self.D)
        S = torch.eye(self.config['n_hidden'], device=self.device, dtype=self.dtype) - temp

        # initialize W
        W_1 = (self.alpha + self.lambda2) / self.alpha * P - torch.mm(temp, P)
        W_k = self.lambda2 / self.alpha * P

        # Hidden layers
        h = []
        h_t_kth_layer = self.h_0

        for t in range(self.config['time_steps']):
            h_t_last_layer = h_t_kth_layer
            # first ISTA step
            h_t_kth_layer = self.soft_l1(torch.mm(h_t_last_layer, W_1.t()) + torch.mm(
                input[t + 1], V.t()), self.lambda0 / self.alpha)

            # 2-k ISTA steps
            for k in range(1, self.config['K']):
                h_t_kth_layer = self.soft_l1(torch.mm(h_t_last_layer, W_k.t()) + \
                                             torch.mm(input[t + 1], V.t()) + \
                                             torch.mm(h_t_kth_layer, S.t()),
                                             self.lambda0 / self.alpha)

            h.append(h_t_kth_layer)

        self.sparse_code = torch.stack(h)

    def forward(self, pre_input, raw_input):
        # Compression
        raw_input_reshape = raw_input.view([-1, self.config['n_features']])
        now_input_reshape = raw_input_reshape.mm(self.A.t())
        self.now_input = now_input_reshape.view([self.config['time_steps'], self.batch_size, -1])

        # Sista graph
        input = torch.cat([pre_input, self.now_input])

        if self.config['lambda1'] > 0:  # L1-L1
            self.build_graph_l1_l1(input)
        else:
            self.build_graph_l1_l2(input)
        zeros_count = torch.sum((self.sparse_code == 0).int()).data.float()
        self.sparsity = zeros_count / self.sparse_code.numel()
        # reconstruction
        sparse_code_reshape = self.sparse_code.view([-1, self.config['n_hidden']])
        z_hat_flattened = torch.mm(sparse_code_reshape, self.D.t())
        z_hat = z_hat_flattened.view([self.config['time_steps'], self.batch_size, -1])
        return z_hat

    def compute_loss(self, input, output):
        return torch.mean((input - output) ** 2)

    def train(self, data_loader):
        optimizer = torch.optim.Adam(self.optimized_params, lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        pre_input = torch.zeros([1, self.batch_size, self.n_input], dtype=self.dtype, device=self.device)
        data_dir = join(self.config['result_path'], str(datetime.now()))
        os.makedirs(data_dir)
        for iter in range(1, self.config['n_iter'] + 1):
            batch = data_loader.load_batch_train(self.batch_size)
            raw_input = torch.tensor(batch, dtype=self.dtype, device=self.device)
            z_hat = self.forward(pre_input, raw_input)
            loss = self.compute_loss(raw_input, z_hat)
            if loss > 1e8:
                print('loss exploded')
                self.writer.write('loss exploded')
                self.writer.close()
                exit()
            if iter == 49000:
                np.save('sparse_code.npy', self.sparse_code.data.cpu().numpy())
                np.save('g.npy', self.G.data.cpu().numpy())
                print(123)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # self.normalize_compression_matrix()
            if iter % self.config['display_each'] == 0:
                to_print = 'iter {}, lr: {}, train_loss {}, ld0: {}, ld1: {}, ld2: {}, sparsity: {}, compressed input range: {} to {} '.format(
                    iter, self.config['lr'],
                    loss,
                    self.lambda0,
                    self.lambda1,
                    self.lambda2,
                    self.sparsity,
                    self.now_input.min(),
                    self.now_input.max())
                print(to_print)
                self.writer.write(to_print + '\n')

                # Test on the evaluation set
                eval_loss = []
                eval_psnr = []
                with torch.no_grad():
                    while True:
                        eval_batch = data_loader.load_batch_eval(self.batch_size)
                        if len(eval_batch) == 0:
                            break
                        eval_raw_input = torch.tensor(eval_batch, dtype=self.dtype, device=self.device)
                        pre_input = torch.zeros([1, self.batch_size, self.n_input], dtype=self.dtype, device=self.device)
                        eval_z_hat = self.forward(pre_input, eval_raw_input)
                        eval_loss.append(self.compute_loss(eval_raw_input, eval_z_hat))
                        eval_psnr.append(psnr(eval_raw_input, eval_z_hat))
                    to_print = 'eval_loss: {}, psnr: {}'.format(sum(eval_loss) / len(eval_loss), sum(eval_psnr) / len(eval_psnr))
                    print(to_print)
                    self.writer.write(to_print + '\n')
                    if iter % (self.config['output_each']) == 0:
                        file_name = join(data_dir, '{}.npy'.format(iter))
                        save_npy = eval_z_hat.data.cpu().reshape(
                            [eval_z_hat.size()[0], eval_z_hat.size()[1], self.config['width'], self.config['height']])
                        np.save(file_name, save_npy)

        # finish training, now testing
        test_loss = []
        test_psnr = []
        with torch.no_grad():
            reconstruct_to_file = []
            while True:
                test_batch = data_loader.load_batch_test(self.batch_size)
                if len(test_batch) == 0:
                    break
                test_raw_input = torch.tensor(test_batch, dtype=self.dtype, device=self.device)
                pre_input = torch.zeros([1, self.batch_size, self.n_input], dtype=self.dtype, device=self.device)
                test_z_hat = self.forward(pre_input, test_raw_input)
                reconstruct_to_file.append(test_z_hat)
                test_loss.append(self.compute_loss(test_raw_input, test_z_hat))
                test_psnr.append(psnr(test_raw_input, test_z_hat))
            to_print = 'test_loss: {}, psnr: {}'.format(sum(test_loss) / len(test_loss),
                                                        sum(test_psnr) / len(test_psnr))
            print(to_print)
            self.writer.write(to_print + '\n')
            file_name = join(data_dir, 'final.npy')
            reconstruct_to_file = torch.cat(reconstruct_to_file, dim=1)
            save_npy = reconstruct_to_file.data.cpu().reshape([reconstruct_to_file.size()[0], reconstruct_to_file.size()[1], self.config['width'], self.config['height']])
            np.save(file_name, save_npy)
        self.writer.close()
