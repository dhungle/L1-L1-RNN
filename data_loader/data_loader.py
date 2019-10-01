import numpy as np
import matplotlib.pyplot as plt


class Loader:
    def __init__(self, path, time_steps, load_only=-1, flatten=True, scale=False):
        '''
        :param path: file path, data format: [time step, index, dimensions of one sample]
        :param load_only: load a limited number of samples, -1 if load all.
        :param flatten: Flatten all frames (images) to one-directional arrays
        :param scale: Scale 8-bit images of range 0-255 to range 0-1
        '''
        self.data = np.load(path).astype('float32')
        print('original data shape {}'.format(self.data.shape[2:]))
        assert load_only != 0, 'load_only should be either -1 (load all) or a positive number'
        assert load_only >= -1, 'load_only should be either -1 (load all) or a positive number'
        assert time_steps <= self.data.shape[0], 'time_steps should be smaller than the number of frames'
        if load_only > 0:
            self.data = self.data[:, :load_only]
        if time_steps < self.data.shape[0]:
            self.data = self.data[:time_steps]
        self.num_frames, self.num_samples, self.size = self.data.shape[0], self.data.shape[1], self.data.shape[2:]

        if flatten:
            self.data = self.data.reshape([self.num_frames, self.num_samples, -1])

        if scale:
            self.data = self.data / 255.

        self.train_cutoff = int(self.num_samples * 0.8)
        self.eval_cutoff = self.train_cutoff + int(self.num_samples * 0.1)
        self.train = self.data[:, :self.train_cutoff, ...]
        self.eval = self.data[:, self.train_cutoff: self.eval_cutoff, ...]
        self.test = self.data[:, self.eval_cutoff:, ...]
        self.current_idx_train = 0
        self.current_idx_eval = 0
        self.current_idx_test = 0

        print('data loaded, training/eval/testing: {}/{}/{}'.format(self.train.shape[1], self.eval.shape[1],
                                                                    self.test.shape[1]))

    def shuffle(self):
        '''
        Like np.random.shuffle but along the second axis
        '''
        indices = np.random.permutation(self.train_cutoff)
        self.train = self.train[:, indices, ...]

    def load_batch_train(self, batch_size):
        if self.current_idx_train + batch_size >= self.train_cutoff:
            self.shuffle()
            self.current_idx_train = 0

        batch = self.train[:, self.current_idx_train:self.current_idx_train + batch_size, ...]
        self.current_idx_train += batch_size
        return batch

    def load_batch_eval(self, batch_size):
        if self.current_idx_eval + batch_size >= self.eval.shape[1]:
            self.current_idx_eval = 0
            return []
        batch = self.eval[:, self.current_idx_eval: self.current_idx_eval + batch_size, ...]
        self.current_idx_eval += batch_size
        return batch

    def load_batch_test(self, batch_size):
        if self.current_idx_test + batch_size >= self.test.shape[1]:
            self.current_idx_test = 0
            return []
        batch = self.test[:, self.current_idx_test: self.current_idx_test + batch_size, ...]
        self.current_idx_test += batch_size
        return batch


class Moving_MNIST_Loader(Loader):
    def __init__(self, path, time_steps=20, load_only=-1, flatten=True, scale=False):
        '''
        :param path: moving mnist file path
        '''
        super(Moving_MNIST_Loader, self).__init__(path, time_steps, load_only, flatten, scale)

    def visualize(self, start=0, end=1):
        for i in range(start, end):
            clip = self.data[:, i, :, :]
            clip = 255 - clip
            plt.figure(1)
            plt.clf()
            plt.title('our method')
            for j in range(7, 8):
                img = clip[j]
                plt.imshow(img, cmap='gray')
                plt.pause(100)
                plt.draw()

class Caltech_Loader(Loader):
    def __init__(self, path, time_steps=128, load_only=-1, flatten=False, scale=False):
        '''
        :param path: Caltech256 file path
        '''
        super(Caltech_Loader, self).__init__(path, time_steps, load_only, flatten, scale)

    def visualize(self, start=0, end=100):
        for i in range(start, end):
            img = self.train[:, i, :]
            plt.figure(1)
            plt.clf()
            plt.title('img {}'.format(i))
            plt.imshow(img, cmap='gray')
            plt.pause(.01)
            plt.draw()


if __name__ == '__main__':
    path = '/home/hung/Desktop/sista_rnn/results/fr/2019-01-29 17:49:45.264019/final.npy'
    loader = Moving_MNIST_Loader(path, flatten=False, scale=False)
    image = loader.visualize(2, 3)

    path = '../data/moving_mnist/mnist_test_seq_16.npy'
    loader2 = Moving_MNIST_Loader(path, flatten=False, scale=False)
    loader2.visualize(9962, 9963)
    # loader2.visualize()
    # data = (loader.data - loader2.data).reshape(-1)
    # non_zeros = np.nonzero(data)
    # print(data.shape)
    # loader.visualize(end=10)
