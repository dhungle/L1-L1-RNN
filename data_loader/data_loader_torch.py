from torch.utils import data
from torchvision.transforms import Compose, ToTensor
from torch.autograd import Variable
import numpy as np


class Moving_MNIST_Loader(data.Dataset):
    def __init__(self, path, time_steps = 20, load_only=-1, flatten=True, scale=False):
        '''
        :param path: moving mnist file path
        :param load_only: load a limited number of samples, -1 if load all.
        :param flatten: Flatten all frames (images) to one-directional arrays
        :param scale: Scale 8-bit images of range 0-255 to range 0-1
        '''
        self.path = np.load(path)
        self.load_all()
        assert load_only != 0, 'load_only should be either -1 (load all) or a positive number'
        assert load_only >= -1 , 'load_only should be either -1 (load all) or a positive number'
        assert time_steps <= self.data.shape[0], 'time_steps should be smaller than the number of frames'
        if load_only > 0:
            self.data = self.data[:, :load_only]
        if time_steps < self.data.shape[0]:
            self.data = self.data[:time_steps]
        self.num_frames, self.num_samples, self.width, self.height = self.data.shape
        if flatten:
            self.data = self.data.reshape([self.num_frames, self.num_samples, -1])
        if scale:
            self.data = self.data/255.

        self.cutoff = int(self.num_samples * 0.8)
        self.train = self.data[:, :self.cutoff, ...]
        self.test = self.data[:, self.cutoff:, ...]
        self.current_idx = 0
        self.exhausted = False

        print('data loaded, training/testing: {}/{}'.format(self.train.shape[1], self.test.shape[1]))


    def shuffle(self):
        '''
        Like np.random.shuffle but with the second axis
        '''
        indices = np.random.permutation(self.cutoff)
        self.train = self.train[:, indices, ...]

    def load_batch_train(self, batch_size):
        if self.current_idx + batch_size >= self.cutoff:
            self.shuffle()
            self.current_idx = 0

        batch = self.train[:, self.current_idx:self.current_idx + batch_size, ...]
        self.current_idx += batch_size
        return batch

    def load_batch_test(self, batch_size):
        if self.current_idx +  batch_size >= self.test.shape[1]:
            self.exhausted = True
        batch = self.test[:, self.current_idx: min(self.current_idx + batch_size, self.test.shape[1]),...]
        self.current_idx += batch_size
        return  batch

    def reset_test_index(self):
        self.current_idx = 0
        self.exhausted = False

    def visualize(self, start=0, end=100):
        if len(self.data) == 0:
            self.load_all()
        for i in range(start, end):
            clip = self.data[:, i, :, :]
            plt.figure(1)
            plt.clf()
            plt.title('clip {}'.format(i))
            for j in range(20):
                img = clip[j]
                plt.imshow(img)
                plt.pause(.01)
                plt.draw()