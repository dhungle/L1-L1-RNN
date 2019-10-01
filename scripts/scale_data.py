from data_loader.data_loader import Moving_MNIST_Loader
import numpy as np
import cv2

size = 16

def main():
    moving_mnist_path = '../data/moving_mnist/mnist_test_seq.npy'
    data_loader = Moving_MNIST_Loader(moving_mnist_path, flatten=False)
    data = data_loader.data
    time_steps, num_samples, width, height = data.shape
    resized_data = np.empty([time_steps, num_samples, size, size])

    for i in range(time_steps):
        for j in range(num_samples):
            img = data[i, j]
            resized_data[i, j] = cv2.resize(img, dsize=(size, size))
    print(resized_data.shape, resized_data.max(), resized_data.min())

    save_path = '../data/moving_mnist/mnist_test_seq_{}_new.npy'.format(size)
    np.save(save_path, resized_data)

if __name__ == '__main__':
    main()