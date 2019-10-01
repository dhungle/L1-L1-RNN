from os import path, listdir
from os.path import join
import numpy as np
import cv2

new_w, new_h = 128, 128
path = '/home/hung/Desktop/image_dataset/caltech256'
processed_images = []
for folder_name in listdir(path):
    folder = join(path, folder_name)
    for name in listdir(folder):
        if 'jpg' in name:
            img_name = join(folder, name)
            img = cv2.imread(img_name)

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            w, h = img_gray.shape[:2]

            # only
            if w > h:
                cutoff = int((w - h)/2)
                img_crop = img_gray[cutoff: cutoff + h]
            else:
                cutoff = int((h - w) / 2)
                img_crop = img_gray[:, cutoff: cutoff + w]
            img_resized = cv2.resize(img_crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            processed_images.append(img_resized)

    print(len(processed_images), 'images processed')
dataset = np.stack(processed_images, axis=1) # shape: time_steps, num_samples, size
save_path = '../data/caltech256/data.npy'
np.save(save_path, dataset)

