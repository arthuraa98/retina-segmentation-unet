"""
Created on Mon Nov  9 13:13:49 2020

@author: arthur de andrade

This creates de hdf5 datasets of the DRIVE database
"""

import h5py
import numpy as np
import glob
from PIL import Image
import os

# ------------Path of the images --------------------------------------------------------------
# train
original_imgs_train = 'train\origin'
groundTruth_imgs_train = 'train\groundtruth'

# test
original_imgs_test = 'test\origin'
groundTruth_imgs_test = 'test\groundtruth'


# ---------------------------------------------------------------------------------------------


def write_dataset(arr, filename):
    with h5py.File(filename, mode='w') as f:
        dataset = f.create_dataset('image', data=arr, dtype=arr.dtype)


def save_as_png(imgs_dir, filename, is_gray=False):
    new_path = filename
    os.makedirs(new_path, exist_ok=True)
    print(f'Path {new_path} created')

    for infile in os.listdir(imgs_dir):
        file, ext = os.path.splitext(infile)
        if is_gray:
            img = Image.open(imgs_dir + '\\' + infile)
        else:
            img = Image.open(imgs_dir + '\\' + infile).convert('RGB')
        new_file = file + '.png'
        img.save(new_path + '\\' + new_file, 'PNG')
        print(f'{new_file} successfully saved.')

    return new_path


def get_datasets(imgs_dir, groundTruth_dir, n_imgs, width, height):
    X_imgs = np.empty((n_imgs, height, width, 3))
    y_imgs = np.empty((n_imgs, height, width))

    for file, i in zip(glob.glob(imgs_dir + '\*.png'), range(len(glob.glob(imgs_dir + '\*.png')))):
        # X set
        img = Image.open(file).convert('RGB')
        X_imgs[i] = np.asarray(img)

    for file, i in zip(glob.glob(groundTruth_dir + '\*.png'), range(len(glob.glob(groundTruth_dir + '\*.png')))):
        # y set
        img = Image.open(file)
        y_imgs[i] = np.asarray(img)

    y_imgs = np.reshape(y_imgs, (n_imgs, height, width, 1))
    assert (np.max(y_imgs) == 255 and np.min(y_imgs) == 0)
    print('y label is correctly within pixel value range 0-255')
    return X_imgs, y_imgs


# Save images as png
original_imgs_train = save_as_png(original_imgs_train, filename='train_png\\origin')
groundTruth_imgs_train = save_as_png(groundTruth_imgs_train, filename='train_png\\groundtruth', is_gray=True)
original_imgs_test = save_as_png(original_imgs_test, filename='test_png\\origin')
groundTruth_imgs_test = save_as_png(groundTruth_imgs_test, filename='test_png\\groundtruth', is_gray=True)

# Create the datasets
X_imgs_train, y_imgs_train = get_datasets(original_imgs_train, groundTruth_imgs_train, 40, 565, 584)
print('saving train datasets')
write_dataset(X_imgs_train, 'dataset_X_train.hdf5')
write_dataset(y_imgs_train, 'dataset_y_train.hdf5')

X_imgs_test, y_imgs_test = get_datasets(original_imgs_test, groundTruth_imgs_test, 20, 565, 584)
print('saving test datasets')
write_dataset(X_imgs_test, 'dataset_X_test.hdf5')
write_dataset(y_imgs_test, 'dataset_y_test.hdf5')
