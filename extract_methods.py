"""
Created on Thu Nov 12 11:22:12 2020

@author: arthur

Functions to extract patches from images, extend images, and verify the data
"""

from utility import read_hdf5
from pre_processing import preProcessing
import numpy as np


def get_training_data(dataset_X_train, dataset_y_train, patch_h, patch_w, n_patches, inside_FOV):
    original_X_train_imgs = read_hdf5(dataset_X_train)
    y_train_imgs = read_hdf5(dataset_y_train)

    X_train_imgs = preProcessing(original_X_train_imgs).pre_processed_imgs()
    # normalize y train images
    y_train_imgs = y_train_imgs / 255.0

    X_train_imgs = X_train_imgs[:, 9:574, :, :]  # To cut bottom and top so that the image can be 565*565
    y_train_imgs = y_train_imgs[:, 9:574, :, :]  # To cut bottom and top so that the image can be 565*565

    assert (np.min(y_train_imgs) == 0 and np.max(y_train_imgs) == 1)

    patches_X_train, patches_y_train = extract_random_patches(X_train_imgs, y_train_imgs, n_patches, patch_h, patch_w,
                                                              inside_FOV)

    # checking the patches
    checking_data(patches_X_train, patches_y_train)
    print("Training patches X/y shape:")
    print(patches_X_train.shape)

    return patches_X_train, patches_y_train


def get_testing_data(dataset_X_test_hdf5, dataset_y_train_hdf5, n_of_testing_imgs, patch_h, patch_w):
    original_X_test_imgs = read_hdf5(dataset_X_test_hdf5)
    y_test_imgs = read_hdf5(dataset_y_train_hdf5)

    X_test_imgs = preProcessing(original_X_test_imgs).pre_processed_imgs()
    y_test_imgs = y_test_imgs / 255.

    print("Test image/labels shape (before extend them): " + str(X_test_imgs.shape))
    # extend both images and labels so they can be divided exactly by the patches dimensions
    X_test_imgs = X_test_imgs[0:n_of_testing_imgs, :, :]
    y_test_imgs = y_test_imgs[0:n_of_testing_imgs, :, :]

    X_test_imgs = extend_img(X_test_imgs, patch_h, patch_w)
    y_test_imgs = extend_img(y_test_imgs, patch_h, patch_w)

    checking_data(X_test_imgs, y_test_imgs)

    # check if labels are within 0-1
    assert (np.max(y_test_imgs) == 1 and np.min(y_test_imgs) == 0)

    print('Test images/labels shape: ' + str(X_test_imgs.shape))

    # extract the TEST patches from the full images
    patches_X_test = extract_ordered_patches(X_test_imgs,
                                             patch_h,
                                             patch_w)
    patches_y_test = extract_ordered_patches(y_test_imgs,
                                             patch_h,
                                             patch_w)

    checking_data(patches_X_test, patches_y_test)

    print('Test Patches images/labels shape: ' + str(patches_X_test.shape))
    print('\n')

    return patches_X_test, patches_y_test


def extract_random_patches(full_X_imgs, full_y_imgs, n_patches, patch_h, patch_w, inside=False):
    assert (len(full_X_imgs.shape) == 4 and len(full_y_imgs.shape) == 4)  # 4-D images
    n_of_imgs = full_X_imgs.shape[0]
    X_patches = np.empty((n_patches, patch_h, patch_w, full_X_imgs.shape[-1]))
    y_patches = np.empty((n_patches, patch_h, patch_w, full_X_imgs.shape[-1]))

    img_h = full_X_imgs.shape[1]
    img_w = full_X_imgs.shape[2]

    # number of patches per image should be equal
    patches_per_img = int(n_patches / n_of_imgs)
    print("Patches per full image: " + str(patches_per_img))

    # iterate over the total number of patches (n_patches)
    iter_patch = 0
    for i in range(n_of_imgs):
        for _ in range(patches_per_img):

            x_center = np.random.randint(int(patch_w / 2), img_w - int(patch_w / 2))
            y_center = np.random.randint(int(patch_h / 2), img_h - int(patch_h / 2))

            X_patch = full_X_imgs[i, y_center - int(patch_h / 2):y_center + int(patch_h / 2),
                      x_center - int(patch_w / 2):x_center + int(patch_w / 2), :]
            y_patch = full_y_imgs[i, y_center - int(patch_h / 2):y_center + int(patch_h / 2),
                      x_center - int(patch_w / 2):x_center + int(patch_w / 2), :]

            # In this way, the patches completely outside the Field Of View (FOV) are not selected
            if inside:
                if not is_patch_inside_FOV(x_center, y_center, img_h, img_w, patch_h):
                    continue

            X_patches[iter_patch] = X_patch
            y_patches[iter_patch] = y_patch

            iter_patch += 1

    return X_patches, y_patches


# Divide all the full images in ordered patches
def extract_ordered_patches(full_imgs, patch_h, patch_w):
    img_h = full_imgs.shape[1]  # height of the full image
    img_w = full_imgs.shape[2]  # width of the full image

    n_of_patches_h = int(img_h / patch_h)
    n_of_patches_w = int(img_w / patch_w)

    if img_h % patch_h != 0:
        rem_pixels = img_h % patch_h
        print(f'{n_of_patches_h} patches in height, with {rem_pixels} pixels leftover')

    print('Number of patches per image: ' + str(n_of_patches_h * n_of_patches_w))

    # Number of patches in all images
    n_patches = (n_of_patches_h * n_of_patches_w) * full_imgs.shape[0]
    patches = np.empty((n_patches, patch_h, patch_w, full_imgs.shape[-1]))

    # iterate over the total number of patches (n_patches)
    iter_patch = 0

    for i in range(full_imgs.shape[0]):  # loop over the full images
        for h in range(n_of_patches_h):
            for w in range(n_of_patches_w):
                patch = full_imgs[i, h * patch_h:(h * patch_h) + patch_h,
                        w * patch_w:(w * patch_w) + patch_w, :]

                patches[iter_patch] = patch
                iter_patch += 1  # total
    assert (iter_patch == n_patches)  # if all patches were extract

    return patches  # array with all the full_imgs divided in patches


# Verify if patches is inside de Field of View (FOV)
def is_patch_inside_FOV(x_center, y_center, img_h, img_w, patch_h):
    new_x = x_center - int(img_w / 2)
    new_y = y_center - int(img_h / 2)
    # patch_h == patch_w == 48
    internal_radius = 270 - int(patch_h * np.sqrt(2.0) / 2.0)
    radius = np.sqrt((new_x ** 2) + (new_y ** 2))
    if radius < internal_radius:
        return True
    else:
        return False


# Extend the full images because patch division is not exact
def extend_img(data, patch_h, patch_w):
    img_h = data.shape[1]
    img_w = data.shape[2]
    new_img_h = 0
    new_img_w = 0

    if img_h % patch_h == 0:
        new_img_h = img_h
    else:
        new_img_h = (int(img_h / patch_h) + 1) * patch_h

    if img_w % patch_w == 0:
        new_img_w = img_w
    else:
        new_img_w = (int(img_w / patch_w) + 1) * patch_w

    new_data = np.zeros((data.shape[0], new_img_h, new_img_w, 1))
    new_data[:, 0:img_h, 0:img_w, :] = data[:, :, :, :]
    return new_data


# Data checking
def checking_data(X_imgs, y_imgs):
    assert (len(X_imgs.shape) == len(y_imgs.shape))
    assert (X_imgs.shape[0] == y_imgs.shape[0])
    assert (X_imgs.shape[1] == y_imgs.shape[1])
    assert (X_imgs.shape[2] == y_imgs.shape[2])
    assert (X_imgs.shape[3] == y_imgs.shape[3])
