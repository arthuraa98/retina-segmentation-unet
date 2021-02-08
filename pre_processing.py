"""
Created on Wed Nov 11 14:13:09 2020

@author: arthur

Performs the pre-processing before the training
"""

import numpy as np
import cv2


class preProcessing:
    """
    Parameters: (array) data

    Return:
        the pre-processed images with the following transformations:
            Gray-scale conversion
            Standardization
            CLAHE
            Gamma adjustment
    """

    def __init__(self, data):
        assert data.shape[-1] == 3  # number of channels
        data = data.astype(np.uint8)
        # Gray-scale conversion
        self.gray_imgs = self.rgb2gray(data)
        # Standardization
        self.standardized_imgs = self.standardization(self.gray_imgs)
        # CLAHE
        self.equalized_imgs = self.clahe(self.standardized_imgs)

        # Gamma adjustment
        output_imgs = self.gamma_adjust(self.equalized_imgs, gamma=0.8)

        # Normalize images
        self.output_imgs = output_imgs / 255.0

    def pre_processed_imgs(self):
        return self.output_imgs

    def rgb2gray(self, imgs):
        assert len(imgs.shape) == 4
        assert imgs.shape[-1] == 3
        bn_imgs = np.empty(imgs.shape[:3])
        for i in range(imgs.shape[0]):
            bn_imgs[i] = np.dot(imgs[i, :, :, :3], [0.2989, 0.5870, 0.1140])
        bn_imgs = np.reshape(bn_imgs, (imgs.shape[0], imgs.shape[1], imgs.shape[2], 1))
        return bn_imgs

    def standardization(self, imgs):
        imgs_standardized = np.empty(imgs.shape)
        imgs_std = np.std(imgs)
        imgs_mean = np.mean(imgs)
        imgs_standardized = (imgs - imgs_mean) / imgs_std
        for i in range(imgs.shape[0]):
            imgs_standardized[i] = ((imgs_standardized[i] - np.min(imgs_standardized[i])) / (
                        np.max(imgs_standardized[i]) - np.min(imgs_standardized[i]))) * 255
            # imgs_standardized[i] = cv2.normalize(imgs[i], imgs_standardized[i], alpha=0, beta=255)
        return imgs_standardized

    def clahe(self, imgs):
        # assert (len(imgs.shape)==4)  #4D arrays
        assert (imgs.shape[-1] == 1)  # check the channel is 1
        # create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        imgs_equalized = np.empty(imgs.shape)
        for i in range(imgs.shape[0]):
            imgs_equalized[i, :, :, 0] = clahe.apply(np.array(imgs[i, :, :, 0], dtype=np.uint8))
        return imgs_equalized

    def gamma_adjust(self, imgs, gamma=1.0):
        # assert (len(imgs.shape)==4)  #4D arrays
        assert (imgs.shape[-1] == 1)  # check the channel is 1
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        new_imgs = np.empty(imgs.shape)
        for i in range(imgs.shape[0]):
            new_imgs[i, :, :, 0] = cv2.LUT(np.array(imgs[i, :, :, 0], dtype=np.uint8), table)
        return new_imgs
