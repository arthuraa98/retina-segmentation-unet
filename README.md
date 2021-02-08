# Retina Blood Vessel Segmentation using a convolution neural network (U-Net)

This repository contains an implementation of a convolutional neural network used to segment blood vessel in a retina fundus images. This is a binary classification task: the model predicts if each pixel in the spectral fundus image is either a vessel or not. The architecture of the model is based on the *U-Net* architecture, described in this [paper](https://researchbank.swinburne.edu.au/file/fce08160-bebd-44ff-b445-6f3d84089ab2/1/2018-xianchneng-retina_blood_vessel.pdf). The performance of this model is tested on the DRIVE database and it achieves an area under the ROC curve near to 1.

## Methods

Before training, the 60 images of the DRIVE training iamges are pre-processed with the following transformations:
* Gray-scale conversion
* Standardization
* Contrast-limited adaptive histogram equalization (CLAHE)
* Gamma adjustment

The proposed U-Net based network was trained using sub-images (patches) of the pre-preocessed full-images. Each patches, of a size 48x48, is obtained randomly selecting its center inside the full-image. Also the patches partially or completely outside the Field Of View (FOV) are selected, in this way the neural network learns how to discriminate the FOV border from blood vessels. A set of X patches is obtained by randomly extracting Y patches in each of the 60 DRIVE training images. The first 90% of the dataset is used for training (dasd patches), while the last 10% is used for validation (dasdad patches).
