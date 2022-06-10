import os
import cv2
import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal.windows import gaussian
import matplotlib.pyplot as plt


INPUT_IMAGE_DIR = "image/"
OUTPUT_IMAGE_DIR = "result/"

IMAGE_EXTENSION = ".jpg"

if not os.path.exists(OUTPUT_IMAGE_DIR):
    os.makedirs(OUTPUT_IMAGE_DIR)


def gaussian_kernel(kernel_size=3):
    h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
    h = np.dot(h, h.transpose())
    h /= np.sum(h)
    return h


def wiener_filter(img, kernel, K):
    kernel /= np.sum(kernel)
    dummy = np.copy(img)
    dummy = fft2(dummy)
    kernel = fft2(kernel, s=img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = np.abs(ifft2(dummy))
    return dummy




for fname in os.listdir(INPUT_IMAGE_DIR):
    ffname = os.path.join(INPUT_IMAGE_DIR, fname)  # full file name
    if os.path.isfile(ffname):

        image = cv2.imread(ffname)

        if len(image.shape) > 2 and image.shape[2] > 1:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        k = 10
        kernel = gaussian_kernel(9)
        imageResult = wiener_filter(image, kernel, 1/k)

        # cv2.imshow('result', imageResult)
        # cv2.waitKey(0)

        cv2.imwrite(os.path.join(OUTPUT_IMAGE_DIR, fname), imageResult)



# imageResult = copy.deepcopy(image)
# cv2.fastNlMeansDenoising(image, imageResult, 100.0, 7, 21)