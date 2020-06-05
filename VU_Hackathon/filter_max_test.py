import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from skimage import data
from skimage.util.dtype import dtype_range
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank

matplotlib.rcParams['font.size'] = 9


def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins)
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')

    xmin, xmax = dtype_range[image.dtype.type]
    ax_hist.set_xlim(xmin, xmax)

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')

    return ax_img, ax_hist, ax_cdf


from skimage.io import imread
import skimage

from skimage.morphology import disk
from skimage.filters.rank import gradient

import cv2
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage.measure import label
from skimage.color import label2rgb
from skimage.morphology import reconstruction

image = imread(r'C:\Users\bunny\Desktop\bc_img.tiff')#[0:1000, 0:1000]
image = skimage.img_as_float64(image, force_copy=True)
img = image

img_bl = cv2.GaussianBlur(img, (7, 7), 0)
# Equalization
# selem = disk(30)
# img_eq = rank.equalize(img_bl, selem=selem)
#
# img_eq = denoise_tv_chambolle(img_eq, weight=0.1, multichannel=True)
# # img_bl = cv2.bilateralFilter(img_eq,20,75,75)
# # img_bl = cv2.medianBlur(img_eq, 15)
# # img_bl = cv2.blur(img_eq,(5,5))
# plt.imshow(img_eq)


ent = gradient(img_bl, disk(9))
seed = np.copy(ent)
seed[1:-1, 1:-1] = ent.max()
mask = ent

filled = reconstruction(seed, mask, method='erosion')

B = np.where(filled > 0, 1, 0)

label_image = label(B)
# image_label_overlay = label2rgb(label_image)

plt.imshow(label_image)
plt.show()

