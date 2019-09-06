import matplotlib.pyplot as plt
from skimage.color import rgb2hed
from matplotlib.colors import LinearSegmentedColormap
from skimage import io

# Create an artificial color close to the orginal one
cmap_hema = LinearSegmentedColormap.from_list('mycmap', ['white', 'purple'])
cmap_dab = LinearSegmentedColormap.from_list('mycmap', ['white', 'darkviolet'])
cmap_eosin = LinearSegmentedColormap.from_list('mycmap', ['darkviolet', 'white'])

image_path1 = 'c:/Users/bunny/Desktop/download (2).png'
ihc_rgb = io.imread(image_path1)[..., :3]
ihc_hed = rgb2hed(ihc_rgb)

import numpy as np
from skimage.exposure import rescale_intensity

# Rescale hematoxylin and DAB signals and give them a fluorescence look
h = rescale_intensity(ihc_hed[:, :, 0], out_range=(0, 1))
d = rescale_intensity(ihc_hed[:, :, 2], out_range=(0, 1))
zdh = np.dstack((np.zeros_like(h), d, h))

fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(ihc_rgb)
ax[0].set_title("Original image")

ax[1].imshow(zdh)
ax[1].set_title("Stain separated image (rescaled)")

ax[2].imshow(zdh[:, :, 1], cmap=cmap_hema)
ax[2].set_title("Hematoxylin")

ax[3].imshow(zdh[:, :, 2], cmap=cmap_dab)
ax[3].set_title("DAB")

for a in ax.ravel():
    a.axis('off')

fig.tight_layout()

from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray

import matplotlib.pyplot as plt

image_gray_1 = zdh[:, :, 1]
image_gray_2 = zdh[:, :, 2]
# image_gray = rgb2gray(image)

blobs_log_1 = blob_log(image_gray_1, max_sigma=50, num_sigma=10, threshold=.1)
blobs_log_2 = blob_log(image_gray_2, max_sigma=30, num_sigma=10, threshold=.1)

# Compute radii in the 3rd column.
blobs_log_1[:, 2] = blobs_log_1[:, 2] * sqrt(2)
blobs_log_2[:, 2] = blobs_log_2[:, 2] * sqrt(2)

blobs_list = [blobs_log_1, blobs_log_2, ]
colors = ['red', 'yellow']
titles = ['blood vessel', 'cell nuclear', ]
sequence = zip(blobs_list, colors, titles)

fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

for idx, (blobs, color, title) in enumerate(sequence):
    ax[idx].set_title(title)
    ax[idx].imshow(zdh[:, :, idx + 1])
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, linewidth=2 - idx, fill=False)
        ax[idx].add_patch(c)
    ax[idx].set_axis_off()

ax[2].set_title('Merge')
ax[2].imshow(zdh)
for blobs, color, title in zip(blobs_list, colors, titles):
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, linewidth=1, fill=False)
        ax[2].add_patch(c)
ax[2].set_axis_off()

plt.tight_layout()

fig, ax = plt.subplots(figsize=(10, 10))
imgplot = plt.imshow(zdh)
for blobs, color, title in zip(blobs_list, colors, titles):
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, linewidth=1, fill=False)
        ax.add_patch(c)
plt.tight_layout()
plt.show()
