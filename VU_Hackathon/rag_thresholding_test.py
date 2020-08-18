from skimage import data, io, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt
from skimage.io import imread
import skimage

# Generate noisy synthetic data
image = imread(r'C:\Users\bunny\Desktop\img_blur.tiff')[0:500, 0:500]
img = skimage.img_as_float64(image, force_copy=True)

labels1 = segmentation.slic(img, compactness=1, n_segments=500)
out1 = color.label2rgb(labels1, img, kind='avg')

g = graph.rag_mean_color(img, labels1)
labels2 = graph.cut_threshold(labels1, g, 29)
out2 = color.label2rgb(labels2, img, kind='avg')

fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True,
                       figsize=(6, 8))

ax[0].imshow(out1)
ax[1].imshow(out2)

for a in ax:
    a.axis('off')

plt.show()