from skimage import io
from skimage.transform import resize
from skimage.color import rgb2hed, hed2rgb
from skimage.exposure import rescale_intensity
import glob
import os
import random
import numpy as np

# 数据集地址
image_path = 'C:/Users/bunny/Desktop/dataset3/test/'

# 将所有的图片resize成100*100
w = 100
h = 100
c = 3


# 读取图片
def resize_img(root_path, new_width, new_height):
    cate = [root_path + folder for folder in os.listdir(root_path) if os.path.isdir(root_path + folder)]
    count = 0
    for idx, folder in enumerate(cate):
        print('reading the images:%s' % folder)
        for im in glob.glob(folder + '/*.jpg'):
            img = io.imread(im)
            img = resize(img, (new_width, new_height))
            io.imsave(im, img)
            del img
            count += 1
            if count % 2500 == 0:
                print(count)


def filter_img_random(root_path, left_count):
    cate = [root_path + folder for folder in os.listdir(root_path) if os.path.isdir(root_path + folder)]
    for idx, folder in enumerate(cate):
        print('deleting the images:%s' % folder)
        count = 0
        file_path_list = [os.path.join(folder, file_name) for file_name in os.listdir(folder)
                          if os.path.isfile(os.path.join(folder, file_name))]
        random.shuffle(file_path_list)
        for path in file_path_list[left_count:]:
            os.remove(path)
            count += 1
            if count % 100 == 0:
                print("\rreading {0}/{1}".format(count, len(file_path_list) - left_count), end='')
        print('\r', end='')


def stain_separate_image(root_path):
    cate = [root_path + folder for folder in os.listdir(root_path) if os.path.isdir(root_path + folder)]
    count = 0
    for idx, folder in enumerate(cate):
        print('reading the images:%s' % folder)
        for im in glob.glob(folder + '/*.jpg'):
            img = io.imread(im)
            ihc_hed = rgb2hed(img)
            # Rescale hematoxylin and DAB signals and give them a fluorescence look
            _h = rescale_intensity(ihc_hed[:, :, 0], out_range=(0.0, 1.0))
            _e = rescale_intensity(ihc_hed[:, :, 1], out_range=(0.0, 1.0))
            _d = rescale_intensity(ihc_hed[:, :, 2], out_range=(0.0, 1.0))
            # zdh = np.dstack((np.zeros_like(_d), _h, _e))
            zdh = np.dstack((_e, _h, (np.zeros_like(_d))))
            io.imsave(im, zdh)
            del img
            count += 1
            if count % 1000 == 0:
                print(count)


def minus_average_image(root_path):
    cate = [root_path + folder for folder in os.listdir(root_path) if os.path.isdir(root_path + folder)]
    count = 0
    for idx, folder in enumerate(cate):
        print('reading the images:%s' % folder)
        for im in glob.glob(folder + '/*.jpg'):
            img = io.imread(im)
            aver = np.mean(img, axis=(0, 1))
            _w = img.shape[0]
            _h = img.shape[1]
            _c = img.shape[2]
            aver_img = np.empty((_w, _h, _c), np.float32)
            aver_img[:, :, 0] = aver[0]
            aver_img[:, :, 1] = aver[1]
            aver_img[:, :, 2] = aver[2]
            minus_aver_img = (img - aver) / 255.0
            io.imsave(im, np.asarray(minus_aver_img, np.float32))
            del img
            count += 1
            if count % 1000 == 0:
                print(count)


# resize_img(image_path, w, h)
# stain_separate_image('C:/Users/bunny/Desktop/d3_eh0/train/')
# filter_img_random('C:/Users/bunny/Desktop/d3/train/', 1000)
minus_average_image('C:/Users/bunny/Desktop/d3_ma/train/')
