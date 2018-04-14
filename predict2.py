from skimage import io, transform
from skimage.color import rgb2hed
from skimage.exposure import rescale_intensity
import tensorflow as tf
import numpy as np
import glob
import os
import random

predict_image_root_path = "C:/Users/bunny/Desktop/test/parenchyma/"
model_root_path = 'c:/Users/bunny/Desktop/dataset1/root/'
model_path = 'c:/Users/bunny/Desktop/dataset1/root/model.ckpt.meta'

name_dict = {0: 'fat', 1: 'parenchyma', 2: 'tumor'}

w = 100
h = 100
c = 3


def read_one_image(image_path):
    img = io.imread(image_path)
    # img = hed_read_image(image_path)
    if img.shape != (w, h, 3):
        print(image_path)
        img = transform.resize(img, (w, h))
    return np.asarray(img)


def hed_read_image(image_path):
    img = io.imread(image_path)
    ihc_hed = rgb2hed(img)
    # Rescale hematoxylin and DAB signals and give them a fluorescence look
    _h = rescale_intensity(ihc_hed[:, :, 0], out_range=(0, 1))
    _d = rescale_intensity(ihc_hed[:, :, 2], out_range=(0, 1))
    zdh = np.dstack((np.zeros_like(_h), _d, _h))
    return zdh


def read_img(folder_path, total_count, size_filter=2000):
    data_list = []
    count = 0
    file_path_list = [os.path.join(predict_image_root_path, x) for x in os.listdir(predict_image_root_path)
                            if os.path.isfile(os.path.join(predict_image_root_path, x))]
    # for im in glob.glob(folder_path + '/*.jpg'):
    while len(data_list) < total_count:
        im = random.choice(file_path_list)
        file_info = os.stat(im)
        file_size = file_info.st_size
        if file_size < size_filter:
            continue
        if file_size > 100 * size_filter:
            continue
        data_list.append(read_one_image(im))
    return data_list


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


saver = tf.train.import_meta_graph(model_path)
saver.restore(sess, tf.train.latest_checkpoint(model_root_path))

graph = tf.get_default_graph()
x = graph.get_tensor_by_name("x:0")

total_correct = 0
total_count = 0
for i in range(20):
    data = read_img(predict_image_root_path, 128)
    feed_dict = {x: data}

    logits = graph.get_tensor_by_name("logits_eval:0")

    classification_result = sess.run(logits, feed_dict)

    # 打印出预测矩阵
    # print(classification_result)
    # 打印出预测矩阵每一行最大值的索引
    # print(tf.argmax(classification_result, 1).eval())
    # 根据索引通过字典对应花的分类
    output = tf.argmax(classification_result, 1).eval()
    # for i in range(len(output)):
    # print("Picture", i + 1, ":" + name_dict[output[i]])
    print(np.count_nonzero(output == 1) * 1.0 / len(output) * 1.0)
    total_correct += np.count_nonzero(output == 1)
    total_count += len(output)
print(total_correct)
print(total_count)
print(1.0 * total_correct / total_count)
sess.close()
