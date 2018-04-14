import glob
import os
import random
from skimage import io
import numpy as np
from tensorflow import keras
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.models import Sequential
import matplotlib.pylab as plt


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


def read_img_random(path, total_count, size_filter=2000):
    cate = [path + folder for folder in os.listdir(path) if os.path.isdir(path + folder)]
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        print('reading the images:%s' % folder)
        count = 0
        file_path_list = [os.path.join(folder, file_name) for file_name in os.listdir(folder)
                          if os.path.isfile(os.path.join(folder, file_name))]
        while count < total_count:
            im = random.choice(file_path_list)
            file_info = os.stat(im)
            file_size = file_info.st_size
            if file_size < size_filter:
                continue
            if file_size > 100 * size_filter:
                continue
            img = io.imread(im)
            if img.shape != (w, h, 3):
                continue
            imgs.append(img)
            labels.append(idx)
            count += 1
            print("\rreading {0}/{1}".format(count, total_count), end='')
        print('\r', end='')
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


def read_img(path, total_count, size_filter=2000):
    cate = [path + folder for folder in os.listdir(path) if os.path.isdir(path + folder)]
    images = []
    labels = []
    for idx, folder in enumerate(cate):
        print('reading the images:%s' % folder)
        count = 0
        for im in glob.glob(folder + '/*.jpg'):
            # print('reading the images:%s'%(im))
            file_info = os.stat(im)
            file_size = file_info.st_size
            if file_size < size_filter:
                continue
            if file_size > 100 * size_filter:
                # print(im)
                continue
            img = io.imread(im)
            if img.shape != (w, h, 3):
                print(im)
                continue
            img = io.imread(im)
            images.append(img)
            labels.append(idx)
            count += 1
            if count >= total_count:
                break
            print("\rreading {0}/{1}".format(count, total_count), end='')
        print('\r', end='')
    return np.asarray(images, np.float32), np.asarray(labels, np.int32)


w = 100
h = 100
c = 3
image_count = 10000
input_shape = (w, h, c)
learning_rate = 0.0001
regularization_rate = 0.0001
category_count = 8
n_epoch = 200
mini_batch_size = 100
# data set path
image_path = 'c:/Users/bunny/Desktop/dataset2/data_50000/'

model = Sequential()

# Layer 1
model.add(Conv2D(32,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Layer 2
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 3
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 4
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# flatten
model.add(Flatten(input_shape=input_shape))

# fc layers
model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(regularization_rate)))
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(regularization_rate)))
model.add(Dense(category_count, activation='softmax', kernel_regularizer=regularizers.l2(regularization_rate)))

# read image
data, label = read_img_random(image_path, image_count)

# shuffle
num_example = data.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
data = data[arr]
label = label[arr]

# divide into train set and test set
ratio = 0.9
s = np.int(num_example * ratio)
data /= 255
x_train = data[:s]
y_train = label[:s]
x_val = data[s:]
y_val = label[s:]

# reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
# because the MNIST is greyscale, we only have a single channel - RGB colour images would have 3
# x_train = x_train.reshape(x_train.shape[0], w, h, c)
# x_val = x_val.reshape(x_val.shape[0], w, h, c)
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_val.shape[0], 'test samples')

# convert class vectors to binary class matrices - this is for use in the
# categorical_crossentropy loss below
y_train = keras.utils.to_categorical(y_train, category_count)
y_val = keras.utils.to_categorical(y_val, category_count)

# train
import tensorflow as tf
from tensorflow.python.keras.backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
# train
history = AccuracyHistory()
model.compile(loss=keras.losses.categorical_crossentropy,
              # optimizer=keras.optimizers.SGD(lr=0.01),
              optimizer=keras.optimizers.Adam(lr=learning_rate),
              metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train,
          batch_size=mini_batch_size,
          epochs=n_epoch,
          verbose=2,
          validation_data=(x_val, y_val),
          callbacks=[history])
model.save_weights(image_path+'/model.h5')
score = model.evaluate(x_val, y_val, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(range(1, n_epoch + 1), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
