
import numpy as np
from tensorflow.keras.utils import Sequence
import cv2
import tensorflow as tf 
class DataGenerator(Sequence):
    def __init__(self, all_filenames, input_size, batch_size, brightness_range= None, random_flip= False, rotation = False, shuffle = True) -> None:
        super(DataGenerator, self).__init__()
        self.all_filenames = all_filenames
        self.input_size = input_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.brightness_range = brightness_range
        self.random_flip = random_flip
        self.rotation = rotation
        self.on_epoch_end()
    def __len__(self):
        return int(np.floor(len(self.all_filenames) / self.batch_size))
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        all_filenames_temp = [self.all_filenames[k] for k in indexes]
        X, Y = self.__data_generation(all_filenames_temp)
        return X, Y
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.all_filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def __data_generation(self, all_filenames_temp):
        batch = len(all_filenames_temp)
        X = np.empty(shape=(batch, *self.input_size,3))
        Y = np.empty(shape=(batch, *self.input_size, 3))
        for i, (fn, label_fn) in enumerate(all_filenames_temp):
            # 
            x = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2RGB)
            x = cv2.resize(x, self.input_size, cv2.INTER_LANCZOS4)
            x = tf.cast(x, tf.float32)
            x = (x - 127.5) / 127.5

            if self.brightness_range:
              scale_bright = np.random.uniform(self.brightness_range[0], self.brightness_range[1])
              x = x*scale_bright
            #
            y = cv2.cvtColor(cv2.imread(label_fn), cv2.COLOR_BGR2RGB)
            y = cv2.resize(y, self.input_size, cv2.INTER_LANCZOS4)
            y = tf.cast(y, tf.float32)
            y = (y - 127.5) / 127.5
            if self.random_flip:
                if tf.random.uniform(()) > 0.5: 
                    x = tf.image.flip_left_right(x)
                    y = tf.image.flip_left_right(y)
                if tf.random.uniform(()) > 0.5: 
                    x = tf.image.flip_up_down(x)
                    y = tf.image.flip_up_down(y)

            if self.rotation:
              (h, w, c) = x.shape
              angle = np.random.uniform(-self.rotation, self.rotation)
              RotMat = cv2.getRotationMatrix2D(center = (w, h), angle=angle, scale=1)
              x = cv2.warpAffine(x.numpy(), RotMat, (w, h))
              y = cv2.warpAffine(y.numpy(), RotMat, (w, h))
            X[i,] = x
            Y[i,] = y

        return X, Y