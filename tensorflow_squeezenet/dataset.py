import numpy as np
import tensorflow as tf

class DataCifar10:
    def __init__(self):
        (x_train, y_train), \
            (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        self.x_train = x_train.astype(np.float32) / 255
        self.x_test = x_test.astype(np.float32) / 255
        self.y_train = tf.keras.utils.to_categorical(y_train)
        self.y_test = tf.keras.utils.to_categorical(y_test)

if __name__ == "__main__":
    pass
