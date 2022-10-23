#!/usr/bin/env python

import random
import numpy as np
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv0 = tf.keras.layers.Conv2D(8, (3, 3), padding="same")
        self.conv1 = tf.keras.layers.Conv2D(8, (3, 3), padding="valid")
        self.conv2 = tf.keras.layers.Conv2D(8, (3, 3), padding="valid")
        self.dense3 = tf.keras.layers.Dense(128)
        self.dense4 = tf.keras.layers.Dense(10)

    def call(self, x):

        x0 = self.conv0(x)
        x0 = tf.nn.relu(x0)
        x0 = tf.nn.max_pool2d(x0, (2, 2), (2, 2), "VALID")

        x1 = self.conv1(x0)
        x1 = tf.nn.relu(x1)
        x1 = tf.nn.max_pool2d(x1, (2, 2), (2, 2), "VALID")

        x2 = self.conv2(x1)
        x2 = tf.nn.relu(x2)
        x2 = tf.nn.max_pool2d(x2, (2, 2), (2, 2), "VALID")

        x3 = tf.keras.layers.Flatten()(x2)
        x3 = self.dense3(x3)
        x3 = tf.nn.relu(x3)

        x4 = self.dense4(x3)
        x4 = tf.nn.softmax(x4)

        y = x4
        return y

seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype(np.float32)[..., None] / 255
x_test = x_test.astype(np.float32)[..., None] / 255
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

model = MyModel()

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalAccuracy()])

model(x_train[:2])
model.summary()

callbacks = [tf.keras.callbacks.EarlyStopping(patience=5,
                                              restore_best_weights=True),
             tf.keras.callbacks.ModelCheckpoint("weights.h5",
                                                save_best_only=True,
                                                save_weights_only=True)]

model.fit(x_train, y_train,
          batch_size=100, epochs=100,
          callbacks=callbacks,
          validation_split=0.1)
score = model.evaluate(x_test, y_test, verbose=0)
print(f"Test: loss={score[0]} accuracy={score[1]}")
