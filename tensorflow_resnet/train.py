#!/usr/bin/env python
import random
import numpy as np
import tensorflow as tf

import resnet
import dataset

class MyModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.shift = tf.keras.layers.RandomTranslation(0.125, 0.125, "constant",
                                                       "nearest")
        self.flip = tf.keras.layers.RandomFlip("horizontal")

        #self.core = resnet.ResNet("resnet18", 10)
        self.core = resnet.ResNet18(10, 5e-4)

    def call(self, inputs):
        x = inputs
        x = self.shift(x)
        x = self.flip(x)
        x = self.core(x)
        outputs = x
        return outputs

# Set seed
seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Load Cifar10
data = dataset.DataCifar10()

# # Normalize data
# data.x_train = (data.x_train - np.mean(data.x_train, axis=(0, 1, 2))) / \
#                 np.std(data.x_train, axis=(0, 1, 2))
# data.x_test = (data.x_test - np.mean(data.x_test, axis=(0, 1, 2))) / \
#                 np.std(data.x_test, axis=(0, 1, 2))

# The model
model = MyModel()

batch_size = 100
epochs = 200

decay_steps = int(epochs*len(data.x_train)*0.9/batch_size)
#cos_decay = tf.keras.optimizers.schedules.CosineDecay(1e-1, decay_steps)
cos_decay = tf.keras.experimental.CosineDecay(1e-1, decay_steps)
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.SGD(cos_decay, momentum=0.9),
              metrics=["accuracy"])

model(data.x_train[:2])
model.summary()

callbacks = [tf.keras.callbacks.ModelCheckpoint("weights.h5",
                                                save_best_only=True,
                                                save_weights_only=True),
             tf.keras.callbacks.TensorBoard(log_dir="logs")]

history = model.fit(data.x_train, data.y_train, batch_size=batch_size,
                    epochs=epochs, validation_split=0.1, callbacks=callbacks,
                    verbose=2)

model.load_weights("weights.h5")
loss, acc = model.evaluate(data.x_test, data.y_test)

print(f"Test loss={loss} Test accruacy={acc}")
