import sys
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import early_scheduler as es

class DataMNIST:
    def __init__(self):
        (x_train, y_train), \
            (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        self.x_train = x_train.astype(np.float32)[..., None] / 255
        self.x_test = x_test.astype(np.float32)[..., None] / 255
        self.y_train = tf.keras.utils.to_categorical(y_train)
        self.y_test = tf.keras.utils.to_categorical(y_test)

class BaseModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv0 = tf.keras.layers.Conv2D(8, (3, 3), (1, 1), "same")
        self.conv1 = tf.keras.layers.Conv2D(8, (3, 3), (1, 1), "valid")
        self.conv2 = tf.keras.layers.Conv2D(8, (3, 3), (1, 1), "valid")
        self.dense3 = tf.keras.layers.Dense(10)

    def call(self, inputs):
        x = inputs

        x = self.conv0(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.nn.relu(x)

        x = self.conv1(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.nn.relu(x)

        x = tf.keras.layers.Flatten()(x)
        x = self.dense3(x)
        x = tf.nn.softmax(x)

        outputs = x
        return outputs
    
def train_model(data, model, opt_type, lr, epochs, batch_size, seed,
                verbose=0):

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    if opt_type == "sgd":
        optimizer = tf.keras.optimizers.SGD(lr)
    elif opt_type == "adam":
        optimizer = tf.keras.optimizers.Adam(lr)
    else:
        assert False, "Invalid optimizer type"

    validation_split = 0.1

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=optimizer, metrics=["accuracy"])

    patience = 15
    #callbacks = [tf.keras.callbacks.EarlyStopping(patience=patience,
    callbacks = [es.EarlyScheduler(patience=patience,
                                   restore_best_weights=True, verbose=1)]

    history = model.fit(data.x_train, data.y_train, batch_size=batch_size,
                        epochs=epochs, validation_split=validation_split, callbacks=callbacks,
                        verbose=verbose)

    loss, acc = model.evaluate(data.x_test, data.y_test)

    return loss, acc, history.history

data = DataMNIST()

model = BaseModel()
model(data.x_train[:2])

opt_type = "adam"
lr = 1e-3
epochs = 500
batch_size = 100
seed = 0
verbose = 2
loss, acc, hist = train_model(data, model, opt_type, lr, epochs, batch_size,
                              seed, verbose)

plt.plot(hist["loss"])
plt.plot(hist["val_loss"])
plt.grid()
plt.show()
