import numpy as np
import tensorflow as tf

class Fire(tf.keras.layers.Layer):
    def __init__(self, ich, och, sch, weight_decay=0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Squeeze layer
        self.convS = tf.keras.layers.Conv2D\
            (
                sch, 1, 1, "SAME",
                kernel_regularizer=tf.keras.regularizers.L2(weight_decay)
            )
        self.bnS = tf.keras.layers.BatchNormalization()

        # Expand 1x1
        self.conv1 = tf.keras.layers.Conv2D\
            (
                och//2, 1, 1, "SAME",
                kernel_regularizer=tf.keras.regularizers.L2(weight_decay)
            )
        self.bn1 = tf.keras.layers.BatchNormalization()

        # Expand 3x3
        self.conv3 = tf.keras.layers.Conv2D\
            (
                och//2, 3, 1, "SAME",
                kernel_regularizer=tf.keras.regularizers.L2(weight_decay)
            )
        self.bn3 = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        x = inputs

        x = self.convS(x)
        x = self.bnS(x)
        x = tf.nn.relu(x)

        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = tf.nn.relu(x1)

        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = tf.nn.relu(x3)

        outputs = tf.keras.layers.Concatenate()([x1, x3])
        return outputs

class SqueezeNet(tf.keras.layers.Layer):
    def __init__(self, num_labels, *args, **kwargs):
        super().__init__(*args, **kwargs)

        weight_decay = 0.0002

        self.conv1 = tf.keras.layers.Conv2D\
            (
                96, 3, 1, "SAME",
                kernel_regularizer=tf.keras.regularizers.L2(weight_decay)
            )
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.fire2 = Fire(96, 128, 16, weight_decay)
        self.fire3 = Fire(128, 128, 16, weight_decay)
        self.fire4 = Fire(128, 256, 32, weight_decay)
        self.fire5 = Fire(256, 256, 32, weight_decay)
        self.fire6 = Fire(256, 384, 48, weight_decay)
        self.fire7 = Fire(384, 384, 48, weight_decay)
        self.fire8 = Fire(384, 512, 64, weight_decay)
        self.fire9 = Fire(512, 512, 64, weight_decay)

        self.conv10 = tf.keras.layers.Conv2D\
            (
                num_labels, 1, 1, "SAME",
                kernel_regularizer=tf.keras.regularizers.L2(weight_decay)
            )

    def call(self, inputs):
        x = inputs

        x = self.conv1(x)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        
        x = self.fire2(x)
        x = self.fire3(x) + x
        x = self.fire4(x)
        x = tf.keras.layers.MaxPooling2D()(x)

        x = self.fire5(x) + x
        x = self.fire6(x)
        x = self.fire7(x) + x
        x = self.fire8(x)
        x = tf.keras.layers.MaxPooling2D()(x)

        x = self.fire9(x)

        x = self.conv10(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.nn.softmax(x)

        outputs = x
        return outputs

if __name__ == "__main__":
    pass
