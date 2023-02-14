import tensorflow as tf

class BasicSame(tf.keras.layers.Layer):
    def __init__(self, och, weight_decay=0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv0 = tf.keras.layers.Conv2D\
                     (
                         och, kernel_size=3, strides=1, padding='same',
                         use_bias=False,
                         kernel_regularizer=tf.keras.regularizers.L2(weight_decay)
                     )
        self.bn0 = tf.keras.layers.BatchNormalization\
                   (
                       beta_regularizer=tf.keras.regularizers.L2(weight_decay),
                       gamma_regularizer=tf.keras.regularizers.L2(weight_decay)
                   )
        self.act0 = tf.keras.layers.ReLU()
        self.conv1 = tf.keras.layers.Conv2D\
                     (
                         och, kernel_size=3, strides=1, padding='same',
                         use_bias=False,
                         kernel_regularizer=tf.keras.regularizers.L2(weight_decay)
                     )
        self.bn1 = tf.keras.layers.BatchNormalization\
                   (
                       beta_regularizer=tf.keras.regularizers.L2(weight_decay),
                       gamma_regularizer=tf.keras.regularizers.L2(weight_decay)
                   )
        self.act1 = tf.keras.layers.ReLU()

    def call(self, inputs):
        x = inputs
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.act0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = tf.keras.layers.Add()([inputs, x])
        x = self.act1(x)
        outputs = x
        return outputs

class BasicChange(tf.keras.layers.Layer):
    def __init__(self, och, strides, weight_decay=0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv0 = tf.keras.layers.Conv2D\
                     (
                         och, kernel_size=3, strides=strides, padding='same',
                         use_bias=False,
                         kernel_regularizer=tf.keras.regularizers.L2(weight_decay)
                     )
        self.bn0 = tf.keras.layers.BatchNormalization\
                   (
                       beta_regularizer=tf.keras.regularizers.L2(weight_decay),
                       gamma_regularizer=tf.keras.regularizers.L2(weight_decay)
                   )
        self.act0 = tf.keras.layers.ReLU()
        self.conv1 = tf.keras.layers.Conv2D\
                     (
                         och, kernel_size=3, strides=1, padding='same',
                         use_bias=False,
                         kernel_regularizer=tf.keras.regularizers.L2(weight_decay)
                     )
        self.bn1 = tf.keras.layers.BatchNormalization\
                   (
                       beta_regularizer=tf.keras.regularizers.L2(weight_decay),
                       gamma_regularizer=tf.keras.regularizers.L2(weight_decay)
                   )

        self.shortcut = tf.keras.layers.Conv2D\
                        (
                            och, kernel_size=1, strides=strides, padding='same',
                            use_bias=False,
                            kernel_regularizer=tf.keras.regularizers.L2(weight_decay)
                        )
        self.bns = tf.keras.layers.BatchNormalization\
                   (
                       beta_regularizer=tf.keras.regularizers.L2(weight_decay),
                       gamma_regularizer=tf.keras.regularizers.L2(weight_decay)
                   )
        
        self.act1 = tf.keras.layers.ReLU()

    def call(self, inputs):
        x = inputs

        # main path
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.act0(x)
        x = self.conv1(x)
        x = self.bn1(x)

        # shortcut path
        sc = self.shortcut(inputs)
        sc = self.bns(sc)

        x = tf.keras.layers.Add()([sc, x])
        x = self.act1(x)
        outputs = x
        return outputs

class ResLayerSame(tf.keras.layers.Layer):
    def __init__(self, och, weight_decay=0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.block0 = BasicSame(och, weight_decay)
        self.block1 = BasicSame(och, weight_decay)

    def call(self, inputs):
        x = inputs
        x = self.block0(x)
        x = self.block1(x)
        outputs = x
        return outputs

class ResLayerChange(tf.keras.layers.Layer):
    def __init__(self, och, strides, weight_decay=0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.block0 = BasicChange(och, strides, weight_decay)
        self.block1 = BasicSame(och, weight_decay)

    def call(self, inputs):
        x = inputs
        x = self.block0(x)
        x = self.block1(x)
        outputs = x
        return outputs

class ResNet18(tf.keras.Model):
    def __init__(self, num_labels, weight_decay=0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv0 = tf.keras.layers.Conv2D\
                     (
                         64, kernel_size=3, strides=1, padding='same',
                         use_bias=False,
                         kernel_regularizer=tf.keras.regularizers.L2(weight_decay)
                     )
        self.bn0 = tf.keras.layers.BatchNormalization\
                   (
                       beta_regularizer=tf.keras.regularizers.L2(weight_decay),
                       gamma_regularizer=tf.keras.regularizers.L2(weight_decay)
                   )

        self.l0 = ResLayerSame(64, weight_decay)
        self.l1 = ResLayerChange(128, 2, weight_decay)
        self.l2 = ResLayerChange(256, 2, weight_decay)
        self.l3 = ResLayerChange(512, 2, weight_decay)

        self.fc = tf.keras.layers.Dense\
                  (
                      num_labels,
                      kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
                      bias_regularizer=tf.keras.regularizers.L2(weight_decay)
                  )

    def call(self, inputs):
        x = inputs
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=4)(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.fc(x)
        x = tf.keras.layers.Softmax()(x)
        outputs = x
        return outputs

if __name__ == "__main__":
    pass
