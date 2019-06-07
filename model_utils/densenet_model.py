# For python 2 support
from __future__ import absolute_import, print_function, division, unicode_literals

# import necessary packages
import tensorflow as tf

# Densenet-121 model
def densenet121_xray_model():
    """
    Functional API to define the DenseNet - 121 for CheXpert dataset
    """
    x = tf.keras.layers.Input(shape = (224, 224, 3), name = "input", dtype = "float32")

    # Define DenseNet - 121
    densenet = tf.keras.applications.DenseNet121(input_shape = (224, 224, 3), include_top = False, weights = "imagenet", pooling = "avg")

    out = densenet(x)
    fc1 = tf.keras.layers.Dense(128, activation = "elu", name = "fc1")(out)

    # y outputs
    y1 = tf.keras.layers.Dense(3, activation = "softmax", name = "y1")(fc1)
    y2 = tf.keras.layers.Dense(3, activation = "softmax", name = "y2")(fc1)
    y3 = tf.keras.layers.Dense(3, activation = "softmax", name = "y3")(fc1)
    y4 = tf.keras.layers.Dense(3, activation = "softmax", name = "y4")(fc1)
    y5 = tf.keras.layers.Dense(3, activation = "softmax", name = "y5")(fc1)
    y6 = tf.keras.layers.Dense(3, activation = "softmax", name = "y6")(fc1)
    y7 = tf.keras.layers.Dense(3, activation = "softmax", name = "y7")(fc1)
    y8 = tf.keras.layers.Dense(3, activation = "softmax", name = "y8")(fc1)
    y9 = tf.keras.layers.Dense(3, activation = "softmax", name = "y9")(fc1)
    y10 = tf.keras.layers.Dense(3, activation = "softmax", name = "y10")(fc1)
    y11 = tf.keras.layers.Dense(3, activation = "softmax", name = "y11")(fc1)
    y12 = tf.keras.layers.Dense(3, activation = "softmax", name = "y12")(fc1)
    y13 = tf.keras.layers.Dense(3, activation = "softmax", name = "y13")(fc1)
    y14 = tf.keras.layers.Dense(3, activation = "softmax", name = "y14")(fc1)


    return tf.keras.models.Model(inputs = x, outputs = [y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14])
