"""#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models


def build_stixel_net(input_shape=(1280, 1920, 3)):

    input_shape -> (height, width, channel)

    img_input = keras.Input(shape=input_shape)

    x = layers.Conv2D(
        64, (3, 3), activation="relu", padding="same", name="block1_conv1"
    )(img_input)
    x = layers.Conv2D(
        64, (3, 3), activation="relu", padding="same", name="block1_conv2"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)

    # Block 2
    x = layers.Conv2D(
        128, (3, 3), activation="relu", padding="same", name="block2_conv1"
    )(x)
    x = layers.Conv2D(
        128, (3, 3), activation="relu", padding="same", name="block2_conv2"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)

    # Block 3
    x = layers.Conv2D(
        256, (3, 3), activation="relu", padding="same", name="block3_conv1"
    )(x)
    x = layers.Conv2D(
        256, (3, 3), activation="relu", padding="same", name="block3_conv2"
    )(x)
    x = layers.Conv2D(
        256, (3, 3), activation="relu", padding="same", name="block3_conv3"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)

    # Block 4
    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block4_conv1"
    )(x)
    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block4_conv2"
    )(x)
    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block4_conv3"
    )(x)

    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1))(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1), padding="same")(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1))(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1))(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Conv2D(2048, (3, 1), strides=(1, 1), padding="valid")(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1))(x)
    x = layers.Conv2D(2048, (1, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1))(x)
    x = layers.Conv2D(2048, (1, 1), strides=(1, 1))(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1))(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Conv2D(160, (1, 1), strides=(1, 1), activation="softmax")(x)

    x = layers.Reshape((240, 160))(x)

    model = models.Model(inputs=img_input, outputs=x)

    return model"""

"""#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models


def build_stixel_net(input_shape=(1280, 1920, 3)):

    input_shape -> (height, width, channel)

    img_input = keras.Input(shape=input_shape)

    x = layers.Conv2D(
        64, (3, 3), activation="relu", padding="same", name="block1_conv1"
    )(img_input)
    x = layers.Conv2D(
        64, (3, 3), activation="relu", padding="same", name="block1_conv2"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)

    # Block 2
    x = layers.Conv2D(
        128, (3, 3), activation="relu", padding="same", name="block2_conv1"
    )(x)
    x = layers.Conv2D(
        128, (3, 3), activation="relu", padding="same", name="block2_conv2"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)

    # Block 3
    x = layers.Conv2D(
        256, (3, 3), activation="relu", padding="same", name="block3_conv1"
    )(x)
    x = layers.Conv2D(
        256, (3, 3), activation="relu", padding="same", name="block3_conv2"
    )(x)
    x = layers.Conv2D(
        256, (3, 3), activation="relu", padding="same", name="block3_conv3"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)

    # Block 4
    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block4_conv1"
    )(x)
    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block4_conv2"
    )(x)
    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block4_conv3"
    )(x)

    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1))(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1), padding="same")(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1))(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1))(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Conv2D(2048, (3, 1), strides=(1, 1), padding="valid")(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1))(x)
    x = layers.Conv2D(2048, (1, 3), strides=(1, 1), padding="same")(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1))(x)
    x = layers.Conv2D(2048, (1, 1), strides=(1, 1))(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1))(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Conv2D(160, (1, 1), strides=(1, 1), activation="softmax")(x)

    x = layers.Reshape((240, 160))(x)

    model = models.Model(inputs=img_input, outputs=x)

    return model"""


#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from keras.layers.merge import concatenate

def inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
    # 1x1 conv
    conv1 = layers.Conv2D(f1, (1, 1), padding='same', activation='relu')(layer_in)
    # 3x3 conv
    conv3 = layers.Conv2D(f2_in, (1, 1), padding='same', activation='relu')(layer_in)
    conv3 = layers.Conv2D(f2_out, (3, 3), padding='same', activation='relu')(conv3)
    # 5x5 conv
    conv5 = layers.Conv2D(f3_in, (1, 1), padding='same', activation='relu')(layer_in)
    conv5 = layers.Conv2D(f3_out, (5, 5), padding='same', activation='relu')(conv5)
    # 3x3 max pooling
    pool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(layer_in)
    pool = layers.Conv2D(f4_out, (1, 1), padding='same', activation='relu')(pool)
    # concatenate filters, assumes filters/channels last
    layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
    return layer_out


def build_stixel_net(input_shape=(1280, 1920, 3)):

    #input_shape -> (height, width, channel)
    #shape=(256, 256, 3))

    img_input = keras.Input(shape=input_shape)

    x = layers.Conv2D(64, (7, 7), activation="relu")(img_input)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), name="block1_pool")(x)
    x = layers.Conv2D(192, (3, 3), activation="relu", strides=(1, 1))(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), name="block1_pool")(x)

    # add inception block 1
    x = inception_module(x, 64, 96, 128, 16, 32, 32)
    # add inception block 2
    x = inception_module(x, 128, 128, 192, 32, 96, 64)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2), name="block2_pool")(x)

    x = inception_module(x, 192, 96, 208, 16, 48, 64)

    x = inception_module(x, 160, 112, 224, 24, 64, 64)

    x = inception_module(x, 128, 128, 256, 24, 64, 64)

    x = inception_module(x, 112, 144, 288, 32, 64, 64)

    x = inception_module(x, 256, 160, 320, 32, 128, 128)

    x = inception_module(x, 160, 112, 224, 24, 64, 64)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2), name="block3_pool")(x)

    x = inception_module(x, 256, 160, 320, 32, 128, 128)

    x = inception_module(x, 384, 192, 384, 48, 128, 128)

    x = layers.MaxPooling2D((7, 7), strides=(1, 1), name="block4_pool")(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Conv2D(2048, (1, 1), activation="relu", strides=(1, 1))(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 1), name="block5_pool")(x)

    x = layers.Conv2D(160, (1, 1), strides=(1, 1), activation="softmax")(x)
    # summarize model

    x = layers.Reshape((240, 160))(x)

    model = models.Model(inputs=img_input, outputs=x)

    return model