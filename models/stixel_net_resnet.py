#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from keras.layers.merge import concatenate


# function for creating an identity or projection residual module
def residual_module(layer_in, n_filters):
    merge_input = layer_in
    # check if the number of filters needs to be increase, assumes channels last format
    if layer_in.shape[-1] != n_filters:
        merge_input = layers.Conv2D(n_filters, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(
            layer_in)
    # conv1
    conv1 = layers.Conv2D(n_filters, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
    # conv2
    conv2 = layers.Conv2D(n_filters, (3, 3), padding='same', activation='linear', kernel_initializer='he_normal')(conv1)
    # add filters, assumes filters/channels last
    layer_out = layers.add([conv2, merge_input])
    # activation function
    layer_out = layers.Activation('relu')(layer_out)
    return layer_out


def build_stixel_net(input_shape=(1280, 1920, 3)):
    # input_shape -> (height, width, channel)
    # shape=(256, 256, 3))

    img_input = keras.Input(shape=input_shape)

    x= layers.Conv2D(64, (7, 7), padding='same', activation='relu')(img_input)
    # add res block 1
    x = residual_module(x, 64)
    x = residual_module(x, 64)
    x = residual_module(x, 64)


    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)

    x = residual_module(x, 128)
    x = residual_module(x, 128)
    x = residual_module(x, 128)
    x = residual_module(x, 128)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)

    x = residual_module(x, 256)
    x = residual_module(x, 256)
    x = residual_module(x, 256)
    x = residual_module(x, 256)
    x = residual_module(x, 256)
    x = residual_module(x, 256)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)

    x = residual_module(x, 512)
    x = residual_module(x, 512)
    x = residual_module(x, 512)

    #mymodel
    x = layers.Dropout(0.4)(x)

    x = layers.Conv2D(1023, (3, 3), strides=(2, 1), padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(1023, (3, 3), strides=(2, 1), padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1))(x)


    x = layers.Dropout(0.4)(x)

    x = layers.Conv2D(2048, (3, 1), strides=(2, 1), padding="valid")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1))(x)
    x = layers.Conv2D(2048, (1, 3), strides=(1, 1), padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1))(x)
    x = layers.Conv2D(2048, (1, 1), strides=(1, 1))(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 1), strides=(2, 1))(x)

    x = layers.Dropout(0.4)(x)
    x = layers.Conv2D(160, (1, 1), strides=(1, 1), activation="softmax")(x)

    x = layers.Reshape((240, 160))(x)

    model = models.Model(inputs=img_input, outputs=x)

    model.summary()
    return model
