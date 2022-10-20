#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
#import visualkeras


def build_stixel_net(input_shape=(1280, 1920, 3), visualize=False):
    """
    input_shape -> (height, width, channel)
    """
    model = models.Sequential()

    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv1", input_shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv2"))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool"))
    if visualize:
        model.add(visualkeras.SpacingDummyLayer())

    # Block 2
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv1"))
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv2"))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool"))
    if visualize:
        model.add(visualkeras.SpacingDummyLayer())

    # Block 3
    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv1"))
    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv2"))
    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv3"))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool"))
    if visualize:
        model.add(visualkeras.SpacingDummyLayer())

    # Block 4
    model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv1"))
    model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv2"))
    model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv3"))
    if visualize:
        model.add(visualkeras.SpacingDummyLayer())

    model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same"))
    model.add(layers.ELU())
    model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same"))
    model.add(layers.ELU())
    model.add(layers.MaxPooling2D((2, 1), strides=(2, 1)))

    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same"))
    model.add(layers.ELU())
    model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same"))
    model.add(layers.ELU())
    model.add(layers.MaxPooling2D((2, 1), strides=(2, 1), padding="same"))

    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same"))
    model.add(layers.ELU())
    model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same"))
    model.add(layers.ELU())
    model.add(layers.MaxPooling2D((2, 1), strides=(2, 1)))

    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same"))
    model.add(layers.ELU())
    model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same"))
    model.add(layers.ELU())
    model.add(layers.MaxPooling2D((2, 1), strides=(2, 1)))

    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(2048, (3, 1), strides=(1, 1), padding="valid"))
    model.add(layers.ELU())
    model.add(layers.MaxPooling2D((2, 1), strides=(2, 1)))
    model.add(layers.Conv2D(2048, (1, 3), strides=(1, 1), padding="same"))
    model.add(layers.ELU())
    model.add(layers.MaxPooling2D((2, 1), strides=(2, 1)))
    model.add(layers.Conv2D(2048, (1, 1), strides=(1, 1)))
    model.add(layers.ELU())
    model.add( layers.MaxPooling2D((2, 1), strides=(2, 1)))

    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(160, (1, 1), strides=(1, 1), activation="softmax"))
    if visualize:
        model.add(visualkeras.SpacingDummyLayer())

    model.add(layers.Reshape((240, 160)))

    return model
