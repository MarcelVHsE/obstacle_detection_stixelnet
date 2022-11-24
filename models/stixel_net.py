#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from keras.layers.merge import concatenate



def build_stixel_net(input_shape=(1280, 1920, 3)):
    model = models.Sequential()
    
    #block 1
    model.add(layers.Conv2D(64, (7, 7), padding='same', activation='relu',name="block1_conv1", input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool"))
    #block 2
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="block2_conv1"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="block2_conv2"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="block2_conv3"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="block2_conv4"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="block2_conv5"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="block2_conv6"))
    #block 3
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="block3_conv1"))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool"))
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="block3_conv2"))
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="block3_conv3"))
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="block3_conv4"))
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="block3_conv5"))
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="block3_conv6"))
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="block3_conv7"))
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="block3_conv8"))
    
    #block 4
    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="block4_conv1"))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool"))
    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="block4_conv2"))
    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="block4_conv3"))
    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="block4_conv4"))
    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="block4_conv5"))
    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="block4_conv6"))
    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="block4_conv7"))
    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="block4_conv8"))
    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="block4_conv9"))
    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="block4_conv10"))
    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="block4_conv11"))
    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="block4_conv12"))

    # Block 5
    model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv1"))
    #model.add(layers.MaxPooling2D((2, 1), strides=(2, 1), name="block5_pool"))
    model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv2"))
    model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv3"))
    model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv4"))
    model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv5"))
    model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv6"))
    
    model.add(layers.Dropout(0.4))
    model.add(layers.Conv2D(1023, (3, 3), strides=(2, 1), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D((2, 1), strides=(2, 1), name="block6_pool"))
    model.add(layers.Conv2D(1023, (3, 3), strides=(2, 1),activation="relu", padding="same", name="block6_conv6"))
    
    model.add(layers.Conv2D(1023, (3, 3), strides=(2, 1), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D((2, 1), strides=(2, 1), name="block7_pool"))
    model.add(layers.Conv2D(1023, (3, 3), strides=(2, 1),activation="relu", padding="same"))
    model.add(layers.Dropout(0.4))
    model.add(layers.Conv2D(1023, (3, 3), strides=(2, 1), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D((2, 1), strides=(2, 1), name="block8_pool"))

    model.add(layers.Conv2D(160, (1, 1), strides=(1, 1), activation="softmax"))

    model.add(layers.Reshape((240, 160)))

    model.summary()
    return model
