#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.losses import Loss
import tensorflow.keras.backend as K
import math
from tensorflow.keras.backend import eval

class StixelLoss(Loss):
    def __init__(
        self, num_bins=160, alpha=1.0, epsilon=0.0001, label_size=(240, 160)
    ):
        super(StixelLoss, self).__init__(name="stixel_loss")
        self._num_bins = num_bins
        self._alpha = alpha
        self._epsilon = epsilon
        self._label_size = label_size

    def call(self, target, predict):
        """
        test
        predict -> (h, w, num_bins)
        target -> (h, w, 2) e.g. shape: (1, 240, 2)
        h(1): is the dimension in this case for the stixel_pos, could be 2 for e.g. stixel_dist
        w(240): is the amount of elements (here: img width) and contains 2 values
            2:  first is if Data (stixel) is available 0=No, 1=Yes
                second is the bin val (e.g. from 0[top] to 160[bottom]
        """
        """
        Depth could look like: target -> (2, 240, 2)
        First Layer is Stixel_Pos
        Second Layer is Stixel_Dist
            [...] 2:first is if Data (stixel) is available 0=No, 1=Yes
                    second is the bin val (e.g. from 0[close] to 160[far]
                - if the dist is more than a specified value it is not available (=0)
                - the distance must be specified dependent on the used lens (hard generalization)
        """

        have_target, stixel_pos = tf.split(target, 2, axis=-1)
        stixel_pos = stixel_pos - 0.5
        stixel_pos = ((stixel_pos - tf.math.floor(stixel_pos)) + tf.math.floor(stixel_pos) + self._epsilon)

        fp = tf.gather(
            predict,
            K.cast(tf.math.floor(stixel_pos), dtype="int32"),
            batch_dims=-1,
        )
        cp = tf.gather(
            predict,
            K.cast(tf.math.ceil(stixel_pos), dtype="int32"),
            batch_dims=-1,
        )

        p = fp * (tf.math.ceil(stixel_pos) - stixel_pos) + cp * (stixel_pos - tf.math.floor(stixel_pos))

        loss = -K.log(p + tf.keras.backend.epsilon()) * have_target
        loss = K.sum(loss) / (K.sum(have_target) + tf.keras.backend.epsilon())
        
        if (tf.math.is_nan(tf.reduce_sum(loss))):
                tf.print(loss)
                tf.print("Loss is NAN")
                tf.print(target)
                tf.print(predict)
                tf.print(fp)
                tf.print(cp)
                tf.print(p)
                tf.print(have_target)
                tf.print(self)

        return loss * self._alpha

    def get_config(self):
        config = {
            "num_bins": self._num_bins,
            "alpha": self._alpha,
            "epsilon": self._epsilon,
            "label_size": self._label_size,
        }

        base_config = super(StixelLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
