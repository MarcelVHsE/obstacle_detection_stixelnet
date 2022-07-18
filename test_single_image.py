#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import cv2
import tqdm as tqdm
import matplotlib.pyplot as plt
from models import build_stixel_net
from data_loader import WaymoStixelDataset
from albumentations import (
    Compose,
    Resize,
    Normalize,
)
import tensorflow.keras.backend as K

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True)
# parser.add_argument(
#     "--image_path", re=True
# )
parsed_args = parser.parse_args()

def rgba(r, r_max=160):
    """Generates a color based on range.
  Args:
    r: the range value of a given point.
    r_max:
  Returns:
    The color for a given range
  """
    c = plt.get_cmap('jet_r')(r / r_max)
    c = list(c)
    c[-1] = 0.7  # alpha
    return c

def test_single_image(model, img, label_size=(240, 320), max_distance=160):
    assert img is not None

    h, w, c = img.shape
    thickness = 2
    val_aug = Compose([Resize(1280, 1920), Normalize(p=1.0)])
    aug_img = val_aug(image=img)["image"]
    aug_img = aug_img[np.newaxis, :]
    predict = model.predict(aug_img, batch_size=1)
    predict = K.reshape(predict, label_size)

    pos_predict = predict[0:240, 0:160]
    pos_predict = K.eval(K.argmax(pos_predict, axis=-1))
    depth_predict = predict[0:240, 160:320]
    depth_predict = K.eval(K.argmax(depth_predict, axis=-1))

    i = 0
    for x, py in enumerate(pos_predict):
        x0 = int(x * w / 240)
        x1 = int(x0 + w / label_size[0])
        range_color = rgba(depth_predict[i], r_max=max_distance)
        range_color = [entry * 255 for entry in range_color]
        start_point = (x0, 0)
        y = int((py + 0.5) * h / 160)
        end_point = (x1, y)
        # paints from top-left to bottom-right
        cv2.rectangle(img, start_point, end_point, range_color, thickness)
        i += 1
    print(depth_predict)
    return img


def main(args):
    print(args.model_path)
    assert os.path.isfile(args.model_path)
    # assert os.path.isfile(args.image_path)
    from config import Config

    dt_config = Config()
    model = build_stixel_net()
    model.load_weights(args.model_path)
    val_set = WaymoStixelDataset(
        data_path=dt_config.DATA_PATH,
        ground_truth_path=os.path.join(dt_config.DATA_PATH, "waymo_val_depth.txt"),
        batch_size=1,
        input_shape=None,
    )

    indices = (
        80,
        300,
    )
    for i, idx in tqdm.tqdm(enumerate(indices)):
        img, _ = val_set[idx]
        img = img[0]

        result = test_single_image(model, img)
        cv2.imwrite("result{}.png".format(i), result)


if __name__ == "__main__":
    main(parsed_args)
