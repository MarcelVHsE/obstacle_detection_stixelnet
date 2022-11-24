#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import cv2
import tqdm as tqdm
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


def test_single_image(model, img, target, idx, label_size=(240, 160)):
    assert img is not None

    h, w, c = img.shape
    color = (0, 0, 255)
    thickness = 1
    val_aug = Compose([Resize(1280, 1920), Normalize(p=1.0)])
    aug_img = val_aug(image=img)["image"]
    aug_img = aug_img[np.newaxis, :]
    predict = model.predict(aug_img, batch_size=1)
    predict = K.reshape(predict, label_size)
    """Save prediction"""
    np.savetxt('predictss/predict_' + str(idx) + ".txt", predict.numpy())
    """Save target"""
    np.savetxt('targetss/target_' + str(idx) + ".txt", np.squeeze(target))
    predict = K.eval(K.argmax(predict, axis=-1))

    for x, py in enumerate(predict):
        x0 = int(x * w / 240)
        x1 = int(x0 + w/label_size[0])
        start_point = (x0, 0)
        y = int((py + 0.5) * h / 160)
        end_point = (x1, y)
        # paints from top-left to bottom-right
        cv2.rectangle(img, start_point, end_point, color, thickness)
    return img


def main(args):
    assert os.path.isfile(args.model_path)
    # assert os.path.isfile(args.image_path)
    from config import Config

    dt_config = Config()
    model = build_stixel_net()
    model.load_weights(args.model_path)
    val_set = WaymoStixelDataset(
        data_path=dt_config.DATA_PATH,
        ground_truth_path=os.path.join(dt_config.DATA_PATH, "waymo_val.txt"),
        batch_size=1,
        input_shape=None,
    )

    #indices = (        80, 42, 333, 404)
    indices = list(range(0,999))
    for i, idx in tqdm.tqdm(enumerate(indices)):
        img, target = val_set[idx]
        img = img[0]
        # full analysis
        cv2.imwrite("imagess/image_{}.png".format(idx), img)
        result = test_single_image(model, img, target, idx)
        cv2.imwrite("resultss/result_{}.png".format(idx), result)


if __name__ == "__main__":
    main(parsed_args)