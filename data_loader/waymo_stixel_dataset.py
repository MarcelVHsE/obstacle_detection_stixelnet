#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.data_utils import Sequence


def visualize_stixel(
        image,
    stixel_pos,
    stixel_width=5,
    stixel_height=100,
    color=(0, 255, 0),
):
    result = np.copy(image)
    [
        cv2.rectangle(
            result, (x, y - stixel_height), (x + stixel_width, y), color
        )
        for (x, y) in stixel_pos
    ]

    return result


class WaymoStixelDataset(Sequence):
    def __init__(
        self,
        data_path,
        ground_truth_path,
        # phase="train",
        batch_size=1,
        label_size=(240, 160),  #label_size=(240, 320),
        shuffle=True,
        transform=None,
        random_seed=2019,
        input_shape=(1280, 1920),
        customized_transform=None,
    ):
        """
        input_shape->(height,width)
        """
        super(WaymoStixelDataset, self).__init__()

        assert os.path.isdir(data_path)
        assert os.path.isfile(ground_truth_path)

        self._data_path = os.path.join(data_path, "waymo_stixel_images")
        #self._data_path = os.path.join(data_path, "")
        self._ground_truth_path = ground_truth_path
        self._batch_size = batch_size
        self._label_size = label_size
        self._shuffle = shuffle
        self._transform = transform
        self._input_shape = input_shape
        self._customized_transform = customized_transform

        # each line in ground truth contains the following information
        # series_date series_id frame_id x y point_type(Train/Test)
        # Eg: 09_26 1 55 242 Train
        lines = [
            line.rstrip("\n") for line in open(self._ground_truth_path, "r")
        ]
        assert len(lines) > 0
        lines = [line.split(" ") for line in lines]

        # assert phase in ("train", "val")
        # phase_dict = {"train": "Train", "val": "Test"}

        self._lines = [
            {
                "path": line[0],
                "x": int(line[1]),
                "y": int(line[2]),
            }
            # Do stuff above if the phase (train or val) fits else ignore it
            for line in lines
        ]
        del lines

        self._image_dict = {}
        for line in self._lines:
            cur_base_image_path = line["path"]
            # if path already exist: append stixel data - one image, multiple stixels- otherwise...
            if cur_base_image_path in self._image_dict.keys():
                self._image_dict[cur_base_image_path].append(
                    [line["x"], line["y"]]
                )
            else:
                # ... add a new dict entry
                self._image_dict[cur_base_image_path] = [[line["x"], line["y"]]]

        # list of all image paths
        self._image_paths = list(self._image_dict.keys())
        # list of ground truth data (stixel position)
        self._stixels_pos = list(self._image_dict.values())
        self._indexes = np.arange(len(self._image_paths))

        np.random.seed(random_seed)

    @property
    def batch_size(self):
        return self._batch_size

    def __getitem__(self, idx):
        assert idx < self.__len__()

        ids = self._indexes[
            idx * self._batch_size : (idx + 1) * self._batch_size
        ]

        X, y = self._data_generation(ids)

        return X, y

    def __len__(self):
        return int(np.floor(len(self._image_dict) / self._batch_size))

    def _data_generation(self, list_ids):
        X = []
        y = []
        for i, idx in enumerate(list_ids):
            img = cv2.imread(
                os.path.join(self._data_path, self._image_paths[idx])
            )
            target = self._generate_label_image(idx)
            if self._customized_transform:
                transformed = self._customized_transform(
                    image=img, target=target
                )
                img = transformed["image"]
                target = transformed["target"]
                del transformed

            if self._input_shape:

                img = cv2.resize(
                    img, (self._input_shape[1], self._input_shape[0])
                )
                if self._transform:
                    img = self._transform(image=img)["image"]

            X.append(img)
            y.append(target)

        X = np.stack(X, axis=0)
        y = np.stack(y, axis=0).astype('float32')

        return X, y

    def _generate_label_image(self, idx):
        img = cv2.imread(os.path.join(self._data_path, self._image_paths[idx]))
        plt.imshow(img)
        positions = np.array(self._stixels_pos[idx], dtype=np.float32)
        height, width = img.shape[:2]

        # Normalized
        positions[:, 0] = positions[:, 0] / width
        positions[:, 1] = positions[:, 1] / height

        colnum, binnum = self._label_size
        have_gt = np.zeros((colnum), dtype=np.int)
        gt = np.zeros((colnum), dtype=np.float32)


        for point in positions:
            col_idx = int(point[0] * colnum)
            row_idx = point[1] * binnum

            if have_gt[col_idx] == 1:
                gt[col_idx] = (gt[col_idx] + row_idx) / 2
            else:
                gt[col_idx] = row_idx
                have_gt[col_idx] = 1

        # ???
        # https://numpy.org/doc/stable/reference/generated/numpy.clip.html
        # @stixel_loss stixel_pos = stixel_pos - 0.5
        # 0.1- 48.99 (0.51, 49.49)
        gt = np.clip(gt, 0.51, 159.49)

        return np.stack((have_gt, gt), axis=1)

    def on_epoch_end(self):
        if self._shuffle:
            np.random.shuffle(self._indexes)
            
    def visualize_one_image(self, idx):
        img = cv2.imread(
            os.path.join(
                self._data_path, self._image_paths[idx * self._batch_size]
            )
        )
        #was bringt transform
        if self._transform:
            img = self._transform(image=img)["image"]

        stixel_pos = self._stixels_pos[idx * self._batch_size]

        return visualize_stixel(img, stixel_pos)

    def get_target(self, idx):
        return self._generate_label_image(idx)

    def get_stixel_pos(self, idx):
        return self._stixels_pos[idx]

    def get_idx_list(self):
        return self._indexes
