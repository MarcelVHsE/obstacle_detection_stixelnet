import numpy as np
import cv2
import os
import yaml
import matplotlib.pyplot as plt


class DataLoader:
    def __init__(self, idx=None):
        with open('config.yaml') as yamlfile:
            self.config = yaml.load(yamlfile, Loader=yaml.FullLoader)

        if idx is not None:
            self.num_datasets = 1
            self.idx = idx
            pred_name = self.config['names']['prediction'] + str(self.idx) + self.config['endings']['prediction']
            trgt_name = self.config['names']['target'] + str(self.idx) + self.config['endings']['target']
            img_name = self.config['names']['image'] + str(self.idx) + self.config['endings']['image']
            # result_name = "result_" + str(self.num) + ".png"

            pred_path = os.path.join(os.getcwd(), self.config['predictions'], pred_name)
            trgt_path = os.path.join(os.getcwd(), self.config['targets'], trgt_name)
            img_path = os.path.join(os.getcwd(), self.config['images'], img_name)

            assert os.path.isfile(pred_path)
            self.predict = np.loadtxt(pred_path)
            """ predict has the shape of the net output thus: 240 (cols), 160 (confidence for GT row)
                    [3.12404239e-07 3.81521829e-07 2.74979442e-07 ... 1.83816603e-03 7.12508219e-04 2.21388228e-03] in sum 160
                    [2.33957076e-09 2.75518475e-09 ...
                    ... 240 times"""
            assert os.path.isfile(trgt_path)
            self.target = np.loadtxt(trgt_path)
            """ targets has the correct GT and shape: 240 (cols), 2 (0 = no gt, 1 = gt + the correct row)
                    [  1.          92.        ]
                    [  0.           0.50999999]
                    ... 240 times"""
            assert os.path.isfile(img_path)
            self.image = cv2.imread(img_path)

            # self.result = cv2.imread(os.path.join(os.getcwd(), result_folder, result_name))
            # self.draw_stixel_on_image(print_prediction=False)
            # self.draw_stixel_on_image()
            """ Extend here to maybe read a complete folder 
                and provide lists of results, targets, ...
            """
        else:
            predicts = []
            targets = []
            prediction_list = [f for f in os.listdir(os.path.join(os.getcwd(), self.config['predictions'])) if
                               f.endswith('.txt')]
            target_list = [f for f in os.listdir(os.path.join(os.getcwd(), self.config['targets'])) if
                               f.endswith('.txt')]
            self.num_datasets = len(target_list)
            num_preds = len(target_list)
            assert self.num_datasets == num_preds
            for predict in prediction_list:
                predicts.append(np.loadtxt(os.path.join(os.getcwd(), self.config['predictions'], predict)))
            for target in target_list:
                targets.append(np.loadtxt(os.path.join(os.getcwd(), self.config['targets'], target)))
            self.predict = np.concatenate(np.asarray(predicts))
            self.target = np.concatenate(np.asarray(targets))

    def draw_stixel_on_image(self, label_size=(240, 160), print_prediction=True):
        h, w, c = self.image.shape
        label_width = int(self.config['label_size']['x'])
        label_height = int(self.config['label_size']['y'])
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        thickness = 1
        color = (0, 0, 255)
        stixel_pos = self.predict
        if print_prediction:
            data = np.argmax(stixel_pos, axis=1)
        else:
            stixel_pos = self.target
            data = [row[1] for row in stixel_pos]
            color = (0, 255, 0)

        for x, py in enumerate(data):
            x0 = int(x * w / label_width)
            x1 = int(x0 + w / label_size[0])
            start_point = (x0, 0)
            y = int((py + 0.5) * h / label_height)
            end_point = (x1, y)
            # paints from top-left to bottom-right
            cv2.rectangle(img, start_point, end_point, color, thickness)

        plt.figure(figsize=(20, 12))
        plt.imshow(img)
        if print_prediction:
            plt.title("Prediction")
        else:
            plt.title("Ground Truth")
        plt.grid("off")
        plt.show()
