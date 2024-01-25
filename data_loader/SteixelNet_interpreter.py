import os
import pandas as pd


class BaseStixel:
    def __init__(self, column=None, top_row=None, bottom_row=None, depth=42.0, grid_step=8):
        self.column = column
        self.top_row = top_row
        self.bottom_row = bottom_row
        self.position_class = -1
        self.depth = depth
        self.grid_step = grid_step

    def __repr__(self):
        return f"{self.column},{self.top_row},{self.bottom_row},{self.depth}"


class SteixelNetInterpreter:
    def __init__(self, export_path, width=800, height=370):
        self.export_path = os.path.join(export_path, 'cityscapes', 'StixelNet', 'testing', 'predictions_from_StixelNet')
        os.makedirs(self.export_path, exist_ok=True)
        self.img_size = {'width': width, 'height': height}
        self.pred_stixels = []

    def get_stixel(self, prediction):
        stixels = []
        for x, py in enumerate(prediction):
            x0 = int(x * self.img_size['width'] / 100)
            y = int((py + 0.5) * self.img_size['height'] / 50)
            stixels.append(BaseStixel(column=x0, top_row=y, bottom_row=y))
        self.pred_stixels.append(stixels)
        return stixels

    def export_stixels_to_csv(self, name, stixels=None):
        if stixels is None:
            stixels = self.pred_stixels
        target_list = []
        for stixel in stixels:
            target_list.append([f"/{name}.png",
                                int(stixel.column),
                                int(stixel.top_row),
                                int(stixel.bottom_row),
                                int(stixel.position_class),
                                round(stixel.depth, 1)])
        target: pd.DataFrame = pd.DataFrame(target_list)
        target.columns = ['img_path', 'x', 'yT', 'yB', 'class', 'depth']
        # save .csv
        target.to_csv(os.path.join(self.export_path, name + ".csv"), index=False)
