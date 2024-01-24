import os
import cv2


class KittiDataLoader:
    def __init__(self, root_dir, img_shape=(800, 370)):
        self.root_dir = root_dir
        self.image_folder = os.path.join(root_dir, 'KITTI-dataset', 'testing', 'STEREO_LEFT')
        self.image_map = [img for img in os.listdir(self.image_folder) if img.endswith(".png")]
        self.img_shape = img_shape

    def __len__(self):
        return len(self.image_map)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_map[idx])
        image = cv2.imread(image_path)
        image = cv2.resize(image, self.img_shape)
        return image
