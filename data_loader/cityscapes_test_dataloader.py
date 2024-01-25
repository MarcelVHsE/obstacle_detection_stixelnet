import os
import cv2


class CityscapesDataLoader:
    def __init__(self, root_dir, img_shape=(800, 370)):
        self.root_dir = root_dir
        self.image_folder = os.path.join(root_dir, 'cityscapes', 'StixelNet', 'testing', 'STEREO_LEFT')
        self.image_map = [img for img in os.listdir(self.image_folder) if img.endswith(".png")]
        self.img_shape = img_shape

    def __len__(self):
        return len(self.image_map)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_map[idx])
        image = cv2.imread(image_path)
        if image.shape[0] != self.img_shape[1] or image.shape[1] != self.img_shape[0]:
            image = cv2.resize(image, self.img_shape)
            print(f"Reshape!, Loaded: {image.shape}, desired: {self.img_shape}")
        cv2.imwrite("resize.png", img=image)
        return image, os.path.splitext(self.image_map[idx])[0]
