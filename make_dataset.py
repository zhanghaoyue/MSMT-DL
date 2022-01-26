import numpy as np
from utils import in_model
from config import opt
import process
import warnings

warnings.filterwarnings("ignore")


class train_Dataset:
    def __init__(self, img_list):
        self.img_path = opt.path_img
        self.img_list = img_list

        return

    def __getitem__(self, idx):
        path_name = self.img_list[idx]

        # for img
        tmp_data = in_model.get_img(self.img_path, path_name)

        img_dict = process.train_preprocess(tmp_data)
        img_list = [img_dict['T1'], img_dict['T2'], img_dict['FL']]
        img = np.array(img_list)
        img = img.astype('float32')

        bbox_array = img_dict['BBOX']
        annot = in_model.get_bbox(bbox_array)
        annot = annot.astype('float32')

        return_list = [path_name, img, annot]

        return return_list

    def __len__(self):
        return len(self.img_list)


class val_Dataset:
    def __init__(self, img_list):
        self.img_path = opt.path_img
        self.img_list = img_list
        return

    def __getitem__(self, idx):
        path_name = self.img_list[idx]

        # for img
        tmp_data = in_model.get_img(self.img_path, path_name)

        img_dict = process.val_preprocess(tmp_data)
        img_list = [img_dict['T1'], img_dict['T2'], img_dict['FL']]
        img = np.array(img_list)
        img = img.astype('float32')

        bbox_array = img_dict['BBOX']
        annot = in_model.get_bbox(bbox_array)
        annot = annot.astype('float32')

        return_list = [path_name, img, annot]

        return return_list

    def __len__(self):
        return len(self.img_list)
