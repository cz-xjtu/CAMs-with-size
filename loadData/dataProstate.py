import torch
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image

CAT_LIST = ['0', '1']
CAT_NAME_TO_NUM = dict(zip(CAT_LIST, range(len(CAT_LIST))))


def load_img_name_list(dataset_index_file):
    img_gt_name_list = open(dataset_index_file).read().splitlines()
    img_name_list = [gt_name.split(' ')[0] for gt_name in img_gt_name_list]
    return img_name_list


def load_label_list_sparse(dataset_index_file):
    img_gt_name_list = open(dataset_index_file).read().splitlines()
    gt_list = []
    for img in img_gt_name_list:
        cls_label = np.zeros(len(CAT_LIST), np.float32)
        cat_name = img.split(' ')[1]
        if cat_name in CAT_LIST:
            cat_num = CAT_NAME_TO_NUM[cat_name]
            cls_label[cat_num] = 1.0
        gt_list.append(cls_label)
    return gt_list


def load_label_list_dense(dataset_index_file):
    img_gt_name_list = open(dataset_index_file).read().splitlines()
    gt_list = []
    for img in img_gt_name_list:
        cls_label = np.zeros(1, np.float32)
        cat_name = img.split(' ')[1]
        if cat_name in CAT_LIST:
            cat_num = CAT_NAME_TO_NUM[cat_name]
            cls_label[0] = cat_num
        gt_list.append(cls_label)
    return gt_list


class ImageDataset(Dataset):
    def __init__(self, dataset_index_file, data_root, transform=None):
        self.img_name_list = load_img_name_list(dataset_index_file)
        self.data_root = data_root
        self.transform = transform

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, item):
        name = self.img_name_list[item]
        img_path = os.path.join(self.data_root, name + '.png')
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return name, img


class ClsDataset(ImageDataset):
    def __init__(self, dataset_index_file, data_root, transform=None):
        super(ClsDataset, self).__init__(dataset_index_file, data_root, transform)
        self.label_list = load_label_list_dense(dataset_index_file)

    def __getitem__(self, item):
        name, img = super(ClsDataset, self).__getitem__(item)
        label = torch.from_numpy(self.label_list[item])
        return name, img, label
