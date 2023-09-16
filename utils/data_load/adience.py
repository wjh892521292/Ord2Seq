import torch
import torch.utils.data as data_utils
import numpy as np
import os
from PIL import Image, ImageDraw
import cv2
from xpinyin import Pinyin
from PIL import Image
import copy

class MyDataset(data_utils.Dataset):

    def __init__(self, img_root, data_root, dataset, transform=None, fold=4):
        self.data_list = []
        self.transform = transform
        '''
        同时倒入frontal和普通的，
        frontal就是对应的aligned
        普通的对应的是faces
        '''
        root = '/data2/chengyi/dataset/ord_reg/AdienceBenchmarkGenderAndAgeClassification/AgeGenderDeepLearning/Folds/train_val_txt_files_per_fold'



        train = 'age_train.txt'
        val = 'age_val.txt'
        test = 'age_test.txt'
        sub = 'age_train_subset.txt'

        if dataset == 'train':
            file_name = [train]
        else:
            file_name = [test]

        for each in file_name:
            f_path = root + '/test_fold_is_' + str(fold) + '/' + each
            with open(f_path, 'r') as f:
                l = f.readlines()
                self.data_list.extend(l)

        label = [0,0,0,0,0,0,0,0]
        for each in self.data_list:
            label[int(each[-2])] += 1

        print(label)




        # print(len(self.data_list))
        # label:
        # [1601, 1172, 1469, 969, 2770, 1271, 413, 401, 0]
        # 15.9---11.64---14.59---9.62---27.51---12.62---4.1---3.98
    def __getitem__(self, idx):
        item = copy.deepcopy(self.data_list[idx])
        img_path = item[:-3]
        label = int(item[-2])
        # img_path = item[2]
        # label = item[-1]
        img_path = '/data2/wangjinhong/data/ord_reg/data/aligned/' + img_path
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        # label = torch.tensor(float(label), dtype=torch.float32)

        return img, label

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    # MyDataset(img_root=None, data_root='/data2/wangjinhong/data/ord_reg/data/', dataset='valid', transform=None, fold=1)
    # MyDataset(img_root=None, data_root='/data2/wangjinhong/data/ord_reg/data/', dataset='train', transform=None, fold=1)

    train = 'age_train.txt'
    data_list = []
    test = 'age_test.txt'
    root = '/data2/chengyi/dataset/ord_reg/AdienceBenchmarkGenderAndAgeClassification/AgeGenderDeepLearning/Folds/train_val_txt_files_per_fold'
    for fold in range(5):
        for each in [train, test]:
            f_path = root + '/test_fold_is_' + str(fold) + '/' + each
            with open(f_path, 'r') as f:
                l = f.readlines()
                data_list.extend(l)

    data_list = list(set(data_list))
    print(len(data_list))
    label = [0 for _ in range(8)]
    for each in data_list:
        label[int(each[-2])] += 1

    print(label)


