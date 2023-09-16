import torch
import torch.utils.data as data_utils
import numpy as np
import os
from PIL import Image, ImageDraw
import cv2
from xpinyin import Pinyin
from PIL import Image
import copy
import pandas as pd
import csv
import random


class MyDataset(data_utils.Dataset):

    def __init__(self, img_root, data_root, dataset, transform=None, fold=3):
        self.data_list = []
        self.items = []
        self.transform = transform

        if dataset == 'train':
            data_num = [i for i in range(10) if i != fold]
        elif dataset == 'valid':
            data_num = [fold]


        for i in data_num:
            f = open('/data2/chengyi/dataset/ord_reg/DR_dataset/ten_fold/fold_{}.csv'.format(i), "r")
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.items.append(row[1:])

        # # ================================ make dataset ================================
        # cls = [[] for _ in range(5)]
        # count = 0
        # with open('/data2/chengyi/dataset/ord_reg/trainLabels.csv', 'r') as f:
        #     reader = csv.reader(f)
        #     next(reader)
        #     for row in reader:
        #         # count += 1
        #         # print(count)
        #         #         if row[1] in remove:
        #         #             count += 1
        #         # else:
        #         cls[int(row[1])].append(row[0])
        #         # self.items.append(row[1:])
        #
        # [random.shuffle(x) for x in cls]
        # # K Fold
        # folds = 10
        # mapping = {5: 'five', 10: 'ten'}
        # for i in range(folds):
        #     fold_i = []
        #     for j in range(5):
        #         current = cls[j]
        #         interval = len(current) // folds
        #         if i != folds-1:
        #             now = current[i * interval: (i + 1) * interval]
        #         else:
        #             now = current[i * interval:]
        #         for each in now:
        #             fold_i.append([each, j])
        #             # fold_i[each] = j
        #     column = ['name', 'label']
        #     test = pd.DataFrame(columns=column, data=fold_i)
        #     test.to_csv('/data2/chengyi/dataset/ord_reg/DR_dataset/{}_fold/fold_{}.csv'.format(mapping[folds], i),
        #                 encoding='gbk')

        # print(len(self.items))
        # cls = [0,0,0,0,0]
        # for each in self.items:
        #     cls[int(each[1])] += 1
        # print(cls)

    def __getitem__(self, idx):
        item = copy.deepcopy(self.items[idx])
        img = item[0]
        label = int(item[1])
        # img_path = '/data2/wangjinhong/data/ord_reg/DR_data/train/' + img + '.jpeg'
        img_path = '/data2/chengyi/dataset/ord_reg/DR_dataset/train/' + img + '.jpg'
        # label = int(item[-1])
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label  # l0, l1, l2#label

    def __len__(self):
        return len(self.items)

if __name__ == '__main__':
    for fold in range(10):
        MyDataset(None,None,'valid',fold=fold)