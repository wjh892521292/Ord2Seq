import random

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

mapping = {'urban': 0, 'people': 1, 'nature': 2, 'animals': 3}

class MyDataset(data_utils.Dataset):

    def __init__(self, img_root, data_root, dataset, transform=None, fold=3):
        self.data_list = []
        self.items = []
        self.transform = transform

        root = '/data2/chengyi/dataset/ord_reg/aesthetics/stratified/'

        '''
        制作分层采样数据集：
        '''
        # subcls = [[[],[],[],[],[]] for _ in range(4)]
        # self.items = []
        # with open(root + 'all.csv', 'r') as f:
        #     reader = csv.reader(f)
        #     next(reader)
        #     for row in reader:
        #         _, id, sub, imgpath, _, label = row
        #         sub = mapping[sub]
        #         item = [id, sub, imgpath, label]
        #         subcls[sub][int(label)].append(item)
        #         # self.items.append(row[1:])
        #
        # train = []
        # valid = []
        # for sub_i in range(4):
        #     for label_j in range(5):
        #         current = subcls[sub_i][label_j]
        #         random.shuffle(current)
        #         interval = len(current) // 4
        #         valid.extend(current[:interval])
        #         train.extend(current[interval:])
        #         pass
        #
        #
        # column = ['id', 'sub_cls', 'img', 'label']
        # test = pd.DataFrame(columns=column, data=valid)
        # test.to_csv(root + 'valid.csv', encoding='gbk')
        #
        # test = pd.DataFrame(columns=column, data=train)
        # test.to_csv(root + 'train.csv', encoding='gbk')

        '''
        加载数据集
        '''
        remove = []
        # f = open('/data2/chengyi/dataset/ord_reg/aesthetics/remove_list_right_poe_stratified.csv', "r")
        # reader = csv.reader(f)
        # next(reader)
        # for row in reader:
        #     remove.append(row[1])
        #
        #
        # f = open('/data2/chengyi/dataset/ord_reg/aesthetics/remove_list_wrong_poe_stratified.csv', "r")
        # reader = csv.reader(f)
        # next(reader)
        # for row in reader:
        #     remove.append(row[1])

        label_list = [0 for _ in range(5)]
        f = open(root + dataset + '_{}.csv'.format(fold), "r")
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row[1] not in remove:
                self.items.append(row[1:])
                label_list[int(row[-1])] += 1

        print(label_list)
        print(len(self.items))

    def __getitem__(self, idx):
        # 'id', 'sub_cls', 'img', 'label'
        item = copy.deepcopy(self.items[idx])
        img_id = item[0]
        img_path = item[2]
        label = int(item[-1])
        img = Image.open(img_path).convert('RGB')
        # sub_cls = mapping[item[1]]
        sub_cls = int(item[1])
        if self.transform:
            img = self.transform(img)
        # mapping[sub_cls],
        return img, label#,img_id,sub_cls  # l0, l1, l2#label

    def __len__(self):
        return len(self.items)


if __name__ == '__main__':
    a = MyDataset(None, None, 'valid')