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

        remove = []
        count = 0.

        fold = int(fold)
        set = fold // 5
        fold = fold % 5
        f = open('/data2/chengyi/dataset/ord_reg/historical/data_265/new_rand{set}/{dataset}_{fold}.csv'.format(set=set,
                                                                                                                dataset=dataset,
                                                                                                                fold=fold), "r")

        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row[1] in remove:
                count += 1
            else:
                self.items.append(row[1:])



        print(count)
        print(len(self.items))

    def __getitem__(self, idx):
        item = copy.deepcopy(self.items[idx])
        img_path = item[0]
        label = int(item[1])
        # img_path = '/data2/wangjinhong/data/ord_reg/DR_data/train/' + img + '.jpeg'
        # img_path = '/data2/chengyi/dataset/ord_reg/DR_dataset/train/' + img + '.jpg'

        # label = int(item[-1])
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label  # l0, l1, l2#label


    def __len__(self):
        return len(self.items)


def make_data_set():
    for i in range(10):
        train = []
        val = []
        mapping = {'1930s': 1, '1940s': 2, '1950s': 3, '1960s': 4, '1970s': 5}
        # data_file = '/data2/chengyi/dataset/ord_reg/historical/HistoricalColor-ECCV2012/data'
        data_file = '/data2/chengyi/dataset/ord_reg/historical/data_265'
        # for root, dirs, files in os.walk(data_file):
        #     dirs.remove('folds')
        #     dirs.remove('folds_55')
        #     dirs.remove('folds_55')
        #     dirs.remove('folds_55')
        dirs = ['1930s', '1940s', '1950s', '1960s', '1970s']
        root = data_file
        for dir in dirs:
            cls = mapping[dir]-1
            dir = os.path.join(root, dir)
            files = os.listdir(dir)
            files = [os.path.join(dir, each_file) for each_file in files]
            val_part = random.sample(files, 55)
            train_part = list(set(files) - set(val_part))
            # train_part = random.sample(train_part, 210)
            for each in train_part:
                train.append([each, cls])
                # train[each] = cls
            for each in val_part:
                val.append([each, cls])
                # val[each] = cls
        # break

        random.shuffle(train)
        random.shuffle(val)

        column = ['name', 'label']
        test = pd.DataFrame(columns=column, data=train)
        test.to_csv('/data2/chengyi/dataset/ord_reg/historical/data_265/new_rand4/train_{}.csv'.format(i),
                    encoding='gbk')

        column = ['name', 'label']
        test = pd.DataFrame(columns=column, data=val)
        test.to_csv('/data2/chengyi/dataset/ord_reg/historical/data_265/new_rand4/valid_{}.csv'.format(i),
                    encoding='gbk')


if __name__ == '__main__':
    make_data_set()
    # for i in range(10):
    #     d = MyDataset(None, None, 'valid', fold=i)