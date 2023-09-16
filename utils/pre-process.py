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

if __name__ == '__main__':

    # ================================ make dataset ================================
    all_img = []
    count = 0
    with open('/data2/chengyi/dataset/ord_reg/trainLabels.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            all_img.append(row[0])

    for img_name in all_img:
        # img = all_img[0]
        img_path = '/data2/wangjinhong/data/ord_reg/DR_data/train/' + img_name + '.jpeg'
        # label = int(item[-1])
        img = Image.open(img_path).convert('RGB')

        width, height = img.size

        shorter_len = min(width, height)

        newsize = (512, 512)
        img = img.resize(newsize)
        save_path = '/data2/chengyi/dataset/ord_reg/DR_dataset/train/' + img_name + '.jpg'
        img.save(save_path, quality=95)
