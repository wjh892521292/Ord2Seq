import torch
import numpy as np
from scipy.stats import beta
import math
import random


__all__ = ['cal']



def cal(func, value, lam, add=False, **kwargs):
    if lam:
        if add:
            return func(value[0], **kwargs) * lam + func(value[1], **kwargs) * (1 - lam)
        else:
            return func(value[0], **kwargs), func(value[1], **kwargs)
    else:
        return func(value, **kwargs)

