from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, cohen_kappa_score
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
def cal_mae_acc_cls(logits, targets, is_sto=True):
    if is_sto:
        r_dim, s_dim, out_dim = logits.shape
        label_arr = torch.arange(0, out_dim).float().cuda()
        probs = F.softmax(logits, -1)
        exp = torch.sum(probs * label_arr, dim=-1)
        exp = torch.mean(exp, dim=0)
        max_a = torch.mean(probs, dim=0)
        max_data = max_a.cpu().data.numpy()
        max_data = np.argmax(max_data, axis=1)
        target_data = targets.cpu().data.numpy()
        exp_data = exp.cpu().data.numpy()
        mae = sum(abs(exp_data - target_data)) * 1.0 / len(target_data)
        acc = sum(np.rint(exp_data) == target_data) * 1.0 / len(target_data)

    else:
        s_dim, out_dim = logits.shape
        probs = F.softmax(logits, -1)
        probs_data = probs.cpu().data.numpy()
        target_data = targets.cpu().data.numpy()
        max_data = np.argmax(probs_data, axis=1)
        label_arr = np.array(range(out_dim))
        exp_data = np.sum(probs_data * label_arr, axis=1)
        mae = sum(abs(exp_data - target_data)) * 1.0 / len(target_data)
        acc = sum(np.rint(exp_data) == target_data) * 1.0 / len(target_data)

    return acc, mae

def accuracy(score, target):
    _, pred = score.max(1)
    equal = (pred == target).float()
    acc = accuracy_score(target.cpu().data.numpy(), pred.cpu().data.numpy())  # the same as equal.mean().data[0]
    # print('equal',equal,equal.mean())
    return equal.mean().item()

def Arg_MAE(score, target):
    _, pred = score.max(1)
    mae = sum((pred-target)**2)
    return mae

def MAE(score, target):
    # _, pred = score.max(1)
    # l1loss = nn.L1Loss()
    # target = target.float()
    # mae = l1loss(pred, target)

    s_dim, out_dim = score.shape
    probs = F.softmax(score, -1)
    probs_data = probs.cpu().data.numpy()
    target_data = target.cpu().data.numpy()
    max_data = np.argmax(probs_data, axis=1)
    label_arr = np.array(range(out_dim))
    exp_data = np.sum(probs_data * label_arr, axis=1)
    mae = sum(abs(exp_data - target_data)) * 1.0 / len(target_data)


    return mae

def MAE2(score, target):
    return abs(target - torch.argmax(score, -1)).float().mean()

def seperate_acc(score, target):
    o4, o2, o1 = score
    _, o4 = o4.squeeze().max(1)
    _, o2 = o2.squeeze().max(1)
    _, o1 = o1.squeeze().max(1)

    target = target.float()
    l4 = (target // 4).unsqueeze(dim=1)
    l2 = ((target % 4) // 2).unsqueeze(dim=1)
    l1 = (target % 2).unsqueeze(dim=1)

    l1loss = nn.L1Loss()

    acc_4 = (o4 == l4).float()
    acc_2 = (o2 == l2).float()
    acc_1 = (o1 == l1).float()

    acc = (o4 * 4 + o2 * 2 + o1 == l4 * 4 + l2 * 2 + l1).float()

    mae = l1loss(target, target)# 没用的

    return acc_4.mean().item(), acc_2.mean().item(), acc_1.mean().item(), acc.mean().item(),mae

def accuracy_new(score, target):
    _, p4 = score[0].squeeze().max(1)
    _, p2 = score[1].squeeze().max(1)
    _, p1 = score[2].squeeze().max(1)
    pred = p1 + p2 * 2 + p4 * 4
    # target = target[0] * 4 + target[1] * 2 + target[2] * 1

    l1loss = nn.L1Loss()
    target = target.float()
    mae = l1loss(pred, target)

    equal = (pred == target).float()
    # acc = accuracy_score(target.cpu().data.numpy(), pred.cpu().data.numpy())  # the same as equal.mean().data[0]
    # print('equal',equal,equal.mean())
    return equal.mean().item(), mae