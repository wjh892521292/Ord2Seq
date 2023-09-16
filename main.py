# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# from utils.data_load.data1 import MyDataset as data1
# from utils.data_load.data2 import MyDataset as data2

import torch
import torchvision.transforms as transforms
from config.cfg import BaseConfig
from training.trainer import DefaultTrainer
import os
from utils import data_load
import numpy as np
from torchvision.transforms import InterpolationMode


def main(args):
    runseed = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    torch.manual_seed(runseed)
    np.random.seed(runseed)

    dataset = getattr(data_load, args.data_name.lower())

    train_data = dataset(
        img_root=args.img_root,
        data_root=args.data_root,
        dataset='train',
        transform=transforms.Compose([
            transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        fold=args.fold
    )

    train_load = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                             drop_last=True)

    val_data = dataset(
        img_root=args.img_root,
        data_root=args.data_root,
        dataset='valid',
        transform=transforms.Compose([
            transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        fold=args.fold
    )

    val_load = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

    trainer = DefaultTrainer(args)
    trainer.train(train_load, val_load)
    trainer.log_wrong()


if __name__ == '__main__':
    cfg = BaseConfig()

    # gpusss = 5
    loss_name = 'CELoss'
    alpha = 0.
    # pvtsingleutk vggsingleutk vgg0527_two_step vgg0528 vgg0528_new
    for model in ['vgg0528_new']:
        fold = 0
        gpusss = 0
        ww = 1
        weights = '{} 1'.format(ww)
        ckpt_name = 'Rebuttal_ICCV/{}/UTK/'.format(model)
        fixed = '--gpu_id {gpusss} ' \
                '--exp_name no_start_token_deepcopy_weight={ww} ' \
                '--optim Adam ' \
                '--model_name {model} ' \
                '--lr 0.0001 ' \
                '--stepvalues 7500 15000 ' \
                '--max_iter 25000 ' \
                '--weights {weights} ' \
                '--fold {fold} ' \
                '--loss_name {loss_name} ' \
                '--data_name utk ' \
                '--alpha {alpha} ' \
                '--val_freq 40 ' \
                '--num_classes 8 ' \
                '--save_folder /data2/chengyi/ord_reg/result/save_model/checkpoint_{ckpt_name}/ ' \
                '--save_log /data2/chengyi/ord_reg/result/save_log/logs_{ckpt_name}/ '.format(fold=fold,
                                                                                              ckpt_name=ckpt_name,
                                                                                              gpusss=gpusss,
                                                                                              loss_name=loss_name,
                                                                                              alpha=alpha,
                                                                                              model=model,
                                                                                              weights=weights,
                                                                                              ww=ww)\
            .split()
        args = cfg.initialize(fixed)
        main(args)
    # ===================================== vgg0526_cacd
    # for model in ['vgg0527_two_step_cacd']:
    #     fold = 0
    #     gpusss = 8
    #     data_name = 'cacd_val'
    #     ckpt_name = 'Rebuttal_ICCV/{}/CADA_{}/'.format(model, data_name)
    #     fixed = '--gpu_id {gpusss} ' \
    #             '--exp_name fold_{fold} ' \
    #             '--optim Adam ' \
    #             '--model_name {model} ' \
    #             '--lr 0.0001 ' \
    #             '--stepvalues 15000 30000 ' \
    #             '--max_iter 50000 ' \
    #             '--fold {fold} ' \
    #             '--loss_name {loss_name} ' \
    #             '--data_name {data_name} ' \
    #             '--alpha {alpha} ' \
    #             '--val_freq 40 ' \
    #             '--num_classes 8 ' \
    #             '--save_folder /data2/chengyi/ord_reg/result/save_model/checkpoint_{ckpt_name}/ ' \
    #             '--save_log /data2/chengyi/ord_reg/result/save_log/logs_{ckpt_name}/ '.format(fold=fold,
    #                                                                                           ckpt_name=ckpt_name,
    #                                                                                           gpusss=gpusss,
    #                                                                                           loss_name=loss_name,
    #                                                                                           alpha=alpha,
    #                                                                                           model=model,
    #                                                                                           data_name=data_name)\
    #         .split()
    #     args = cfg.initialize(fixed)
    #     main(args)





    # for model in ['vggtrans_tiny']:
    #     fold = 4
    #     gpusss = fold-3
    #     # 'set=256-8-6-1024/'
    #     ckpt_name = 'Rebuttal/{}/set=512-8-3-512_version2/Adience/'.format(model)
    #     fixed = '--gpu_id {gpusss} ' \
    #             '--exp_name fold_{fold} ' \
    #             '--optim Adam ' \
    #             '--model_name {model} ' \
    #             '--lr 0.0001 ' \
    #             '--stepvalues 7500 ' \
    #             '--max_iter 15000 ' \
    #             '--fold {fold} ' \
    #             '--loss_name {loss_name} ' \
    #             '--data_name faces_final ' \
    #             '--alpha {alpha} ' \
    #             '--val_freq 40 ' \
    #             '--num_classes 8 ' \
    #             '--save_folder /data2/chengyi/ord_reg/result/save_model/checkpoint_{ckpt_name}/ ' \
    #             '--save_log /data2/chengyi/ord_reg/result/save_log/logs_{ckpt_name}/ '.format(fold=fold,
    #                                                                                           ckpt_name=ckpt_name,
    #                                                                                           gpusss=gpusss,
    #                                                                                           loss_name=loss_name,
    #                                                                                           alpha=alpha,
    #                                                                                           model=model)\
    #         .split()
    #     args = cfg.initialize(fixed)
    #     main(args)

    # for fold in range(5):

    #     gpusss = 2
    #     mix_mode = 'fmix'
    #     model = 'PvtMix'
    #     # mixup cutmix fmix
    #     # 'set=256-8-6-1024/'
    #     ckpt_name = 'Mixups/{}/Adience_Remake2/'.format(mix_mode)
    #     fixed = '--gpu_id {gpusss} ' \
    #             '--exp_name fold_{fold} ' \
    #             '--optim Adam ' \
    #             '--mix_mode {mix_mode} ' \
    #             '--model_name {model} ' \
    #             '--lr 0.0001 ' \
    #             '--stepvalues 7500 ' \
    #             '--max_iter 15000 ' \
    #             '--fold {fold} ' \
    #             '--loss_name {loss_name} ' \
    #             '--data_name faces_final ' \
    #             '--alpha {alpha} ' \
    #             '--val_freq 400 ' \
    #             '--num_classes 8 ' \
    #             '--save_folder /data2/chengyi/ord_reg/result/save_model/checkpoint_{ckpt_name}/ ' \
    #             '--save_log /data2/chengyi/ord_reg/result/save_log/logs_{ckpt_name}/ '.format(fold=fold,
    #                                                                                           ckpt_name=ckpt_name,
    #                                                                                           gpusss=gpusss,
    #                                                                                           loss_name=loss_name,
    #                                                                                           alpha=alpha,
    #                                                                                           model=model,
    #                                                                                           mix_mode=mix_mode)\
    #         .split()
    #     args = cfg.initialize(fixed)
    #     main(args)


    # for fold in range(20):
    #     model = 'vgglstm'
    #     gpusss = 8
    #     ckpt_name = 'Rebuttal/{}/His'.format(model)
    #     fixed = '--gpu_id {gpusss} ' \
    #             '--exp_name fold_{fold} ' \
    #             '--optim Adam ' \
    #             '--model_name {model} ' \
    #             '--lr 0.0001 ' \
    #             '--stepvalues 16000 ' \
    #             '--max_iter 1700 ' \
    #             '--fold {fold} ' \
    #             '--loss_name {loss_name} ' \
    #             '--data_name historical_all ' \
    #             '--alpha {alpha} ' \
    #             '--val_freq 10 ' \
    #             '--num_classes 5 ' \
    #             '--save_folder /data2/chengyi/ord_reg/result/save_model/checkpoint_{ckpt_name}/ ' \
    #             '--save_log /data2/chengyi/ord_reg/result/save_log/logs_{ckpt_name}/ '.format(fold=fold,
    #                                                                                           ckpt_name=ckpt_name,
    #                                                                                           gpusss=gpusss,
    #                                                                                           loss_name=loss_name,
    #                                                                                           alpha=alpha,
    #                                                                                           model=model)\
    #         .split()
    #     args = cfg.initialize(fixed)
    #     main(args)

    # ckpt_name = 'Aesthetics_Stratified'
    # gpusss = 2
    # opti = ['Adam', 'SGD']
    # l = [0.0001, 0.01]
    # for i in range(2):
    #     optimizer = opti[i]
    #     lr = l[i]
    #     fixed = '--gpu_id {gpusss} ' \
    #             '--exp_name sord_{optimizer} ' \
    #             '--optim {optimizer} ' \
    #             '--model_name sord ' \
    #             '--lr {lr} ' \
    #             '--stepvalues 7500 ' \
    #             '--max_iter 15000 ' \
    #             '--loss_name CELoss ' \
    #             '--data_name aesthetics3 ' \
    #             '--val_freq 40 ' \
    #             '--num_classes 5 ' \
    #             '--save_folder /data2/chengyi/ord_reg/result/save_model/checkpoint_{ckpt_name}/ ' \
    #             '--save_log /data2/chengyi/ord_reg/result/save_log/logs_{ckpt_name}/ '.format(ckpt_name=ckpt_name,
    #                                                                                           gpusss=gpusss,
    #                                                                                           lr=lr,
    #                                                                                           optimizer=optimizer)\
    #         .split()
    #     args = cfg.initialize(fixed)
    #     main(args)

    # gpusss = 3
    # for fold in [9]:
    #     ckpt_name = 'PVT/DR'
    #     fixed = '--gpu_id {gpusss} ' \
    #             '--exp_name re_pvt_trans_fold_{fold} ' \
    #             '--optim Adam ' \
    #             '--model_name pvt_trans ' \
    #             '--lr 0.0001 ' \
    #             '--max_iter 10000 ' \
    #             '--fold {fold} ' \
    #             '--loss_name CELoss ' \
    #             '--data_name dr ' \
    #             '--stepvalues 30000 60000 ' \
    #             '--num_classes 5 ' \
    #             '--val_freq 250 ' \
    #             '--save_folder /data2/chengyi/ord_reg/result/save_model/checkpoint_{ckpt_name}/ ' \
    #             '--save_log /data2/chengyi/ord_reg/result/save_log/logs_{ckpt_name}/ '.format(fold=fold,
    #                                                                                           ckpt_name=ckpt_name,
    #                                                                                           gpusss=gpusss)\
    #         .split()
    #     args = cfg.initialize(fixed)
    #     main(args)
'--data_name aesthetics ' \
'--img_root /data2/wangjinhong/data/ord_reg/aesthetics/ ' \
'--data_root /data2/wangjinhong/data/ord_reg/beauty-icwsm15-dataset.tsv  ' \
 \
'''
cd /data2/chengyi/ord_reg
source activate torch18
python main.py

'''
