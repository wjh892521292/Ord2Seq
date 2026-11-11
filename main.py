



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

    loss_name = 'CELoss'
    alpha = 0.
    model_name = os.environ.get('vggtrans')
   
    for model in [model_name]:
        fold = 0
        gpusss = 0
        ww = 1
        weights = '{} 1'.format(ww)
        ckpt_name = '{}/UTK/'.format(model)
        fixed = '--gpu_id {gpusss} ' \
                '--exp_name fold_{fold}_weight_{ww} ' \
                '--optim Adam ' \
                '--model_name {model} ' \
                '--lr 0.0001 ' \
                '--stepvalues 7500 15000 ' \
                '--max_iter 25000 ' \
                '--weights {weights} ' \
                '--fold {fold} ' \
                '--loss_name {loss_name} ' \
                '--data_name utk ' \
                '--img_root /path/to/utk/images ' \
                '--data_root /path/to/utk/annotations ' \
                '--alpha {alpha} ' \
                '--val_freq 40 ' \
                '--num_classes 8 ' \
                '--save_folder ./outputs/checkpoints/{ckpt_name}/ ' \
                '--save_log ./outputs/logs/{ckpt_name}/ '.format(fold=fold,
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
