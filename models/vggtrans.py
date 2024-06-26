import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16_bn
import torch.nn.functional as F
import numpy as np
# from models import soft_target_and_lossfunc
from models.mixup import mix, cal
import os

'''
计算loss的部分改为CELoss
label1和label2改为01

现在是这样的：
mask1为11100-00011
mask2为11000-00100-00010-00001
loss是BCELoss
'''



def _get_activation_fn(activation):
    if activation is None:
        return nn.Identity()
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class multi_layer_feature(nn.Module):
    def __init__(self, dim):
        super(multi_layer_feature, self).__init__()
        backbone = nn.Sequential(*list(vgg16_bn(pretrained=True).children()))
        self.layer1 = backbone[:6]
        self.layer2 = backbone[6]
        self.layer3 = backbone[7]
        self.fc1 = nn.Linear(512, dim)
        self.fc2 = nn.Linear(1024, dim)
        self.fc3 = nn.Linear(2048, dim)

    def forward(self, x):
        bs = x.size(0)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        x1, x2, x3 = self.avgpool(x1), self.avgpool(x2), self.avgpool(x3)
        x1, x2, x3 = x1.reshape(bs, -1), x2.reshape(bs, -1), x3.reshape(bs, -1)

        x1, x2, x3 = self.fc1(x1), self.fc2(x2), self.fc3(x3)

        out = torch.cat([x1.unsqueeze(1), x2.unsqueeze(1), x3.unsqueeze(1)], dim=1)
        # shape = bs * 3 * 2048
        return out


class single_feature(nn.Module):
    def __init__(self, dim):
        super(single_feature, self).__init__()
        self.backbone = nn.Sequential(*list(vgg16_bn(pretrained=True).children()))
        self.feature = self.backbone[0]
        self.down = nn.Linear(512, dim)

    def forward(self, x):
        feature = self.feature(x)
        BS, C, H, W = feature.shape
        # shape = B * 2048 * 7 * 7
        feature = feature.reshape(BS, C, H * W).permute((2, 0, 1))
        feature = self.down(feature)
        return feature


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.name = 'ResTrans5'
        dim = 512
        self.cls = args.num_classes
        self.need_mask_in_training = False
        self.feature = single_feature(dim=dim)
        self.nhead = 8
        self.transformer = nn.Transformer()
        encoder_layer = nn.TransformerEncoderLayer(512, 8, 2048, 0.1, 'relu')
        encoder_norm = nn.LayerNorm(512)
        self.encoder = nn.TransformerEncoder(encoder_layer, 6, encoder_norm)
        decoder_layer = nn.TransformerDecoderLayer(512, 8, 2048, 0.1, 'relu')
        decoder_norm = nn.LayerNorm(512)
        self.decoder = nn.TransformerDecoder(decoder_layer, 6, decoder_norm)
        self.Embed = nn.Embedding(15, dim)

        self.fc1 = nn.Linear(dim, self.cls)
        self.fc2 = nn.Linear(dim, self.cls)
        self.fc3 = nn.Linear(dim, self.cls)
        self.fc = [self.fc1, self.fc2, self.fc3]
        self.acti = nn.Identity()

        self.weight = [1., 1., 1.]

    def model_name(self):
        return self.name

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def make_binary_gt(self, tgt):
        # 起始符 = 14
        if self.cls == 5:
            mapping = {
                0: [0, 2, 6],
                1: [0, 2, 7],
                2: [0, 3, 8],
                3: [1, 4, 9],
                4: [1, 5, 10]
            }
            # DR:
            # mapping = {
            #     0: [0, 2, 6],
            #     1: [0, 3, 7],
            #     2: [1, 4, 8],
            #     3: [1, 5, 9],
            #     4: [1, 5, 10]
            # }
        elif self.cls == 8:
            mapping = {
                0: [0, 2, 6],
                1: [0, 2, 7],
                2: [0, 3, 8],
                3: [0, 3, 9],
                4: [1, 4, 10],
                5: [1, 4, 11],
                6: [1, 5, 12],
                7: [1, 5, 13],
            }
        else:
            print('我没有设置这个mapping')
            raise ValueError
        # [5 [01234...] [01234...] [01234...]]
        # [14 [00001]]
        bs, tgt_len = tgt.shape
        start_token = (torch.ones([bs, 1]) * 14).long()
        if tgt_len == 1:
            converted_tgt = self.Embed(start_token.cuda())
        else:
            if tgt_len == 2:
                cls = torch.tensor([mapping[x.item()][0] for x in tgt[:, 1]]).unsqueeze(1)
            elif tgt_len == 3:
                # cls1 = torch.tensor([mapping[x.item()][0] for x in tgt[:, 1]])
                # cls

                cls = torch.tensor(
                    [[mapping[x.item()][0] for x in tgt[:, 1]],
                     # [mapping[x.item()][1] for x in tgt[:, 2]]],
                     [mapping[x.item()][1] for x in tgt[:, 2]]],
                ).permute((1, 0))
            else:
                cls = torch.tensor([mapping[x.item()] for x in tgt[:, 1]])

            converted_tgt = torch.cat([start_token, cls], dim=1)
            converted_tgt = self.Embed(converted_tgt.long().cuda())
        return converted_tgt.permute((1, 0, 2))

    def make_tgt(self, tgt):
        tgt = tgt.unsqueeze(dim=1)
        BS = tgt.size(0)
        start_token = torch.ones([BS, 1]) * self.cls
        tgt = torch.cat([start_token.cuda(), tgt, tgt, tgt], dim=1).long()
        tgt = self.make_binary_gt(tgt)
        return tgt

    def make_mask_0712_l1(self, out1):
        '''
        这是直接11100 --- 00011这样的mask
        :param out1:
        :return:
        '''
        if len(out1.shape) == 3:
            out1 = out1.squeeze(0)
        out1 = F.softmax(out1, dim=1)
        if self.cls == 5:
            left, right = out1.split([3, 2], dim=1)
            left = left.sum(dim=1)
            right = right.sum(dim=1)
            comparison = (left > right)
            mask = torch.stack([comparison, comparison, comparison, ~comparison, ~comparison]).float().permute(1, 0)
            # left, right = out1.split([2, 3], dim=1)
            # left = left.sum(dim=1)
            # right = right.sum(dim=1)
            # comparison = (left > right)
            # mask = torch.stack([comparison, comparison, ~comparison, ~comparison, ~comparison]).float().permute(1, 0)
            # 不知道为什么DR用0-1234编码第一层会导致无法拟合
        elif self.cls == 8:
            left, right = out1.split([4, 4], dim=1)
            left = left.sum(dim=1)
            right = right.sum(dim=1)
            comparison = (left > right)
            mask = torch.stack([comparison, comparison, comparison, comparison,
                                ~comparison, ~comparison, ~comparison, ~comparison]).float().permute(1, 0)
        else:
            print('我没写')
            raise ValueError
        return mask.float()

    def make_mask_0712_l2(self, out2):
        '''
        这是直接11000 --- 00100 --- 00010 --- 00001这样的mask
        :param out1:
        :return:
        '''
        bs = out2.size(0)
        if len(out2.shape) == 3:
            out2 = out2.squeeze(0)
        out2 = F.softmax(out2, dim=1)
        if self.cls == 5:

            # tmp = torch.zeros([bs, 4]).cuda()
            # tmp[:, 0] = out2[:, 0] + out2[:, 1]
            # tmp[:, 1] = out2[:, 2]
            # tmp[:, 2] = out2[:, 3]
            # tmp[:, 3] = out2[:, 4]
            # # out2 = tmp
            # _, indices = torch.max(tmp, dim=1)
            # indices = F.one_hot(indices, num_classes=4)
            # mask = torch.cat([indices[:, 0].unsqueeze(1), indices[:, 0].unsqueeze(1),
            #                   indices[:, 1].unsqueeze(1), indices[:, 2].unsqueeze(1),
            #                   indices[:, 3].unsqueeze(1)], dim=1)



            out2[:, 1] = out2[:, 1] + out2[:, 0]
            out2 = out2[:, 1:]  # 0-1，2，3，4
            _, indices = torch.max(out2, dim=1)
            indices = F.one_hot(indices, num_classes=4)
            mask = torch.cat([indices[:, 0].unsqueeze(1), indices], dim=1)

            # # -->
            # pass

            # tmp = torch.zeros([bs, 3]).cuda()
            # tmp[:, 0] = out2[:, 0]
            # tmp[:, 1] = out2[:, 1] + out2[:, 2]
            # tmp[:, 2] = out2[:, 3] + out2[:, 4]
            # # out2 = tmp
            # _, indices = torch.max(tmp, dim=1)
            # indices = F.one_hot(indices, num_classes=3)
            # mask = torch.cat([indices[:, 0].unsqueeze(1),
            #                   indices[:, 1].unsqueeze(1), indices[:, 1].unsqueeze(1),
            #                   indices[:, 2].unsqueeze(1), indices[:, 2].unsqueeze(1)], dim=1)

            # tmp = torch.zeros([bs, 4]).cuda()
            # tmp[:, 0] = out2[:, 0]
            # tmp[:, 1] = out2[:, 1]
            # tmp[:, 2] = out2[:, 2]
            # tmp[:, 3] = out2[:, 3] + out2[:, 4]
            # # out2 = tmp
            # _, indices = torch.max(tmp, dim=1)
            # indices = F.one_hot(indices, num_classes=4)
            # mask = torch.cat([indices[:, 0].unsqueeze(1),
            #                   indices[:, 1].unsqueeze(1), indices[:, 2].unsqueeze(1),
            #                   indices[:, 3].unsqueeze(1), indices[:, 3].unsqueeze(1)], dim=1)
        elif self.cls == 8:
            tmp = torch.zeros([bs, 4]).cuda()
            tmp[:, 0] = out2[:, 0] + out2[:, 1]
            tmp[:, 1] = out2[:, 2] + out2[:, 3]
            tmp[:, 2] = out2[:, 4] + out2[:, 5]
            tmp[:, 3] = out2[:, 6] + out2[:, 7]
            out2 = tmp
            _, indices = torch.max(out2, dim=1)
            indices = F.one_hot(indices, num_classes=4)
            mask = torch.cat([indices[:, 0].unsqueeze(1), indices[:, 0].unsqueeze(1),
                              indices[:, 1].unsqueeze(1), indices[:, 1].unsqueeze(1),
                              indices[:, 2].unsqueeze(1), indices[:, 2].unsqueeze(1),
                              indices[:, 3].unsqueeze(1), indices[:, 3].unsqueeze(1)], dim=1)
        else:
            print('我没写')
            raise ValueError
        return mask.float()

    def forward(self, x, tgt):
        tgt = tgt.long()
        BS = x.size(0)
        lam = None

        feature = self.feature(x)
        memory = self.encoder(feature)

        label_3 = F.one_hot(tgt, num_classes=self.cls).float()
        label_1 = self.make_mask_0712_l1(label_3)
        label_2 = self.make_mask_0712_l2(label_3)

        if self.training:
            tgt_mask = self.generate_square_subsequent_mask(4).cuda()
            tgt = cal(self.make_tgt, tgt, lam, add=True)
            output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
            out1, out2, out3, _ = output.split(1, dim=0)
            out1 = self.fc[0](self.acti(out1)).squeeze(0)
            out2 = self.fc[1](self.acti(out2)).squeeze(0)
            out3 = self.fc[2](self.acti(out3)).squeeze(0)

            loss1 = nn.BCEWithLogitsLoss()(out1, label_1)

            if self.need_mask_in_training:
                mask1 = self.make_mask_0712_l1(out1)
                out2 = mask1 * out2

            loss2 = nn.BCEWithLogitsLoss()(out2, label_2)

            if self.need_mask_in_training:
                mask2 = self.make_mask_0712_l2(out2)
                out3 = mask2 * out3

            loss3 = nn.BCEWithLogitsLoss()(out3, label_3)

            loss = loss1 + loss2 + loss3

        else:
            dec_input = torch.zeros(BS, 0).long().cuda()
            next_symbol = (torch.ones([BS, 1]) * self.cls).long().cuda()
            # probout = []
            output_hard = []
            for i in range(3):
                dec_input = torch.cat([dec_input, next_symbol], -1)
                tgt_mask = self.generate_square_subsequent_mask(i + 1).cuda()

                # ====================== 这里需要改成和training一样的部分 ======================
                tgt = self.make_binary_gt(dec_input)

                output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
                projected = self.fc[i](self.acti(output))
                # probout.append(projected)
                '''
                mask?
                '''
                # projected = projected.squeeze(0)
                projected = projected[-1]
                if i == 0:
                    mask = self.make_mask_0712_l1(projected)
                    loss1 = nn.BCEWithLogitsLoss()(projected, label_1)
                    out1 = projected
                if i == 1:
                    projected = mask * projected
                    mask = self.make_mask_0712_l2(projected)
                    loss2 = nn.BCEWithLogitsLoss()(projected, label_2)
                    out2 = projected
                if i == 2:
                    # projected = mask * projected
                    # loss3 = nn.BCEWithLogitsLoss()(projected, label_3)
                    # out3 = projected
                    # 对于最后一层输出加入mask会导致MAE很高 0718：
                    # -->
                    # projected = projected
                    loss3 = nn.BCEWithLogitsLoss()(projected, label_3)
                    out3 = projected

                # TODO: 这里也改了
                if i in [0, 1]:
                    prob = mask.max(dim=-1, keepdim=False)[1]
                else:
                    prob = projected.max(dim=-1, keepdim=False)[1]
                # prob = projected.max(dim=-1, keepdim=False)[1]

                next_word = prob.data[-1].unsqueeze(dim=-1) if len(prob.shape) > 1 else prob.unsqueeze(
                    dim=-1).data
                next_symbol = next_word.clone()
                output_hard.append(next_symbol)

            # out1, out2, out3 = probout[0][-1], probout[1][-1], probout[2][-1]

            loss = loss1 * self.weight[0] + loss2 * self.weight[1] + loss3 * self.weight[2]

        return out3, loss

class arg_test():
    def __init__(self):
        self.num_classes = 8
        self.optim = 'Adam'

if __name__ == '__main__':
    args = arg_test()
    # backbone = nn.Sequential(*list(vgg16_bn(pretrained=True).children()))
    # seq = torch.rand(32, 3, 3)
    # mask = get_attn_subsequence_mask(seq)
    # print(mask)
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    model = Model(args).cuda()

    total = sum([param.nelement() for param in model.parameters()])

    print("Number of parameter: %.2fM" % (total / 1e6))


    import time

    # ============= 模拟inference =============
    # src = torch.rand(1, 3, 224, 224).cuda()
    # tgt = torch.Tensor([0]).cuda()
    # model.eval()
    # T1 = time.time()
    # for i in range(100):
    #     out = model(src, tgt)
    # T2 = time.time()
    # print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))
    # ============= 模拟训练 =============
    src = torch.rand(32, 3, 224, 224).cuda()
    tgt = torch.Tensor([0 for _ in range(32)]).cuda()
    optim = getattr(torch.optim, args.optim) \
                    (filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=0.0001)
    model.train()
    T1 = time.time()
    for i in range(10):
        _, loss = model(src, tgt)
        loss.backward()
        optim.step()
        model.zero_grad()
    T2 = time.time()
    print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))



    #
    # src = torch.rand(8, 3, 224, 224).cuda()
    # tgt = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7]).cuda()
    #
    # out = model(src, tgt)
    # model.eval()
    # out = model(src, tgt)
    # print(out)
