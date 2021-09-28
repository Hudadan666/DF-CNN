import torch
import models
import os
from data_loader import BSDS_RCFLoader
from torch.utils.data import DataLoader
import numpy as np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

model = models.resnet101(pretrained=True).cuda()

init_lr = 1e-2
batch_size = 3

# resume = '/home/HDD/project/meeting/rcf-edge-detection-master/ckpt/only-final-lr-0.01-iter-50000.pth'
# checkpoint = torch.load(resume)
# model.load_state_dict(checkpoint)


def adjust_lr(init_lr, now_it, total_it):
    power = 0.9
    lr = init_lr * (1 - float(now_it) / total_it) ** power
    return lr


def make_optim(model, lr):
    optim = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    return optim


def save_ckpt(model, name):
    print('saving checkpoint ... {}'.format(name), flush=True)
    if not os.path.isdir('ckpt'):
        os.mkdir('ckpt')
    torch.save(model.state_dict(), os.path.join('ckpt', '{}.pth'.format(name)))


train_dataset = BSDS_RCFLoader(split="train")
# test_dataset = BSDS_RCFLoader(split="test")
train_loader = DataLoader(
    train_dataset, batch_size=batch_size,
    num_workers=8, drop_last=True, shuffle=True)


def MSE_DF(prediction, label):
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(prediction.float(), label.float())
    return loss


def cross_entropy_loss_RCF(prediction, label):
    label = label.long()
    mask = label.float()
    num_positive = torch.sum((mask == 1).float()).float()
    num_negative = torch.sum((mask == 0).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0

    # print('num pos', num_positive)
    # print('num neg', num_negative)
    # print(1.0 * num_negative / (num_positive + num_negative), 1.1 * num_positive / (num_positive + num_negative))

    cost = torch.nn.functional.binary_cross_entropy(
            prediction.float(), label.float(), weight=mask, reduce=False)
    return torch.sum(cost) / (num_negative + num_positive)


model.train()
total_epoch = 30
each_epoch_iter = len(train_loader)
total_iter = total_epoch * each_epoch_iter

print_cnt = 10
ckpt_cnt = 10000
cnt = 0

for epoch in range(total_epoch):
    avg_loss = 0.
    for i, (image, label, mask) in enumerate(train_loader):
        cnt += 1
        optim = make_optim(model, adjust_lr(init_lr, cnt, total_iter))
        image, label, mask = image.cuda(), label.cuda(), mask.cuda()
        outs, df = model(image, label.size()[2:])

        # final_result = outs[-1]/(df+0.000001)
        # final_result = torch.clamp(final_result, 0, 1)

        logit_loss = cross_entropy_loss_RCF(outs[-1], label)
        df_loss = MSE_DF(df, mask)
        total_loss = logit_loss + 0.5*df_loss
        # final_result = torch.from_numpy(final_result).cuda()
        # print(final_result.shape)
        # print(label.shape)
        # final_result = final_result.squeeze().detach().cpu()
        # final_result = np.clip(final_result, 0, 1)
        # logit_loss = cross_entropy_loss_RCF(final_result, label)
        # df_loss = MSE_DF(df, mask)
        # total_loss = logit_loss + 0.25*df_loss
        # total_loss = logit_loss
        # for each in outs:
        #     loss = cross_entropy_loss_RCF(each, label)
        #     total_loss += loss
        optim.zero_grad()
        total_loss.backward()
        optim.step()

        avg_loss += float(total_loss)
        if cnt % print_cnt == 0:
            print('[{}/{}] loss:{} avg_loss: {}'.format(cnt, total_iter, float(total_loss), avg_loss / print_cnt),
                  flush=True)
            avg_loss = 0

        if cnt % ckpt_cnt == 0:
            save_ckpt(model, 'only-final-lr-{}-iter-{}'.format(init_lr, cnt))

