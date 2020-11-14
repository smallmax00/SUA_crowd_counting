import torch
from torch.utils import data
from Datasets.dataset_semantic_QNRF_JHU import Dataset as full_supervise_Dataset
from Models.model import Model
import os
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
checkpoint_logs_name = 'JHU'

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--dataset', default='JHU', type=str, help='dataset')  # SHB
parser.add_argument('--data_path', default='./Data_Crowd_Counting/', type=str, help='path to dataset')
parser.add_argument('--load', default=True, action='store_true', help='load checkpoint')
parser.add_argument('--save_path', default='./checkpoints/' + checkpoint_logs_name, type=str, help='path to save checkpoint')  # seman_SHB
parser.add_argument('--gpu', default=0, type=int, help='gpu id')

args = parser.parse_args()

def normalize(image, MIN_BOUND, MAX_BOUND):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    reverse_image = 1 - image
    return reverse_image


test_dataset = full_supervise_Dataset(args.data_path, args.dataset, False)
test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device('cuda:' + str(args.gpu))

def create_model(ema=False):
    # Network definition
    net = Model()
    model = net.to(device)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

model = create_model()


if args.load:
    checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint_best.pth'))
    model.load_state_dict(checkpoint['model'])

iter_num = 0
model.eval()

print('start validation')
with torch.no_grad():
    mae, mse = 0.0, 0.0
    for i, (image, gt, den_val_gt, att_val_gt) in enumerate(test_loader):

        image = image.to(device)

        predict, dmp_to_att_val, seg_val = model(image)

        # unc
        T = 8
        volume_batch_r = image.repeat(2, 1, 1, 1)
        stride = volume_batch_r.shape[0] // 2
        preds = torch.zeros([stride * T, 2, image.shape[2], image.shape[3]]).cuda()
        for i in range(T // 2):
            ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
            with torch.no_grad():
                _, _, ema_seg = model(ema_inputs)
                preds[2 * stride * i:2 * stride * (i + 1)] = ema_seg
        preds = F.softmax(preds, dim=1)
        preds = preds.reshape(T, stride, 2, image.shape[2], image.shape[3])
        preds = torch.mean(preds, dim=0)  # (batch /2, 1, 128, 128)
        uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)  # (batch/2, 1, 128, 128)

        uncertainty_norm = normalize(uncertainty, 0, np.log(2)) * 7

        mae += torch.abs(predict.sum() - den_val_gt.sum()).item()
        mse += ((predict.sum() - den_val_gt.sum()) ** 2).item()

        # save GT
        save_img = np.transpose(image.cpu().numpy().squeeze(), [1, 2, 0]) * 0.2 + 0.45
        density_gt = den_val_gt.cpu().numpy().squeeze().astype('float32')
        attention_gt = att_val_gt.cpu().numpy().squeeze()

        # density
        save_pre_den = predict.data
        save_pre_den = save_pre_den.cpu().numpy().squeeze().astype('float32')

        # dmp_to_seg
        save_pre_dmp_to_att = dmp_to_att_val.data
        save_pre_dmp_to_att[save_pre_dmp_to_att >= 0.5] = 1.0
        save_pre_dmp_to_att[save_pre_dmp_to_att < 0.5] = 0.0
        save_pre_dmp_to_att = save_pre_dmp_to_att.cpu().numpy().squeeze()  # .astype('uint8')

        # seg
        save_pre_att_2 = seg_val.data
        save_pre_att_2 = save_pre_att_2.cpu().numpy().squeeze().astype('uint8')
        save_pre_att_2 = np.transpose(save_pre_att_2, [1, 2, 0])
        save_pre_att_2 = np.argmin(save_pre_att_2, axis=2)

        # unc
        uncertainty = uncertainty.cpu().numpy().squeeze().astype('float32')
        uncertainty = uncertainty * (uncertainty > 0.5)
        uncertainty_norm = uncertainty_norm.cpu().numpy().squeeze().astype('float32')
        uncertainty_norm = uncertainty_norm

        plt.figure()
        plt.subplot(1, 6, 1)
        plt.imshow(save_pre_den)
        plt.subplot(1, 6, 2)
        plt.imshow(density_gt)
        plt.subplot(1, 6, 3)
        plt.imshow(save_pre_dmp_to_att)
        plt.subplot(1, 6, 4)
        plt.imshow(save_pre_att_2)
        plt.subplot(1, 6, 5)
        plt.imshow(uncertainty, cmap='inferno')
        plt.subplot(1, 6, 6)
        plt.imshow(uncertainty_norm, cmap='inferno')
        plt.show()

    mae /= len(test_loader)
    mse /= len(test_loader)
    mse = mse ** 0.5
    print('MAE:', mae, 'MSE:', mse)



