import cv2
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torchvision
from deeplabv3 import DeepLabv3
from PIL import Image
from torch.utils.data import Dataset, DataLoader

random.seed(0)

from pspnet import PSPNet

readvdnames = lambda x: open(x).read().rstrip().split('\n')


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default="Adam")
    parser.add_argument('--num-epochs', type=float, default=30)
    parser.add_argument('--ckpt-dir', type=str, default="./work_dirs/test")
    return parser.parse_args()


################################# DEFINE DATASET #################################
class TinySegData(Dataset):
    def __init__(self, db_root="TinySeg", img_size=256, phase='train'):
        classes = ['person', 'bird', 'car', 'cat', 'plane', ]
        seg_ids = [1, 2, 3, 4, 5]

        templ_image = db_root + "/JPEGImages/{}.jpg"
        templ_mask = db_root + "/Annotations/{}.png"

        ids = readvdnames(db_root + "/ImageSets/" + phase + ".txt")

        # build training and testing dbs
        samples = []
        for i in ids:
            samples.append([templ_image.format(i), templ_mask.format(i)])
        self.samples = samples
        self.phase = phase
        self.db_root = db_root
        self.img_size = img_size

        self.color_transform = torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2)

        if not self.phase == 'train':
            print("resize and augmentation will not be applied...")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.phase == 'train':
            return self.get_train_item(idx)
        else:
            return self.get_test_item(idx)

    def get_train_item(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample[0])

        if random.randint(0, 1) > 0:
            image = self.color_transform(image)
        image = np.asarray(image)[..., ::-1]  # to BGR
        seg_gt = (np.asarray(Image.open(sample[1]).convert('P'))).astype(np.uint8)

        image = image.astype(np.float32)
        image = image / 127.5 - 1  # -1~1

        if random.randint(0, 1) > 0:
            image = image[:, ::-1, :]  # HWC
            seg_gt = seg_gt[:, ::-1]

        # random crop to 256x256
        height, width = image.shape[0], image.shape[1]
        if height == width:
            miny, maxy = 0, 256
            minx, maxx = 0, 256
        elif height > width:
            miny = np.random.randint(0, height - 256)
            maxy = miny + 256
            minx = 0
            maxx = 256
        else:
            miny = 0
            maxy = 256
            minx = np.random.randint(0, width - 256)
            maxx = minx + 256
        image = image[miny:maxy, minx:maxx, :].copy()
        seg_gt = seg_gt[miny:maxy, minx:maxx].copy()

        if self.img_size != 256:
            new_size = (self.img_size, self.img_size)
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
            seg_gt = cv2.resize(seg_gt, new_size, interpolation=cv2.INTER_NEAREST)

        image = np.transpose(image, (2, 0, 1))  # To CHW

        # cv2.imwrite("test.png", np.concatenate([(image[0]+1)*127.5, seg_gt*255], axis=0))
        return image, seg_gt, sample

    def get_test_item(self, idx):
        sample = self.samples[idx]
        image = cv2.imread(sample[0])
        seg_gt = (np.asarray(Image.open(sample[1]).convert('P'))).astype(np.uint8)

        image = image.astype(np.float32)
        image = image / 127.5 - 1  # -1~1
        image = np.transpose(image, (2, 0, 1))

        # cv2.imwrite("test.png", np.concatenate([(image[0]+1)*127.5, seg_gt*255], axis=0))
        return image, seg_gt, sample


################################# FUNCTIONS #################################
def get_confusion_matrix(gt_label, pred_label, class_num):
    """
        Calcute the confusion matrix by given label and pred
        :param gt_label: the ground truth label
        :param pred_label: the pred label
        :param class_num: the number of class
        :return: the confusion matrix
        """
    index = (gt_label * class_num + pred_label).astype('int32')

    label_count = np.bincount(index)
    confusion_matrix = np.zeros((class_num, class_num))

    for i_label in range(class_num):
        for i_pred_label in range(class_num):
            cur_index = i_label * class_num + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix


def get_confusion_matrix_for_3d(gt_label, pred_label, class_num):
    confusion_matrix = np.zeros((class_num, class_num))

    for sub_gt_label, sub_pred_label in zip(gt_label, pred_label):
        sub_gt_label = sub_gt_label[sub_gt_label != 255]
        sub_pred_label = sub_pred_label[sub_pred_label != 255]
        cm = get_confusion_matrix(sub_gt_label, sub_pred_label, class_num)
        confusion_matrix += cm
    return confusion_matrix


if __name__ == "__main__":
    args = init_args()
    print(args)

    IMG_SIZE = 128
    print("=> the training size is {}".format(IMG_SIZE))

    train_loader = DataLoader(TinySegData(img_size=IMG_SIZE, phase='train'), batch_size=args.batch_size,
                              shuffle=True, num_workers=8)
    val_loader = DataLoader(TinySegData(img_size=IMG_SIZE, phase='val'), batch_size=1, shuffle=False, num_workers=8)

    model = PSPNet(n_classes=6, pretrained=True)
    # model = DeepLabv3()

    if args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        raise EOFError

    criterion = torch.nn.CrossEntropyLoss()
    mkdirs = lambda x: os.makedirs(x, exist_ok=True)
    # model.load_state_dict(torch.load("ckpt_seg/epoch_79_iou0.88.pth"))

    ckpt_dir = args.ckpt_dir
    mkdirs(ckpt_dir)
    epoch = args.num_epochs
    best_iou = 0

    for i in range(0, epoch):
        # train
        model.train()
        epoch_iou = []
        val_iou = []
        epoch_loss = []
        for j, (images, seg_gts, rets) in enumerate(train_loader):
            images = images.cuda()
            seg_gts = seg_gts.cuda()
            optimizer.zero_grad()

            seg_logit = model(images)
            loss_seg = criterion(seg_logit, seg_gts.long())
            loss = loss_seg
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

            if j % 10 == 0:
                seg_preds = torch.argmax(seg_logit, dim=1)
                seg_preds_np = seg_preds.detach().cpu().numpy()
                seg_gts_np = seg_gts.cpu().numpy()

                confusion_matrix = get_confusion_matrix_for_3d(seg_gts_np, seg_preds_np, class_num=6)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IU = IU_array.mean()

                # log_str = "[E{}/{} - {:3d}] ".format(i, epoch, j)
                # log_str += "loss[seg]: {:0.4f}, miou: {:0.4f}, ".format(loss_seg.item(), mean_IU)
                # print(log_str)

                # images_np = np.transpose((images.cpu().numpy() + 1) * 127.5, (0, 2, 3, 1))
                # n, h, w, c = images_np.shape
                # images_np = images_np.reshape(n * h, w, -1)[:, :, 0]
                # seg_preds_np = seg_preds_np.reshape(n * h, w)
                # visual_np = np.concatenate([images_np, seg_preds_np * 40], axis=1)  # NH * W
                # cv2.imwrite('visual.png', visual_np)

                epoch_iou.append(mean_IU)

        with torch.no_grad():
            model.eval()
            for j, (images, seg_gts, rets) in enumerate(val_loader):
                images = images.cuda()
                seg_gts = seg_gts.cuda()

                seg_logit = model(images)
                # loss_seg = criterion(seg_logit, seg_gts.long())
                # loss = loss_seg

                seg_preds = torch.argmax(seg_logit, dim=1)
                seg_preds_np = seg_preds.detach().cpu().numpy()
                seg_gts_np = seg_gts.cpu().numpy()

                confusion_matrix = get_confusion_matrix_for_3d(seg_gts_np, seg_preds_np, class_num=6)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IU = IU_array.mean()

                val_iou.append(mean_IU)

        epoch_iou = np.mean(epoch_iou)
        epoch_loss = np.mean(epoch_loss)
        val_iou = np.mean(val_iou)
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), "{}/epoch_{}_iou{:0.4f}.pth".format(ckpt_dir, i, val_iou))
        print("[Epoch %2d] train loss: %.4f, train iou: %.4f, val iou: %.4f, best iou: %.4f" % (
            i + 1, epoch_loss, epoch_iou, val_iou, best_iou))
        # print("=> saving to {}".format("{}/epoch_{}_iou{:0.2f}.pth".format(ckpt_dir, i, epoch_iou)))
