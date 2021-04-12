import argparse
import torch.utils.data

from models import *
from train import train
from dataset import VidDataset
import os
import numpy as np
import random
from test import test

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def init_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default="train", help='train/test')
    # parser.add_argument('--mode', type=str, default="test", help='train/test')

    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default="Adam")
    parser.add_argument('--sche', type=str, default="None")
    parser.add_argument('--box', type=float, default=10)
    # parser.add_argument('--sche', type=str, default="cos")
    # parser.add_argument('--sche', type=str, default="reduce")

    parser.add_argument('--backbone', type=str, default="res18")
    # parser.add_argument('--backbone', type=str, default="cnn")
    # parser.add_argument('--head', type=str, default="base")
    parser.add_argument('--head', type=str, default="fpn")

    parser.add_argument('--num-epochs', type=float, default=20)

    parser.add_argument('--h', type=int, default=128)
    parser.add_argument('--w', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=8)

    parser.add_argument('--data-path', type=str, default="./tiny_vid/")
    parser.add_argument('--ckp-path', type=str, default="./log/res18_3737.tar")
    parser.add_argument('--save-path', type=str, default="./log")
    if not os.path.exists(parser.parse_args().save_path):
        os.mkdir(parser.parse_args().save_path)

    return parser.parse_args()


if __name__ == "__main__":

    args = init_args()
    setup_seed(0)
    print(args)

    if args.backbone == "cnn":
        backbone = CNN()
    elif args.backbone == "res18":
        backbone = resnet18(pretrained=True)
    else:
        raise ValueError

    if args.head == "base":
        model = Head(backbone.cuda()).cuda()
    elif args.head == "fpn":
        model = FPNHead(backbone.cuda()).cuda()
    else:
        raise ValueError

    if args.mode == "train":
        train_dataset = VidDataset(args.data_path, (args.h, args.w))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   num_workers=args.num_workers, shuffle=True, drop_last=True)
        eval_dataset = VidDataset(args.data_path, (args.h, args.w), mode="eval")
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size,
                                                  num_workers=args.num_workers, shuffle=False, drop_last=False)
        train(model, train_loader, eval_loader, args)

    if args.mode == "test":
        eval_dataset = VidDataset(args.data_path, (args.h, args.w), mode="eval")
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size,
                                                  num_workers=args.num_workers, shuffle=False, drop_last=False)
        test(model, eval_loader, args)
