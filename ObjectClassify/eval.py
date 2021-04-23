import os
import time

import torch.utils.data
from torch import nn
from torch import optim
from tqdm import tqdm


def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


def evaluate(model, eval_loader, args):

    model.eval()

    criterion_class = nn.CrossEntropyLoss()
    criterion_box = nn.L1Loss()

    running_loss, acc, iou, n = 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        for i, data in enumerate(eval_loader):
            image = data["image"].cuda()
            boxes = data["box"].cuda()
            classes = data["class"].cuda()

            predict_box, predict_class = model(image)

            box_loss = criterion_box(predict_box, boxes)
            class_loss = criterion_class(predict_class, classes)
            loss = class_loss + args.box * box_loss
            running_loss += loss.item()

            predict_class = torch.argmax(predict_class, dim=1)
            for j in range(boxes.shape[0]):
                n += 1
                p_iou = compute_iou(predict_box[j, :], boxes[j, :])
                iou += p_iou
                if p_iou > 0.5 and predict_class[j] == classes[j]:
                    acc += 1

        print("[eval] loss = %.4f, acc = %.4f, iou = %.4f"
              % (running_loss / len(eval_loader), acc / n, iou / n))
    return acc / n
