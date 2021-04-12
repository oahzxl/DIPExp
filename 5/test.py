import os
import time

import torch.utils.data
from torch import nn
from torch import optim
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt


def draw_box_cv(img, boxes, labels, scores, c=0):
    LABEL_NAME_MAP = ['bird', 'car', 'dog', 'lizard', 'turtle']

    left_line = max(boxes[1], boxes[1])
    right_line = min(boxes[3], boxes[3])
    top_line = max(boxes[0], boxes[0])
    bottom_line = min(boxes[2], boxes[2])
    y_c = (left_line + right_line) // 2
    x_c = (top_line + bottom_line) // 2
    h = -left_line + right_line
    w = -top_line + bottom_line

    color = [0, 0, 0]
    color[c] = 255

    rect = ((x_c, y_c), (w, h), 0)
    # 根据四边形的中心x_c, y_c，w，h以及偏移角度theta，恢复出四个点点的坐标
    rect = cv2.boxPoints(rect)
    rect = np.int0(rect)  # 转为int
    cv2.drawContours(img, [rect], -1, color, 2)  # 在图中根据rect绘制

    category = LABEL_NAME_MAP[labels]  # 类别

    if scores is not None:
        cv2.rectangle(img,
                      pt1=(bottom_line - 52, right_line - 9),
                      pt2=(bottom_line, right_line),
                      color=color,
                      thickness=-1)
        cv2.putText(img,
                    text=category + ": " + "%.2f" % float(scores),
                    org=(bottom_line - 50, right_line - 2),
                    fontFace=1,
                    fontScale=0.5,
                    thickness=1,
                    color=(color[1], color[2], color[0]))
    return img


def test(model, eval_loader, args):

    model.load_state_dict(torch.load(args.ckp_path)["model_state_dict"])
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(eval_loader):
            image = data["image"].cuda()
            boxes = data["box"].cuda()
            classes = data["class"].cuda()
            ori_image = [cv2.imread(data["path"][j]) for j in range(image.shape[0])]
            predict_box, predict_class = model(image)
            predict = torch.argmax(predict_class, dim=1)
            predict_class = torch.softmax(predict_class, dim=1)

            boxes = boxes * 64 + 64
            predict_box = predict_box * 64 + 64

            boxes = boxes.cpu().numpy().astype(np.int64)
            predict_box = predict_box.cpu().numpy().astype(np.int64)
            predict = predict.cpu().numpy().astype(np.int32)
            predict_class = predict_class.cpu().numpy().astype(np.float32)
            classes = classes.cpu().numpy().astype(np.int32)
            img = np.array(ori_image, np.uint8)

            for j in range(image.shape[0]):
                # img = draw_box_cv(ori_image[j], boxes[j], predict[j], predict_class[j, predict[j]])
                img[j] = draw_box_cv(img[j], boxes[j], classes[j], 1, c=0)
                img[j] = draw_box_cv(img[j], predict_box[j], predict[j], predict_class[j, predict[j]], c=1)
                plt.imshow(cv2.cvtColor(img[j], cv2.COLOR_RGB2BGR))
                plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
                plt.show()

