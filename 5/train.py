import os
import time

import torch.utils.data
from torch import nn
from torch import optim
from eval import evaluate


def train(model, train_loader, eval_loader, args):

    criterion_class = nn.CrossEntropyLoss()
    criterion_box = nn.MSELoss()
    # criterion = LabelSmoothCELoss()
    # criterion = WeightedLabelSmoothCELoss(1978, 2168, 1227)

    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    else:
        raise ValueError

    global_step, best_acc, loss, t_remain, best_loss = 0, 0, 0, 0, 999.0

    for epoch in range(0, args.num_epochs, 1):
        model.train()
        running_loss = 0.0
        t = time.time()

        for i, data in enumerate(train_loader):
            image = data["image"].cuda()
            boxes = data["box"].cuda()
            classes = data["class"].cuda()

            optimizer.zero_grad()

            predict_box, predict_class = model(image)

            box_loss = criterion_box(predict_box, boxes)
            class_loss = criterion_class(predict_class, classes)
            loss = class_loss + args.box * box_loss
            running_loss += loss.item()

            loss.backward()
            optimizer.step()
            global_step += 1

        print("[train] epoch = %2d, loss = %.4f, lr = %.1e, time per picture = %.2fs, remaining time = %.0fs"
              % (epoch + 1, running_loss / len(train_loader), optimizer.state_dict()['param_groups'][0]['lr'],
                 (time.time() - t) / len(train_loader) / args.batch_size,
                 ((time.time() - t_remain) * (args.num_epochs - epoch - 1)) if t_remain != 0 else -1))
        t_remain = time.time()

        best_acc = evaluate(model, eval_loader, args)

        torch.save({
                "model_state_dict": model.state_dict(),
            }, os.path.join(args.save_path, args.backbone + "_acc_%.5f" % best_acc + ".tar"))

    print("Done.")
