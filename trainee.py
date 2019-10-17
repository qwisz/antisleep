import torch
from utils import warmup_lr_scheduler


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    loss_accum = 0
    for i_step, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_accum += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()


    ave_loss = loss_accum / i_step

    return ave_loss

def train_model(model, optimizer, lr_scheduler, data_loader, device, num_epochs):
    loss_history = []

    for epoch in range(num_epochs):
        loss = train_one_epoch(model, optimizer, data_loader, device, epoch)
        loss_history.append(loss)
        lr_scheduler.step()

    return loss_history