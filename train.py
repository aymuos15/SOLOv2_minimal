import torch
from torch.nn.utils import clip_grad

from model.solov2 import SOLOv2
from configs import *
from data_loader.build_loader import make_data_loader

if __name__ == '__main__':
    # Get the model configuration based on MODEL_CHOICE
    model_class = globals()[MODEL_CHOICE]
    cfg = model_class(mode='train')

    model = SOLOv2(cfg).cuda()
    model.train()

    data_loader = make_data_loader(cfg)
    len_loader = len(data_loader)

    start_epoch = 1
    step = 1
    start_lr = cfg.lr

    optimizer = torch.optim.SGD(model.parameters(), lr=start_lr, momentum=0.9, weight_decay=0.0001)

    for epoch in range(start_epoch, cfg.epochs + 1):
        epoch_loss_cate = 0.0
        epoch_loss_ins = 0.0

        for img, gt_labels, gt_bboxes, gt_masks in data_loader:
            img = img.cuda().detach()
            gt_labels = [one.cuda().detach() for one in gt_labels]
            gt_bboxes = [one.cuda().detach() for one in gt_bboxes]

            if cfg.warm_up_iters > 0 and step <= cfg.warm_up_iters:  # warm up learning rate.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = (cfg.lr - cfg.warm_up_init) * (step / cfg.warm_up_iters) + cfg.warm_up_init

            if epoch in cfg.lr_decay_steps:  # learning rate decay.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = cfg.lr * 0.1 ** (cfg.lr_decay_steps.index(epoch) + 1)

            loss_cate, loss_ins = model(img, gt_labels, gt_bboxes, gt_masks)
            loss_total = loss_cate + loss_ins

            optimizer.zero_grad()
            loss_total.backward()
            clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                                      max_norm=35, norm_type=2)
            optimizer.step()

            epoch_loss_cate += loss_cate.item()
            epoch_loss_ins += loss_ins.item()
            step += 1

        cur_lr = optimizer.param_groups[0]['lr']
        avg_loss_cate = epoch_loss_cate / len_loader
        avg_loss_ins = epoch_loss_ins / len_loader
        print(f'epoch: {epoch} | lr: {cur_lr:.2e} | avg_l_class: {avg_loss_cate:.3f} | avg_l_ins: {avg_loss_ins:.3f}')

        if epoch % cfg.val_interval == 0 and epoch > cfg.start_save:
            print(f'weights/{cfg.name()}_{epoch}.pth saved.')
            torch.save(model.state_dict(), f'weights/{cfg.name()}_{epoch}.pth')