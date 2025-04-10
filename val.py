import os
import json
import numpy as np
import torch
import pycocotools.mask as mask_util
from model.solov2 import SOLOv2
from configs import *
from metrics.cocoeval import SelfEval
from data_loader.build_loader import make_data_loader


def val(cfg, model=None):
    if model is None:
        model = SOLOv2(cfg).cuda()
        state_dict = torch.load(cfg.val_weight)
        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict, strict=True)
        print(f'Evaluating "{cfg.val_weight}".')

    cfg.eval()
    model.eval()
    data_loader = make_data_loader(cfg)
    dataset = data_loader.dataset

    json_results = []
    dt_id = 1

    for i, (img, ori_shape, resize_shape, _) in enumerate(data_loader):
        img = img.cuda().detach()

        with torch.no_grad():
            seg_result = model(img, ori_shape=ori_shape, resize_shape=resize_shape, post_mode='val')[0]

        if seg_result is not None:
            seg_pred = seg_result[0].cpu().numpy()
            cate_label = seg_result[1].cpu().numpy()
            cate_score = seg_result[2].cpu().numpy()

            for j in range(seg_pred.shape[0]):
                data = dict()
                cur_mask = seg_pred[j, ...]
                data['image_id'] = dataset.ids[i]
                data['score'] = float(cate_score[j])
                rle = mask_util.encode(np.array(cur_mask[:, :, np.newaxis], order='F'))[0]
                rle['counts'] = rle['counts'].decode()
                data['segmentation'] = rle

                if 'Coco' in dataset.__class__.__name__:
                    data['category_id'] = dataset.cate_ids[cate_label[j] + 1]
                else:
                    data['category_id'] = int(cate_label[j] + 1)
                    data['id'] = dt_id
                    data['iscrowd'] = 0
                    data['area'] = int(cur_mask.sum())

                    hs, ws = np.where(cur_mask > 0)
                    x1, x2 = float(ws.min()), float(ws.max())
                    y1, y2 = float(hs.min()), float(hs.max())
                    data['bbox'] = [x1, y1, x2 - x1, y2 - y1]

                dt_id += 1
                json_results.append(data)

    file_path = f'{'results/'}/{cfg.name()}.json'
    with open(file_path, "w") as f:
        json.dump(json_results, f)

    coco_dt = dataset.coco.loadRes(file_path)
    segm_eval = SelfEval(dataset.coco, coco_dt, all_points=True, iou_type='segmentation')

    segm_eval.evaluate()
    segm_eval.accumulate()
    segm_eval.summarize()


if __name__ == '__main__':
    # Get the model configuration based on MODEL_CHOICE
    model_class = globals()[MODEL_CHOICE]
    cfg = model_class(mode='val')
    cfg.print_cfg()
    val(cfg)
