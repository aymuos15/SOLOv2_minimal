import pdb

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import normal_init, ConvModule, bias_init_with_prob, matrix_nms, multi_apply


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    assert avg_factor is not None, 'avg_factor can not be None'

    # if reduction is mean, then average the loss by avg_factor
    if reduction == 'mean':
        loss = loss.sum() / avg_factor
    # if reduction is 'none', then do nothing, otherwise raise an error
    elif reduction != 'none':
        raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def py_sigmoid_focal_loss(pred, target, weight=None, gamma=2.0, alpha=0.25, reduction='mean',
                          avg_factor=None, loss_weight=1.):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred).unsqueeze(1)

    num_classes = pred_sigmoid.shape[1]
    class_range = torch.arange(1, num_classes + 1, dtype=pred_sigmoid.dtype, device='cuda').unsqueeze(0)
    target = (target == class_range).float()

    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)

    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss_weight * loss


def center_of_mass(bitmasks):
    _, h, w = bitmasks.size()
    ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks.device)
    xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks.device)

    m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
    m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
    m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)
    center_x = m10 / m00
    center_y = m01 / m00
    return center_x, center_y


def points_nms(heat, kernel=2):
    # kernel must be 2
    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=1)
    keep = (hmax[:, :, :-1, :-1] == heat).float()
    return heat * keep


def dice_loss(input, target):
    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1).float()

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    return 1 - d


class SOLOv2Head(nn.Module):
    def __init__(self, num_classes, in_channels=256, stacked_convs=4, seg_feat_channels=256,
                 strides=(8, 8, 16, 32, 32), scale_ranges=((8, 32), (16, 64), (32, 128), (64, 256), (128, 512)),
                 sigma=0.2, num_grids=(40, 36, 24, 16, 12), ins_out_channels=64, mode='train'):
        super(SOLOv2Head, self).__init__()
        self.num_grids = num_grids
        self.cate_out_ch = num_classes - 1
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.sigma = sigma
        self.kernel_out_ch = ins_out_channels * 1 * 1
        self.scale_ranges = scale_ranges
        self.ins_loss_weight = 3.0
        self.max_num = 100
        self.mode = mode
        self._init_layers()

    def _init_layers(self):
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        self.cate_convs = nn.ModuleList()
        self.kernel_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels + 2 if i == 0 else self.seg_feat_channels
            self.kernel_convs.append(ConvModule(chn,
                                                self.seg_feat_channels,
                                                3,
                                                stride=1,
                                                padding=1,
                                                norm_cfg=norm_cfg,
                                                bias=norm_cfg is None))

            chn = self.in_channels if i == 0 else self.seg_feat_channels
            self.cate_convs.append(ConvModule(chn,
                                              self.seg_feat_channels,
                                              3,
                                              stride=1,
                                              padding=1,
                                              norm_cfg=norm_cfg,
                                              bias=norm_cfg is None))

        self.solo_cate = nn.Conv2d(self.seg_feat_channels, self.cate_out_ch, 3, padding=1)

        self.solo_kernel = nn.Conv2d(self.seg_feat_channels, self.kernel_out_ch, 3, padding=1)

    def init_weights(self):
        for m in self.cate_convs:
            normal_init(m.conv, std=0.01)
        for m in self.kernel_convs:
            normal_init(m.conv, std=0.01)

        bias_cate = bias_init_with_prob(0.01)
        normal_init(self.solo_cate, std=0.01, bias=bias_cate)
        normal_init(self.solo_kernel, std=0.01)

    def forward(self, feats):
        all_feats = (F.interpolate(feats[0], scale_factor=0.5, mode='bilinear'),
                     feats[1],
                     feats[2],
                     feats[3],
                     F.interpolate(feats[4], size=feats[3].shape[-2:], mode='bilinear'))

        cate_pred_all, kernel_pred_all = [], []
        for i in range(len(all_feats)):
            kernel_feat = all_feats[i]

            # ins branch
            # concat coord
            x_range = torch.linspace(-1, 1, kernel_feat.shape[-1], device=kernel_feat.device)
            y_range = torch.linspace(-1, 1, kernel_feat.shape[-2], device=kernel_feat.device)
            y, x = torch.meshgrid(y_range, x_range)
            y = y.expand([kernel_feat.shape[0], 1, -1, -1])
            x = x.expand([kernel_feat.shape[0], 1, -1, -1])
            coord_feat = torch.cat([x, y], 1)
            kernel_feat = torch.cat([kernel_feat, coord_feat], 1)

            # kernel branch
            kernel_feat = F.interpolate(kernel_feat, size=self.num_grids[i], mode='bilinear').contiguous()
            cate_feat = kernel_feat[:, :-2, :, :].contiguous()

            for i, kernel_layer in enumerate(self.kernel_convs):
                kernel_feat = kernel_layer(kernel_feat)

            kernel_pred = self.solo_kernel(kernel_feat)

            # cate branch
            for i, cate_layer in enumerate(self.cate_convs):
                cate_feat = cate_layer(cate_feat)

            cate_pred = self.solo_cate(cate_feat)

            if not self.training:  # This step decreases memory usage during the inference
                # by keeping the higher scored category predictions. It has a negligible effect on results and speed.
                # Don't understand it clearly, ~540MB memory is saved for "Solov2_light_res50".
                cate_pred = points_nms(cate_pred.sigmoid(), kernel=2).permute(0, 2, 3, 1)
                # cate_pred = cate_pred.sigmoid().permute(0, 2, 3, 1)

            cate_pred_all.append(cate_pred)
            kernel_pred_all.append(kernel_pred)

        return cate_pred_all, kernel_pred_all

    def target_single(self, gt_bboxes_raw, gt_labels_raw, gt_masks_raw, mask_feat_size):
        device = gt_labels_raw[0].device
        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))

        ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list = [], [], [], []
        for (lower_bound, upper_bound), stride, num_grid in zip(self.scale_ranges, self.strides, self.num_grids):
            hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
            num_ins = len(hit_indices)

            ins_label = []
            grid_order = []
            cate_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device)
            ins_ind_label = torch.zeros([num_grid ** 2], dtype=torch.bool, device=device)

            if num_ins == 0:
                ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label)
                ins_ind_label_list.append(ins_ind_label)
                grid_order_list.append([])
                continue

            gt_bboxes = gt_bboxes_raw[hit_indices]
            gt_labels = gt_labels_raw[hit_indices]
            gt_masks = gt_masks_raw[hit_indices.cpu().numpy(), ...]

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

            # mass center
            gt_masks_pt = torch.from_numpy(gt_masks).to(device=device)
            center_ws, center_hs = center_of_mass(gt_masks_pt)
            valid_mask_flags = gt_masks_pt.sum(dim=-1).sum(dim=-1) > 0

            for gt_mask, gt_label, half_h, half_w, center_h, center_w, valid_mask_flag in zip(gt_masks, gt_labels,
                                                                                              half_hs, half_ws,
                                                                                              center_hs, center_ws,
                                                                                              valid_mask_flags):
                if not valid_mask_flag:
                    continue

                upsampled_size = (mask_feat_size[0] * 4, mask_feat_size[1] * 4)
                coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))
                coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))

                # left, top, right, down
                top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))
                down_box = min(num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid)))
                left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))
                right_box = min(num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid)))

                top = max(top_box, coord_h - 1)
                down = min(down_box, coord_h + 1)
                left = max(coord_w - 1, left_box)
                right = min(right_box, coord_w + 1)

                cate_label[top:(down + 1), left:(right + 1)] = gt_label

                h, w = gt_mask.shape[:2]
                scale = 1. / 4
                new_w, new_h = int(w * float(scale) + 0.5), int(h * float(scale) + 0.5)
                gt_mask = cv2.resize(gt_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)  # todo: why LINEAR?
                gt_mask = torch.from_numpy(gt_mask).to(device=device)

                for i in range(top, down + 1):
                    for j in range(left, right + 1):
                        index = int(i * num_grid + j)

                        cur_ins_label = torch.zeros([mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8,
                                                    device=device)
                        cur_ins_label[:gt_mask.shape[0], :gt_mask.shape[1]] = gt_mask
                        ins_label.append(cur_ins_label)
                        ins_ind_label[index] = True
                        grid_order.append(index)

            if len(ins_label) == 0:
                ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
            else:
                ins_label = torch.stack(ins_label, 0)

            ins_label_list.append(ins_label)
            cate_label_list.append(cate_label)
            ins_ind_label_list.append(ins_ind_label)
            grid_order_list.append(grid_order)

        return ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list

    def loss(self, cate_preds, kernel_preds, ins_pred, gt_bbox_list, gt_label_list, gt_mask_list):
        ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list = multi_apply(
            self.target_single,
            gt_bbox_list,
            gt_label_list,
            gt_mask_list,
            mask_feat_size=ins_pred.size()[-2:])

        # ins
        ins_labels = [torch.cat([ins_labels_level_img for ins_labels_level_img in ins_labels_level], 0)
                      for ins_labels_level in zip(*ins_label_list)]

        kernel_preds = [[kernel_preds_level_img.view(kernel_preds_level_img.shape[0], -1)[:, grid_orders_level_img]
                         for kernel_preds_level_img, grid_orders_level_img in
                         zip(kernel_preds_level, grid_orders_level)]
                        for kernel_preds_level, grid_orders_level in zip(kernel_preds, zip(*grid_order_list))]

        # generate masks
        ins_pred_list = []
        for b_kernel_pred in kernel_preds:
            b_mask_pred = []
            for idx, kernel_pred in enumerate(b_kernel_pred):
                if kernel_pred.size()[-1] == 0:
                    continue

                cur_ins_pred = ins_pred[idx, ...]
                H, W = cur_ins_pred.shape[-2:]
                N, I = kernel_pred.shape
                cur_ins_pred = cur_ins_pred.unsqueeze(0)
                kernel_pred = kernel_pred.permute(1, 0).view(I, -1, 1, 1)
                cur_ins_pred = F.conv2d(cur_ins_pred, kernel_pred, stride=1).view(-1, H, W)
                b_mask_pred.append(cur_ins_pred)

            if len(b_mask_pred) == 0:
                b_mask_pred = None
            else:
                b_mask_pred = torch.cat(b_mask_pred, 0)

            ins_pred_list.append(b_mask_pred)

        ins_ind_labels = [torch.cat([ins_ind_labels_level_img.flatten()
                                     for ins_ind_labels_level_img in ins_ind_labels_level])
                          for ins_ind_labels_level in zip(*ins_ind_label_list)]
        flatten_ins_ind_labels = torch.cat(ins_ind_labels)

        num_ins = flatten_ins_ind_labels.sum()

        # dice loss
        loss_ins = []
        for input, target in zip(ins_pred_list, ins_labels):
            if input is None:
                continue
            input = torch.sigmoid(input)
            loss_ins.append(dice_loss(input, target))

        loss_ins = torch.cat(loss_ins).mean()
        loss_ins = loss_ins * self.ins_loss_weight

        # cate
        cate_labels = [torch.cat([cate_labels_level_img.flatten() for cate_labels_level_img in cate_labels_level])
                       for cate_labels_level in zip(*cate_label_list)]

        flatten_cate_labels = torch.cat(cate_labels)

        cate_preds = [cate_pred.permute(0, 2, 3, 1).reshape(-1, self.cate_out_ch) for cate_pred in cate_preds]
        flatten_cate_preds = torch.cat(cate_preds)

        loss_cate = py_sigmoid_focal_loss(flatten_cate_preds, flatten_cate_labels, avg_factor=num_ins + 1)
        return loss_cate, loss_ins

    def get_seg(self, cate_preds, kernel_preds, seg_pred, ori_shape=None, resize_shape=None, cfg=None,
                detect_thre=None, mask_thre=None):
        batch_size = seg_pred.shape[0]

        num_levels = len(cate_preds)
        featmap_size = seg_pred.size()[-2:]

        result_batch_list = []

        for j in range(batch_size):
            cate_pred_list = [cate_preds[i][j].view(-1, self.cate_out_ch).detach() for i in range(num_levels)]
            seg_pred_list = seg_pred[j, ...].unsqueeze(0)
            kernel_pred_list = [kernel_preds[i][j].permute(1, 2, 0).view(-1, self.kernel_out_ch).detach()
                                for i in range(num_levels)]

            cate_pred_list = torch.cat(cate_pred_list, dim=0)
            kernel_pred_list = torch.cat(kernel_pred_list, dim=0)

            result = self.get_seg_single(cate_pred_list, seg_pred_list, kernel_pred_list,
                                         featmap_size, resize_shape, ori_shape, cfg, detect_thre, mask_thre)
            result_batch_list.append(result)

        return result_batch_list

    def get_seg_single(self, cate_preds, seg_preds, kernel_preds, featmap_size, resize_shape, ori_shape, cfg,
                       detect_thre, mask_thre):

        detect_thre = detect_thre if self.mode == 'onnx' else cfg['update_thr']
        mask_thre = mask_thre if self.mode == 'onnx' else cfg['mask_thr']

        # process.
        inds = (cate_preds > cfg['score_thr'])

        cate_scores = cate_preds[inds]
        if cate_scores.shape[0] == 0:
            return

        # cate_labels & kernel_preds
        inds = inds.nonzero()
        cate_labels = inds[:, 1]
        kernel_preds = kernel_preds[inds[:, 0]]

        # trans vector.
        size_trans = cate_labels.new_tensor(self.num_grids).pow(2).cumsum(0)
        strides = kernel_preds.new_ones(size_trans[-1])

        n_stage = len(self.num_grids)
        strides[:size_trans[0]] *= self.strides[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_ - 1]:size_trans[ind_]] *= self.strides[ind_]
        strides = strides[inds[:, 0]]

        # mask encoding.
        I, N = kernel_preds.shape
        kernel_preds = kernel_preds.view(I, N, 1, 1)
        seg_preds = F.conv2d(seg_preds, kernel_preds, stride=1).squeeze(0).sigmoid()

        # mask.
        seg_masks = seg_preds > mask_thre
        sum_masks = seg_masks.sum((1, 2)).float()

        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            return None

        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # maskness.
        seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores

        # sort and keep top nms_pre
        sort_inds = torch.argsort(cate_scores, descending=True)
        if sort_inds.shape[0] > cfg['nms_pre']:
            sort_inds = sort_inds[:cfg['nms_pre']]

        seg_masks = seg_masks[sort_inds, :, :]
        seg_preds = seg_preds[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        # Matrix NMS
        cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores, sum_masks=sum_masks)

        # filter.
        keep = cate_scores >= detect_thre
        if keep.sum() == 0:
            return None
        seg_preds = seg_preds[keep, :, :]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # sort and keep top_k
        sort_inds = torch.argsort(cate_scores, descending=True)

        if not self.training:
            if sort_inds.shape[0] > self.max_num:
                sort_inds = sort_inds[:self.max_num]

            seg_preds = seg_preds[sort_inds, :, :]
            cate_scores = cate_scores[sort_inds]
            cate_labels = cate_labels[sort_inds]

        h, w = resize_shape
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)
        seg_masks = F.interpolate(seg_preds.unsqueeze(0), size=upsampled_size_out, mode='bilinear')[:, :, :h, :w]

        if not self.training:
            seg_masks = F.interpolate(seg_masks, size=ori_shape, mode='bilinear')

        seg_masks = seg_masks.squeeze(0)
        seg_masks = (seg_masks > mask_thre).to(torch.uint8)
        if self.mode in ('detect', 'onnx'):
            mask_density = seg_masks.sum(dim=(1, 2))
            orders = torch.argsort(mask_density, descending=True)
            seg_masks = seg_masks[orders]
            cate_labels = cate_labels[orders]
            cate_scores = cate_scores[orders]

        return seg_masks, cate_labels, cate_scores
