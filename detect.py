import os
import torch
import matplotlib.pyplot as plt

from model.solov2 import SOLOv2
from configs import *
from data_loader.build_loader import make_data_loader

plt.switch_backend('Agg')

# Get the model configuration based on MODEL_CHOICE
model_class = globals()[MODEL_CHOICE]
cfg = model_class(mode='val')
model = SOLOv2(cfg).cuda()

state_dict = torch.load(cfg.val_weight, weights_only=True)
model.load_state_dict(state_dict, strict=True)
model.eval()

data_loader = make_data_loader(cfg)
dataset = data_loader.dataset
val_num = len(data_loader)

for i, (img, resize_shape, img_name, img_resized) in enumerate(data_loader):

    with torch.no_grad():
        seg_result = model(img.cuda().detach(), resize_shape=resize_shape, post_mode='detect')[0]

        if seg_result is not None:
            seg_pred = seg_result[0].cpu().numpy()
            cate_label = seg_result[1].cpu().numpy()
            cate_score = seg_result[2].cpu().numpy()

            fig, ax = plt.subplots(1, 2, figsize=(15, 5))
            ax[0].imshow(img[0, 0, ...])
            ax[0].set_title('Original Image')
            ax[0].axis('off')
            ax[1].imshow(seg_pred[0])
            ax[1].set_title(f'Segmentation Result | {cate_label[0]} | {cate_score[0]:.2f}')
            ax[1].axis('off')

            # Save the figure to the result directory
            result_path = os.path.join('./results', f'{i}_result.png')
            plt.savefig(result_path)
            plt.close(fig)
        else:
            print(f"Image {img_name[0]} has no segmentation result.")