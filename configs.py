import numpy as np
from data_loader.dataset import CocoIns
from data_loader.augmentations import TrainAug, ValAug

TrainBatchSize = 12

############## MODEL CONFIGURATION ##############

class Solov2_res50:
    def __init__(self, mode):
        self.mode = mode
        
        # Data configurations
        self.setup_data_configs()
        
        # Model hyperparameters
        self.setup_model_hyperparams()
        
        # Training parameters
        self.setup_training_params()
        
        # Validation/Testing parameters
        self.setup_validation_params()
        
        # Detection/inference settings
        self.setup_detection_params()

    def setup_data_configs(self):
        """Setup data-related configurations"""
        self.dataset = CocoIns
        self.data_root = 'data/dummy/'
        self.train_imgs = self.data_root + 'train/'
        self.train_ann = self.data_root + 'annotations/annotations_train.json'
        self.val_imgs = self.data_root + 'test/'
        self.val_ann = self.data_root + 'annotations/annotations_test.json'
        self.class_names = DUMMY_CLASSES
        self.num_classes = len(self.class_names)

    def setup_model_hyperparams(self):
        """Setup model architecture hyperparameters"""
        self.resnet_depth = 50
        self.fpn_in_c = [256, 512, 1024, 2048]
        self.pretrained = 'weights/backbone_resnet34.pth'
        self.break_weight = ''
        
        # Head configurations
        self.head_stacked_convs = 4
        self.head_seg_feat_c = 512
        self.head_scale_ranges = ((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048))
        self.head_ins_out_c = 256
        self.mask_feat_num_classes = 256

    def setup_training_params(self):
        """Setup training-related parameters"""
        self.epochs = 5
        self.train_bs = TrainBatchSize
        self.lr = 0.01 * (self.train_bs / 16)
        self.warm_up_init = self.lr * 0.01
        self.warm_up_iters = int(500 * (16 / self.train_bs))
        self.lr_decay_steps = (27, 33)
        self.train_aug = TrainAug(img_scale=[(1333, 800), (1333, 768), (1333, 736),
                                           (1333, 704), (1333, 672), (1333, 640)])
        self.train_workers = 8
        self.start_save = 0

    def setup_validation_params(self):
        """Setup validation/testing parameters"""
        self.val_interval = 1
        self.val_weight = 'weights/Solov2_light_res34_5.pth'
        self.val_bs = 1
        self.val_aug = ValAug(img_scale=[(1333, 800)])
        self.val_num = -1
        self.postprocess_para = {'nms_pre': 500, 'score_thr': 0.3, 'mask_thr': 0.5, 'update_thr': 0.05}

    def setup_detection_params(self):
        """Setup detection/inference parameters"""
        if self.mode in ('detect', 'onnx'):
            self.postprocess_para['update_thr'] = 0.3  # for detect score threshold
        self.detect_mode = 'overlap'

    def print_cfg(self):
        print()

    def name(self):
        return self.__class__.__name__

    def train(self):
        self.mode = 'train'

    def eval(self):
        self.mode = 'val'


class Solov2_light_res50(Solov2_res50):
    def __init__(self, mode):
        super().__init__(mode)
    
    def setup_model_hyperparams(self):
        """Override model hyperparameters"""
        super().setup_model_hyperparams()
        self.head_stacked_convs = 2
        self.head_seg_feat_c = 256
        self.head_ins_out_c = 128
        self.head_scale_ranges = ((1, 56), (28, 112), (56, 224), (112, 448), (224, 896))
        self.mask_feat_num_classes = 128
    
    def setup_training_params(self):
        """Override training parameters"""
        super().setup_training_params()
        self.train_aug = TrainAug(img_scale=[(768, 512), (768, 480), (768, 448),
                                           (768, 416), (768, 384), (768, 352)])
    
    def setup_validation_params(self):
        """Override validation parameters"""
        super().setup_validation_params()
        self.val_weight = 'weights/Solov2_light_res34_36.pth'
        self.val_aug = ValAug(img_scale=[(768, 448)])


class Solov2_light_res34(Solov2_res50):
    def __init__(self, mode):
        super().__init__(mode)
    
    def setup_model_hyperparams(self):
        """Override model hyperparameters"""
        super().setup_model_hyperparams()
        self.resnet_depth = 34
        self.pretrained = 'weights/backbone_resnet34.pth'
        self.break_weight = ''
        self.fpn_in_c = [64, 128, 256, 512]
        self.head_stacked_convs = 2
        self.head_seg_feat_c = 256
        self.head_ins_out_c = 128
        self.head_scale_ranges = ((1, 56), (28, 112), (56, 224), (112, 448), (224, 896))
        self.mask_feat_num_classes = 128
    
    def setup_training_params(self):
        """Override training parameters"""
        super().setup_training_params()
        self.train_aug = TrainAug(img_scale=[(768, 512), (768, 480), (768, 448),
                                           (768, 416), (768, 384), (768, 352)])
        self.pretrained = 'weights/backbone_resnet34.pth'
    
    def setup_validation_params(self):
        """Override validation parameters"""
        super().setup_validation_params()
        self.val_weight = 'weights/Solov2_light_res34_1.pth'
        self.val_aug = ValAug(img_scale=[(768, 448)])
    
    def setup_detection_params(self):
        """Override detection parameters"""
        super().setup_detection_params()
        self.detect_images = self.val_imgs  # '/home/feiyu/Data/nanjiao/nanjiao_seg/语义分割/bgs'
        if self.mode in ('detect', 'onnx'):
            self.postprocess_para['update_thr'] = 0.9
        self.postprocess_para['mask_thr'] = 0.9

############## DATASET CONFIGURATION ##############
'''COCO'''

CocoIdMap = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
             9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
             18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
             27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
             37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
             46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
             54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
             62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
             74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
             82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}

COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
                'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
                'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush')

''' Dummy '''

DummyIdMap = {1: 1}

DUMMY_CLASSES = ('Square',)

''''''

COLORS = np.array([[0, 0, 0], [244, 67, 54], [233, 30, 99], [156, 39, 176], [103, 58, 183], [100, 30, 60],
                   [63, 81, 181], [33, 150, 243], [3, 169, 244], [0, 188, 212], [20, 55, 200],
                   [0, 150, 136], [76, 175, 80], [139, 195, 74], [205, 220, 57], [70, 25, 100],
                   [255, 235, 59], [255, 193, 7], [255, 152, 0], [255, 87, 34], [90, 155, 50],
                   [121, 85, 72], [158, 158, 158], [96, 125, 139], [15, 67, 34], [98, 55, 20],
                   [21, 82, 172], [58, 128, 255], [196, 125, 39], [75, 27, 134], [90, 125, 120],
                   [121, 82, 7], [158, 58, 8], [96, 25, 9], [115, 7, 234], [8, 155, 220],
                   [221, 25, 72], [188, 58, 158], [56, 175, 19], [215, 67, 64], [198, 75, 20],
                   [62, 185, 22], [108, 70, 58], [160, 225, 39], [95, 60, 144], [78, 155, 120],
                   [101, 25, 142], [48, 198, 28], [96, 225, 200], [150, 167, 134], [18, 185, 90],
                   [21, 145, 172], [98, 68, 78], [196, 105, 19], [215, 67, 84], [130, 115, 170],
                   [255, 0, 255], [255, 255, 0], [196, 185, 10], [95, 167, 234], [18, 25, 190],
                   [0, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [155, 0, 0],
                   [0, 155, 0], [0, 0, 155], [46, 22, 130], [255, 0, 155], [155, 0, 255],
                   [255, 155, 0], [155, 255, 0], [0, 155, 255], [0, 255, 155], [18, 5, 40],
                   [120, 120, 255], [255, 58, 30], [60, 45, 60], [75, 27, 244], [128, 25, 70]], dtype='uint8')