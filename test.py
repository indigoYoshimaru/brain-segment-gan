import json 
import argparse
import os
from datetime import datetime
import logging
from tqdm import tqdm
import time
from utils.run_util import *
from utils.test_util3d import *

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from torchvision import transforms
from torchvision.utils import make_grid

from dataloaders.datasets3d import *
# from networks.multiscale_generator import MultiscaleGenerator as Gener
from optimization.losses import *
import importlib

"""
This file is the testing file for all UNet models
"""

logFormatter = logging.Formatter(
    '[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
rootLogger = logging.getLogger()
while rootLogger.handlers:
    rootLogger.handlers.pop()
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)
rootLogger.setLevel(logging.NOTSET)
rootLogger.propagate = False

# 1. Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--datacfg', dest = 'data_config_dir', type=str, 
                    help = 'Data and training params configuration file directory')
parser.add_argument('--testcfg', dest = 'test_config_dir', type=str, 
                    help = 'Model hypeparameters configuration file directory')
parser.add_argument('--seg_cp', dest= 'seg_cp_dir', type = str, 
                    help = 'Segmentation model checkpoint directory')
parser.add_argument('--writer', dest = 'writer_dir', type= str, 
                    help = 'Summary writer directory')

args = parser.parse_args()

# 2. Read config files (json) -> dict 
data_cfg = read_json(args.data_config_dir)
root = data_cfg.get('data_dir')
data_cfg = data_cfg.get('brats', {})
params_cfg = data_cfg.get('params', {})
test_cfg = read_json(args.model_config_dir)

# 3. Save directory config
writer_dir = os.path.join(args.writer_dir, test_cfg['model_name'], f'ver{test_cfg["version"]}')
writer = SummaryWriter(writer_dir)
timestamp = datetime.now().strftime('%m%d%H%M')

# 4. Load dataset
transform = compose(
        RandomRotFlip(),
        RandomCrop(data_cfg['orig_patch_size']),
        ToTensor())

ds_name = data_cfg['test_ds_name']
test_data_path = os.path.join(root, ds_name)
has_mask = bool(data_cfg['has_mask'][ds_name])
db_test = BratsSet(base_dir=test_data_path,
                    split='all',
                    mode='test',
                    ds_weight=1,
                    xyz_permute=None,
                    transform=transform,
                    chosen_modality=data_cfg['chosen_modality'],
                    binarize=False,
                    train_loc_prob=data_cfg['localization_prob'],
                    min_output_size=data_cfg['orig_patch_size']
                    )
logging.info(f'{ds_name}: {len(db_test)} images, has_mask: {has_mask}')

test_loader = DataLoader(db_test, batch_size=params_cfg['batch_size'], sampler=None,
                            num_workers=params_cfg['num_workers'], pin_memory=False, shuffle=False)

# 5. Init model (dynamically)

segmentation_module = importlib.import_module(test_cfg['module'])
SegmentationModelClass = getattr(segmentation_module, test_cfg['class'])
seg_net = SegmentationModelClass(in_channels=4, num_classes = data_cfg['num_classes'])

# 6. Load checkpoint 
iter_num, epoch = load_model(args.seg_cp_dir, seg_net, optimizer = None)
seg_net.cuda()

allcls_avg_metric = test_all_cases(seg_net, db_test,
                                    net_type=test_cfg['model_name'],
                                    num_classes=data_cfg['num_classes'],
                                    batch_size=data_cfg['batch_size'],
                                    orig_patch_size=data_cfg['orig_patch_size'],
                                    input_patch_size=data_cfg['input_patch_size'],
                                    stride_xy=data_cfg['orig_patch_size'][0] // 2,
                                    stride_z=data_cfg['orig_patch_size'][2] // 2,
                                    save_result=data_cfg['save_result'],
                                    test_save_path=test_save_path,
                                    preproc_fn=preproc_fn,
                                    test_interp=args.test_interp,
                                    has_mask=has_mask)

print("%d scores:" % iter_num)
for cls in range(1, args.num_classes):
    dice, jc, hd, asd = allcls_avg_metric[cls-1]
    print('%d: dice: %.3f, jc: %.3f, hd: %.3f, asd: %.3f' %
            (cls, dice, jc, hd, asd))
