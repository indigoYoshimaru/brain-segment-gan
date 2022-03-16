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
from prettytable import PrettyTable

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
brats_cfg = data_cfg.get('brats', {})
params_cfg = data_cfg.get('params', {})
print(params_cfg)
test_cfg = read_json(args.test_config_dir)

# 3. Init model (dynamically)

segmentation_module = importlib.import_module(test_cfg['module'])
SegmentationModelClass = getattr(segmentation_module, test_cfg['class'])
seg_net = SegmentationModelClass(in_channels=4, num_classes = brats_cfg['num_classes'])

# 4. Load checkpoint 
iter_num, epoch = load_model(args.seg_cp_dir, seg_net, optimizer = None)
epoch-=1
seg_net.cuda()

# 5. Save directory config: writer directory, log directory, save images
test_dir = f'{test_cfg["model_name"]}-ver{test_cfg["version"]}-epoch{epoch}'
writer_dir = os.path.join(args.writer_dir, 'runs', test_dir)
writer = SummaryWriter(writer_dir)
timestamp = datetime.now().strftime('%m%d%H%M')
image_save_path = os.path.join(args.writer_dir, 'predicted', test_dir) 

if not os.path.isdir(image_save_path):
    os.makedirs(image_save_path, exist_ok=True)
# 6. Load dataset

ds_name = brats_cfg['test_ds_name']
test_data_path = os.path.join(root, ds_name)
has_mask = bool(brats_cfg['has_mask'][ds_name])
db_test = BratsSet(base_dir=test_data_path,
                    split='all',
                    mode='test',
                    ds_weight=1,
                    xyz_permute=None,
                    transform=ToTensor(),
                    chosen_modality=brats_cfg['chosen_modality'],
                    binarize=False,
                    train_loc_prob=brats_cfg['localization_prob'],
                    min_output_size=brats_cfg['orig_patch_size']
                    )
logging.info(f'{ds_name}: {len(db_test)} images, has_mask: {has_mask}')

test_loader = DataLoader(db_test, batch_size=1, sampler=None,
                            num_workers=params_cfg['num_workers'], pin_memory=False, shuffle=False)

avg_scores = test_all_cases(seg_net, db_test, writer,
                                    num_classes=brats_cfg['num_classes'],
                                    batch_size=1,
                                    orig_patch_size=brats_cfg['orig_patch_size'],
                                    input_patch_size=brats_cfg['input_patch_size'],
                                    stride_xy=brats_cfg['orig_patch_size'][0] // 2,
                                    stride_z=brats_cfg['orig_patch_size'][2] // 2,
                                    save_result=test_cfg['save_result'],
                                    test_save_path=image_save_path,
                                    preproc_fn=None,
                                    test_interp=None,
                                    has_mask=has_mask)

print(f'Model: {test_cfg["model_name"]} - epoch: {epoch} - iteration: {iter_num} scores:')
class_name = ['ET', 'WT', 'TC']
class_name = np.array(class_name)
table = PrettyTable()
table.field_names=['Class', 'Dice', 'Jaccard', 'Hausdorff', 'Avg Surface Distance']
for cls in range(brats_cfg['num_classes']-1): 
    table.add_row([class_name[cls], avg_scores[cls][0], avg_scores[cls][1], avg_scores[cls][2], avg_scores[cls][3]])

print(table)
