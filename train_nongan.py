import json 
import argparse
import os
from datetime import datetime
import logging
from tqdm import tqdm
import time
import importlib
from utils.run_util import *

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from torchvision import transforms
from torchvision.utils import make_grid

from dataloaders.datasets3d import *
import optimization.losses as losses

"""
This file is the training file for any non-GAN model

"""

def worker_init_fn(worker_id):
    random.seed(params_cfg['seed']+worker_id)

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
parser.add_argument('--modelcfg', dest = 'model_config_dir', type=str, 
                    help = 'Model hypeparameters configuration file directory')
parser.add_argument('--cp', dest= 'cp_dir', type = str, 
                    help = 'Model checkpoint directory')
parser.add_argument('--writer', dest = 'writer_dir', type= str, 
                    help = 'Summary writer directory')

args = parser.parse_args()

# 2. Read config files (json) -> dict 
train_cfg = read_json(args.data_config_dir)
root = train_cfg.get('data_dir')
data_cfg = train_cfg.get('brats', {})
params_cfg = train_cfg.get('params', {})
model_cfg = read_json(args.model_config_dir)

# 3. Save directory config

writer_dir = os.path.join(args.writer_dir, model_cfg['model_name'], f'ver{model_cfg["version"]}')
writer = SummaryWriter(writer_dir)
timestamp = datetime.now().strftime('%m%d%H%M')
checkpoint_root = os.path.join('model', model_cfg['model_name'], f'ver{model_cfg["version"]}')

if not os.path.isdir(checkpoint_root):
    os.makedirs(checkpoint_root, exist_ok=True)


fileHandler = logging.FileHandler(checkpoint_root+f"/{timestamp}-log.txt")
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

logging.info(f'Checkpoints to be saved at: {checkpoint_root}')

# 4. Load dataset 
transform = compose(
        RandomRotFlip(),
        RandomCrop(data_cfg['orig_patch_size']),
        ToTensor(),
    )

db_trains = []
for ds_name in data_cfg['train_ds_names']: 
    train_data_path = os.path.join(root, ds_name)
    has_mask = bool(data_cfg['has_mask'][ds_name])

    db_train = BratsSet(base_dir=train_data_path,
                            split='all',
                            mode='train',
                            ds_weight=1,
                            xyz_permute=None,
                            transform=transform,
                            chosen_modality=data_cfg['chosen_modality'],
                            binarize=False,
                            train_loc_prob=data_cfg['localization_prob'],
                            min_output_size=data_cfg['orig_patch_size']
                            )

    db_trains.append(db_train)
    logging.info(f'{ds_name}: {len(db_train)} images, has_mask: {has_mask}')

db_train_combo = ConcatDataset(db_trains)
logging.info(f'Combined dataset: {len(db_train_combo)} images')

train_sampler = None
shuffle = True

trainloader = DataLoader(db_train_combo, batch_size=params_cfg['batch_size'], sampler=train_sampler,
                            num_workers=params_cfg['num_workers'], pin_memory=False, shuffle=shuffle,
                            worker_init_fn=worker_init_fn)

# 5. Init models
segmentation_module = importlib.import_module(model_cfg['module'])
SegmentationModelClass = getattr(segmentation_module, model_cfg['class'])
net = SegmentationModelClass(in_channels=4, num_classes = data_cfg['num_classes']) 
if params_cfg['device']=='cuda': 
    net.cuda()

net.train()
out_size = None

# 6. Init optimizer and loss 

OptClass = optim.__dict__[model_cfg['opt']]
optimizer = OptClass(net.parameters(), lr = model_cfg['lr'], weight_decay = model_cfg['decay'], betas = tuple(model_cfg['betas']))

### region based loss: including dice loss, log cosh dice loss, focal tversky loss  
region_loss = getattr(losses, model_cfg['region_loss'])
class_weights = torch.ones(data_cfg['num_classes']).cuda()
class_weights[0] = 0
class_weights /= class_weights.sum()
logging.info(f'Segmentation loss: {region_loss}')

# 7. Load checkpoint if there is
iter_num = 0
dis_iter_num = 0 
start_epoch = 0
max_epoch = params_cfg['epochs']
Tensor = torch.cuda.FloatTensor

if args.cp_dir: 
    iter_num, start_epoch = load_model(args.cp_dir, net, optimizer)
    # to_train_epoch = params_cfg['epochs']-start_epoch

logging.info(f'Total epochs: {params_cfg["epochs"]}')
logging.info(f'Iterations per epoch: {len(trainloader)}')
logging.info(f'Starting at iter/epoch: {iter_num}/{start_epoch}')

for epoch in tqdm(range(start_epoch, max_epoch), ncols=70):

    time1 = time.time()
    epoch_loss = 0
    epoch_region_loss = 0
    # for now region loss equals total loss 
    for idx, sampled_batch in enumerate(trainloader):
        volume_batch, mask_batch = sampled_batch['image'].cuda(), sampled_batch['mask'].cuda()
        mask_batch = brats_map_label(mask_batch, binarize = False)
        volume_batch = F.interpolate(volume_batch, size=data_cfg['input_patch_size'], mode='trilinear', align_corners=False)

        outputs_raw = net(volume_batch)
        outputs_soft = torch.sigmoid(outputs_raw)
        outputs_hard = convert_to_hard(outputs_soft)
                   
        # voxel-wise loss 
        total_region_loss = 0
        region_losses = []
        logging.debug(f'Pred_soft: {outputs_soft.unique()}')
        logging.debug(f'Pred_hard: {outputs_hard.unique()}')

        for cls in range(1, data_cfg['num_classes']):
            # bce_loss_func is actually nn.BCEWithLogitsLoss(), so use raw scores as input.
            # dice loss always uses sigmoid/softmax transformed probs as input.
            loss_val = region_loss(
                outputs_soft[:, cls], mask_batch[:, cls])
            region_losses.append(loss_val.item())
            total_region_loss += loss_val * class_weights[cls]

        # outputs_raw return the feature maps, while output_soft is the black and white version    
        if idx % 50 ==0:
            draw_image(writer, volume_batch,'image', c_start=0, c_end=1, iter_num=iter_num, size = params_cfg['coords']) 
            draw_image(writer, outputs_soft,'predicted_label', c_start=1, c_end=2, iter_num=iter_num, size = params_cfg['coords']) 
            draw_image(writer, outputs_hard, 'output_hard', c_start=1, c_end=2, iter_num=iter_num, size=params_cfg['coords'])
            draw_image(writer, mask_batch,'groundtruth_label', c_start=1, c_end=2, iter_num=iter_num, size = params_cfg['coords'])
        
        del volume_batch, mask_batch
        
        # total loss is total batch loss of all classes
        loss_g = total_region_loss
        iter_num += 1 
        optimizer.zero_grad()
        loss_g.backward()
        optimizer.step()
        epoch_region_loss += total_region_loss
        epoch_loss += loss_g

        logging.info(f'{idx}/{epoch}: \n  Region loss - {model_cfg["region_loss"]}: {total_region_loss}\n')
        writer.add_scalar('sample/region_loss',
                            total_region_loss.item(), iter_num)
        writer.add_scalar('epoch/total_loss', loss_g, epoch)
        
    if epoch%params_cfg['save_epoch']==0: 
        save_model(checkpoint_root,model_cfg['model_name'], net, optimizer, epoch, iter_num)

    writer.add_scalar('epoch/region_loss', epoch_region_loss /len(trainloader), epoch)
    writer.add_scalar('epoch/total_loss', epoch_loss /len(trainloader), epoch)
