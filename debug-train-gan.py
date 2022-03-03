import json 
import argparse
import os
from datetime import datetime
import logging
from tqdm import tqdm
import time
from utils.run_util import *

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from torchvision import transforms
from torchvision.utils import make_grid

from dataloaders.datasets3d import *
from networks.unet3d import Modified3DUNet as Unet3D
# from networks.discriminator_mask_only import Discriminator
from networks.v2v_discriminator import Discriminator
# from networks.another_dis_version import Discriminator
from optimization.losses import dice_loss_indiv

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
parser.add_argument('--gen_cp', dest= 'gen_cp_dir', type = str, 
                    help = 'Model checkpoint directory')
parser.add_argument('--dis_cp', dest= 'dis_cp_dir', type = str, 
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
#print(model_cfg)

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
gen_net = Unet3D(in_channels=4, num_classes = data_cfg['num_classes']) 
dis_net = Discriminator(in_channels = 4)
if params_cfg['device']=='cuda': 
    gen_net.cuda()
    dis_net.cuda()

gen_net.train()
dis_net.train()

# 6. Init optimizer and loss 
gen_cfg = model_cfg['generator']
dis_cfg = model_cfg['discriminator']
GOptClass = optim.__dict__[gen_cfg['opt']]
DOptClass = optim.__dict__[dis_cfg['opt']]
gen_opt = GOptClass(gen_net.parameters(), lr = gen_cfg['lr'], weight_decay = gen_cfg['decay'], betas = tuple(gen_cfg['betas']))
dis_opt = GOptClass(dis_net.parameters(), lr = dis_cfg['lr'], weight_decay = dis_cfg['decay'], betas = tuple(dis_cfg['betas']))
print(dis_cfg)
dis_epoch = dis_cfg['epochs']
### bce loss
# bce_weight = torch.tensor(data_cfg['bce_weight']).cuda()
# bce_weight = bce_weight * (data_cfg['num_classes'] - 1) / bce_weight.sum()
# bce_loss_func = nn.BCEWithLogitsLoss(  # weight=weights_batch.view(-1,1,1,1),
#         pos_weight=bce_weight)
### dice loss 
dice_loss_func = dice_loss_indiv
class_weights = torch.ones(data_cfg['num_classes']).cuda()
class_weights[0] = 0
class_weights /= class_weights.sum()

mse_loss_func = nn.MSELoss()

# 7. Load checkpoint if there is
gen_iter_num = 0
dis_iter_num = 0 
start_epoch = 0
max_epoch = params_cfg['epochs']
Tensor = torch.cuda.FloatTensor

if args.gen_cp_dir: 
    gen_iter_num, start_epoch = load_model(args.gen_cp_dir, gen_net, gen_opt)
    start_epoch -=1
if args.dis_cp_dir: 
    dis_iter_num, start_epoch = load_model(args.dis_cp_dir, dis_net, dis_opt)
    # to_train_epoch = params_cfg['epochs']-start_epoch

logging.info(f'Total epochs: {params_cfg["epochs"]}')
logging.info(f'Iterations per epoch: {len(trainloader)}')
logging.info(f'Starting at iter/epoch: {gen_iter_num}/{start_epoch}')

for epoch in tqdm(range(start_epoch, max_epoch), ncols=70):

    time1 = time.time()
    epoch_gen_loss = 0
    epoch_dis_loss = 0 
    epoch_ce_loss= 0
    epoch_dice_loss= 0
    epoch_gen_gan_loss = 0
    for de in tqdm(range(dis_epoch)): 
        if epoch == 0: 
            continue
        for idx, sampled_batch in enumerate(trainloader):
            volume_batch, mask_batch = sampled_batch['image'].cuda(), sampled_batch['mask'].cuda()
            mask_batch = brats_map_label(mask_batch, binarize = False)
            volume_batch = F.interpolate(volume_batch, size=data_cfg['input_patch_size'], mode='trilinear', align_corners=False)
            dis_opt.zero_grad()

            # Adversarial grounde truths, real = 1, fake = 0
            # valid_gt = Tensor(np.ones((volume_batch.size(0),1,7,7,6)))
            # fake_gt = Tensor(np.zeros((volume_batch.size(0),1,7,7,6)))
            valid_gt = Tensor(np.ones((volume_batch.size(0),1,14,14,12)))
            fake_gt = Tensor(np.zeros((volume_batch.size(0), 1,14,14,12)))
            # compute loss for real prediction: L2[D(x,y), 1], x is image, y is mask
            pred_real = dis_net(volume_batch, mask_batch)
            logging.debug(f'Predict real val: {torch.unique(pred_real)}')
            loss_real = mse_loss_func(pred_real, valid_gt)
            # compute loss for fake prediction: L2[D(x, y^), 0], x is image, y^ is generated mask
            softmax_pred, fake_mask = gen_net(volume_batch)
            # softmax_pred = torch.round(softmax_pred)
            # logging.debug(f'difference: {softmax_pred[mask_batch!= softmax_pred]}')
            pred_fake = dis_net(volume_batch, fake_mask)
            logging.debug(f'Predict fake val: {torch.unique(pred_fake)}')

            loss_fake = mse_loss_func(pred_fake, fake_gt)
            loss_d = loss_real + loss_fake
            epoch_dis_loss+=loss_d

            dis_iter_num+=1
            logging.info(f'Discriminator {idx}/{de}/{epoch}: loss real: {loss_real} -- loss fake: {loss_fake}')
            loss_d.backward()
            dis_opt.step()
            
            writer.add_scalar('discriminator/loss_real', loss_real.item(), dis_iter_num)
            writer.add_scalar('discriminator/loss_fake', loss_fake.item(), dis_iter_num)
            writer.add_scalar('discriminator/total_iter_loss', loss_d, dis_iter_num)
            del volume_batch, mask_batch

    if dis_epoch!=0: 
        writer.add_scalar('discriminator/epoch_loss', epoch_dis_loss/(len(trainloader)*dis_epoch), epoch)
    if epoch%params_cfg['save_epoch']==0:
        save_model(checkpoint_root, 'discriminator', dis_net, dis_opt, epoch, dis_iter_num)

    for idx, sampled_batch in enumerate(trainloader):
        volume_batch, mask_batch = sampled_batch['image'].cuda(), sampled_batch['mask'].cuda()
        mask_batch = brats_map_label(mask_batch, binarize = False)
        volume_batch = F.interpolate(volume_batch, size=data_cfg['input_patch_size'], mode='trilinear', align_corners=False)

        # Adversarial grounde truths, real = 1, fake = 0
        valid_gt = Tensor(np.ones((volume_batch.size(0),1,14,14,12)))
        softmax_pred, fake_mask = gen_net(volume_batch)
        pred_fake = dis_net(volume_batch, fake_mask)
        loss_GAN  = mse_loss_func(pred_fake, valid_gt)
        # voxel-wise loss 
        # total_ce_loss = bce_loss_func(fake_mask.permute([0, 2, 3, 4, 1]),
        #                                   # after brats_map_label(), dim 1 of mask_batch is segmantation class.
        #                                   # It's permuted to the last dim to align with outputs for bce loss computation.
        #                                   mask_batch.permute([0, 2, 3, 4, 1]))
        total_dice_loss = 0
        dice_losses = []
        outputs_soft = torch.sigmoid(fake_mask)

        for cls in range(1, data_cfg['num_classes']):
            # bce_loss_func is actually nn.BCEWithLogitsLoss(), so use raw scores as input.
            # dice loss always uses sigmoid/softmax transformed probs as input.
            dice_loss = dice_loss_func(
                outputs_soft[:, cls], mask_batch[:, cls])
            dice_losses.append(dice_loss.item())
            total_dice_loss += dice_loss * class_weights[cls]
        
        if gen_iter_num % 50 ==0:
            draw_image(writer, volume_batch,'train/Image', from_slice=0, to_slice=1, iter_num=gen_iter_num, coords = params_cfg['coords']) 
            draw_image(writer, softmax_pred,'train/Softmax_label', from_slice=1, to_slice=2, iter_num=gen_iter_num, coords = params_cfg['coords']) 
            draw_image(writer, outputs_soft,'train/Predicted_label', from_slice=1, to_slice=2, iter_num=gen_iter_num, coords = params_cfg['coords']) 
            draw_image(writer, mask_batch,'train/Groundtruth_label', from_slice=1, to_slice=2, iter_num=gen_iter_num, coords = params_cfg['coords'])
        
        del volume_batch, mask_batch
        
        # total loss is total batch loss of all classes
        loss_g =  5*total_dice_loss + loss_GAN
        gen_iter_num += 1 
        gen_opt.zero_grad()
        loss_g.backward()
        gen_opt.step()
        # epoch_ce_loss += total_ce_loss
        epoch_dice_loss += total_dice_loss
        epoch_gen_gan_loss += loss_GAN
        epoch_gen_loss += loss_g

        logging.info(f'Generator {idx}/{epoch}: \n  Dice loss: {total_dice_loss}\n   L2 loss: {loss_GAN}')
        # logging.info(f'Class dice loss: {dice_losses}')
        # writer.add_scalar('generator-sample/cross entropy loss', total_ce_loss.item(), gen_iter_num)
        writer.add_scalar('generator-sample/dice loss',
                            total_dice_loss.item(), gen_iter_num)
        writer.add_scalar('generator-sample/GAN mse loss', loss_GAN.item(), gen_iter_num)
        writer.add_scalar('generator-sample/Total sample loss', loss_g, gen_iter_num)
   

    if epoch%params_cfg['save_epoch']==0: 
        save_model(checkpoint_root,'generator', gen_net, gen_opt, epoch, gen_iter_num)

    writer.add_scalar('generator-epoch/epoch dice', epoch_dice_loss/len(trainloader), epoch)
    # writer.add_scalar('generator-epoch/cross entropy', epoch_ce_loss/len(trainloader), epoch)
    writer.add_scalar('generator-epoch/GAN mse', epoch_gen_gan_loss/len(trainloader), epoch)
    writer.add_scalar('generator-epoch/Total epoch loss',epoch_gen_loss/len(trainloader),epoch)
