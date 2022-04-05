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
from networks.v2v_generator import Generator
from networks.multiscale_discriminator import MultiscaleDiscriminator as Discriminator
# from networks.v2v_discriminator import Discriminator
import optimization.losses as losses


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
gen_net = Generator(in_channels=4, num_classes = data_cfg['num_classes']) 
dis_net = Discriminator(in_channels = 4)
if params_cfg['device']=='cuda': 
    gen_net.cuda()
    dis_net.cuda()

gen_net.train()
dis_net.train()
out_size = None

# 6. Init optimizer and loss 
gen_cfg = model_cfg['generator']
dis_cfg = model_cfg['discriminator']
GOptClass = optim.__dict__[gen_cfg['opt']]
DOptClass = optim.__dict__[dis_cfg['opt']]
gen_opt = GOptClass(gen_net.parameters(), lr = gen_cfg['lr'], weight_decay = gen_cfg['decay'], betas = tuple(gen_cfg['betas']))
dis_opt = GOptClass(dis_net.parameters(), lr = dis_cfg['lr'], weight_decay = dis_cfg['decay'], betas = tuple(dis_cfg['betas']))

dis_epoch =dis_cfg['epochs']

# if model_cfg['dynamic_train']: 
#     dis_epoch =0

### region based loss: including dice loss, log cosh dice loss, focal tversky loss  
region_loss = getattr(losses, gen_cfg['region_loss'])
class_weights = torch.ones(data_cfg['num_classes']).cuda()
class_weights[0] = 0
class_weights /= class_weights.sum()

gan_loss_func = getattr(losses, dis_cfg['gan_loss'])()
logging.info(f'Segmentation loss: {region_loss} - GAN loss: {gan_loss_func}')

# 7. Load checkpoint if there is
gen_iter_num = 0
dis_iter_num = 0 
start_epoch = 0
max_epoch = params_cfg['epochs']
Tensor = torch.cuda.FloatTensor
loss_trend_iters = []
loss_trend_vals = []

if args.gen_cp_dir: 
    gen_iter_num, start_epoch, loss_trend_iters, loss_trend_vals = load_model(args.gen_cp_dir, gen_net, gen_opt)
if args.dis_cp_dir: 
    dis_iter_num, start_epoch, _, _ = load_model(args.dis_cp_dir, dis_net, dis_opt)
    # to_train_epoch = params_cfg['epochs']-start_epoch

logging.info(f'Total epochs: {params_cfg["epochs"]}')
logging.info(f'Iterations per epoch: {len(trainloader)}')
logging.info(f'Starting at iter/epoch: {gen_iter_num}/{start_epoch}')

valid_gt = None
fake_gt = None

for epoch in tqdm(range(start_epoch, max_epoch), ncols=70):

    time1 = time.time()
    epoch_gen_loss = 0
    epoch_dis_loss = 0 
    epoch_region_loss = 0
    epoch_gen_gan_loss_func = 0
    
    # wipe out current array of GAN loss periodically 
    if model_cfg['dynamic_train'] and epoch % model_cfg['dy_epoch']==0: 
        logging.info('Wiping out loss trend vals...')
        loss_trend_iters = []
        loss_trend_vals = []

    #==========TRAIN DISCRIMINATOR==========#
    for de in tqdm(range(dis_epoch)): 
        # if epoch ==0: 
        #     continue
        for idx, sampled_batch in enumerate(trainloader):
            volume_batch, mask_batch = sampled_batch['image'].cuda(), sampled_batch['mask'].cuda()
            print(volume_batch.unique())
            mask_batch = brats_permute(mask_batch, binarize = False)
            volume_batch = F.interpolate(volume_batch, size=data_cfg['input_patch_size'], mode='trilinear', align_corners=False)
            dis_opt.zero_grad()
            fake_mask = gen_net(volume_batch)
            outputs_soft = torch.sigmoid(fake_mask)
            outputs_hard = convert_to_hard(outputs_soft)
            
            real_input = mask_batch
            fake_input = outputs_hard
           
            if model_cfg['matmul']: 
                # print_size('volume', volume_batch)
                # print_size('label', mask_batch)
                # print_size('pred', outputs_soft)
                real_input = torch.mul(mask_batch, volume_batch)
                fake_input = torch.mul(fake_input, volume_batch)
            pred_real = dis_net(real_input)
            pred_fake = dis_net(fake_input)

            # Adversarial grounde truths, real = 1, fake = 0
            if fake_gt is None: 
                out_size = pred_real.size()
                valid_gt = Tensor(np.ones((volume_batch.size(0), out_size[1],out_size[2],out_size[3],out_size[4])))
                fake_gt = Tensor(np.zeros((volume_batch.size(0), out_size[1],out_size[2],out_size[3],out_size[4])))
            # compute loss for real prediction: L2[D(x,y), 1], x is image, y is mask
            loss_real = gan_loss_func(pred_real, valid_gt)
            # compute loss for fake prediction: L2[D(x, y^), 0], x is image, y^ is generated mask
            loss_fake = gan_loss_func(pred_fake, fake_gt)

            logging.debug(f'Pred_real: {pred_real.unique()}')
            logging.debug(f'Pred_fake: {pred_fake.unique()}')

            loss_d = loss_real + loss_fake
            epoch_dis_loss+=loss_d

            dis_iter_num+=1
            logging.info(f'Discriminator {idx}/{de}/{epoch}: loss real: {loss_real} -- loss fake: {loss_fake}')
            loss_d.backward()
            dis_opt.step()
            
            writer.add_scalar('discriminator/real_sample_loss', loss_real.item(), dis_iter_num)
            writer.add_scalar('discriminator/fake_sample_loss', loss_fake.item(), dis_iter_num)
            writer.add_scalar('discriminator/total_sample_loss', loss_d, dis_iter_num)
            if idx %50==0: 
                draw_image(writer, volume_batch,'discriminator/image', c_start=0, c_end=1, iter_num=dis_iter_num, size = params_cfg['coords']) 
                draw_image(writer, mask_batch,'discriminator/groundtruth_label', c_start=1, c_end=2, iter_num=dis_iter_num, size = params_cfg['coords'])
                draw_image(writer, outputs_hard, 'discriminator/outputs_hard', c_start=1, c_end=2, iter_num=dis_iter_num, size=params_cfg['coords'])
                draw_image(writer, pred_real, 'discriminator/real_feat_map', iter_num=dis_iter_num, size = [2,11,2], mode='map')
                draw_image(writer, pred_fake, 'discriminator/fake_feat_map', iter_num=dis_iter_num, size = [2,11,2], mode='map')

            del volume_batch, mask_batch

    if dis_epoch!=0: 
        writer.add_scalar('discriminator/total_epoch_loss', epoch_dis_loss/(len(trainloader)*dis_epoch), epoch)
    if epoch%params_cfg['save_epoch']==0:
        save_model(checkpoint_root, 'discriminator', dis_net, dis_opt, epoch, dis_iter_num)

    #==========TRAIN GENERATOR==========#
    for idx, sampled_batch in enumerate(trainloader):
        volume_batch, mask_batch = sampled_batch['image'].cuda(), sampled_batch['mask'].cuda()
        mask_batch = brats_permute(mask_batch, binarize = False)
        volume_batch = F.interpolate(volume_batch, size=data_cfg['input_patch_size'], mode='trilinear', align_corners=False)

        fake_mask = gen_net(volume_batch)
        outputs_soft = torch.sigmoid(fake_mask)
        outputs_hard = convert_to_hard(outputs_soft)
        fake_input = outputs_hard
        if model_cfg['matmul']: 
            fake_input = torch.mul(fake_input, volume_batch)
        pred_fake = dis_net(fake_input)
        
        if valid_gt is None: 
            out_size = pred_fake.size()
            valid_gt = Tensor(np.ones((volume_batch.size(0), out_size[1],out_size[2],out_size[3],out_size[4])))

        loss_GAN  = gan_loss_func(pred_fake, valid_gt)
        
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

        # fake_mask return the feature maps, while output_soft is the black and white version    
        if idx % 50 ==0:
            draw_image(writer, volume_batch,'generator/image', c_start=0, c_end=1, iter_num=gen_iter_num, size = params_cfg['coords']) 
            draw_image(writer, outputs_soft,'generator/predicted_label', c_start=1, c_end=2, iter_num=gen_iter_num, size = params_cfg['coords']) 
            draw_image(writer, outputs_hard, 'generator/output_hard', c_start=1, c_end=2, iter_num=gen_iter_num, size=params_cfg['coords'])
            draw_image(writer, mask_batch,'generator/groundtruth_label', c_start=1, c_end=2, iter_num=gen_iter_num, size = params_cfg['coords'])
        
        del volume_batch, mask_batch
        
        # total loss is total batch loss of all classes
        loss_g =  3*total_region_loss + loss_GAN
        gen_iter_num += 1 
        gen_opt.zero_grad()
        loss_g.backward()
        gen_opt.step()
        epoch_region_loss += total_region_loss
        epoch_gen_gan_loss_func += loss_GAN
        epoch_gen_loss += loss_g

        if model_cfg['dynamic_train']: 
            loss_trend_iters.append(gen_iter_num)
            loss_trend_vals.append(loss_GAN.item())
            assert len(loss_trend_iters)==len(loss_trend_vals)
        
        logging.info(f'Generator {idx}/{epoch}: \n  Region loss - {gen_cfg["region_loss"]}: {total_region_loss}\n   L2 loss: {loss_GAN}')
        writer.add_scalar('generator_sample/region_loss',
                            total_region_loss.item(), gen_iter_num)
        writer.add_scalar('generator_sample/GAN_loss', loss_GAN.item(), gen_iter_num)
        writer.add_scalar('generator_sample/total_sample_loss', loss_g, gen_iter_num)

    if epoch%params_cfg['save_epoch']==0: 
        save_model(checkpoint_root,'generator', gen_net, gen_opt, epoch, gen_iter_num, loss_iter=loss_trend_iters, loss_vals=loss_trend_vals)

    # if the slopeficient > 0, generator GAN loss increases. -> train generator <-> dis_epoch = 0
    # else, the loss is stable or reduced, the generator is either being too good or cannot learn anything 
    # -> train discriminator to fool the generator <-> dis_epoch = dis_cfg['epochs']
    if model_cfg['dynamic_train']: 
        if model_cfg['simple_dynamic']: 
            avg_epoch_gan_loss = epoch_gen_gan_loss_func/len(trainloader) 
            dis_epoch = simple_dynamic(avg_epoch_gan_loss, dis_cfg['epochs'])
            logging.info(f'Using simple dynamic:\nAvg GAN loss of 1 epoch is: {avg_epoch_gan_loss} - Discriminator epoch: {dis_epoch}')
        else: 
            logging.info('Evaluating trend... ')
            slope, y_inter, dis_epoch = calculate_trend(loss_trend_iters, loss_trend_vals, dis_cfg['epochs'])
            logging.info(f'Coeficient: {slope} - Y-intercept: {y_inter} - Discriminator epoch: {dis_epoch}')
            writer.add_scalar('generator-epoch/slope', slope, epoch)
            writer.add_scalar('generator-epoch/y-intercept', y_inter, epoch)
    
    writer.add_scalar('generator-epoch/region_loss', epoch_region_loss /len(trainloader), epoch)
    writer.add_scalar('generator-epoch/GAN_loss', epoch_gen_gan_loss_func/len(trainloader), epoch)
    writer.add_scalar('generator-epoch/total_epoch_loss',epoch_gen_loss/len(trainloader),epoch)
