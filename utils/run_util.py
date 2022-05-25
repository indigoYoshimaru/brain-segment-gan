import random
import json 
from functools import reduce
import os
import torch
import numpy as np
import logging
from dataloaders.datasets3d import *
from sklearn import linear_model as lm 

from torchvision.utils import make_grid

def print_size(name,start): 
    print(f'Block {name} size: {start.size()}')

def print_arg(*print_args, **kwargs):
    #if args.local_rank == 0:
    print(*print_args, **kwargs)

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

def compose2(f, g):
    return lambda *a, **kw: g(f(*a, **kw))

def compose(*fs):
    return reduce(compose2, fs)

def warmup_constant(start, warmup=500):
    if start < warmup:
        return start/warmup
    return 1

def set_deterministic_mode(): 
    ...

def read_json(file_dir): 
    with open(file_dir, "r") as content:
        data = json.load(content)
    return data

def load_model(checkpoint_path, net, optimizer): 
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cuda'))
    params = net.state_dict()
    model_state_dict = state_dict['model']
    optim_state_dict = state_dict['optim_state']
    iter_num = state_dict['iter_num'] + 1
    epoch_num = state_dict['epoch_num'] + 1
    loss_iters = state_dict.get('loss_iters', [])
    loss_vals = state_dict.get('loss_vals', []) 

    params.update(model_state_dict)
    net.load_state_dict(params)
    if optim_state_dict is not None and optimizer is not None:
        optimizer.load_state_dict(optim_state_dict)
    logging.info(f'model loaded from {checkpoint_path}')
    # warmup info is mainly in optim_state_dict. So after loading the checkpoint,
    # the optimizer won't do warmup already.
    return iter_num, epoch_num, loss_iters, loss_vals


def save_model(checkpoint_dir, model_type,net, optimizer, epoch_num,iter_num, loss_iter=[], loss_vals=[]):   
    save_model_path = os.path.join(
        checkpoint_dir, f'{model_type}_epoch{epoch_num}.pth')

    torch.save({'iter_num': iter_num, 'epoch_num': epoch_num, 'model': net.state_dict(),
                'optim_state': optimizer.state_dict(), 'loss_iters': loss_iter, 'loss_vals': loss_vals},
                # 'args': vars(args)},
                save_model_path)

    logging.info(f'save model to {save_model_path}')

def draw_image(writer, arr, img_name: str, iter_num: int, size: list, c_start=0, c_end =0, n_grid = 5, mode='train', is_norm=True): 
    """Draw image to tensorboard"""
    start, end, step = size
    
    if mode =='test': 
        image = arr[c_start:c_end, : , : , start:end:step]
    elif mode=='map': 
        image = arr[0, : , : , : , start:end:step]
        is_norm = False
    else:
        image = arr[0, c_start:c_end, : , : , start:end:step]
    
    image = image.permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
    grid_image = make_grid(image, n_grid, normalize=is_norm)
    writer.add_image(img_name, grid_image, iter_num)

def convert_to_hard(preds_soft, thres=0.5): 
    preds_hard = torch.zeros_like(preds_soft)
    preds_hard[preds_soft>=thres]=1
    return preds_hard

def convert_to_img(preds_soft): 
    converted = torch.zeros_like(preds_soft)
    for idx, img in enumerate(preds_soft):
        inv_probs = brats_inv_map_label(img)
        preds_hard = torch.argmax(inv_probs, dim=0)
        # Map 3 to 4, and keep 0, 1, 2 unchanged.
        preds_hard += (preds_hard == 3)
        converted[idx] = preds_hard
    # print(f'converted shape: {converted.shape}')
    # print(f'converted unique: {converted.unique()}')
    # test = converted[0]
    # pair1 = (test[0]==test[1])
    # pair2 = (test[1]==test[2])
    # pair3 = (test[2]==test[3])
    # print(f'equal: pair1: {pair1.unique()}; pair2: {pair2.unique()}; pair3: {pair3.unique()}')
    return converted

def rotate_image(img, axes): 
    img = img.gpu()
    img = np.rot90(img, 1, axes)
    return torch.from_numpy(img).copy()

def rotate_visual(img, gt, pred, axes: tuple): 
    """
    Rotate brain image and masks for perspective visualization
    Args: 
        img: image tensor
        gt: groundtruth tensor
        pred: predicted tumor tensor
        axes: tuple of axis to rotate image. 
            the initial perspective is from the top
    """
    if axes == (0,0): 
        return {'image': img, 'gt': gt, 'pred': pred}
    img = img.cpu()
    gt = gt.cpu()
    pred = pred.cpu()
    img_rot = np.rot90(img, 1, axes)
    gt_rot = np.rot90(gt, 1, axes)
    pred_rot = np.rot90(pred, 1, axes)
    img_rot = np.flip(img_rot, 3)
    gt_rot = np.flip(gt_rot, 3)
    pred_rot = np.flip(pred_rot, 3)
    img_rot = torch.from_numpy(img_rot.copy())
    gt_rot = torch.from_numpy(gt_rot.copy())
    pred_rot = torch.from_numpy(pred_rot.copy())
    return {'image': img_rot, 'gt': gt_rot, 'pred': pred_rot}


def calculate_trend(iters: list, vals: list, dis_epoch:int):
    """
    Calculate the slope (trend) to determine discriminator training. 
    if the slope > 0 or slope ~ 0 and intercp~1 -> dis_epoch = 0. 
    else dis_epoch = dis_epoch (!= 0)

    Args: 
        iters: list of iterations during k epochs of generator
        vals: list of loss values during k epochs of generator
    
    """
    reg_model = lm.LinearRegression()
    iters = np.asarray(iters).reshape(1, -1).transpose()
    vals = np.asarray(vals)
    
    reg_model.fit(iters, vals)
    slope = reg_model.coef_[0]
    y_inter = reg_model.intercept_
    del reg_model
    # if (abs(slope)<0.1 and y_inter>0.7) or (slope > 0): 
    #     return slope, y_inter, 0

    if slope < -0.0001: 
        return slope, y_inter, dis_epoch

    if abs(slope)<=0.0001: 
        if y_inter >0.7: 
            return slope, y_inter, 0
        # if y_inter <0.5: 
        else: 
            return slope, y_inter, dis_epoch
    
    return slope, y_inter, 0 

def simple_dynamic(avg_gan: int, dis_epoch: int): 
    if avg_gan>0.7: 
        return 0
    return dis_epoch