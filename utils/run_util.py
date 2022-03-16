import random
import json 
from functools import reduce
import os
import torch
import logging
from dataloaders.datasets3d import *

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

    params.update(model_state_dict)
    net.load_state_dict(params)
    if optim_state_dict is not None and optimizer is not None:
        optimizer.load_state_dict(optim_state_dict)
    logging.info(f'model loaded from {checkpoint_path}')
    # warmup info is mainly in optim_state_dict. So after loading the checkpoint,
    # the optimizer won't do warmup already.
    return iter_num, epoch_num


def save_model(checkpoint_dir, model_type,net, optimizer, epoch_num,iter_num):   
    save_model_path = os.path.join(
        checkpoint_dir, f'{model_type}_epoch{epoch_num}.pth')
    torch.save({'iter_num': iter_num, 'epoch_num': epoch_num, 'model': net.state_dict(),
                'optim_state': optimizer.state_dict()},
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