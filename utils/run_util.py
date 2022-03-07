import random
import json 
from functools import reduce
import os
import torch
import logging

from torchvision.utils import make_grid

def print_size(name,x): 
    print(f'Block {name} size: {x.size()}')

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

def warmup_constant(x, warmup=500):
    if x < warmup:
        return x/warmup
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

def draw_image(writer, arr, img_name: str, from_slice: int, to_slice: int, iter_num: int, coords: list): 
    """Draw image to tensorboard"""
    x, y, z = coords
    # if type=='brain':
    #     image = arr[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
    # else: 
    #     image =arr[0, 1:2, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
    
    image = arr[0, from_slice:to_slice, : , : , x:y:z].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
    grid_image = make_grid(image, 5, normalize=True)
    writer.add_image(img_name, grid_image, iter_num)

