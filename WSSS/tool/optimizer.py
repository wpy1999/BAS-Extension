import torch
from tool import pyutils, imutils, torchutils, visualization

def get_optimizer(model, args, max_step):
    param_groups = ([], [], [], [])
    for name, value in model.named_parameters():
        if 'classifier' in name : 
            if 'weight' in name:
                param_groups[2].append(value)
            elif 'bias' in name:
                param_groups[3].append(value)
        else:
            if 'weight' in name:
                param_groups[0].append(value)
            elif 'bias' in name:
                param_groups[1].append(value)

    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)
    return optimizer