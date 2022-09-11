import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os

import voc12.dataloader
from misc import torchutils, imutils

cudnn.enabled = True

def _work(process_id, model,model_ori, dataset, args):

    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()
        model_ori.cuda()

        for iter, pack in enumerate(data_loader):

            img_name = pack['name'][0]
            label = pack['label'][0] 
            size = pack['size']

            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)

            BAS_outputs = [model(img[0].cuda(non_blocking=True))
                       for img in pack['img']]

            strided_BAS = torch.sum(torch.stack(
                [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o
                 in BAS_outputs]), 0)

            highres_BAS = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                         mode='bilinear', align_corners=False) for o in BAS_outputs]
            highres_BAS = torch.sum(torch.stack(highres_BAS, 0), 0)[:, 0, :size[0], :size[1]]

            valid_cat = torch.nonzero(label)[:, 0] 

            strided_BAS = strided_BAS[valid_cat]
            strided_BAS /= F.adaptive_max_pool2d(strided_BAS, (1, 1)) + 1e-5

            highres_BAS = highres_BAS[valid_cat]
            highres_BAS /= F.adaptive_max_pool2d(highres_BAS, (1, 1)) + 1e-5

            if True:  ## combine with CAM
                cam = [model_ori(img[0].cuda(non_blocking=True))
                       for img in pack['img']]

                strided_cam = torch.sum(torch.stack(
                    [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o
                    in cam]), 0)

                highre_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                            mode='bilinear', align_corners=False) for o in cam]
                highre_cam = torch.sum(torch.stack(highre_cam, 0), 0)[:, 0, :size[0], :size[1]]

                strided_cam = strided_cam[valid_cat]
                strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

                strided_BAS = strided_cam + strided_BAS
                strided_BAS /= F.adaptive_max_pool2d(strided_BAS, (1, 1)) + 1e-5

                highre_cam = highre_cam[valid_cat]
                highre_cam /= F.adaptive_max_pool2d(highre_cam, (1, 1)) + 1e-5

                highres_BAS = highre_cam + highres_BAS
                highres_BAS /= F.adaptive_max_pool2d(highres_BAS, (1, 1)) + 1e-5

            # save cams
            np.save(os.path.join(args.cam_out_dir, img_name + '.npy'),
                    {"keys": valid_cat, "cam": strided_BAS.cpu(), "high_res": highres_BAS.cpu().numpy()})

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')

def run(args):
    model_ori = getattr(importlib.import_module("net.resnet50_cam"), 'CAM')()
    model_ori.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)
    model_ori.eval()

    model = getattr(importlib.import_module(args.cam_network), 'model')(args)
    checkpoint = torch.load('sess/epoch_10.pth.tar')  
    pretrained_dict = {k[7:]: v for k, v in checkpoint.items()}
    model.load_state_dict(pretrained_dict)
    model.eval()

    n_gpus = torch.cuda.device_count()

    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.train_list,
                                                             voc12_root=args.voc12_root, scales=args.cam_scales)
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model,model_ori, dataset, args), join=True)
    print(']')

    torch.cuda.empty_cache()