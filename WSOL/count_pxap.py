from cProfile import label
import numpy as np
import cv2
import os 
from DataLoader import *
import argparse
import torch 
from torch.autograd import Variable
from Model import *
from utils.accuracy import *
import matplotlib.pyplot as plt
_RESIZE_LENGTH = 224

def load_mask_image(file_path, resize_size):
    mask = np.float32(cv2.imread(file_path, cv2.IMREAD_GRAYSCALE))
    mask = cv2.resize(mask, resize_size, interpolation=cv2.INTER_NEAREST)
    return mask

            
def get_mask(mask_root, mask_paths, ignore_path):
    mask_all_instances = []
    for mask_path in mask_paths:
        mask_file = os.path.join(mask_root, mask_path)
        mask = load_mask_image(mask_file, (_RESIZE_LENGTH, _RESIZE_LENGTH))
        mask_all_instances.append(mask > 0.5)
    mask_all_instances = np.stack(mask_all_instances, axis=0).any(axis=0)

    ignore_file = os.path.join(mask_root, ignore_path)
    ignore_box_mask = load_mask_image(ignore_file,
                                      (_RESIZE_LENGTH, _RESIZE_LENGTH))
    ignore_box_mask = ignore_box_mask > 0.5

    ignore_mask = np.logical_and(ignore_box_mask,
                                 np.logical_not(mask_all_instances))

    if np.logical_and(ignore_mask, mask_all_instances).any():
        raise RuntimeError("Ignore and foreground masks intersect.")

    return (mask_all_instances.astype(np.uint8) +
            255 * ignore_mask.astype(np.uint8))

def get_mask_paths(metadata):
    mask_paths = {}
    ignore_paths = {}
    with open(metadata) as f:
        for line in f.readlines():
            image_id, mask_path, ignore_path = line.strip('\n').split(',')
            if image_id in mask_paths:
                mask_paths[image_id].append(mask_path)
                assert (len(ignore_path) == 0)
            else:
                mask_paths[image_id] = [mask_path]
                ignore_paths[image_id] = ignore_path
    return mask_paths, ignore_paths

def get_mask_paths_cubmask(metadata):
    mask_paths = {}
    ignore_paths = {}
    with open(metadata) as f:
        for line in f.readlines():
            image_id, mask_path, ignore_path = line.strip('\n').split(',')
            if image_id in mask_paths:
                mask_paths[image_id].append(mask_path)
                assert (len(ignore_path) == 0)
            else:
                mask_paths['/'.join(image_id.split('/')[1:])] = [mask_path]
                ignore_paths['/'.join(image_id.split('/')[1:])] = ignore_path
    return mask_paths, ignore_paths

def normalize_map(atten_map,shape):
    min_val = np.min(atten_map)
    max_val = np.max(atten_map)
    atten_norm = (atten_map - min_val)/(max_val - min_val + 1e-5)
    atten_norm = cv2.resize(atten_norm, dsize=(shape[1],shape[0]))
    return atten_norm


def val_pxap_one_epoch(args, Val_Loader, model, epoch=0):
    if args.evaluate == False:
        return None, None, None 
    cam_threshold_list = list(np.arange(0, 1, 0.001))
    threshold_list_right_edge = np.append(cam_threshold_list, [1.0, 2.0, 3.0])
    num_bins = len(cam_threshold_list) + 2
    gt_true_score_hist = np.zeros(num_bins, dtype=np.float)
    gt_false_score_hist = np.zeros(num_bins, dtype=np.float)
    if args.root == 'OpenImage':
        mask_paths, ignore_paths = get_mask_paths('/home2/Datasets/OpenImage/test_mask.txt')
    elif args.root == 'CUB_200_2011':
         mask_paths, ignore_paths = get_mask_paths_cubmask('/home2/Datasets/CUB_200_2011/CUBMask/test/localization.txt')
    for step, (img_batch, img_tencrop, img_class, box, path_batch) in enumerate(Val_Loader): ## batch_size=1
        with torch.no_grad():
            batch = img_batch.size(0)
            img_batch = Variable(img_batch).cuda() 
            output1, _, _, fmap_batch = model(img_batch, img_class, args.topk, evaluate=True)
            for i in range(batch):
                shape = (224, 224)
                path = path_batch[i].replace('\\','/')
                img_name = path.split('/')[-1]
                img_path = path.split('/')[-2] + '/' + path.split('/')[-1]
                if args.root == 'OpenImage':
                    gt_mask = get_mask('/home2/Datasets/OpenImage/mask/',
                                        mask_paths['test/' + img_path],
                                        ignore_paths['test/' + img_path])
                elif args.root == 'CUB_200_2011':
                    gt_mask = get_mask('/home2/Datasets/CUB_200_2011/CUBMask/',
                                        mask_paths[img_path],
                                        ignore_paths[img_path])
                fmap = np.array(fmap_batch.data.cpu())[i][0]
                fmap = normalize_map(fmap, shape) 
                
                gt_true_scores = fmap[gt_mask == 1]
                gt_false_scores = fmap[gt_mask == 0]
                # histograms in ascending order
                gt_true_hist, _ = np.histogram(gt_true_scores,
                                            bins=threshold_list_right_edge)
                gt_true_score_hist += gt_true_hist.astype(np.float)

                gt_false_hist, _ = np.histogram(gt_false_scores,
                                                bins=threshold_list_right_edge)
                gt_false_score_hist += gt_false_hist.astype(np.float)

                if args.save_img_flag:
                    ori_img = cv2.imread(path)
                    ori_img = cv2.resize(ori_img, (224,224))
                    heatmap = np.uint8(255 * fmap)
                    heatmap = heatmap.astype(np.uint8)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    img_add = cv2.addWeighted(ori_img.astype(np.uint8), 0.5, heatmap.astype(np.uint8), 0.5, 0)
                    if os.path.exists('logs/' + args.root + '/' + args.arch + '/save_pxap_img') == 0:
                        os.mkdir('logs/' + args.root + '/' + args.arch + '/save_pxap_img')
                    cv2.imwrite('logs/' + args.root + '/' + args.arch + '/save_pxap_img/' + img_name, img_add)
        

    num_gt_true = gt_true_score_hist.sum()
    tp = gt_true_score_hist[::-1].cumsum()
    fn = num_gt_true - tp

    num_gt_false = gt_false_score_hist.sum()
    fp = gt_false_score_hist[::-1].cumsum()
    tn = num_gt_false - fp

    if ((tp + fn) <= 0).all():
        raise RuntimeError("No positive ground truth in the eval set.")
    if ((tp + fp) <= 0).all():
        raise RuntimeError("No positive prediction in the eval set.")

    non_zero_indices = (tp + fp) != 0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    IoU = tp / (fp + fn + tp)
    IoU*= 100

    PxAP = (precision[1:] * np.diff(recall))[non_zero_indices[1:]].sum()
    PxAP *= 100

    pIoU = max(IoU)
    print('Val Epoch : [{}] \tPxAP: {:.2f}  \t pIoU : {:.2f}  \n'.format(epoch, PxAP, pIoU))

    return PxAP, pIoU, 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--resize_size', type=int, default=224) 
    parser.add_argument('--root', type=str, help="[CUB_200_2011, OpenImage]", 
                                default='OpenImage')
    parser.add_argument('--num_classes', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument('--arch', type=str, default='resnet')  ##  choose  [ vgg, resnet, inception, mobilenet ]
    parser.add_argument('--evaluate', type=bool, default=True)
    parser.add_argument('--save_img_flag', type=bool, default=False)
    parser.add_argument('--tencrop', type=bool, default=False)
    parser.add_argument('--gpu', type=str, default='3')           
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    MyData = eval(args.root).ImageDataset(args, phase='test')
    Val_Loader = torch.utils.data.DataLoader(dataset=MyData, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    ##  model
    model = eval(args.arch).model(args, pretrained=False)
    model.cuda(device=0)
    checkpoint = torch.load('logs/' + args.root + '/' + args.arch + '/' + 'best_loc.pth.tar') ## best_top1  best_gt  best_cls  epoch_99
    checkpoint['model'] = {k[7:]: v for k, v in checkpoint['model'].items()}
    model.load_state_dict(checkpoint['model'])
    model.eval()

    val_pxap_one_epoch(args, Val_Loader, model)

