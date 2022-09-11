import os
import argparse
import torch
import torch.nn as nn 
from Model import *
from DataLoader import *
from torch.autograd import Variable
from evaluator import val_loc_one_epoch 
from count_pxap import val_pxap_one_epoch
from utils.accuracy import *
from utils.lr import *
from utils.optimizer import *
from utils.func import *
from utils.util import *
from utils.hyperparameters import get_hyperparameters
import pprint
import os
import random
import shutil
seed = 6
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        ##  path
        self.parser.add_argument('--root', type=str, help="[CUB_200_2011, OpenImage, Fgvc_aircraft_2013b, Standford_Car, Stanford_Dogs]", 
                                  default='CUB_200_2011')
        self.parser.add_argument('--num_classes', type=int, default=200)      
        ##  save
        self.parser.add_argument('--save_path', type=str, default='logs')
        self.parser.add_argument('--log_file', type=str, default='log.txt')
        self.parser.add_argument('--log_code_dir', type=str, default='save_code')
        ##  dataloader
        self.parser.add_argument('--crop_size', type=int, default=224)
        self.parser.add_argument('--resize_size', type=int, default=256) 
        self.parser.add_argument('--num_workers', type=int, default=8)
        ##  train
        self.parser.add_argument('--batch_size', type=int, default=32)
        self.parser.add_argument('--epochs', type=int, default=120)
        self.parser.add_argument('--decay_epoch', type=int, default=80)
        self.parser.add_argument('--phase', type=str, default='train')  
        self.parser.add_argument('--lr', type=float, default=0.001)
        self.parser.add_argument('--weight_decay', type=float, default=5e-4)
        self.parser.add_argument('--power', type=float, default=0.9)
        self.parser.add_argument('--momentum', type=float, default=0.9)
        ## evaluate 
        self.parser.add_argument('--evaluate_epoch', type=int, default=30)
        self.parser.add_argument('--topk', type=int, default=200)
        self.parser.add_argument('--threshold', type=float, default=[0.1])
        self.parser.add_argument('--tencrop', type=bool, default=False)
        self.parser.add_argument('--evaluate', type=bool, default=False)
        self.parser.add_argument('--save_img_flag', type=bool, default=False)
        ## parameters
        self.parser.add_argument('--alpha', type=float, default=0.5)
        self.parser.add_argument('--beta', type=float, default=1.5)
        ##  model
        self.parser.add_argument('--arch', type=str, help="[vgg, resnet, inception, mobilenet]",
                                    default='mobilenet')           
        ##  GPU'
        self.parser.add_argument('--gpu', type=str, default='5')
        
    def parse(self):
        opt = self.parser.parse_args()
        opt.arch = opt.arch     
        return opt

args = opts().parse()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

args = get_hyperparameters(args)

## save_log_txt
makedirs(args.save_path + '/' + args.root + '/' + args.arch + '/' + args.log_code_dir)
sys.stdout = Logger(args.save_path + '/' + args.root + '/' + args.arch +  '/' + args.log_code_dir + '/' + args.log_file)
sys.stdout.log.flush()

##  save_code
save_file = ['train.py', 'train_ILSVRC.py', 'evaluator.py', 'count_pxap.py', 'show_loc.py']
for file_name in save_file:
    shutil.copyfile(file_name, args.save_path + '/' + args.root + '/' + args.arch + '/' + args.log_code_dir + '/' + file_name)
save_dir = ['Model', 'utils', 'DataLoader']
for dir_name in save_dir:
    copy_dir(dir_name, args.save_path + '/' + args.root + '/' + args.arch + '/' + args.log_code_dir + '/' + dir_name)

if __name__ == '__main__':
    TrainData = eval(args.root).ImageDataset(args, phase='train')
    ValData = eval(args.root).ImageDataset(args, phase='test')
    Train_Loader = torch.utils.data.DataLoader(dataset=TrainData, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    Val_Loader = torch.utils.data.DataLoader(dataset=ValData, batch_size=64,shuffle=False, num_workers=args.num_workers, pin_memory=True)
    ##  model
    model = eval(args.arch).model(args, pretrained=True)
    model.cuda(device=0)

    ##  optimizer 
    optimizer = get_optimizer(model, args)
    loss_func = nn.CrossEntropyLoss().cuda()
    best_gt, best_top1, best_loc = 0, 0, 0
    print('Train begining!')
    for epoch in range(0, args.epochs):
        ##  accuracy
        cls_acc_1 = AverageMeter()
        cls_acc_2 = AverageMeter()
        loss_epoch_1 = AverageMeter()
        loss_epoch_2 = AverageMeter()
        loss_epoch_3 = AverageMeter()
        loss_epoch_4 = AverageMeter()
        
        poly_lr_scheduler(optimizer, epoch, decay_epoch=args.decay_epoch)
        model.train()
        for step, (path, imgs, label) in enumerate(Train_Loader):                
            imgs, label = Variable(imgs).cuda(), label.cuda()
            ##  backward
            optimizer.zero_grad()
            output1, output2, bas, area, map = model(imgs, label, 1)
            label = label.long()
            pred = torch.max(output1, 1)[1]  
            loss_1 = loss_func(output1, label).cuda()
            loss_2 = loss_func(output2, label).cuda()
            bas, area = bas.mean(0), area.mean(0)
            loss =  loss_1 + loss_2 * args.alpha  + bas + area * args.beta 
            loss.backward()
            optimizer.step() 
            ##  count_accuracy
            cur_batch = label.size(0)
            cur_cls_acc_1 = 100. * compute_cls_acc(output1, label) 
            cls_acc_1.updata(cur_cls_acc_1, cur_batch)
            cur_cls_acc_2 = 100. * compute_cls_acc(output2, label) 
            cls_acc_2.updata(cur_cls_acc_2, cur_batch)
            loss_epoch_1.updata(loss_1.data, 1)
            loss_epoch_2.updata(loss_2.data, 1)
            loss_epoch_3.updata(bas.data, 1)
            loss_epoch_4.updata(area.data, 1)

        print('Epoch:[{}/{}]\tstep:[{}/{}]\tCLS:{:.3f}\tFRG:{:.3f}\tBAS:{:.2f}\tArea:{:.2f}\tepoch_acc:{:.2f}%\tepoch_acc2:{:.2f}%'.format(
                        epoch+1, args.epochs, step+1, len(Train_Loader), loss_epoch_1.avg,loss_epoch_2.avg,loss_epoch_3.avg,loss_epoch_4.avg,cls_acc_1.avg,cls_acc_2.avg
                ))
        sys.stdout.log.flush()    
        torch.save({'model':model.state_dict(),
                    'best_thr':0,
                    'epoch':epoch+1,
                    }, os.path.join(args.save_path, args.root + '/' +args.arch + '/' + 'epoch_'+ str(epoch+1) +'.pth.tar'), _use_new_zipfile_serialization=False)

        ##  test_acc
        if epoch >= args.evaluate_epoch:
            args.evaluate = True
        top1_acc, gt_acc, thr = {"CUB_200_2011": val_loc_one_epoch,  
                             "Fgvc_aircraft_2013b": val_loc_one_epoch,
                             "Standford_Car": val_loc_one_epoch,
                             "Stanford_Dogs": val_loc_one_epoch,
                             "OpenImage": val_pxap_one_epoch,
                            }[args.root](args, Val_Loader, model, epoch+1)
        if args.evaluate:
            loc_acc = top1_acc + gt_acc
            if top1_acc > best_top1:
                best_top1 = top1_acc
                torch.save({'model':model.state_dict(),
                            'best_thr':thr,
                            'epoch':epoch+1,
                            }, os.path.join(args.save_path, args.root + '/' + args.arch + '/' + 'best_top1.pth.tar'),_use_new_zipfile_serialization=False)
            if gt_acc > best_gt:
                best_gt = gt_acc
                torch.save({'model':model.state_dict(),
                            'best_thr':thr,
                            'epoch':epoch+1,
                            }, os.path.join(args.save_path, args.root + '/' + args.arch + '/' + 'best_gt.pth.tar'),_use_new_zipfile_serialization=False)
            if loc_acc > best_loc:
                best_loc = loc_acc
                torch.save({'model':model.state_dict(),
                            'best_thr':thr,
                            'epoch':epoch+1,
                            }, os.path.join(args.save_path, args.root + '/' + args.arch + '/' + 'best_loc.pth.tar'),_use_new_zipfile_serialization=False)

            sys.stdout.log.flush()

    if args.evaluate:
        ##  test best_gt
        checkpoint = torch.load(args.save_path  + '/' + args.root + '/' + args.arch + '/' + 'best_gt.pth.tar') ## best_top1  best_gt  epoch_99
        model.load_state_dict(checkpoint['model'])
        args.threshold = [checkpoint['best_thr']] if checkpoint['best_thr']!=0 else args.threshold
        print('\ntest best gt checkpoint')
        top1_acc, gt_acc, thr = {"CUB_200_2011": val_loc_one_epoch,
                             "Fgvc_aircraft_2013b": val_loc_one_epoch,
                             "Standford_Car": val_loc_one_epoch,
                             "Stanford_Dogs": val_loc_one_epoch,
                             "OpenImage": val_pxap_one_epoch,
                            }[args.root](args, Val_Loader, model, checkpoint['epoch'])

        ##  test best_top1
        checkpoint = torch.load(args.save_path  + '/' + args.root + '/' + args.arch + '/' + 'best_top1.pth.tar') ## best_top1  best_gt  epoch_99
        model.load_state_dict(checkpoint['model'])
        args.threshold = [checkpoint['best_thr']] if checkpoint['best_thr']!=0 else args.threshold
        print('\ntest best top1 checkpoint')
        top1_acc, gt_acc, thr = {"CUB_200_2011": val_loc_one_epoch,
                             "Fgvc_aircraft_2013b": val_loc_one_epoch,
                             "Standford_Car": val_loc_one_epoch,
                             "Stanford_Dogs": val_loc_one_epoch,
                             "OpenImage": val_pxap_one_epoch,
                            }[args.root](args, Val_Loader, model, checkpoint['epoch'])
        sys.stdout.log.flush()

        checkpoint = torch.load(args.save_path  + '/' + args.root + '/' + args.arch + '/' + 'best_loc.pth.tar') ## best_top1  best_gt  epoch_99
        model.load_state_dict(checkpoint['model'])
        args.threshold = [checkpoint['best_thr']] if checkpoint['best_thr']!=0 else args.threshold
        print('\ntest best loc checkpoint')
        top1_acc, gt_acc, thr = {"CUB_200_2011": val_loc_one_epoch,
                             "Fgvc_aircraft_2013b": val_loc_one_epoch,
                             "Standford_Car": val_loc_one_epoch,
                             "Stanford_Dogs": val_loc_one_epoch,
                             "OpenImage": val_pxap_one_epoch,
                            }[args.root](args, Val_Loader, model, checkpoint['epoch'])
        sys.stdout.log.flush()
    






