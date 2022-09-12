import os
import argparse
import torch
import torch.nn as nn 
from Model import *
from DataLoader import *
from torch.autograd import Variable
#from evaluator import val_loc_one_epoch 
from skimage import measure
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.hyperparameters import get_hyperparameters
from utils.accuracy import *
from utils.lr import *
from utils.optimizer import *
from utils.func import *
from utils.util import *
import pprint
import os
import random
import shutil
def set_seed(seed):
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
        self.parser.add_argument('--root', type=str, help="[CUB_200_2011, ILSVRC, OpenImage, Fgvc_aircraft_2013b, Standford_Car, Stanford_Dogs]", 
                                  default='ILSVRC')
        self.parser.add_argument('--num_classes', type=int, default=1000)    
        self.parser.add_argument('--seed', type=int, default=0)  
        ##  save
        self.parser.add_argument('--save_path', type=str, default='logs')
        self.parser.add_argument('--log_file', type=str, default='log.txt')
        self.parser.add_argument('--log_code_dir', type=str, default='save_code')
        self.parser.add_argument('--threshold', type=float, default=[0.5])
        self.parser.add_argument('--evaluate', type=bool, default=False)
        self.parser.add_argument('--save_img_flag', type=bool, default=False)
        self.parser.add_argument('--tencrop', type=bool, default=False)
        ##  dataloader
        self.parser.add_argument('--crop_size', type=int, default=224)
        self.parser.add_argument('--resize_size', type=int, default=256) 
        self.parser.add_argument('--num_workers', type=int, default=2)
        self.parser.add_argument("--local_rank", type=int,default=-1)
        ##  train
        self.parser.add_argument('--batch_size', type=int, default=64)
        self.parser.add_argument('--epochs', type=int, default=9)
        self.parser.add_argument('--phase', type=str, default='train') ## train / test
        self.parser.add_argument('--lr', type=float, default=0.001)
        self.parser.add_argument('--weight_decay', type=float, default=5e-4)
        self.parser.add_argument('--power', type=float, default=0.9)
        self.parser.add_argument('--momentum', type=float, default=0.9)
        ##  model
        self.parser.add_argument('--arch', type=str,  help="[vgg, resnet, inception, mobilenet]",
                                    default='resnet')           
        ##  GPU'
        
    def parse(self):
        opt = self.parser.parse_args()
        opt.arch = opt.arch     
        return opt

args = opts().parse()

## default setting (topk, threshold)
args = get_hyperparameters(args)
args.batch_size = args.batch_size // int(torch.cuda.device_count())
print(args.batch_size)

## save_log_txt
makedirs(args.save_path + '/' + args.root + '/' + args.arch + '/' + args.log_code_dir)
sys.stdout = Logger(args.save_path + '/' + args.root + '/' + args.arch +  '/' + args.log_code_dir + '/' + args.log_file)
sys.stdout.log.flush()

##  save_code
save_file = ['train.py', 'train_ILSVRC.py', 'evaluator.py', 'count_pxap.py']
for file_name in save_file:
    shutil.copyfile(file_name, args.save_path + '/' + args.root + '/' + args.arch + '/' + args.log_code_dir + '/' + file_name)
save_dir = ['Model', 'utils', 'DataLoader']
for dir_name in save_dir:
    copy_dir(dir_name, args.save_path + '/' + args.root + '/' + args.arch + '/' + args.log_code_dir + '/' + dir_name)

if __name__ == '__main__':
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')   
    device = torch.device("cuda", local_rank)
    set_seed(args.seed + dist.get_rank())
    TrainData = eval(args.root).ImageDataset(args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(TrainData)
    Train_Loader = torch.utils.data.DataLoader(dataset=TrainData, batch_size=args.batch_size , sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    
    ##  model
    model = eval(args.arch).model(args, pretrained=True) 
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
    model = DDP(model, find_unused_parameters=False , device_ids=[local_rank], output_device=local_rank)
    
    ##  optimizer 
    optimizer = get_optimizer(model, args)
    loss_func = nn.CrossEntropyLoss().to(local_rank)
    best_gt = 0
    best_top1 = 0
    best_cls = 0
    if dist.get_rank() == 0:
        print('Train begining!')
    for epoch in range(0, args.epochs):
        Train_Loader.sampler.set_epoch(epoch)
        ##  accuracy
        cls_acc_1 = AverageMeter()
        cls_acc_2 = AverageMeter()
        loss_epoch_1 = AverageMeter()
        loss_epoch_2 = AverageMeter()
        loss_epoch_3 = AverageMeter()
        loss_epoch_4 = AverageMeter()
        poly_lr_scheduler(optimizer, epoch, decay_epoch=3)
        model.train()
        for step, (path, imgs, label) in enumerate(Train_Loader):       
            imgs, label = Variable(imgs).to(local_rank), label.to(local_rank)
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
            if (step+1) % (len(Train_Loader) // 5) == 0 and args.root=='ILSVRC' and dist.get_rank() == 0:    
                iter = int((step+1) // (len(Train_Loader) // 5)) 
                print('Epoch:[{}/{}]\tstep:[{}/{}]\tCLS:{:.3f}\tFRG:{:.3f}\tBAS:{:.2f}\tArea:{:.2f}\tepoch_acc:{:.2f}%\tepoch_acc2:{:.2f}%'.format(
                        epoch+1, args.epochs, step+1, len(Train_Loader), loss_epoch_1.avg,loss_epoch_2.avg,loss_epoch_3.avg,loss_epoch_4.avg,cls_acc_1.avg,cls_acc_2.avg
                ))
                sys.stdout.log.flush() 
        if dist.get_rank() == 0:
            print('Epoch:[{}/{}]\tstep:[{}/{}]\tCLS:{:.3f}\tFRG:{:.3f}\tBAS:{:.2f}\tArea:{:.2f}\tepoch_acc:{:.2f}%\tepoch_acc2:{:.2f}%'.format(
                        epoch+1, args.epochs, step+1, len(Train_Loader), loss_epoch_1.avg,loss_epoch_2.avg,loss_epoch_3.avg,loss_epoch_4.avg,cls_acc_1.avg,cls_acc_2.avg
                ))
            sys.stdout.log.flush()    
            torch.save({'model':model.state_dict(),
                        'best_thr':0,
                        'epoch':epoch+1,
                        }, os.path.join(args.save_path, args.root + '/' +args.arch + '/' + 'epoch_'+ str(epoch+1) +'.pth.tar'), _use_new_zipfile_serialization=False)

        
