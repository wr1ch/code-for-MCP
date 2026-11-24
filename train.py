from torch.utils.tensorboard import SummaryWriter
import os, utils, glob, losses
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from model import MCPNet
class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def main():
    batch_size = 1
    train_dir = '../NLST/imagesTrnorm'
    val_dir = '../NLST/imagesValnorm'
    labels_dir = '../NLST/masksTr'
    points_dir= '../NLST/keypointsTr'
    weights = [1, 1] # loss weights
    save_dir = '{}_norm{}_diffusion_un/'.format(weights[0], weights[1])
    save_dir2 = '{}_norm{}_diffusion_un/'.format(weights[0], weights[1])
    if not os.path.exists('experiments/'+save_dir):
        os.makedirs('experiments/'+save_dir)
    if not os.path.exists('experiments/'+save_dir2):
        os.makedirs('experiments/'+save_dir2)
    if not os.path.exists('logs/'+save_dir):
        os.makedirs('logs/'+save_dir)
    sys.stdout = Logger('logs/'+save_dir)
    lr = 0.0001 # learning rate
    epoch_start = 0
    max_epoch = 500 #max traning epoch
    cont_training = False #if continue training

    '''
    Initialize model
    '''
    imgshape = (224, 192, 224)
    model = MCPNet(imgshape)
    model.cuda()
    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model((224, 192, 224), 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model((224, 192, 224), 'bilinear')
    reg_model_bilin.cuda()

    '''
    If continue from previous training
    '''
    if cont_training:
        model_dir = 'experiments/'+save_dir2
        epoch_start = torch.load(model_dir + natsorted(os.listdir(model_dir))[1],weights_only=False)['epoch']
        print("strat:{}".format(epoch_start))
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[1],weights_only=False)['state_dict']
        print('Model: {} loaded!'.format(natsorted(os.listdir(model_dir))[1]))
        model.load_state_dict(best_model)
    else:
        updated_lr = lr

    '''
    Initialize training
    '''
    #train_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16)), ])

    #val_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
    train_set = datasets.ABDataset(train_dir)
    val_set = datasets.ABtestDataset(val_dir,labels_dir,points_dir)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    criterion_mi = losses.MutualInformation()
    criterion_reg = losses.Grad3d(penalty='l2')
    stoptranin=False
    writer = SummaryWriter(log_dir='logs/'+save_dir)
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.cuda() for t in data]
            x = data[0].cuda()
            y = data[1].cuda()



            #x_in = torch.cat((x,y), dim=1)
            output, flow = model(x,y)
            loss_ncc = criterion_mi(output, y) * weights[0]
            loss_reg = criterion_reg(flow,y) * weights[1]
            loss = loss_ncc + loss_reg
            loss_all.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del  loss

            #y_in = torch.cat((y, x), dim=1)
            output, flow = model(y, x)
            loss_ncc = criterion_mi(output, x) * weights[0]
            loss_reg = criterion_reg(flow,y) * weights[1]
            loss = loss_ncc  + loss_reg
            loss_all.update(loss.item(), x.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f},  Reg: {:.6f}'.format(idx, len(train_loader),
                                                                                                loss.item(),
                                                                                                loss_ncc.item(),
                                                                                                loss_reg.item()))
            if torch.isnan(loss_reg):
                stoptranin=True 
                break
        if(stoptranin==True):
            break    

        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))









def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)





if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 1
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    torch.manual_seed(0)
    main()