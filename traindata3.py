from torch.utils.tensorboard import SummaryWriter
import os, utils, glob, losses
import sys
from torch.utils.data import DataLoader
from data import datasets2, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from Models.modelmcp import Mymodel
import gc
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
    weights = [1, 1, 1] # loss weightsa
    save_dir = 'mcpnetMIND_test{}_dsc{}_diffusion_{}un/'.format(weights[0], weights[1], weights[2])
    if not os.path.exists('experiments/'+save_dir):
        os.makedirs('experiments/'+save_dir)
    if not os.path.exists('logs/'+save_dir):
        os.makedirs('logs/'+save_dir)
    sys.stdout = Logger('logs/'+save_dir)
    lr = 0.0001 # learning rate
    epoch_start = 0
    max_epoch = 40 #max traning epoch
    cont_training = False#if continue training

    '''
    Initialize model
    '''

    imgshape = (192, 160, 192)
    head_dim = 6
    num_heads = [8,8,8,8,1]
    model = Mymodel(imgshape, head_dim=head_dim, num_heads=num_heads, scale=1)
    model.cuda()
    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model((192, 160, 192), 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model((192, 160, 192), 'bilinear')
    reg_model_bilin.cuda()

    '''
    If continue from previous training
    '''
    if cont_training:
        model_dir = 'experiments/'+save_dir
        epoch_start = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1],weights_only=False)['epoch']
        print("strat:{}".format(epoch_start))
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1],weights_only=False)['state_dict']
        print('Model: {} loaded!'.format(natsorted(os.listdir(model_dir))[-1],weights_only=False))
        model.load_state_dict(best_model)
    else:
        updated_lr = lr

    '''
    Initialize training
    '''
    #train_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16)), ])

    #val_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
    train_set = datasets2.ABDataset(train_base_dir='../train', transform=None)

    test_set = datasets2.ABtestDataset(test_base_dir='../test', transform=None)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    #val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    #criterion_ncc = losses.NCC_vxm()
    criterion_mind = losses.MIND_loss().cuda()
#    criterion_dsc = losses.DiceLoss()
    #criterion_regu = losses.Regu_loss
    criterion_reg = losses.Grad3d(penalty='l2')
    best_dsc = 0
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
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]
            output, flow= model(x,y)

            loss_ncc = criterion_mind(output, y) * weights[0]
            #loss_ncc_affine = criterion_ncc(affine, y) * weights[0]
            loss_reg = criterion_reg(flow, y) * weights[2]
            loss = loss_ncc + loss_reg
            loss_all.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader),
                                                                                                loss.item(),
                                                                                                loss_ncc.item(),
                                                                                                loss_reg.item()
                                                                                                ))
            if(idx%400==0):
                writer.add_scalar('Loss/train', loss_all.avg, epoch)
                print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
                '''
                Validation
                '''
                eval_dsc = utils.AverageMeter()
                with torch.no_grad():
                    for data in test_loader:
                        model.eval()
                        data = [t.cuda() for t in data]
                        x = data[0]
                        y = data[1]
                        x_seg = data[2]
                        y_seg = data[3]
                        #grid_img = mk_grid_img(8, 1, (160, 192, 224))
                        output, flow= model(x,y)
                        def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
                        #def_grid = reg_model_bilin([grid_img.float(), flow.cuda()])
                        dsc = utils.dice_val_VOI(def_out.long(), y_seg.long())
                        dscoral = utils.dice_val_VOI(x_seg.long(), y_seg.long())
                        eval_dsc.update(dsc.item(), x.size(0))
                        print("before{}_after{}".format(dscoral,dsc))
                        del def_out, output, flow, x, y, x_seg, y_seg
                best_dsc = max(eval_dsc.avg, best_dsc)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_dsc': best_dsc,
                    'optimizer': optimizer.state_dict(),
                }, save_dir='experiments/'+save_dir, filename='dsc{:.4f}.pth.tar'.format(eval_dsc.avg))
                writer.add_scalar('DSC/validate', eval_dsc.avg, epoch)
                print('After_all{}'.format(eval_dsc.avg))
                loss_all.reset()
                gc.collect()               
                torch.cuda.empty_cache()   

        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))

    writer.close()

def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

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
    torch.manual_seed(3407)
    main()