'''Train CIFAR10 with PyTorch.'''
import argparse
import os
import sys
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import copy

import os
import argparse

import models.resnet_filter as resnet_filter

import models.params_resnet as params_resnet
import collections

from visdom import Visdom
import numpy as np

import cls

CIFAR_PATH = '/home/charles/data'

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
parser.add_argument('--expname', default='give_me_a_name_cifar', type=str,
                    help='name of experiment')
parser.add_argument('--granularity', default='low', type=str)
parser.add_argument('--dependent', default='True', type=str)
parser.add_argument('--gate-weight-decay', default=10, type=float,
                    help='weight decay numerator for gates')
parser.add_argument('--constquad', dest='constquad', action='store_true')

parser.add_argument('--lrfact', default=1, type=float,
                    help='learning rate factor')
parser.add_argument('--lossfact', default=2, type=float,
                    help='loss factor')
parser.add_argument('--target', default=0.6, type=float, help='target rate')
parser.add_argument('--batchsize', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='number of start epoch (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--test', dest='test', action='store_true',
                    help='To only run inference on test set')
parser.add_argument('--visdom', dest='visdom', action='store_true',
                    help='Use visdom to track and plot')
parser.add_argument('--print-freq', '-p', default=25, type=int,
                    help='print frequency (default: 10)')
parser.set_defaults(test=False)
parser.set_defaults(visdom=False)

best_prec1 = 0

def main():
    global args, best_prec1, model
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    torch.cuda.manual_seed(args.seed)
    
    # Data loading code
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if args.dependent == 'True':
        dependent_gates = True
    elif args.dependent == 'False':
        dependent_gates = False
    else:
        assert(False)


    RESUME_CKPT = '/home/charles/dev/convnet-aig/checkpoints/cifar_cleanres111.pth.tar'
    model = resnet_filter.ResNet110_cifar_111(params_resnet.Params_ResnetFilter_Cifar(args.granularity), dependent_gates=dependent_gates)

    args.resume = RESUME_CKPT

    # optionally resume from a checkpoint
    if args.resume:
        latest_checkpoint = args.resume
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
#            args.start_epoch = checkpoint['epoch']
            args.start_epoch = 0
            best_prec1 = 0#checkpoint['best_prec1']
            state = model.state_dict()
            loaded_state_dict = checkpoint['state_dict']
            for k in state:
                if 'gate' not in k:
                    print k               
            print '\n\n'

            for k in loaded_state_dict:
                if k in state:
                    newk = k
                else:
                    newk = k
                    print 'here', k
                    if newk == 'conv1.weight':
                        newk = 'conv1.block.0.weight'
                    else:
                        newk = newk.replace('conv1', 'conv.0')
                    newk = newk.replace('bn1', 'conv.1')
                    newk = newk.replace('conv2', 'conv.2')
                    newk = newk.replace('bn2', 'conv.3')
                    newk = newk.replace('module.', '')
                if 'gate' in k:
                    print k, loaded_state_dict[k]
                print newk
                assert(newk in state)

                state[newk] = loaded_state_dict[k]
            model.load_state_dict(state, strict=True)
            print("=> loaded checkpoint '{}'")
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    USE_JUNK_LABEL = False
    valid_labels = (0, 1)


    kwargs = {'num_workers': 2, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(CIFAR_PATH, train=True, download=True, transform=transform_train),
        batch_size=args.batchsize, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(CIFAR_PATH, train=False, transform=transform_test),
        batch_size=args.batchsize, shuffle=False, **kwargs)
    print('gflops', model.possible_gflops)    

#    model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

#    for name, param in model.state_dict().items():
#        print(name)
    cudnn.benchmark = True


    num_gates = model.num_gates_total
    num_gates_per_layer = model.num_gates_per_layer
    print 'number of gates:', num_gates
    print 'number of gates per layer:', num_gates_per_layer

    param_settings = []

    gate_weight_decay = args.gate_weight_decay / num_gates * args.weight_decay
    #gate_weight_decay = 0
    param_dict = {'params': [param for name, param in model.named_parameters() if 'gate' in name], 'lr': args.lr, 'weight_decay': gate_weight_decay}
    param_settings.append(param_dict)
    param_dict = {'params': [param for name, param in model.named_parameters() if 'gate' not in name], 'lr': args.lr, 'weight_decay': args.weight_decay}
    param_settings.append(param_dict)

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(param_settings, momentum=args.momentum)


    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    if args.test:
        test_acc = validate(val_loader, model, criterion, 350)
        sys.exit()


    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)
    print('Best accuracy: ', best_prec1)

def train(train_loader, model, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_c = AverageMeter()
    losses_t = AverageMeter()
    top1 = AverageMeter()
    activations = AverageMeter()
    gflops_ave = AverageMeter()

    temp = 1

    # switch to train mode
    model.train()

    end = time.time()
    
    ttt = torch.FloatTensor(33).fill_(0)
    ttt = ttt.cuda()
    ttt = torch.autograd.Variable(ttt, requires_grad=False)
    prev_batch_mean = args.target
    tr = args.target
 
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output, gflops, bm = model(input_var, temperature=temp)

        # classification loss
        loss_classify = criterion(output, target_var)


#        tr = prev_batch_mean

        # target rate loss - gives tr to each gate to distribute among inputs
        #print(len(activation_rates), activation_rates[0].shape)
        if args.constquad:
            batch_acts = bm.gt(tr).type(torch.cuda.FloatTensor) * torch.pow(tr - bm, 2)
        else:
            batch_acts = torch.pow(tr - bm, 2)
        acts = torch.mean(batch_acts)

        act_loss = args.lossfact * acts
        loss = loss_classify  + act_loss

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data[0], input.size(0))
        losses_c.update(loss_classify.data[0], input.size(0))
        losses_t.update(act_loss.data[0], input.size(0))
        
        top1.update(prec1[0], input.size(0))
        activations.update(float(bm.data[0]), 1)
        gflops_ave.update(gflops.data[0]/float(1e6), 1)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i+1 == len(train_loader):
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f}) c({lossc.avg:.4f}) t({losst.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Gflops: {gf.val:.3f} ({gf.avg:.3f})\t'
                  'BM: {act.val:.3f} ({act.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses, lossc=losses_c, losst=losses_t, top1=top1, gf=gflops_ave, act=activations))



def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    accumulator = resnet_filter.ActivationAccum(epoch=epoch)
    activations = AverageMeter()
    gflops_ave = AverageMeter()
    acts_accumulator = DictAverageMeter() 
    
    # Temperature of Gumble Softmax 
    # We simply keep it fixed
    temp = 1

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input = input.cuda()

        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output, gflops, bm = model(input_var, temperature=temp)
        activation_rate = model.layer_activation_rates

        # classification loss
        loss = criterion(output, target_var)

        # target rate loss
        acts = torch.mean(bm)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        activations.update(float(bm.data[0]), 1)
        gflops_ave.update(gflops.data[0]/float(1e6), 1)
        acts_accumulator.update(activation_rate)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i+1 == len(val_loader):
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Gflops: {gf.val:.3f} ({gf.avg:.3f})\t'
                  'Activations: {act.val:.3f} ({act.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1, gf=gflops_ave, act=activations))
    print('Activations {')
    for k,v in acts_accumulator.avg.iteritems():
        print k, ':' , str(v)+','
    print('}')
    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    #state = model.state_dict()
    #for k, v in state.iteritems():
        #if 'gate' in k:
            #print k, v

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.expname) + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class DictAverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = collections.OrderedDict()
        self.sum = collections.OrderedDict()
        self.count = 0

    def update(self, val, n=1):
        self.count += 1
        if len(self.sum) == 0:
            self.sum = copy.deepcopy(val)
        else:
            self.avg = collections.OrderedDict()
            for k, v in val.iteritems():
                self.avg[k] = [0, 0, 0, 0]
                for vk, vv in enumerate(v):
                    #print vk, self.sum[k]
                    self.sum[k][vk] += vv
                    self.avg[k][vk] = self.sum[k][vk] / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""

#### BEGIN CHANGE
    lr = args.lr
    factor = args.lrfact
#    if epoch >= 150:
    if epoch >= 40:
        lr = 0.1 * lr
#    if epoch >= 250:
    if epoch >= 70:
        lr = 0.1 * lr
#    lr = base_lr + base_height * last_batch_iteration
#### END CHANGE    
 

    optimizer.param_groups[0]['lr'] = factor * lr
    optimizer.param_groups[1]['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    print(sys.argv)
    main()
