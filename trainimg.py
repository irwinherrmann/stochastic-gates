'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import collections
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

import model_select

import math
from visdom import Visdom
import numpy as np

DEFAULT_DIR = '/media/ministorage'


try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

# Training settings
parser = argparse.ArgumentParser(description='PyTorch ImageNet Example')
parser.add_argument('--dir', default=DEFAULT_DIR)
parser.add_argument('--model', default='res')

parser.add_argument('--expname', default='give_me_a_name', type=str, metavar='n',
                    help='name of experiment (default: test')
parser.add_argument('--batchrate', type=bool, default=False)
parser.add_argument('--constquad', type=bool, default=False)

parser.add_argument('--timing', default=False)
parser.add_argument('--data', default='',
                    help='path to dataset')
parser.add_argument('--pretrained', default='', type=str,
                    help='path to pretrained checkpoint (default: none)')


parser.add_argument('--batchsize', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--lrdecay', default=30, type=int,
                    help='epochs to decay lr')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='number of start epoch (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--lrfact', default=1, type=float,
                    help='learning rate factor')
parser.add_argument('--lossfact', default=1, type=float,
                    help='loss factor')
parser.add_argument('--target', default=0.7, type=float, help='target rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='folder path to save checkpoint (default: none)')
parser.add_argument('--test', dest='test', action='store_true',
                    help='To only run inference on test set')
parser.add_argument('--visdom', dest='visdom', action='store_true',
                    help='Use visdom to track and plot')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    help='print frequency (default: 10)')


parser.set_defaults(test=False)
parser.set_defaults(visdom=False)

best_prec1 = 0

def round_down(num,digits):
    factor = 10.0 ** digits
    return math.floor(num * factor) / factor


def main():
    global args, best_prec1, model
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    torch.cuda.manual_seed(args.seed)

    directory = "runs/%s/"%(args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)

    model_module, model, pretrained_end = model_select.get_imgnet(args.model)

    # either a pretrained or a resume for imagenet. nothing from scratch
    if args.pretrained == '':
        args.pretrained = os.path.join(args.dir, pretrained_end)
    if args.data == '':
        args.data = os.path.join(args.dir, 'imagenet/raw-data')

    # optionally initialize from pretrained
    if args.pretrained and not args.resume:
        latest_checkpoint = args.pretrained
        if os.path.isfile(latest_checkpoint):
            print("=> loading checkpoint '{}'".format(latest_checkpoint))
            checkpoint = torch.load(latest_checkpoint)
            state = model.state_dict()
            loaded_state_dict = checkpoint
            for k in loaded_state_dict:
                if k in state:
                    state[k] = loaded_state_dict[k]
                else:
                    if 'fc' in k:
                        state[k.replace('fc', 'linear')] = loaded_state_dict[k]
                    if 'downsample' in k:
                        state[k.replace('downsample', 'shortcut')] = loaded_state_dict[k]
            model.load_state_dict(state) 
            print("=> loaded checkpoint '{}'".format(latest_checkpoint))
        else:
            print("=> no checkpoint found at '{}'".format(latest_checkpoint))
    

    model = torch.nn.DataParallel(model).cuda()

    # ImageNet Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
#    valdir = os.path.join(args.data, 'val1000')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    print('Start to load train...')
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=args.batchsize, shuffle=True,
        num_workers=10, pin_memory=True)
    print('Done loading train...')

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batchsize, shuffle=False,
        num_workers=10, pin_memory=True)
    print('Loaded val...')


    # optionally resume from a checkpoint
    if args.resume:
        latest_checkpoint = args.resume
        if os.path.isfile(latest_checkpoint):
            print("=> loading checkpoint '{}'".format(latest_checkpoint))
            checkpoint = torch.load(latest_checkpoint)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD([{'params': [param for name, param in model.named_parameters() if 'fc' in name],
                            'lr': args.lrfact * args.lr, 'weight_decay': args.weight_decay},
                            {'params': [param for name, param in model.named_parameters() if 'fc' not in name],
                            'lr': args.lr, 'weight_decay': args.weight_decay}
                            ], momentum=args.momentum)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    if args.test:
        test_acc = validate(val_loader, model, criterion, 60, target_rates)
        sys.exit()

    start_epoch = args.start_epoch
    end_epoch = args.epochs
    for epoch in range(start_epoch, end_epoch):
        adjust_learning_rate(optimizer, epoch)

        tr = args.target
        target_rates = collections.defaultdict(lambda : tr)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, target_rates)

        # evaluate on validation set
        prec1 = validate(model_module, val_loader, model, criterion, epoch, target_rates)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)

    print('Best accuracy: ', best_prec1)

#    vis.save([args.expname]) 

def train(train_loader, model, criterion, optimizer, epoch, target_rates):
    """Train for one epoch on the training set"""
    losses = AverageMeter()
    losses_c = AverageMeter()
    losses_t = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    activations = AverageMeter()

    
    if args.timing:
        batch_time = AverageMeter()
        batch_time_data = AverageMeter()
        batch_time_model = AverageMeter()
        batch_time_accum = AverageMeter()
        batch_time_step = AverageMeter()


    # Temperature of Gumble Softmax 
    # We simply keep it fixed
    temp = 1

    # switch to train mode
    model.train()

    end_total = time.time()

    for i, (input, target) in enumerate(train_loader):

        if args.timing:
            end = time.time()

        target = target.cuda(async=True)
        input = input.cuda()

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        if args.timing:
            time_data = time.time() - end
            end = time.time()

        # compute output
        output, activation_rates = model(input_var, temperature=temp)


        if args.timing:
            time_model = time.time() - end
            end = time.time()

        # classification loss
        loss_classify = criterion(output, target_var)

        # target rate loss
        acts = 0
        acts_plot = 0
        acts_sum = 0
        gate_acts = []
        for j, act in enumerate(activation_rates):
            if target_rates[j] < 1:
                tr = target_rates[j]

                acts_plot += torch.mean(act)
                if args.constquad:
                    act_mean = torch.mean(act)
                    acts += act_mean.gt(tr).type(torch.cuda.FloatTensor) * torch.pow(tr - act_mean, 2)
                else:
                    acts += torch.pow(tr - torch.mean(act), 2)
                acts_sum += torch.sum(act)
                gate_acts.append(torch.mean(act).data[0])                
            else:
                acts_plot += 1
        
        # batch rate
        batch_mean = acts_sum / (len(activation_rates) * activation_rates[0].shape[0])
        if args.constquad:
            tr = target_rates[0]
            batch_acts = batch_mean.gt(tr).type(torch.cuda.FloatTensor) * len(activation_rates) * torch.pow(tr - batch_mean, 2)
        else:
            tr = target_rates[0]
            batch_acts = len(activation_rates) * torch.pow(tr - batch_mean, 2)

        if args.batchrate:
            acts = batch_acts

        # this is important when using data DataParallel
        acts_plot = torch.mean(acts_plot / len(activation_rates))
        acts = torch.mean(acts / len(activation_rates))

        act_loss = args.lossfact * acts
        loss = loss_classify + act_loss

        # The following comment from AIG's code.
        # Sometimes this value is nan 
        # If someone can find out why, please add a pull request
        # For now, we skip the batch and move on
        if math.isnan(acts_plot.data[0]):
            print(activation_rates)
            optimizer.zero_grad()
            loss.backward()
            continue 

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        losses_c.update(loss_classify.data[0], input.size(0))
        losses_t.update(act_loss.data[0], input.size(0))

        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        activations.update(acts_plot.data[0], 1)

        if args.timing:
            time_accum = time.time() - end
            end = time.time()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_time = time.time() - end_total
        end_total = time.time()

        if args.timing:
            # measure elapsed time
            time_step = time.time() - end
            end = time.time()

            # measure elapsed time
            batch_time.update(total_time)
            batch_time_data.update(time_data)
            batch_time_model.update(time_model)
            batch_time_accum.update(time_accum)
            batch_time_step.update(time_step)

        if i % args.print_freq == 0:
            if args.timing:
                print('Epoch: [{0}][{1}/{2}]\t'
    #                  'Time {bt:.3f}\t'
                      'Time t({bt.val:.3f}) d({bt_data.val:.3f}) m({bt_model.val:.3f}) a({bt_accum.val:.3f}) s({bt_step.val:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f}) c({lossc.avg:.4f}) a({lossa.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                      'Activations: {act.val:.3f} ({act.avg:.3f})'.format(
                          epoch, i, len(train_loader),
                          #bt=batch_time,
                          bt=batch_time, bt_data=batch_time_data, bt_model=batch_time_model, bt_accum=batch_time_accum, bt_step=batch_time_step,
                          loss=losses, lossa=losses_t, lossc=losses_c, top1=top1, top5=top5, act=activations))
            else:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {bt:.3f}\t'
#                      'Time t({bt.val:.3f}) d({bt_data.val:.3f}) m({bt_model.val:.3f}) a({bt_accum.val:.3f}) s({bt_step.val:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f}) c({lossc.avg:.4f}) a({lossa.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                      'Activations: {act.val:.3f} ({act.avg:.3f})'.format(
                          epoch, i, len(train_loader),
                          bt=total_time,
#                          bt=batch_time, bt_data=batch_time_data, bt_model=batch_time_model, bt_accum=batch_time_accum, bt_step=batch_time_step,
                          loss=losses, lossa=losses_t, lossc=losses_c, top1=top1, top5=top5, act=activations))
            #print('gate activation rates: ' + str({k:round_down(v,3) for k,v in enumerate(gate_acts)}))


def validate(model_module, val_loader, model, criterion, epoch, target_rates):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    accumulator = model_module.ActivationAccum_img(epoch)
    activations = AverageMeter()

    # Temperature of Gumble Softmax 
    # We simply keep it fixed
    temp = 1

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input = input.cuda()

        target_var = torch.autograd.Variable(target, volatile=True)
        input_var = torch.autograd.Variable(input, volatile=True)

        # compute output
        output, activation_rates = model(input_var, temperature=temp)

        # classification loss
        loss = criterion(output, target_var)

        acts = 0
        for j, act in enumerate(activation_rates):
            if target_rates[j] < 1:
                acts += torch.mean(act)
            else:
                acts += 1
        # this is important when using data DataParallel
        acts = torch.mean(acts / len(activation_rates))
 
        # see above
        if math.isnan(acts.data[0]):
             continue

        # accumulate statistics over eval set
        accumulator.accumulate(activation_rates, target_var, target_rates)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        activations.update(acts.data[0], 1)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'Activations: {act.val:.3f} ({act.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1, top5=top5, act=activations))

    activ_output = accumulator.getoutput()

    print('gate activation rates:')
    print({k:round_down(v,3) for k,v in activ_output[0].items()})

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint_%d.pth.tar'):
    directory = "runs/%s/"%(args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename % state['epoch']
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.expname) + 'model_best.pth.tar')

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, x, y, env=None):
        if env is not None:
            print_env = env
        else:
            print_env = self.env
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=print_env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.updateTrace(X=np.array([x]), Y=np.array([y]), env=print_env, win=self.plots[var_name], name=split_name)
    def plot_heatmap(self, map, epoch):
        self.viz.heatmap(X=map,
                         env=self.env,
                         opts=dict(title='activations {}'.format(epoch),
                                   xlabel='modules',
                                   ylabel='classes'
                                   ))

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

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lrdecay))
    factor = args.lrfact
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
