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
import collections

import copy

import models.params_resnet as params
import models.resnet_filter as resnetfilter
import models.resnet_filter_basicblock as resnetfilter_basic


import math
import numpy as np

DEFAULT_DIR = '/nfs01/data/imagenet-original'

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
parser.add_argument('--constquad', default=False)
parser.add_argument('--usedropout', type=str, default='True')

parser.add_argument('--timing', default=False)
parser.add_argument('--data', default='',
                    help='path to dataset')
parser.add_argument('--pretrained', default='', type=str,
                    help='path to pretrained checkpoint (default: none)')
parser.add_argument('--granularity', type=str, default='low')
parser.add_argument('--dependent_gates', type=str, default='True')
parser.add_argument('--gate-weight-decay', type=float, default=40)

parser.add_argument('--resize', type=int, default=224)
parser.add_argument('--num_workers', type=int, default=15)

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
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    help='print frequency (default: 10)')


parser.set_defaults(test=False)

best_prec1 = 0

def round_down(num,digits):
        factor = 10.0 ** digits
        return math.floor(num * factor) / factor


def main():
    global args, best_prec1, model, weight_per_gate, constant_gflops
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    torch.cuda.manual_seed(args.seed)

    directory = "runs/%s/"%(args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if args.dependent_gates == 'True':
        dependent_gates = True
    elif args.dependent_gates == 'False':
        dependent_gates = False
    else:
        print('Not valid entry for dependent_gates! Explode!')
        assert(False)

    if args.usedropout == 'True':
        use_dropout = True
    elif args.usedropout == 'False':
        use_dropout = False
    else:
      assert(False)

    if args.model == 'res18':
        model_module = resnetfilter_basic
        model = model_module.ResNet18_ImageNet(params.Params_ResnetFilter_Img(args.granularity), dependent_gates=dependent_gates)
        args.pretrained = 'pretrained_imagenet/resnet18.pth'#os.path.join(args.dir, pretrained_end)
    elif args.model == 'res34':
        model_module = resnetfilter_basic
        model = model_module.ResNet34_ImageNet(params.Params_ResnetFilter_Img(args.granularity), dependent_gates=dependent_gates)
        args.pretrained = 'pretrained_imagenet/resnet34.pth'#os.path.join(args.dir, pretrained_end)
    elif args.model == 'res50':
        model_module = resnetfilter
        model = model_module.ResNet50_ImageNet(params.Params_ResnetFilter_Img(args.granularity), dependent_gates=dependent_gates)
        pretrained_end = 'pretrained_imagenet/resnet50.pth'
        args.pretrained = os.path.join(args.dir, pretrained_end)
    elif args.model == 'res101':
        model_module = resnetfilter
        model = model_module.ResNet101_ImageNet(params.Params_ResnetFilter_Img(args.granularity), dependent_gates=dependent_gates)
        pretrained_end = 'pretrained_imagenet/resnet101.pth'
        args.pretrained = os.path.join(args.dir, pretrained_end)
    else:
        print('Could not find model! Explode!')
        exit(1) # explode

    # either a pretrained or a resume for imagenet. nothing from scratch
    if args.data == '':
        args.data = os.path.join(args.dir, 'imagenet/raw-data')

    print('gated_gflops    %10d' % sum(model.get_gate_weights()))
    print('constant_gflops %10d' % model.get_constant_gflops())
    print('Total gflops    %10d' % model.get_possible_gflops())



#### BEGIN CHANGE target rate
    target_rates = collections.defaultdict(lambda : args.target)

#### END CHANGE
    #args.pretrained = '/home/richard/convnet-aig/checkpoints/cleanmobile96.pth.tar'

    # optionally initialize from pretrained
    if args.pretrained and not args.resume:
        latest_checkpoint = args.pretrained
        if os.path.isfile(latest_checkpoint):
            print("=> loading checkpoint '{}'".format(latest_checkpoint))
            # TODO: clean this part up
            checkpoint = torch.load(latest_checkpoint)
            state = model.state_dict()
            loaded_state_dict = checkpoint

            for k in state:
                print k
            print '\n\n'

            #for k in loaded_state_dict:
            #    print k
            #print '\n\n'

            for k in loaded_state_dict:
                if k in state:
                    state[k] = loaded_state_dict[k]
                else:
                    newk = k
                    newk = newk.replace('conv1', 'conv.0')
                    newk = newk.replace('bn1', 'conv.1')
                    newk = newk.replace('conv2', 'conv.2')
                    newk = newk.replace('bn2', 'conv.3')
                    newk = newk.replace('conv3', 'conv.4')
                    newk = newk.replace('bn3', 'conv.5')
                    newk = newk.replace('downsample', 'shortcut')
                    newk = newk.replace('fc', 'linear')
#                    k = k.replace('conv1', 'conv.0')
                    #print k, newk
                    assert(newk in state)
                    state[newk] = loaded_state_dict[k]

#                print '\n\n'
#                for k in state:
#                    print k

            model.load_state_dict(state) 
            print("=> loaded checkpoint '{}'".format(latest_checkpoint))
        else:
            print("=> no checkpoint found at '{}'".format(latest_checkpoint))

    # optionally resume from a checkpoint
    if args.resume:
        model = torch.nn.DataParallel(model).cuda()
        latest_checkpoint = args.resume
        if os.path.isfile(latest_checkpoint):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(latest_checkpoint)
            args.start_epoch = 60#checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            state = model.state_dict()
            loaded_state = checkpoint['state_dict']
            for k in state.keys():
                if k in loaded_state:
                    state[k] = loaded_state[k]
                else:
                    print k
#            print '\n\n____'
#            for k in loaded_state.keys():
#                if k in state:
#                    state[k] = loaded_state[k]
#                if k not in state:
#                    print k
            model.load_state_dict(state, strict=True)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        model = torch.nn.DataParallel(model).cuda()

    # ImageNet Data loading code
    #args.data = '/nfs01/data/imagenet-original/ILSVRC2012_img_'
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    print('Start to load train...')
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(args.resize),
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=args.batchsize, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)
    print('Done loading train...')

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Resize(args.resize),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batchsize, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    print('Loaded val...')

    cudnn.benchmark = True

    gate_weight_decay = args.weight_decay * args.gate_weight_decay / model.module.num_gates
    print 'num_gates:', model.module.num_gates
    print 'gate_weight_decay param:', args.gate_weight_decay
    print 'gate_weight_decay:', gate_weight_decay
    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD([{'params': [param for name, param in model.named_parameters() if 'gate' in name],
                            'lr': args.lrfact * args.lr, 'weight_decay': gate_weight_decay},
                            {'params': [param for name, param in model.named_parameters() if 'gate' not in name],
                            'lr': args.lr, 'weight_decay': args.weight_decay}
                            ], momentum=args.momentum)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    if args.test:
        test_acc = validate(model_module, val_loader, model, criterion, 0, target_rates)
        sys.exit()

    start_epoch = args.start_epoch
    end_epoch = args.epochs

    for epoch in range(start_epoch, end_epoch):
        adjust_learning_rate(optimizer, epoch)

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
    gflops = AverageMeter()
    bms = AverageMeter()
    
    if args.timing:
        batch_time = AverageMeter()
        batch_time_data = AverageMeter()
        batch_time_model = AverageMeter()
        batch_time_accum = AverageMeter()
        batch_time_back = AverageMeter()
        batch_time_del = AverageMeter()


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
        output, gflops_plot, batch_mean = model(input_var, temperature=temp)
#        print(activation_rates)

#        z = make_dot(output.mean(), params=dict(model.named_parameters()))
#        with open('tmp.dot', 'w') as f:
#            f.write(str(z))

        if args.timing:
            time_model = time.time() - end
            end = time.time()

        # classification loss
        loss_classify = criterion(output, target_var)

        if args.constquad:
            tr = target_rates[0]
            acts = batch_mean.gt(tr).type(torch.cuda.FloatTensor) * torch.pow(tr - batch_mean, 2)
        else:
            tr = target_rates[0]
            acts = torch.pow(tr - batch_mean, 2)
        
#### END CHANGE

        # this is important when using data DataParallel
        gflops_plot = torch.mean(gflops_plot)
        batch_mean = torch.mean(batch_mean)
        acts = torch.mean(acts)

#        print('acts', acts)

        act_loss = args.lossfact * acts
        loss = loss_classify + act_loss

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        losses_c.update(loss_classify.data[0], input.size(0))
        losses_t.update(act_loss.data[0], input.size(0))

        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        gflops.update(gflops_plot.data[0]/1e6, 1)
        bms.update(batch_mean.data[0], 1)

        if args.timing:
            time_accum = time.time() - end
            end = time.time()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.timing:
            time_back = time.time() - end
            end = time.time()

        del loss
        del loss_classify
        del output
        del act_loss
        del acts
        del batch_mean

        total_time = time.time() - end_total
        end_total = time.time()


        if args.timing:
            # measure elapsed time
            time_del = time.time() - end
            end = time.time()

            batch_time.update(total_time)
            batch_time_data.update(time_data)
            batch_time_model.update(time_model)
            batch_time_accum.update(time_accum)
            batch_time_back.update(time_back)
            batch_time_del.update(time_del)

        if i % args.print_freq == 0:
            if args.timing:
                print('Epoch: [{0}][{1}/{2}]\t'
    #                  'Time {bt:.3f}\t'
                      'Time t({bt.val:.3f}) data({bt_data.val:.3f}) model({bt_model.val:.3f}) accum({bt_accum.val:.3f}) back({bt_back.val:.3f}) del({bt_del.val:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f}) c({lossc.avg:.4f}) a({lossa.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                      'Gflops: {act.val:.3f} ({act.avg:.3f})\t'
                      'BM: {bm.val:.3f} ({bm.avg:.3f})'.format(
                          epoch, i, len(train_loader),
                          #bt=batch_time,
                          bt=batch_time, bt_data=batch_time_data, bt_model=batch_time_model, bt_accum=batch_time_accum, bt_back=batch_time_back, bt_del=batch_time_del,
                          loss=losses, lossa=losses_t, lossc=losses_c, top1=top1, top5=top5, act=gflops, bm=bms))
            else:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {bt:.3f}\t'
#                      'Time t({bt.val:.3f}) d({bt_data.val:.3f}) m({bt_model.val:.3f}) a({bt_accum.val:.3f}) s({bt_step.val:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f}) c({lossc.avg:.4f}) a({lossa.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                      'Gflops: {act.val:.3f} ({act.avg:.3f})\t'
                      'BM: {bm.val:.3f} ({bm.avg:.3f})'.format(
                          epoch, i, len(train_loader),
                          bt=total_time,
#                          bt=batch_time, bt_data=batch_time_data, bt_model=batch_time_model, bt_accum=batch_time_accum, bt_step=batch_time_step,
                          loss=losses, lossa=losses_t, lossc=losses_c, top1=top1, top5=top5, act=gflops, bm=bms))
            #print('gate activation rates: ' + str({k:round_down(v,3) for k,v in enumerate(gate_acts)}))
            #break


def validate(model_module, val_loader, model, criterion, epoch, target_rates):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    gflops_accumulator = AverageMeter()
    bm_accumulator = AverageMeter()
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

        target_var = torch.autograd.Variable(target, volatile=True)
        input_var = torch.autograd.Variable(input, volatile=True)

        # compute output
        output, gflops_plot, bm = model(input_var, temperature=temp)
        activation_rate = model.module.layer_activation_rates

        # classification loss
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        bm_accumulator.update(bm.data[0], input.size(0))
        gflops_accumulator.update(gflops_plot.data[0]/1e6, input.size(0))
        acts_accumulator.update(activation_rate)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i+1 == len(val_loader):
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'Gflops: {act.val:.3f} ({act.avg:.3f})\t'
                  'BM: {bm.val:.3f} ({bm.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1, top5=top5, act=gflops_accumulator, bm=bm_accumulator))

    print('Activations {')
    for k,v in acts_accumulator.avg.iteritems():
        print k, ':' , str(v)+','
    print('}')
    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = "runs/%s/"%(args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename# % state['epoch']
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.expname) + 'model_best.pth.tar')
    if state['epoch'] % 10 == 0 or state['epoch'] == 75:
        shutil.copyfile(filename, 'runs/%s/'%(args.expname) + 'checkpoint_%d.pth.tar' % state['epoch'])

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
#                    print vk, self.sum[k]
                    self.sum[k][vk] += vv
                    self.avg[k][vk] = self.sum[k][vk] / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lrdecay))
    factor = args.lrfact
    print('learning_rate', lr, 'epoch', epoch)
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
    print(' '.join(sys.argv))
    main()
