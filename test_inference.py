'''
Evaluate the checkpoints with various inference strategies.

Note that the script currently uses the filename of the checkpoint to identify which model we're testing.
data-independent model should include the term "un" in its name.
res50 should include the term "res" in its name.
res101 should include the term "res101" in its name.
'''
from __future__ import print_function

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
import math
import collections

import os
import argparse

import models.test_models.test_convnet as testres
import models.test_models.test_unattach_convnet as testunres

from visdom import Visdom
import numpy as np
import gate_utils


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def round_down(num,digits):
    factor = 10.0 ** digits
    return math.floor(num * factor) / factor


parser = argparse.ArgumentParser(description='PyTorch ImageNet Evaluator')

# required arguments
parser.add_argument('--resume', required=True, type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--dir', required=True, type=str,
                    help='path to ImageNet directory containing the folders "train" and "val"')


# arguments for testing inference techniques
parser.add_argument('--test_batchsize', type=int, default=120, metavar='N',
                    help='input batch size for testing (default: 150)')
parser.add_argument('--num_seeds', type=int, default=5,
                    help='number of times to repeat the stochastic experiment')


# used for resetting the batchnorm, can skip if you're just evaluating the given checkpoints
parser.add_argument('--resetbatchnorm', default=False, type=boolean_string,
                    help='reset batchnorm')
parser.add_argument('--train_batchsize', type=int, default=90, metavar='N',
                    help='input batch size for reseting batch norm (default: 95)')
parser.add_argument('--num_trained', default=210, type=int,
                    help='Num batches to train for lr=0. This resets BN.')
parser.add_argument('--ckptname', default='model_final.pth.tar', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=0, type=float,
                    help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')


args = parser.parse_args()

start_epoch=0
best_prec1 = 0

def main():
    global args, best_prec1, start_epoch
    args = parser.parse_args()
    print('args.resume %s' % args.resume)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    if 'res' not in args.resume:
        print('Did not recognize model name')
        exit(1)

    is_independent = 'un' in args.resume
    is_res101 = 'res101' in args.resume

    if is_independent:
        model_module = testunres
    else:
        model_module = testres

    if is_res101:
        model = model_module.ResNet101_ImageNet()
    else:
        model = model_module.ResNet50_ImageNet()

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

    kwargs = {'num_workers': 2, 'pin_memory': True}

    model = torch.nn.DataParallel(model).cuda() # DataParallel before loading
    latest_checkpoint = args.resume
    if os.path.isfile(args.resume):
        checkpoint = torch.load(latest_checkpoint)
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(latest_checkpoint))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        exit(1)


#    with open('multitest_afterload.txt', 'w') as F:
#        for name, param in model.state_dict().items():
#            if 'bn' in name:
#                F.write('%s, %s\n' % (str(name), str(param)))

#    for module in model.modules():
#        if isinstance(module, torch.nn.modules.BatchNorm1d):
#            module.eval()
#        if isinstance(module, torch.nn.modules.BatchNorm2d):
#            module.eval()
#        if isinstance(module, torch.nn.modules.BatchNorm3d):
#            module.eval() 

 
    cudnn.benchmark = True
    LR = 0
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD([{'params': [param for name, param in model.named_parameters() if 'fc' in name],
                            'lr': 0, 'weight_decay': 0},
                            {'params': [param for name, param in model.named_parameters() if 'fc' not in name],
                            'lr': 0, 'weight_decay': 0}
                            ], momentum=args.momentum)

    target_rates = collections.defaultdict(lambda : 0) # the value doesn't matter.


    # We printed out the batchnorms number before this and after this and they change.
    # We make sure that LR = 0 and weight_decay = 0 in order to not change the model.
    # Make sure you run this on 1 GPU.
    # This strongly affects the test time performance.

    # Here are links of people who seem to be observing the same problem.
    # https://discuss.pytorch.org/t/model-eval-gives-incorrect-loss-for-model-with-batchnorm-layers/7561/30
    # https://discuss.pytorch.org/t/performance-highly-degraded-when-eval-is-activated-in-the-test-phase/3323/15
    # We will change to groupnorm in the future.


    if args.resetbatchnorm:
        traindir = os.path.join(args.dir, 'train')
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])),
            batch_size=args.train_batchsize, shuffle=True,
            num_workers=10, pin_memory=True)

        print('Using training with lr=0 to reset the batchnorm values...')
        train(train_loader, model, criterion, optimizer, 0, target_rates, args.num_trained)

        dirname = os.path.split(args.resume)[0]

        save_checkpoint({
            'epoch': 100,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, dirname, args.ckptname)

        print('Done resetting batchnorms...')


    valdir = os.path.join(args.dir, 'val')
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batchsize, shuffle=False,
        num_workers=10, pin_memory=True)


    print('Start testing runs...')
    validate(model_module, val_loader, model, criterion, start_epoch, long_form=True)
#    with open('multitest_aftervalidate.txt', 'w') as F:
#        for name, param in model.state_dict().items():
#            if 'bn' in name:
#                F.write('%s, %s\n' % (str(name), str(param)))


def save_checkpoint(state, dirname, filename='model_final.pth.tar'):
    filename = os.path.join(dirname, filename)
    torch.save(state, filename)


def train(train_loader, model, criterion, optimizer, epoch, target_rates, num_trained):
    """Train for one epoch on the training set"""
    losses = AverageMeter()
    losses_c = AverageMeter()
    losses_t = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    activations = AverageMeter()
    

    # Temperature of Gumble Softmax 
    # We simply keep it fixed
    temp = 1

    # switch to train mode
    model.train()

    end_total = time.time()

    for i, (input, target) in enumerate(train_loader):

        target = target.cuda(async=True)
        input = input.cuda()

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
#        output = model(input_var, temperature=temp)
        output, activation_rates = model(input_var, temperature=temp)
#        activation_rates = [torch.autograd.Variable(torch.cuda.FloatTensor([1.0,1.0]))]


        # classification loss
        loss_classify = criterion(output, target_var)

        # target rate loss
        acts = 0
        acts_plot = 0
        acts_sum = 0
        gate_acts = []
        for j, act in enumerate(activation_rates):
            if target_rates[j] < 1:
                acts_plot += torch.mean(act)
                acts_sum += torch.sum(act)
                gate_acts.append(torch.mean(act).data[0])                
            else:
                acts_plot += 1
        
        # batch rate
        batch_mean = acts_sum / (len(activation_rates) * activation_rates[0].shape[0])
        tr = target_rates[0]
        batch_acts = len(activation_rates) * torch.pow(tr - batch_mean, 2)
        

#### BEGIN CHANGE batch rate
        acts = batch_acts

#### END CHANGE

        # this is important when using data DataParallel
        acts_plot = torch.mean(acts_plot / len(activation_rates))
        acts = torch.mean(acts / len(activation_rates))

        act_loss = acts
        loss = loss_classify + act_loss

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

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_time = time.time() - end_total
        end_total = time.time()

        if i % 50 == 0 or i+1 == min(len(train_loader), num_trained):
            print('\tEpoch: [{0}][{1}/{2}]\t'
                  'Time {bt:.3f}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f}) c({lossc.avg:.4f}) a({lossa.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'Activations: {act.val:.3f} ({act.avg:.3f})'.format(
                      epoch, i, min(len(train_loader), num_trained),
                      bt=total_time,
                      loss=losses, lossa=losses_t, lossc=losses_c,
                      top1=top1, top5=top5, act=activations))
        if i > num_trained:
            break

def validate_one(model_module, val_loader, model, seed, gate_mode, precision_map, pred_map, prob=1, threshold=.5):
    top1 = AverageMeter()
    top5 = AverageMeter()
    activations = AverageMeter()

    accumulator = model_module.ActivationAccum_img(0)

    target_rates = collections.defaultdict(lambda : .0)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    pred_map[seed] = [[], []]
    gate_activations = []
    targets = []
    acts_nan_counter = 0
    num_correct = 0
    num_total = 0
    num_acts = 0
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        output, activation_rates, w1bns_list = model(input_var, gate_mode = gate_mode, prob = prob, threshold = threshold)


        # compute output
        preds = {}
        label = {}

        _, pred1 = output.topk(1, 1, True, True)
        pred_map[seed][0].append(pred1)

        _, pred5 = output.topk(5, 1, True, True)
        pred_map[seed][1].append(pred5)

        targets.append(target)

        # target rate loss
        acts = 0
        for act in activation_rates:
            acts += torch.mean(act)

 
        # this is important when using data DataParallel
        acts = torch.mean(acts / len(activation_rates))

        if math.isnan(acts.data[0]):
            acts_nan_counter += 1

        accumulator.accumulate(activation_rates, target_var, target_rates)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        num_acts += acts * target.shape[0]
        num_correct += prec1 * target.shape[0] / 100
        num_total += target.shape[0]
 
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        activations.update(acts.data[0], 1)

        if i % 20 == 0 or i + 1 == len(val_loader):
            print('\tEpoch: [{0}/{1}]\t\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'Activations: {act.val:.3f} ({act.avg:.3f})'.format(
                      i, len(val_loader)-1, # need to subtract one since we zero index
                      top1=top1, top5=top5, act=activations))

    activ_output = accumulator.getoutput()
    print('gate activation rates:')
    print({k:round_down(v,4) for k,v in activ_output[0].items()})

    precision_map[seed] = [float(top1.avg), float(top5.avg)]
    return num_correct, num_total, num_acts, acts_nan_counter, targets, activ_output

def validate(module_model, val_loader, model, criterion, epoch, prob=1, long_form=True):
    """Perform validation on the validation set"""    
    # Temperature of Gumble Softmax 
    # We simply keep it fixed
    temp = 1

    # switch to evaluate mode
    model.eval()

#    with open('multitest_aftercalltoeval.txt', 'w') as F:
#        for name, param in model.state_dict().items():
#            if 'bn' in name:
#                F.write('%s, %s\n' % (str(name), str(param)))


    from scipy import stats

    if False:
        precision_map = {}
        pred_map = {}
        ensemble_acc = []
        outputs = []

        prob = .8
        seeds = range(args.num_seeds)
        acts = []
        gate_mode = 'stochastic-variable'
        print('Begin %s run...' % gate_mode)
        for seed in seeds:
            num_correct, num_total, num_acts, acts_nan_counter, targets, activ_output = validate_one(module_model, val_loader, model, seed, gate_mode, precision_map, pred_map, prob=prob)
            print('\t%s seed %d: prec@1 %.4f prec@5 %.4f act %.4f' % (gate_mode, seed, precision_map[seed][0], precision_map[seed][1], float(num_acts)/ num_total))
            acts.append(float(num_acts)/ num_total)
        prec1s = []
        prec5s = []
        for k in precision_map.keys():
            prec1s.append(precision_map[k][0])
            prec5s.append(precision_map[k][1])
        print('stochastic stats prec@1\t mean %.4f stddev %.4f' % (np.mean(prec1s), np.std(prec1s)))
        print('stochastic stats prec@5\t mean %.4f stddev %.4f' % (np.mean(prec5s), np.std(prec5s)))
        print('stochastic stats act\t mean %.4f stddev %.4f' % (np.mean(acts), np.std(acts)))
        print()

    if True:
        precision_map = {}
        pred_map = {}
        ensemble_acc = []
        outputs = []
        THRESHOLD = .5
        seed = 0
        gate_mode = 'argmax'
        print('Begin %s run...' % gate_mode)
        num_correct, num_total, num_acts, acts_nan_counter, targets, activ_output = validate_one(module_model, val_loader, model, seed, gate_mode, precision_map, pred_map, threshold=THRESHOLD)
        print('%s seed %d: prec@1 %.4f prec@5 %.4f act %.4f' % (gate_mode, seed, precision_map[seed][0], precision_map[seed][1], float(num_acts)/ num_total))
        print()
        exit(1)

        seed = 0
        gate_mode = 'always_on'
        print('Begin %s run...' % gate_mode)
        num_correct, num_total, num_acts, acts_nan_counter, targets, activ_output = validate_one(module_model, val_loader, model, seed, gate_mode, precision_map, pred_map)
        print('%s seed %d: prec@1 %.4f prec@5 %.4f act %.4f' % (gate_mode, seed, precision_map[seed][0], precision_map[seed][1], float(num_acts)/ num_total))
        print()

    precision_map = {}
    pred_map = {}
    ensemble_acc = []
    outputs = []

    seeds = range(args.num_seeds)
    acts = []
    gate_activations = {}
    gate_mode = 'stochastic'
    print('Begin %s run...' % gate_mode)
    for seed in seeds:
        num_correct, num_total, num_acts, acts_nan_counter, targets, activ_output = validate_one(module_model, val_loader, model, seed, gate_mode, precision_map, pred_map)
        print('\t%s seed %d: prec@1 %.4f prec@5 %.4f act %.4f' % (gate_mode, seed, precision_map[seed][0], precision_map[seed][1], float(num_acts)/ num_total))
        acts.append(float(num_acts)/ num_total)
        for k, v in activ_output[0].items():
            if k not in gate_activations:
                gate_activations[k] = []
            gate_activations[k].append(v)

    prec1s = []
    prec5s = []
    for k in precision_map.keys():
        prec1s.append(precision_map[k][0])
        prec5s.append(precision_map[k][1])

    print('stochastic stats prec@1\t mean %.4f stddev %.4f' % (np.mean(prec1s), np.std(prec1s)))
    print('stochastic stats prec@5\t mean %.4f stddev %.4f' % (np.mean(prec5s), np.std(prec5s)))
    print('stochastic stats act\t mean %.4f stddev %.4f' % (np.mean(acts), np.std(acts)))
    print('stochastic acts means: ', {k: np.mean(v) for k,v in gate_activations.items()})
    print('stochastic acts stddev: ', {k: np.std(v) for k,v in gate_activations.items()})
    print()


    num_correct = [0, 0]
    num_total = 0
    ensemble_acc = []
    print('Begin %s run...' % 'ensemble')
    for batch_index in range(len(targets)):
        for in_batch_index in range(targets[batch_index].shape[0]):
            val1 = []
            val5 = []
            for seed in seeds:
                val1.append(pred_map[seed][0][batch_index][in_batch_index].data[0])
                val5.extend(pred_map[seed][1][batch_index][in_batch_index].data)
            ensemble_pred, ensemble_count = stats.mode(val1)
            top5_dict = collections.Counter(val5)
            num_correct[0] += int(ensemble_pred[0] == targets[batch_index][in_batch_index])
            num_correct[1] += int(targets[batch_index][in_batch_index] in [k for k,v in top5_dict.most_common(5)])
            num_total += 1

    print('Ensemble: ', 100 * float(num_correct[0]) / num_total, 100 * float(num_correct[1]) / num_total, num_total)
    print()



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

if __name__ == '__main__':
    print(sys.argv)
    main()


