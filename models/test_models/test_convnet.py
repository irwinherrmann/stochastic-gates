'''ConvNet-AIG in PyTorch.

Residual Network is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

Adaptive Inference Graphs is from the original ConvNet-AIG paper:
[2] Andreas Veit, Serge Belognie
    Convolutional Networks with Adaptive Inference Graphs. ECCV 2018

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable

from models.gumbelmodule import GumbleSoftmax
from collections import OrderedDict
import collections


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class Sequential_ext(nn.Module):
    """A Sequential container extended to also propagate the gating information
    that is needed in the target rate loss.
    """

    def __init__(self, *args):
        super(Sequential_ext, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def forward(self, input, temperature=1, gate_mode='stochastic', openings=None):
        gate_activations = []
        w1bns = []
        for i, module in enumerate(self._modules.values()):
            input, gate_activation, w1bn = module(input, temperature, gate_mode)
            gate_activations.append(gate_activation)
            w1bns.append(w1bn)
        return input, gate_activations, w1bns


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, test=False):
        super(BasicBlock, self).__init__()
        self.test = test
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        # Gate layers
        self.fc1 = nn.Conv2d(in_planes, 16, kernel_size=1)
        self.fc1bn = nn.BatchNorm1d(16)
        self.fc2 = nn.Conv2d(16, 2, kernel_size=1)
        # initialize the bias of the last fc for 
        # initial opening rate of the gate of about 85%
        self.fc2.bias.data[0] = 0.1
        self.fc2.bias.data[1] = 2
        self.gs = GumbleSoftmax()
        self.gs.cuda()

    def forward(self, x, temperature=1, gate_mode='stochastic'):
        assert(gate_mode in ['stochastic', 'always_on', 'argmax'])

        # Compute relevance score
        w = F.avg_pool2d(x, x.size(2))
        w = F.relu(self.fc1bn(self.fc1(w)))
        w = self.fc2(w)
        # Sample from Gumble Module
#        print 'fc before gumble', w.shape

        if gate_mode == "argmax":
          _, max_value_indexes = w.data.max(1, keepdim=True) #max_values_indices is batchsize x 1 and is 0 or 1.
          output_multiplier = max_value_indexes.unsqueeze(1)
        elif gate_mode == "stochastic":
          w = self.gs(w, temp=temperature, force_hard=True)
          output_multiplier = w[:,1].unsqueeze(1)
        elif gate_mode == "always_on":
          output_multiplier = torch.ones(w[:,1].unsqueeze(1).size())
        else:
          assert(False) # Error: added a possible gate mode without implementing it.

        # TODO(chi): Write the test code
        #print(w[:,1].unsqueeze(1))
        #if self.test and w[:,1].unsqueeze(1) == 0:
        #    out = self.shortcut(x)
        #    return out, w[:,1]

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.shortcut(x) + out * output_multiplier
        out = F.relu(out)
        # Return output of layer and the value of the gate
        # The value of the gate will be used in the target rate loss

        return out, output_multiplier.squeeze(1)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, test=False):
        super(Bottleneck, self).__init__()
        self.test = test
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        # Gate layers
        self.fc1 = nn.Conv2d(in_planes, 16, kernel_size=1)
        self.fc1bn = nn.BatchNorm1d(16)
        self.fc2 = nn.Conv2d(16, 2, kernel_size=1)
        # initialize the bias of the last fc for 
        # initial opening rate of the gate of about 85%
        self.fc2.bias.data[0] = 0.1
        self.fc2.bias.data[1] = 2

        self.gs = GumbleSoftmax()
        self.gs.cuda()

    def forward(self, x, temperature=1, gate_mode='stochastic', threshold=.5):
        assert(gate_mode in ['stochastic', 'always_on', 'argmax'])

        # Compute relevance score
        w = F.avg_pool2d(x, x.size(2))
        w1bn = self.fc1(w)
        w = self.fc1bn(w1bn)
        w = F.relu(w)
        w = self.fc2(w)

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))

        if gate_mode == "argmax":
            _, max_value_indexes = w.data.max(1, keepdim=True) #max_values_indices is batchsize x 1 and is 0 or 1.
            output_multiplier = torch.autograd.Variable(max_value_indexes.float(), volatile=True)
        elif gate_mode == 'threshold':
            output_multiplier = torch.autograd.Variable(torch.gt(w[:,1], threshold).unsqueeze(1), volatile=True)
        elif gate_mode == "stochastic":
            w = self.gs(w, temp=temperature, force_hard=True)
            output_multiplier = w[:,1].unsqueeze(1)
        elif gate_mode == "always_on":
            output_multiplier = torch.autograd.Variable(torch.ones(out.size()).cuda(), volatile=True)
        else:
          assert(False) # Error: added a possible gate mode without implementing it.



        out = self.shortcut(x) + out * output_multiplier
        out = F.relu(out, inplace=True)
        # Return output of layer and the value of the gate
        # The value of the gate will be used in the target rate loss
        return out, output_multiplier.squeeze(1), w1bn

    
class ResNet_ImageNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, test=False):
        self.in_planes = 64
        super(ResNet_ImageNet, self).__init__()
        self.test = test
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        for k, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if 'fc2' in str(k):
                    # Initialize last layer of gate with low variance
                    m.weight.data.normal_(0, 0.001)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.test))
            self.in_planes = planes * block.expansion
        return Sequential_ext(*layers)

    def forward(self, out, temperature=1, gate_mode = 'stochastic', prob=1, threshold=.5):
        gate_activations = []
        w1bns = []

        out = self.relu(self.bn1(self.conv1(out)))
        out = self.maxpool(out)

        out, a, w1 = self.layer1(out, temperature, gate_mode)
        gate_activations.extend(a)
        w1bns.extend(w1)

        out, a, w1 = self.layer2(out, temperature, gate_mode)
        gate_activations.extend(a)
        w1bns.extend(w1)

        out, a, w1 = self.layer3(out, temperature, gate_mode)
        gate_activations.extend(a)
        w1bns.extend(w1)

        out, a, w1 = self.layer4(out, temperature, gate_mode)
        gate_activations.extend(a)
        w1bns.extend(w1)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, gate_activations, w1bns

class ResNet_cifar(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, test=False):
        super(ResNet_cifar, self).__init__()
        self.in_planes = 16
        self.test = test

        self.conv1 = conv3x3(3,16)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*block.expansion, num_classes)

        for k, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'fc2' in str(k):
                    # Initialize last layer of gate with low variance
                    m.weight.data.normal_(0, 0.001)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.test))
            self.in_planes = planes * block.expansion
        return Sequential_ext(*layers)

    def forward(self, x, temperature=1, prob=1, openings=None):
        gate_activations = []
        out = F.relu(self.bn1(self.conv1(x)))
        out, a = self.layer1(out, temperature)
        gate_activations.extend(a)
        out, a = self.layer2(out, temperature)
        gate_activations.extend(a)
        out, a = self.layer3(out, temperature)
        gate_activations.extend(a)
#        print 'before avg_pool2d', out.shape
        out = F.avg_pool2d(out, 8)
#        print 'before view', out.shape
        out = out.view(out.size(0), -1)
#        print 'before linear', out.shape
        out = self.linear(out)
#        exit(1)
        return out, gate_activations

def ResNet50_ImageNet(test=False):
    return ResNet_ImageNet(Bottleneck, [3,4,6,3], test=test)

def ResNet101_ImageNet(test=False):
    return ResNet_ImageNet(Bottleneck, [3,4,23,3], test=test)

def ResNet152_ImageNet(test=False):
    return ResNet_ImageNet(Bottleneck, [3,8,36,3], test=test)


class ActivationAccum_img():
    def __init__(self, epoch):
        self.gates = collections.defaultdict(lambda : 0)
        self.batchsize = 0
        self.epoch = epoch

    def accumulate(self, actives, targets, target_rates):
        self.batchsize += len(targets)
        for j, act in enumerate(actives):
            self.gates[j] += torch.sum(act)

    def getoutput(self):
        for k in list(self.gates.keys()):
            if type(self.gates[k]) != int or type(self.gates[k]) != float:
                self.gates[k] = self.gates[k].data.cpu().numpy()[0]
        
        return([{k: float(self.gates[k]) / self.batchsize for k in self.gates}])


def test():
    net = ResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test()
