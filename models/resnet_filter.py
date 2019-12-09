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
import time
import collections

from models.gumbelmodule import GumbleSoftmax
import models.gates as gates
from collections import OrderedDict 

from seq_with_gumbel import SpecialGumble


num_total_time = 6
total_time = [0 for _ in range(num_total_time)]
gate_time = 0

def run_gate(num_gates, gate, x):
    if num_gates > 0:
        return gate(x)
    else:
        return -1

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

    def forward(self, input, temperature=1, openings=None):
        first_cat = True
        for i, module in enumerate(self._modules.values()):
            input, gate_activation, shortcut_activation, _, _ = module(input, temperature)
            if first_cat:
                gate_activations = gate_activation
                shortcut_activations = shortcut_activation
                first_cat = False
            else:
                gate_activations = torch.cat([gate_activations, gate_activation], 1)
        return input, gate_activations, shortcut_activations


class Sequential_withGumble(nn.Module):
    """A Sequential container extended to also propagate the gating information
    that is needed in the target rate loss.
    """

    def __init__(self, *args):
        super(Sequential_withGumble, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            gumble_count = 0
            for idx, module in enumerate(args):
                if type(module) == SpecialGumble or type(module) == nn.ReLU:
                    self.add_module('gumble_%d'%gumble_count, module)
                    gumble_count += 1
                else:
                    self.add_module(str(idx-gumble_count), module)

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

    def forward(self, input, weights, splits, temperature=1, openings=None):
        gate_index = 0
        first_cat = True
        for i, module in enumerate(self._modules.values()):
            if type(module) == SpecialGumble:
                assert(gate_index < len(weights))
                #print len(weights), splits[gate_index], splits[gate_index+1]
                w = weights[:, splits[gate_index]: splits[gate_index+1], :]
                input, activations, activations_soft = module(input, w, temperature)
                gate_index += 1
                activations = torch.sum(activations, 1).view((-1, 1, 1))
                activations_soft = torch.sum(activations_soft, 1).view((-1, 1, 1))
                if first_cat:
                    gate_activations = activations
                    gate_activations_soft = activations_soft
                    first_cat = False
                else:
                    gate_activations = torch.cat([gate_activations, activations], 2)
                    gate_activations_soft = torch.cat([gate_activations_soft, activations_soft], 2)
            else:
                input = module(input)
        return input, gate_activations, gate_activations_soft

class IdentityModule(nn.Module):
    def __init__(self):
        super(IdentityModule, self).__init__()

    def forward(self, x):
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, filter_info_inp, filter_info_mid, filter_info_oup, filter_info_shortcut, num_gates_fixed_open, dependent_gates=True, stride=1):
        super(BasicBlock, self).__init__()
        #print 'stride', stride
        self.num_gates_per_sht = 0
        self.shortcut = nn.Sequential()
        self.has_shortcut = stride != 1 or in_planes != self.expansion*planes
        if self.has_shortcut:
            num_filters_per_sht, self.num_gates_per_sht = filter_info_shortcut
            self.gating_check(self.expansion*planes, num_filters_per_sht, self.num_gates_per_sht, num_gates_fixed_open)
            self.shortcut = Sequential_withGumble(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes),
                SpecialGumble(num_gates_fixed_open, self.num_gates_per_sht, num_filters_per_sht),
            )
            print 'has_shortcut'

        num_filters_per_inp, num_gates_per_inp = filter_info_inp
        num_filters_per_mid, num_gates_per_mid = filter_info_mid
        num_filters_per_oup, num_gates_per_oup = filter_info_oup
        self.num_gates_per_inp = num_gates_per_inp
        self.num_gates_per_mid = num_gates_per_mid

        self.gating_check(in_planes, num_filters_per_inp, num_gates_per_inp, num_gates_fixed_open)
        self.gating_check(planes, num_filters_per_mid, num_gates_per_mid, num_gates_fixed_open)
        self.gating_check(planes, num_filters_per_oup, num_gates_per_oup, num_gates_fixed_open)

        self.conv = Sequential_withGumble(
            SpecialGumble(num_gates_fixed_open, num_gates_per_inp, num_filters_per_inp),
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            SpecialGumble(num_gates_fixed_open, num_gates_per_mid, num_filters_per_mid),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            SpecialGumble(num_gates_fixed_open, num_gates_per_oup, num_filters_per_oup),
        )

        self.total_gates = self.num_gates_per_sht + num_gates_per_inp + num_gates_per_mid + num_gates_per_oup
        self.splits = (0, self.num_gates_per_inp, self.num_gates_per_inp + self.num_gates_per_mid, self.total_gates)
        self.shortcut_splits = (0, self.num_gates_per_sht)
        print 'dependent_gates', dependent_gates
        if dependent_gates:
            self.gate = gates.DependentGate_Block(in_planes, self.total_gates)
        else:
            self.gate = gates.IndependentGate(in_planes, self.total_gates)


    def gating_check(self, num_filters, num_filters_per_gate, num_gates, num_gates_fixed_open):
        #print num_filters, num_filters_per_gate, num_gates, num_gates_fixed_open
        assert(num_filters % num_filters_per_gate == 0)
        assert(num_filters == num_filters_per_gate * (num_gates + num_gates_fixed_open))
        assert(num_filters / num_filters_per_gate == num_gates+num_gates_fixed_open)


    def forward(self, x, temperature=1):
        w = run_gate(self.total_gates, self.gate, x)

        if self.has_shortcut:
            w_shortcut = w[:, :self.num_gates_per_sht, :]
            w_conv = w[:, self.num_gates_per_sht:, :]
            sh, shortcut_act, shortcut_act_soft = self.shortcut(x, w_shortcut, self.shortcut_splits, temperature)
        else:
            sh = self.shortcut(x)
            w_conv = w
            shortcut_act = None
            shortcut_act_soft = None

        out, act, act_soft = self.conv(x, w_conv, self.splits, temperature)

        out = sh + out

        out = F.relu(out)
        # Return output of layer and the value of the gate
        # The value of the gate will be used in the target rate loss
        return out, act, shortcut_act, act_soft, shortcut_act_soft


class PlainBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, filter_info_inp, filter_info_mid, filter_info_oup, filter_info_shortcut, num_gates_fixed_open, dependent_gates=True, stride=1):
        super(PlainBasicBlock, self).__init__()
        #print 'stride', stride
        self.num_gates_per_sht = 0
        self.shortcut = nn.Sequential()

        num_filters_per_inp, num_gates_per_inp = filter_info_inp
        num_filters_per_mid, num_gates_per_mid = filter_info_mid
        num_filters_per_oup, num_gates_per_oup = filter_info_oup
        self.num_gates_per_inp = num_gates_per_inp
        self.num_gates_per_mid = num_gates_per_mid

        self.gating_check(in_planes, num_filters_per_inp, num_gates_per_inp, num_gates_fixed_open)
        self.gating_check(planes, num_filters_per_mid, num_gates_per_mid, num_gates_fixed_open)
        self.gating_check(planes, num_filters_per_oup, num_gates_per_oup, num_gates_fixed_open)

        self.conv = Sequential_withGumble(
            SpecialGumble(num_gates_fixed_open, num_gates_per_inp, num_filters_per_inp),
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            SpecialGumble(num_gates_fixed_open, num_gates_per_mid, num_filters_per_mid),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            SpecialGumble(num_gates_fixed_open, num_gates_per_oup, num_filters_per_oup),
        )

        self.total_gates = self.num_gates_per_sht + num_gates_per_inp + num_gates_per_mid + num_gates_per_oup
        self.splits = (0, self.num_gates_per_inp, self.num_gates_per_inp + self.num_gates_per_mid, self.total_gates)
        print 'dependent_gates', dependent_gates
        if dependent_gates:
            self.gate = gates.DependentGate_Block(in_planes, self.total_gates)
        else:
            self.gate = gates.IndependentGate(in_planes, self.total_gates)


    def gating_check(self, num_filters, num_filters_per_gate, num_gates, num_gates_fixed_open):
        #print num_filters, num_filters_per_gate, num_gates, num_gates_fixed_open
        assert(num_filters % num_filters_per_gate == 0)
        assert(num_filters == num_filters_per_gate * (num_gates + num_gates_fixed_open))
        assert(num_filters / num_filters_per_gate == num_gates+num_gates_fixed_open)


    def forward(self, x, temperature=1):
        w = run_gate(self.total_gates, self.gate, x)

        shortcut_act = None
        w_conv = w

        out, act = self.conv(x, w_conv, self.splits, temperature)

        out = F.relu(out)
        # Return output of layer and the value of the gate
        # The value of the gate will be used in the target rate loss
        return out, act, shortcut_act


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, filter_infos, dependent_gates=True, stride=1):
        super(Bottleneck, self).__init__()

        self.filter_info_list, self.num_gates_fixed_open = filter_infos

        assert(len(self.filter_info_list) == 5)
        for i in range(len(self.filter_info_list)):
            assert(len(self.filter_info_list[i]) == 2)

        self.conv = Sequential_withGumble(
            SpecialGumble(self.num_gates_fixed_open, self.filter_info_list[1][1], self.filter_info_list[1][0]), # expand
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            SpecialGumble(self.num_gates_fixed_open, self.filter_info_list[2][1], self.filter_info_list[2][0]), # contract
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            SpecialGumble(self.num_gates_fixed_open, self.filter_info_list[3][1], self.filter_info_list[3][0]), # contract
            nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.expansion*planes),
            SpecialGumble(self.num_gates_fixed_open, self.filter_info_list[4][1], self.filter_info_list[4][0]), # expand
        )

        self.shortcut = nn.Sequential()
        self.has_shortcut = stride != 1 or in_planes != self.expansion*planes
        if self.has_shortcut:
            print 'has shortcut!'
            num_filters_per_sht, self.num_gates_per_sht = self.filter_info_list[0]
            self.gating_check(self.expansion*planes, num_filters_per_sht, self.num_gates_per_sht, self.num_gates_fixed_open)
            self.shortcut = Sequential_withGumble(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes),
                SpecialGumble(self.num_gates_fixed_open, self.num_gates_per_sht, num_filters_per_sht),
            )
        else:
            self.num_gates_per_sht = 0

        filters_per = [in_planes, planes, planes, self.expansion*planes]

        # Gate layers
        self.total_gates = self.num_gates_per_sht
        self.splits = [0]
        for i, filter_info in enumerate(self.filter_info_list[1:]):
            num_filters_per_gatingstructure, num_gates_per_gatingstructure = filter_info
            self.total_gates += num_gates_per_gatingstructure
            self.gating_check(filters_per[i], num_filters_per_gatingstructure, num_gates_per_gatingstructure, self.num_gates_fixed_open)
            self.splits.append(self.splits[-1] + num_gates_per_gatingstructure)

        assert(self.splits[-1] == self.total_gates - self.num_gates_per_sht), str(self.splits) + ' ' + str(self.total_gates) + ' ' + str(self.num_gates_per_sht)
        self.shortcut_splits = (0, self.num_gates_per_sht)

        print 'dependent_gates', dependent_gates
        if dependent_gates:
            self.gate = gates.DependentGate_Block(in_planes, self.total_gates)
        else:
            self.gate = gates.IndependentGate(in_planes, self.total_gates)


    def gating_check(self, num_filters, num_filters_per_gate, num_gates, num_gates_fixed_open):
        #print num_filters, num_filters_per_gate, num_gates, num_gates_fixed_open
        assert(num_filters % num_filters_per_gate == 0)
        assert(num_filters == num_filters_per_gate * (num_gates + num_gates_fixed_open))
        assert(num_filters / num_filters_per_gate == num_gates+num_gates_fixed_open)

    def forward(self, x, temperature=1):
        w = run_gate(self.total_gates, self.gate, x)

        if self.has_shortcut:
            w_shortcut = w[:, :self.num_gates_per_sht, :]
            w_conv = w[:, self.num_gates_per_sht:, :]
            sh, shortcut_act, shorcut_act_soft = self.shortcut(x, w_shortcut, self.shortcut_splits, temperature)
        else:
            sh = self.shortcut(x)
            w_conv = w
            shortcut_act = None

        out, acts, _ = self.conv(x, w_conv, self.splits, temperature)

        out = sh + out
        out = F.relu(out, inplace=True)

        # Return output of layer and the value of the gate
        # The value of the gate will be used in the target rate loss
        return out, acts, shortcut_act

    
class ResNet_ImageNet(nn.Module):

    def __init__(self, params, block, layers, num_classes=1000, dependent_gates=True):
        self.in_planes = 64
        super(ResNet_ImageNet, self).__init__()

        self.dependent_gates = dependent_gates

        num_filters_expand = params.granularity[0]
        num_filters_contract = params.granularity[1]
        gates_fixed_open = params.gates_fixed_open

        self.layer_activation_rates = OrderedDict()
        self.possible_gflops_per_layer = OrderedDict()
        self.possible_gflops = 0
        self.constant_gflops = 0
        self.gated_gflops = 0
        self.gflops_for_gates = 0
        self.gflops_for_pools = 0
        self.num_gates_per_layer = []
        self.num_gates = 0
        self.input_size = 224
        size = self.input_size

        self.sizes = []
        self.strides = []
        self.shortcut_sizes = []
        self.shortcut_inputchannels = []
        self.sizes = []
        self.strides = []

        # build layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        size = size / 2
        self.first_size = size
        layer_gflops = 3 * 64 * 7 * 7 * size * size
        print 'Layer0 gflops:', layer_gflops
        self.possible_gflops += layer_gflops
        self.constant_gflops += layer_gflops

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        size = size / 2

        print 'Layer1 ----------'
        self.layer1 = self._make_layer(block, 64, layers[0], size, num_filters_expand[0], num_filters_contract[0], gates_fixed_open, stride=1)
        print 'Layer2 ----------'
        self.layer2 = self._make_layer(block, 128, layers[1], size, num_filters_expand[1], num_filters_contract[1], gates_fixed_open, stride=2)
        size = size / 2
        print 'Layer3 ----------'
        self.layer3 = self._make_layer(block, 256, layers[2], size, num_filters_expand[2], num_filters_contract[2], gates_fixed_open, stride=2)
        size = size / 2
        print 'Layer4 ----------'
        self.layer4 = self._make_layer(block, 512, layers[3], size, num_filters_expand[3], num_filters_contract[3], gates_fixed_open, stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        print 'GFLOPs in gates: %d' % self.gflops_for_gates
        print 'GFLOPs in pools: %d' % self.gflops_for_pools

        layer_gflops = 512 * block.expansion * num_classes
        self.possible_gflops += layer_gflops
        self.constant_gflops += layer_gflops

        self.register_buffer('sizesquared_tensor', torch.pow(torch.Tensor(self.sizes).cuda(), 2))
        self.register_buffer('stridesquared_tensor', torch.pow(torch.Tensor(self.strides).cuda(), 2))
        self.register_buffer('shortcut_sizesquared_tensor', torch.pow(torch.Tensor(self.shortcut_sizes).cuda(), 2))
        self.register_buffer('shortcut_inputchannels_tensor', torch.Tensor(self.shortcut_inputchannels).cuda())

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

    def _make_layer(self, block, planes, num_blocks, size, num_filters_per_expand, num_filters_per_contract, num_gates_fixed_open, stride=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        # layer creation
        for stride in strides:
            print 'size', size, 'stride', stride

            # gating divisions
            num_gates_per_inp = self.in_planes / num_filters_per_expand - num_gates_fixed_open
            filter_info_inp = num_filters_per_expand, num_gates_per_inp
            self.gflops_for_gates += num_gates_per_inp * (self.in_planes * 16  + 16 * 2)
            self.gflops_for_pools += num_gates_per_inp * size * size

            num_gates_per_mid = planes / num_filters_per_contract - num_gates_fixed_open
            filter_info_mid1 = num_filters_per_contract, num_gates_per_mid
            filter_info_mid2 = num_filters_per_contract, num_gates_per_mid
            self.gflops_for_gates += num_gates_per_mid * (planes * 16  + 16 * 2)
            self.gflops_for_gates += num_gates_per_mid * (planes * 16  + 16 * 2)

            num_gates_per_oup = planes * block.expansion / num_filters_per_expand - num_gates_fixed_open
            filter_info_oup = num_filters_per_expand, num_gates_per_oup
            self.gflops_for_gates += num_gates_per_oup * (planes * block.expansion * 16  + 16 * 2)

            num_gates_per_sht = planes * block.expansion / num_filters_per_expand - num_gates_fixed_open
            filter_info_sht = num_filters_per_expand, num_gates_per_sht

            filter_info_list = [filter_info_sht, filter_info_inp, filter_info_mid1, filter_info_mid2, filter_info_oup]
            filter_infos = (filter_info_list, num_gates_fixed_open)
            print 'make_layer in_planes, num_filter_per_layer, num_gates_per_inp ', self.in_planes, num_filters_per_expand, num_gates_per_inp
            print 'make_layer planes, num_filters_per_layer, num_gates_per_mid1   ', planes, num_filters_per_contract, num_gates_per_mid
            print 'make_layer planes, num_filters_per_layer, num_gates_per_mid2   ', planes, num_filters_per_contract, num_gates_per_mid
            print 'make_layer planes, num_filters_per_layer, num_gates_per_oup   ', planes*block.expansion, num_filters_per_expand, num_gates_per_mid
            if stride != 1 or self.in_planes != block.expansion*planes :
                print 'make_layer planes, num_filters_per_layer, num_gates_per_sht   ', planes*block.expansion, num_filters_per_expand, num_gates_per_sht

            filters_fixed_open_expand = num_filters_per_expand * num_gates_fixed_open
            filters_fixed_open_contract = num_filters_per_contract * num_gates_fixed_open

            num_gates_this_layer = num_gates_per_inp + num_gates_per_mid + num_gates_per_mid + num_gates_per_oup

            self.strides.append(stride)
            self.sizes.append(size)

            if stride == 1 and self.in_planes == block.expansion*planes:
                possible_layer_gflops = pow(size, 2) * (1 * self.in_planes * planes + 9 * planes * planes / pow(stride, 2) + 1 * planes * planes * block.expansion / pow(stride, 2))
                constant_layer_gflops = pow(size, 2) * (1 * filters_fixed_open_expand * filters_fixed_open_contract + 9 * pow(filters_fixed_open_contract, 2) / pow(stride, 2) + 1 * filters_fixed_open_contract * filters_fixed_open_expand / pow(stride, 2))
                print 'layer_gflops', possible_layer_gflops

            else:
                possible_layer_gflops = pow(size, 2) * (1 * self.in_planes * planes + 9 * planes * planes / pow(stride, 2) + 1 * planes * planes * block.expansion / pow(stride, 2))
                constant_layer_gflops = pow(size, 2) * (1 * filters_fixed_open_expand * filters_fixed_open_contract + 9 * pow(filters_fixed_open_contract, 2) / pow(stride, 2) + 1 * filters_fixed_open_contract * filters_fixed_open_expand / pow(stride, 2))

                shortcut_layer_gflops = pow(size / stride, 2) * 1 * self.in_planes * planes * block.expansion
                constant_shortcut_layer_gflops = pow(size / stride, 2) * 1 * self.in_planes * filters_fixed_open_expand
                print 'layer_gflops', possible_layer_gflops

                possible_layer_gflops += shortcut_layer_gflops
                constant_layer_gflops += constant_shortcut_layer_gflops
                print 'layer_gflops with shortcut', possible_layer_gflops

                self.gflops_for_gates += num_gates_per_sht * (planes * block.expansion * 16  + 16 * 2)

                size = size / stride

                self.shortcut_sizes.append(size)
                self.shortcut_inputchannels.append(self.in_planes) 

                num_gates_this_layer += num_gates_per_sht

                print 'make_layers shortcut'

#                print 'miniactive', 1 * self.in_planes * planes * size * size
#            print 'possible_gflops', possible_layer_gflops

            #self.possible_gflops_per_layer[(layer_index, index)] = [self.in_planes, planes, planes * block.expansion]

            self.possible_gflops += possible_layer_gflops
            self.constant_gflops += constant_layer_gflops
            self.gated_gflops += possible_layer_gflops - constant_layer_gflops

            self.num_gates += num_gates_this_layer
            self.num_gates_per_layer.append(num_gates_this_layer)

            layers.append(block(self.in_planes, planes, filter_infos, self.dependent_gates, stride))
            self.in_planes = planes * block.expansion
        return Sequential_ext(*layers)

    def forward(self, out, temperature=1):
        gate_activations = []
        out = self.relu(self.bn1(self.conv1(out)))
        out = self.maxpool(out)
        out, a, sh_a = self.layer1(out, temperature)
        gate_activations = a
        shortcut_activations = sh_a

        out, a, sh_a = self.layer2(out, temperature)
        gate_activations = torch.cat((gate_activations, a), 1)
        shortcut_activations = torch.cat((shortcut_activations, sh_a), 1)

        out, a, sh_a = self.layer3(out, temperature)
        gate_activations = torch.cat((gate_activations, a), 1)
        shortcut_activations = torch.cat((shortcut_activations, sh_a), 1)

        out, a, sh_a = self.layer4(out, temperature)
        gate_activations = torch.cat((gate_activations, a), 1)
        shortcut_activations = torch.cat((shortcut_activations, sh_a), 1)
        #print 'shortcut_activations', shortcut_activations.shape

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        gflops, bm = self.get_gflops(gate_activations, shortcut_activations)

        return out, gflops, bm

    def get_gflops(self, gate_activations, shortcut_activations):
        gflops_sum = 0
        layer_gflops = 3 * 64 * 7 * 7 * self.first_size * self.first_size
        gflops_sum += layer_gflops

        sizesquared = torch.autograd.Variable(self.sizesquared_tensor, requires_grad=False) 
        stridesquared = torch.autograd.Variable(self.stridesquared_tensor, requires_grad=False) 
        shortcut_sizesquared = torch.autograd.Variable(self.shortcut_sizesquared_tensor, requires_grad=False)
        shortcut_inputchannels = torch.autograd.Variable(self.shortcut_inputchannels_tensor, requires_grad=False)

        assert(shortcut_activations.shape[1] == 4), shortcut_activations.shape
        assert(shortcut_inputchannels.shape == torch.Size([4])), shortcut_inputchannels.shape
        assert(shortcut_sizesquared.shape == torch.Size([4])), shortcut_sizesquared.shape
        shortcut_activations = shortcut_activations.view((-1, 4))
        shortcut_gflops = shortcut_sizesquared * shortcut_inputchannels * shortcut_activations
        #print shortcut_gflops.shape
        gflops_sum += torch.mean(torch.sum(shortcut_gflops, 1), 0)
        #print 'after shortcut', gflops_sum

        acts_inp = gate_activations[:,:,0]
        acts_mid1 = gate_activations[:,:,1]
        acts_mid2 = gate_activations[:,:,2]
        acts_oup = gate_activations[:,:,3]

        # from above: pow(size, 2) * (1 * self.in_planes * planes + 9 * planes * planes + 1 * planes * planes * block.expansion)
        layer_gflops = sizesquared * (acts_inp * acts_mid1 + 9 * acts_mid1 * acts_mid2 / stridesquared + 1 * acts_mid2 * acts_oup / stridesquared) # (batchsize, num_layers)
        #print torch.mean(layer_gflops, 0)
        gflops_sum += torch.mean(torch.sum(layer_gflops, 1), 0)

        layer_gflops = 512 * 4 * 1000
        gflops_sum += layer_gflops

#        print 'final', float(gflops_sum.data[0]), float(self.possible_gflops)
#        print '\n\n'
        return gflops_sum, (gflops_sum - self.constant_gflops)/ self.gated_gflops

    def get_gate_weights(self):
        return [self.gated_gflops]

    def get_constant_gflops(self):
        return self.constant_gflops

    def get_possible_gflops(self):
        return self.possible_gflops


class ResNet_cifar(nn.Module):
    def __init__(self, params, block, num_blocks, num_classes=10, width=1., dependent_gates=True):
        super(ResNet_cifar, self).__init__()
        self.in_planes = 16
        self.dependent_gates = dependent_gates

        granularity = params.granularity
        gates_fixed_open = params.gates_fixed_open

        self.input_size = 32
        self.num_classes = num_classes

        self.layer_activation_rates = OrderedDict()
        self.possible_gflops_per_layer = OrderedDict()

        self.sizes = []
        self.shortcut_sizes = []
        self.shortcut_inputchannels = []

        self.num_gates_per_layer = []
        self.num_gates_total = 0
        self.possible_gflops = 0
        self.constant_gflops = 0
        self.gated_gflops = 0
        size = self.input_size
        self.layer_filters = [int(16*width), int(32*width), int(64*width)]
        self.stride = [1,2,2]

        layer_gflops = 9 * 3 * int(16*width) * size * size
        self.possible_gflops += layer_gflops
        self.constant_gflops += layer_gflops

        self.conv1 = conv3x3(3,int(16*width))
        self.bn1 = nn.BatchNorm2d(int(16*width))
        self.layer1 = self._make_layer(block, int(16*width), num_blocks[0], granularity[0], stride=1, size=size, gates_fixed_open=gates_fixed_open, layer_index=0)
        self.layer2 = self._make_layer(block, int(32*width), num_blocks[1], granularity[1],stride=2, size=size, gates_fixed_open=gates_fixed_open, layer_index=1)
        self.layer3 = self._make_layer(block, int(64*width), num_blocks[2], granularity[2], stride=2, size=size/2, gates_fixed_open=gates_fixed_open, layer_index=2)
        self.linear = nn.Linear(int(64*width)*block.expansion, num_classes)

        layer_gflops = pow(int(64*width) * block.expansion * num_classes, 2)
        self.possible_gflops += layer_gflops
        self.constant_gflops += layer_gflops
        print 'possible_gflops   %5d' % self.possible_gflops
        print 'constant_gflops   %5d' % self.constant_gflops
        print 'gated_gflops      %5d' % self.gated_gflops
        print 'granularity', granularity

        self.sizesquared_tensor = torch.pow(torch.Tensor(self.sizes).cuda(), 2)
        self.shortcut_sizesquared_tensor = torch.pow(torch.Tensor(self.shortcut_sizes).cuda(), 2)
        self.shortcut_inputchannels_tensor = torch.Tensor(self.shortcut_inputchannels).cuda()

        for k, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'fc2' in str(k):
                    # Initialize last layer of gate with low variance
                    m.weight.data.normal_(0, 0.001)

    def _make_layer(self, block, planes, num_blocks, num_filter_per_layer, stride, size, gates_fixed_open, layer_index):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        # gating divisions
        num_gates_per_inp = self.in_planes / num_filter_per_layer - gates_fixed_open
        filter_info_inp = num_filter_per_layer, num_gates_per_inp
        num_gates_per_mid = planes / num_filter_per_layer - gates_fixed_open
        filter_info_mid = num_filter_per_layer, num_gates_per_mid
        num_gates_per_sht = planes * block.expansion / num_filter_per_layer - gates_fixed_open
        filter_info_sht = num_filter_per_layer, num_gates_per_sht
        print 'make_layer in_planes, num_filter_per_layer, num_gates_per_inp ', self.in_planes, num_filter_per_layer, num_gates_per_inp
        print 'make_layer planes, num_filters_per_layer, num_gates_per_mid   ', planes, num_filter_per_layer, num_gates_per_mid
        print 'make_layer planes, num_filters_per_layer, num_gates_per_oup   ', planes, num_filter_per_layer, num_gates_per_mid
        print 'make_layer planes, num_filters_per_layer, num_gates_per_sht   ', planes, num_filter_per_layer, num_gates_per_sht

        # layer creation
        for index, stride in enumerate(strides):
            if stride == 1:
                possible_layer_gflops = 9 * self.in_planes * planes * size * size + 9 * planes * planes * size * size
                constant_layer_gflops = 9 * pow(num_filter_per_layer * gates_fixed_open, 2) * size * size + 9 * pow(num_filter_per_layer * gates_fixed_open, 2) * size * size
            else:
                size = size / stride
                channels_fixed_open = num_filter_per_layer * gates_fixed_open

                shortcut_layer_gflops = 1 * self.in_planes * planes * size * size
                constant_shortcut_layer_gflops = 1 * self.in_planes * channels_fixed_open * size * size

                possible_layer_gflops = shortcut_layer_gflops + 9 * self.in_planes * planes * size * size + 9 * planes * planes * size * size
                constant_layer_gflops = constant_shortcut_layer_gflops + 9 * pow(channels_fixed_open, 2) * size * size + 9 * pow(channels_fixed_open, 2) * size * size

                self.shortcut_sizes.append(size)
                self.shortcut_inputchannels.append(self.in_planes)
#                print 'miniactive', 1 * self.in_planes * planes * size * size
#            print 'possible_gflops', possible_layer_gflops
            self.possible_gflops_per_layer[(layer_index, index)] = [self.in_planes, planes, planes]

            self.sizes.append(size)

            self.possible_gflops += possible_layer_gflops
            self.constant_gflops += constant_layer_gflops
            self.gated_gflops += possible_layer_gflops - constant_layer_gflops

            if index == 0:
                fi_inp = filter_info_inp
                fi_mid = filter_info_mid
                fi_oup = filter_info_mid
                fi_shortcut = filter_info_sht
                num_gates_this_layer = num_gates_per_inp + num_gates_per_mid + num_gates_per_mid
            else:
                fi_inp = filter_info_mid
                fi_mid = filter_info_mid
                fi_oup = filter_info_mid
                fi_shortcut = filter_info_sht
                num_gates_this_layer = num_gates_per_mid + num_gates_per_mid + num_gates_per_mid
            if stride != 1:
                num_gates_this_layer += num_gates_per_sht
            self.num_gates_total += num_gates_this_layer
            self.num_gates_per_layer.append(num_gates_this_layer)
            
            layers.append(block(self.in_planes, planes, fi_inp, fi_mid, fi_oup, fi_shortcut, gates_fixed_open, self.dependent_gates, stride))
            self.in_planes = planes * block.expansion

        return Sequential_ext(*layers)

    def forward(self, x, temperature=1, openings=None):
#        global global_total
        out = F.relu(self.bn1(self.conv1(x)))
        #out, a, sh_a = self.layer1(out, temperature)
        out, a, sh_a = self.layer1(out, temperature)
        gate_activations = a
        assert(sh_a == None)
#        print 'after layer1'

        #out, a, sh_a = self.layer2(out, temperature)
        out, a, sh_a = self.layer2(out, temperature)
        gate_activations = torch.cat((gate_activations, a), 1)
        shortcut_activations = sh_a
#        print 'after layer2'

        #out, a, sh_a = self.layer3(out, temperature)
        out, a, sh_a = self.layer3(out, temperature)
        gate_activations = torch.cat((gate_activations, a), 1)
        shortcut_activations = torch.cat((shortcut_activations, sh_a), 1)
#        print 'after layer3'

#        print 'before avg_pool2d', out.shape
        out = F.avg_pool2d(out, 8)
#        print 'before view', out.shape
        out = out.view(out.size(0), -1)
#        print 'before linear', out.shape
        out = self.linear(out)

        gflops, bm = self.get_gflops(gate_activations, shortcut_activations)

        return out, gflops, bm


    def get_gflops(self, gate_activations, shortcut_activations):
        gflops_sum = 0
        size = self.input_size
        layer_gflops = 9 * 3 * 16 * size * size
        gflops_sum += layer_gflops

        sizesquared = torch.autograd.Variable(self.sizesquared_tensor, requires_grad=False) 
        shortcut_sizesquared = torch.autograd.Variable(self.shortcut_sizesquared_tensor, requires_grad=False)
        shortcut_inputchannels = torch.autograd.Variable(self.shortcut_inputchannels_tensor, requires_grad=False)

        shorcut_gflops = shortcut_sizesquared * shortcut_inputchannels * shortcut_activations

        acts_inp = gate_activations[:,:,0]
        acts_mid = gate_activations[:,:,1]
        acts_oup = gate_activations[:,:,2]

        layer_gflops = 9 * acts_inp * acts_mid * sizesquared + 9 * acts_mid * acts_oup * sizesquared # (batchsize, num_layers)
        gflops_sum += torch.mean(torch.sum(layer_gflops, 1), 0)

        layer_gflops = pow(64 * 1 * self.num_classes, 2)
        gflops_sum += layer_gflops

#        print 'final', float(gflops_sum.data[0]), float(self.possible_gflops)
#        print '\n\n'
        return gflops_sum, (gflops_sum - self.constant_gflops)/ self.gated_gflops

def Plain20_cifar(params, nclass=10, dependent_gates=True):
    return PlainNet_cifar(params, PlainBasicBlock, [3,3,3], num_classes=nclass, dependent_gates=dependent_gates)

def ResNet20_cifar(params, nclass=10, dependent_gates=True):
    return ResNet_cifar(params, BasicBlock, [3,3,3], num_classes=nclass, dependent_gates=dependent_gates)

def ResNet110_cifar(params, nclass=10, dependent_gates=True):
    return ResNet_cifar(params, BasicBlock, [18,18,18], num_classes=nclass, dependent_gates=dependent_gates)

def ResNet56_cifar(params, nclass=10, dependent_gates=True):
    return ResNet_cifar(params, BasicBlock, [9,9,9], num_classes=nclass, dependent_gates=dependent_gates)

def ResNet110_cifar_111(params, nclass=10, dependent_gates=True):
    return ResNet_cifar(params, BasicBlock, [1,1,1], num_classes=nclass, dependent_gates=dependent_gates)


def ResNet18_ImageNet(params, dependent_gates=True):
    return ResNet_ImageNet(params, BasicBlock, [2,2,2,2], dependent_gates=dependent_gates)


def ResNet34_ImageNet(params, dependent_gates=True):
    return ResNet_ImageNet(params, BasicBlock, [3,4,6,3], dependent_gates=dependent_gates)

def ResNet50_ImageNet(params, dependent_gates=True):
    return ResNet_ImageNet(params, Bottleneck, [3,4,6,3], dependent_gates=dependent_gates)

def ResNet101_ImageNet(params, dependent_gates=True):
    return ResNet_ImageNet(params, Bottleneck, [3,4,23,3], dependent_gates=dependent_gates)

def ResNet152_ImageNet(params, dependent_gates=True):
    return ResNet_ImageNet(params, Bottleneck, [3,8,36,3], dependent_gates=dependent_gates)


class ActivationAccum():
    def __init__(self, epoch):
        self.numblocks = [18,18,18]
#        self.numblocks = [1,1,1]
        self.gates = {i: 0 for i in range(np.sum(self.numblocks))}
        self.classes = {i: 0 for i in range(10)}
        self.epoch = epoch

        if self.epoch % 25 == 0:
            self.heatmap = torch.cuda.FloatTensor(len(self.classes), len(self.gates))
            self.heatmap[:, :] = 0

    def accumulate(self, actives, targets):
        for j, act in enumerate(actives):
            self.gates[j] += torch.sum(act)

            if self.epoch % 25 == 0:
                for k in range(10):
                    self.classes[k] += torch.sum(act[targets==k])
                    self.heatmap[k, j] += torch.sum(act[targets==k]).data[0]
            
    def getoutput(self):
        if self.epoch % 25 == 0:
            return([{k: self.gates[k].data[0] / 10000 for k in self.gates},
                {k: self.classes[k].data[0] / 1000 / np.sum(self.numblocks) for k in self.classes},
                self.heatmap.cpu().numpy() / 1000])
        else:
            return([{k: self.gates[k].data[0] / 10000 for k in self.gates}])


class ActivationAccum_img():
    def __init__(self, epoch, batchsize = 50000):
        self.numblocks = [3,4,6,3]
        self.gates = collections.defaultdict(lambda : 0)
        self.classes = {i: 0 for i in range(1000)}
        self.batchsize = batchsize
        self.epoch = epoch

        if epoch in [30, 60, 99,149]:
            self.heatmap = torch.cuda.FloatTensor(len(self.classes), len(self.gates))
            self.heatmap[:, :] = 0

    def accumulate(self, actives, targets, target_rates):
        for j, act in enumerate(actives):
            if target_rates[j] < 1:
                self.gates[j] += torch.sum(act)
            else:
                self.gates[j] += targets.size(0)



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
