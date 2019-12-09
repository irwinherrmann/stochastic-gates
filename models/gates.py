
import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import unittest

import numpy as np

gate_time = 0

TOP_VALUE = 4
#TOP_VALUE = 8
#TOP_VALUE = 16

class DependentGate_List(nn.Module):
    def __init__(self, inp, num_gates, unit_test_init=False):
        super(DependentGate_List, self).__init__()

        self.num_gates = num_gates

        self.fc1s = nn.ModuleList([])
        self.fc1bns = nn.ModuleList([])
        self.fc2s = nn.ModuleList([])
        for i in range(self.num_gates):
            self.fc1s.append(nn.Conv2d(inp, 16, kernel_size=1))
            self.fc1bns.append(nn.BatchNorm2d(16))
            if unit_test_init:
                self.fc1bns[i].weight.data.fill_(0.1)

            fc2 = nn.Conv2d(16, 2, kernel_size=1)
            if unit_test_init:
                self.fc1s[i].weight.data.fill_(0.1)
                self.fc1s[i].bias.data.fill_(0.1)
                fc2.weight.data.fill_(0.1)
                fc2.bias.data.fill_(0.1)
            else:
                fc2.bias.data[0] = .1
                fc2.bias.data[1] = TOP_VALUE
            self.fc2s.append(fc2)

    def forward(self, x):
        global gate_time
        end_time = time.time()

        w_out = F.avg_pool2d(x, x.size(2))
        if False:
            print 'w_out', w_out.shape
            print w_out
            print 'self.num_gates', self.num_gates
            print 'w_out_expand', w_out.shape
            print w_out
        start_ws = []
        for i in range(self.num_gates):
            w = self.fc1s[i](w_out)

#            print 'list_before', i, w

            w = F.relu(self.fc1bns[i](w))

#            print 'list_after', i, w

            w = self.fc2s[i](w)
#                print 'w', w.shape
            start_ws.append(w.unsqueeze(1))
            if i == 0:
                w_cat = w.unsqueeze(1)
            else:
                w_cat = torch.cat((w_cat, w.unsqueeze(1)), 0)
        gate_time += time.time() - end_time
        return w_cat



class DependentGate_Block(nn.Module):
    def __init__(self, inp, num_gates, unit_test_init=False):
        super(DependentGate_Block, self).__init__()

        self.inp = inp

        self.num_gates = num_gates
        self.fc1_weights = nn.Parameter(torch.zeros((1, num_gates, self.inp, 16)), requires_grad=True)
        self.fc1_bias = nn.Parameter(torch.zeros((1, num_gates, 1, 16)), requires_grad=True)
#        print 'fc1_bias', self.fc1_bias

        self.fc1bn = nn.BatchNorm2d(num_gates*16)

        self.fc2_weights = nn.Parameter(torch.zeros((1, num_gates, 16, 2)), requires_grad=True)
        self.fc2_bias = nn.Parameter(torch.zeros((1, num_gates,1,2)), requires_grad=True)

        if unit_test_init:
            self.fc1_weights.data.fill_(0.1)
            self.fc1_bias.data.fill_(0.1)
            self.fc1bn.weight.data.fill_(0.1)
            self.fc2_weights.data.fill_(0.1)
            self.fc2_bias.data.fill_(0.1)
        else:
            torch.nn.init.xavier_uniform(self.fc1_weights)
            torch.nn.init.xavier_uniform(self.fc1_bias)
            torch.nn.init.xavier_uniform(self.fc2_weights)
            self.fc2_bias.data[:,:,:,0] = .1
            self.fc2_bias.data[:,:,:,1] = TOP_VALUE
            #for i in range(num_gates):
            #    self.fc2_bias.data[:,2*i] = .1
            #    self.fc2_bias.data[:,2*i+1] = 8

    def forward(self, x):
#        global gate_time
#        end_gate_time = time.time()
#        end_time = time.time()

        batchsize = x.shape[0]
#        print 'batchsize', batchsize, 'num_gates', self.num_gates
        w = F.avg_pool2d(x, x.size(2)).squeeze(2).squeeze(2) # (bs, inp)
#        print 'x_pool', w.shape

#        print 'setup', time.time() - end_time
#        end_time = time.time()

#        print 'for1', time.time() - end_time
#        end_time = time.time()

#        print 'fc1_mult', self.fc1_weights.shape, type(self.fc1_weights), self.fc1_weights.requires_grad
#        print 'fc1_add', self.fc1_weights.shape, type(self.fc1_weights)

        w = torch.matmul(w.view(batchsize,1,1,self.inp), self.fc1_weights)
        assert(w.shape == (batchsize, self.num_gates, 1, 16))
        w = w + self.fc1_bias
        assert(w.shape == (batchsize, self.num_gates, 1, 16))

#        print 'fc1apply', time.time() - end_time
#        end_time = time.time()

#        print 'batch_before', w

# PT1
#        w = w.unsqueeze(2).unsqueeze(2)

        w = F.relu(self.fc1bn(w.view((batchsize, self.num_gates*16))).view((batchsize, self.num_gates, 1, 16)))

# PT1
#        w = w.squeeze(2).squeeze(2)

#        print 'batch_after', w

#        print 'fc1bn', time.time() - end_time
#        end_time = time.time()

        #for i in range(self.num_gates):
        #    if i == 0:
        #        w_new = w[:,16*i:16*(i+1)]
        #    else:
        #        w_new = torch.cat((w_new, w[:,16*i:16*(i+1)]), 0)

#        print 'for2', time.time() - end_time
#        end_time = time.time()

        #w = w_new

        #del w_new
#        print 'w.shape', w.shape
#        print 'fc2_mult', self.fc2_weights.shape
#        print 'fc2_add', self.fc2_bias.shape

        w = torch.matmul(w, self.fc2_weights)
        w = w + self.fc2_bias 
        assert(w.shape == (batchsize, self.num_gates, 1 ,2))
        w = w.view(batchsize, self.num_gates, 2)

#        print 'fc2apply', time.time() - end_time
#        end_time = time.time()

        return w

class DependentGate_Block_Memory(nn.Module):
    def __init__(self, inp, num_gates, unit_test_init=False):
        super(DependentGate_Block_Memory, self).__init__()

        self.inp = inp

        self.num_gates = num_gates
        self.fc1_weights = nn.Parameter(torch.zeros((self.inp, num_gates*16)), requires_grad=True)
        self.fc1_bias = nn.Parameter(torch.zeros((1, num_gates*16)), requires_grad=True)

        self.fc1bn = nn.BatchNorm2d(num_gates*16)

        self.fc2_weights = nn.Parameter(torch.zeros((16, num_gates*2)), requires_grad=True)
        self.fc2_bias = nn.Parameter(torch.zeros((1, num_gates*2)), requires_grad=True)

        self.linear1 = nn.Linear(num_gates*2, num_gates*2, bias=False)
        self.linear2 = nn.Linear(num_gates*2, num_gates*2, bias=False)

        if unit_test_init:
            self.fc1_weights.data.fill_(0.1)
            self.fc1_bias.data.fill_(0.1)
            self.fc1bn.weight.data.fill_(0.1)
            self.fc2_weights.data.fill_(0.1)
            self.fc2_bias.data.fill_(0.1)
        else:
            torch.nn.init.xavier_uniform(self.fc1_weights)
            torch.nn.init.xavier_uniform(self.fc1_bias)
            torch.nn.init.xavier_uniform(self.fc2_weights)
            for i in range(num_gates):
                self.fc2_bias.data[:,2*i] = .1
                self.fc2_bias.data[:,2*i+1] = TOP_VALUE

    def forward(self, x):
#        global gate_time
#        end_gate_time = time.time()
#        end_time = time.time()

        batchsize = x.shape[0]
#        print 'batchsize', batchsize, 'num_gates', self.num_gates
        w = F.avg_pool2d(x, x.size(2)).squeeze(2).squeeze(2) # (bs, inp)
#        print 'x_pool', w.shape

#        print 'setup', time.time() - end_time
#        end_time = time.time()

#        print 'for1', time.time() - end_time
#        end_time = time.time()

#        print 'fc1_mult', self.fc1_weights.shape, type(self.fc1_weights), self.fc1_weights.requires_grad
#        print 'fc1_add', self.fc1_weights.shape, type(self.fc1_weights)

        w = torch.mm(w, self.fc1_weights)
        w = w + self.fc1_bias # (bs * num_gates, 16, 1)

#        print 'fc1apply', time.time() - end_time
#        end_time = time.time()

#        print 'batch_before', w

# PT1
#        w = w.unsqueeze(2).unsqueeze(2)

        w = F.relu(self.fc1bn(w))

# PT1
#        w = w.squeeze(2).squeeze(2)

#        print 'batch_after', w

#        print 'fc1bn', time.time() - end_time
#        end_time = time.time()

        for i in range(self.num_gates):
            if i == 0:
                w_new = w[:,16*i:16*(i+1)]
            else:
                w_new = torch.cat((w_new, w[:,16*i:16*(i+1)]), 0)

#        print 'for2', time.time() - end_time
#        end_time = time.time()

        w = w_new

        del w_new
#        print 'w.shape', w.shape
#        print 'fc2_mult', self.fc2_weights.shape
#        print 'fc2_add', self.fc2_bias.shape

        w = torch.mm(w, self.fc2_weights)
        w = w + self.fc2_bias # (bs * num_gates, 2, 1)

#        print 'fc2apply', time.time() - end_time
#        end_time = time.time()

        w = self.linear2(self.linear1(w))

        for i in range(self.num_gates):
            if i == 0:
                w_new = w[batchsize*i:batchsize*(i+1), 2*i:2*(i+1)]
            else:
                w_new = torch.cat((w_new, w[batchsize*i:batchsize*(i+1), 2*i:2*(i+1)]), 0)
        w = w_new

        del w_new
#        print 'w.shape', w.shape
        w = w.unsqueeze(2).unsqueeze(2)
#        print 'w.shape', w.shape
#        print 'for3', time.time() - end_time
#        end_time = time.time()

#        gate_time += time.time() - end_gate_time
#        print 'w', w
        return w


class IndependentGate(nn.Module):
    def __init__(self, inp, num_gates, unit_test_init=False):
        super(IndependentGate, self).__init__()
#        print 'Hello from IndependentGate!'
        self.inp = inp
        self.num_gates = num_gates

        self.w = nn.Parameter(torch.zeros((1, num_gates, 2)), requires_grad=True)
        for i in range(num_gates):
            self.w.data[:, i, 0] = .1
            self.w.data[:, i, 1] = TOP_VALUE

    def forward(self, x):
        w = self.w.expand(x.shape[0], self.num_gates, 2)
        return w


class DependentGate_BlockEfficient(nn.Module):
    def __init__(self, inp, num_gates, unit_test_init=False):
        super(DependentGate_BlockEfficient, self).__init__()
#        print 'Hello from DependentGate_BlockEfficient!'
        self.inp = inp

        self.num_gates = num_gates
        self.fc1_weights = nn.Parameter(torch.zeros((self.inp, num_gates*16)), requires_grad=True)
        self.fc1_bias = nn.Parameter(torch.zeros((1, num_gates*16)), requires_grad=True)
#        print 'fc1_bias', self.fc1_bias

        self.fc1bn = nn.BatchNorm2d(num_gates*16)

        self.fc2_weights = nn.Parameter(torch.zeros((num_gates, 16, 2)), requires_grad=True)
        self.fc2_bias = nn.Parameter(torch.zeros((num_gates, 1, 2)), requires_grad=True)

        if unit_test_init:
            self.fc1_weights.data.fill_(0.1)
            self.fc1_bias.data.fill_(0.1)
            self.fc1bn.weight.data.fill_(0.1)
            self.fc2_weights.data.fill_(0.1)
            self.fc2_bias.data.fill_(0.1)
        else:
            self.fc1_weights.data.fill_(0.1)
            self.fc1_bias.data.fill_(0.1)
            self.fc2_weights.data.fill_(0.1)
            for i in range(num_gates):
                self.fc2_bias.data[i,:,0] = .1
                self.fc2_bias.data[i,:,1] = TOP_VALUE

    def forward(self, x):
#        global gate_time
#        end_gate_time = time.time()
#        end_time = time.time()

        batchsize = x.shape[0]
#        print 'batchsize', batchsize, 'num_gates', self.num_gates
        w = F.avg_pool2d(x, x.size(2)).squeeze(2).squeeze(2) # (bs, inp)
#        print 'x_pool', w.shape

#        print 'setup', time.time() - end_time
#        end_time = time.time()

#        print 'for1', time.time() - end_time
#        end_time = time.time()

#        print 'fc1_mult', self.fc1_weights.shape, type(self.fc1_weights), self.fc1_weights.requires_grad
#        print 'fc1_add', self.fc1_weights.shape, type(self.fc1_weights)

        w = torch.mm(w, self.fc1_weights)
        w = w + self.fc1_bias # (bs * num_gates, 16, 1)

#        print 'fc1apply', time.time() - end_time
#        end_time = time.time()

#        print 'batch_before', w

# PT1
#        w = w.unsqueeze(2).unsqueeze(2)

        w = F.relu(self.fc1bn(w))

# PT1
#        w = w.squeeze(2).squeeze(2)


#        print 'batch_after', w

#        print 'fc1bn', time.time() - end_time
#        end_time = time.time()

        for i in range(self.num_gates):
            if i == 0:
                w_new = w[:,16*i:16*(i+1)].unsqueeze(0)
            else:
                w_new = torch.cat((w_new, w[:,16*i:16*(i+1)].unsqueeze(0)), 0)


#        print 'for2', time.time() - end_time
#        end_time = time.time()

        w = w_new

        del w_new
#        print 'w.shape', w.shape
#        print 'fc2_mult', self.fc2_weights.shape
#        print 'fc2_add', self.fc2_bias.shape

#        print 'w.shape', w.shape
        w = torch.matmul(w, self.fc2_weights)
#        print 'w_aftermul.shape', w.shape
        w = w + self.fc2_bias # (bs * num_gates, 2, 1)

#        print 'fc2apply', time.time() - end_time
#        end_time = time.time()

        for i in range(self.num_gates):
            if i == 0:
                w_new = w[i,:,:]
            else:
                w_new = torch.cat((w_new, w[i,:,:]), 0)
        w = w_new

        del w_new
#        print 'w.shape', w.shape
        w = w.unsqueeze(2).unsqueeze(2)
#        print 'w.shape', w.shape
#        print 'for3', time.time() - end_time
#        end_time = time.time()

#        gate_time += time.time() - end_gate_time
        return w
