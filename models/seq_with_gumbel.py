import torch
import torch.nn as nn
from models.gumbelmodule import GumbleSoftmax

class SpecialGumble(nn.Module):
    def __init__(self, num_gates_fixed_open, num_gates, num_filters_per_gate):
        super(SpecialGumble, self).__init__()
        self.num_gates_fixed_open = num_gates_fixed_open
        self.num_gates = num_gates
        self.num_filters_per_gate = num_filters_per_gate

        self.gs = GumbleSoftmax()

    def forward(self, x, start_w, temperature):
        batchsize = x.shape[0]
        if self.num_gates == 0:
            return x, []

        assert (start_w.shape == (batchsize, self.num_gates, 2)), str(start_w.shape) + str((batchsize, self.num_gates, 2))
        w_soft = self.gs(start_w, temp=temperature, force_hard=False)
        w = self.gs(start_w, temp=temperature, force_hard=True)
        assert w.shape == (batchsize, self.num_gates,2), w.shape

        w = w[:,:,1]
        assert(w.shape == (batchsize, self.num_gates)), w.shape

        w = nn.functional.pad(w, (self.num_gates_fixed_open, 0), value=1)
        assert(w.shape == (batchsize, self.num_gates + self.num_gates_fixed_open))
        assert(x.shape[1] == (self.num_gates+self.num_gates_fixed_open)*self.num_filters_per_gate), (x.shape , self.num_gates, self.num_filters_per_gate, self.num_gates_fixed_open)

        height = x.shape[2]
        width  = x.shape[3]

        x = x.view(batchsize, self.num_gates+self.num_gates_fixed_open, self.num_filters_per_gate, height, width)*\
            w.view(batchsize, self.num_gates+self.num_gates_fixed_open, 1, 1, 1)
        x = x.view(batchsize, self.num_gates+self.num_gates_fixed_open, self.num_filters_per_gate, height, width)+\
            (2*(1-w.view(batchsize, self.num_gates+self.num_gates_fixed_open, 1, 1, 1)))
        x = x.view(batchsize, (self.num_gates+self.num_gates_fixed_open)*self.num_filters_per_gate, height, width)

        assert(w.shape == (batchsize, self.num_gates+self.num_gates_fixed_open)), w.shape
        # multiply by num_filters_per_gate so that w shows the number of "on" channels.
        w = w*self.num_filters_per_gate
        return x, w, w_soft

class GumbleRelu(nn.Module):
    def __init__(self, shape, unit_test_init=False):
        super(GumbleRelu, self).__init__()
        self.gs = GumbleSoftmax()

        self.fc1_weights = nn.Parameter(torch.zeros((1, shape[1], shape[2], shape[3], 1)), requires_grad=True)
        self.fc1_bias_initial = nn.Parameter(torch.zeros((1, shape[1], shape[2], shape[3], 1)), requires_grad=True)
        self.fc1_bias = nn.Parameter(torch.zeros((1, shape[1], shape[2], shape[3], 2)), requires_grad=True)
        if unit_test_init:
            self.fc1_weights.data.fill_(0.1)
            self.fc1_bias.data.fill_(0.1)
        else:
            torch.nn.init.xavier_uniform(self.fc1_weights)
            self.fc1_bias.data[:,:,:,:,1] = 4

    def forward(self, x, temperature=1):
        #print x.shape, self.fc1_weights.shape
        w = x.unsqueeze(4) * self.fc1_weights
        #print w
        w = torch.cat((self.fc1_bias_initial.expand(w.shape), w), 4)
        w = w + self.fc1_bias
        w = self.gs(w, temp=temperature, force_hard=True)
        w = w[:,:,:,:,1]
        return x * w

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
                if type(module) == SpecialGumble:
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
        gate_activations = []
        gate_index = 0
        batchsize = input.shape[0]
        assert weights.shape == (batchsize, splits[-1], 2), str(weights.shape) + str((batchsize, splits[-1], 2)) + str(splits)
        for i, module in enumerate(self._modules.values()):
            if type(module) == SpecialGumble:
                assert(gate_index < len(weights))
                w = weights[:, splits[gate_index]: splits[gate_index+1], :]
                input, activations = module(input, w, temperature)
                gate_index += 1
                gate_activations.append(activations)
            else:
                input = module(input)
        return input, gate_activations
