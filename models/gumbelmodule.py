import torch
import torch.nn.functional as F
from torch.autograd import Variable

import time

"""
Gumbel Softmax Sampler
Requires 2D input [batchsize, number of categories]

Does not support sinlge binary category. Use two dimensions with softmax instead.
"""

verbose = False

class GumbleSoftmax(torch.nn.Module):
    def __init__(self, hard=False):
        super(GumbleSoftmax, self).__init__()
        self.hard = hard
        self.gpu = False
        
    def cuda(self):
        self.gpu = True
    
    def cpu(self):
        self.gpu = False
        
    def sample_gumbel(self, shape, eps=1e-10):
        """Sample from Gumbel(0, 1)"""
        #noise = torch.cuda.FloatTensor(shape).uniform_()
        noise = torch.rand(shape)
        noise.add_(eps).log_().neg_()
        noise.add_(eps).log_().neg_()
        return Variable(noise)

#        noise = torch.rand(shape)
#        noise.add_(eps).log_().neg_()
#        noise.add_(eps).log_().neg_()
#        if self.gpu:
#            return Variable(noise).cuda()
#        else:
#            return Variable(noise)

    def sample_gumbel_like(self, template_tensor, eps=1e-10):
        end_time = time.time()
        uniform_samples_tensor = torch.cuda.FloatTensor(template_tensor.shape).uniform_()
        if verbose:
            print 'random', time.time() - end_time
            end_time = time.time()

        gumble_samples_tensor = - torch.log(eps - torch.log(uniform_samples_tensor + eps))
        if verbose:
            print 'log', time.time() - end_time
            end_time = time.time()
        return gumble_samples_tensor

    def gumbel_softmax_sample(self, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        dim = len(logits.shape) - 1
        end_time = time.time()

        gumble_samples_tensor = self.sample_gumbel_like(logits.data)

        if verbose:
            print 'gumble_sample', time.time() - end_time
            end_time = time.time()

        gumble_trick_log_prob_samples = logits + Variable(gumble_samples_tensor)

        if verbose:
            print 'gumble_trick_log_prob_samples', time.time() - end_time
            end_time = time.time()

        soft_samples = F.softmax(gumble_trick_log_prob_samples / temperature, dim)

        if verbose:
            print 'soft_samples', time.time() - end_time
            end_time = time.time()
        return soft_samples
    
    def gumbel_softmax(self, logits, temperature, hard=False, index=False):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
        logits: [ ..., n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
        [..., n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
        """

        end_time = time.time()
        dim = len(logits.shape) - 1

        y = self.gumbel_softmax_sample(logits, temperature)

        if verbose:
            print 'gumbel_softmax_sample', time.time() - end_time

        if hard:
            end_time = time.time()

            _, max_value_indexes = y.data.max(dim, keepdim=True)
#            y_hard = torch.zeros_like(logits).scatter_(1, max_value_indexes, 1)


            if verbose:
                print 'max_value_indexes', time.time() - end_time
                end_time = time.time()

            y_hard = logits.data.clone().zero_().scatter_(dim, max_value_indexes, 1)


            if verbose:
                print 'y_hard', time.time() - end_time
                end_time = time.time()

            y = Variable(y_hard - y.data) + y


            if verbose:
                print 'y', time.time() - end_time
                end_time = time.time()
#            exit(1)

            if index:
                return idx
        return y
        
    def forward(self, logits, temp=1, force_hard=False):
        samplesize = logits.size()

        if self.training and not force_hard:
            return self.gumbel_softmax(logits, temperature=1, hard=False)
        else:
            return self.gumbel_softmax(logits, temperature=1, hard=True) 
