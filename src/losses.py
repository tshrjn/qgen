from torch import nn
import torch.nn.functional as F

from constants import END_TOKEN
print("END_TOKEN", END_TOKEN)

def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as not requiring gradients"

class MaskedNLLLoss(nn.NLLLoss):
    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=False):
        super(MaskedNLLLoss, self).__init__(weight, size_average, ignore_index, reduce)

    def forward(self, input, target):
        _assert_no_grad(target)
        curr_loss = F.nll_loss(input, target, self.weight, self.size_average, self.ignore_index, self.reduce)
        loss_mask = target == END_TOKEN
        loss_mask = 1 - loss_mask.float()
        curr_loss = curr_loss * loss_mask
        curr_loss = curr_loss.sum() / (loss_mask.data.sum() + 1e-8)
        return curr_loss

class Perplexity(nn.NLLLoss):
    """ Language model perplexity loss.
    Perplexity is the token averaged likelihood.  When the averaging options are the same, it is the exponential of negative log-likelihood.
    Args:
        weight (torch.Tensor, optional): refer to http://pytorch.org/docs/master/nn.html#nllloss
        mask (int, optional): index of masked token, i.e. weight[mask] = 0.
    """
    _NAME = "Perplexity"
    _MAX_EXP = 100

    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=False):
        super(Perplexity, self).__init__(weight, size_average, ignore_index, reduce)

    def forward(self, input, target):
        _assert_no_grad(target)
        curr_loss = F.nll_loss(input, target, self.weight, self.size_average, self.ignore_index, self.reduce)
        loss_mask = target == END_TOKEN
        loss_mask = 1 - loss_mask.float()
        curr_loss = curr_loss * loss_mask
        return (curr_loss.sum(), loss_mask.data.sum())
