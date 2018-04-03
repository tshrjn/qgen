import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

END_TOKEN = 2

class WordEmbedder(nn.Module):
    def __init__(self, wt_params):
        super(WordEmbedder, self).__init__()
        self.input_size, self.output_size = wt_params.shape
        self.embedding = nn.Embedding(self.input_size, self.output_size)

        # TODO: Verify
        self.embedding.weight = nn.Parameter(torch.from_numpy(wt_params).float())
        self.embedding.weight.requires_grad = False
    def forward(self, x):
        return self.embedding(x)


class BaseRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False):
        super(BaseRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(input_size, hidden_size, batch_first= True, bidirectional=self.bidirectional)

    def forward(self, x, hidden):
        output, hidden = self.gru(x, hidden)
        return output, hidden


class DocumentEncoder(BaseRNN):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False):
        super(DocumentEncoder, self).__init__(input_size, hidden_size, num_layers, bidirectional)
        self.output_size = self.hidden_size*2
        self.fc = nn.Linear(self.output_size, 1)

    def forward(self,x ,h):
        o, h = self.gru(x, h)
        x = self.fc(o)
        x = F.sigmoid(x)
        return x, o, h

class QuestionDecoder(BaseRNN):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, bidirectional=False):
        super(QuestionDecoder, self).__init__(input_size, hidden_size, num_layers, bidirectional)
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x, h):
        o, h = self.gru(x, h)
        x = self.fc(x)
        x = F.log_softmax(x, dim=-1)
        return x, h

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
        loss_mask = loss_mask.float()
        curr_loss = curr_loss * loss_mask
        curr_loss = curr_loss.sum() / len(loss_mask)
        return curr_loss
