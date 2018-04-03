import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

END_TOKEN = 2

class WordEmbedder(nn.Module):
    def __init__(self, wt_params):
        super(WordEmbedder, self).__init__()
        self.input_size, self.output_size = wt_params.shape
        self.word_vec_size = self.output_size
        self.embedding = nn.Embedding(self.input_size, self.output_size)

        # TODO: Verify
        self.embedding.weight = nn.Parameter(torch.from_numpy(wt_params).float())
        self.embedding.weight.requires_grad = False
    def forward(self, x):
        return self.embedding(x)


class BaseRNN(nn.Module):
    '''
    Ref: https://github.com/howardyclo/pytorch-seq2seq-example/blob/master/seq2seq.ipynb
    '''
    def __init__(self, input_size, hidden_size, rnn_type='GRU', num_layers=1, dropout=0.3, bidirectional=False,batch_first=True):
        super(BaseRNN, self).__init__()

        self.batch_first = batch_first
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.input_size = input_size
        self.hidden_size = hidden_size // self.num_directions

        self.rnn_type = rnn_type
        self.rnn = getattr(nn, self.rnn_type)(
                           input_size=self.input_size,
                           hidden_size=self.hidden_size,
                           num_layers=self.num_layers,
                           batch_first=self.batch_first,
                           dropout=self.dropout,
                           bidirectional=self.bidirectional)

    def forward(self, x, hidden):
        '''
        Args:
            - x:
            - hidden:
        Returns:
            - output: (*x.shape, hidden_size * num_directions)
            - hidden: (num_layers, batch_size, hidden_size * num_directions)
        '''

        # - hidden: (num_layers * num_directions, batch_size, hidden_size)
        outputs, hidden = self.rnn(x, hidden)

        if self.bidirectional:
            # (num_layers * num_directions, batch_size, hidden_size)
            # => (num_layers, batch_size, hidden_size * num_directions)
            hidden = self._cat_directions(hidden)

        return outputs, hidden

def _cat_directions(self, hidden):
    """ If the encoder is bidirectional, do the following transformation.
        Ref: https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/DecoderRNN.py#L176
        -----------------------------------------------------------
        In: (num_layers * num_directions, batch_size, hidden_size)
        (ex: num_layers=2, num_directions=2)

        layer 1: forward__hidden(1)
        layer 1: backward_hidden(1)
        layer 2: forward__hidden(2)
        layer 2: backward_hidden(2)

        -----------------------------------------------------------
        Out: (num_layers, batch_size, hidden_size * num_directions)

        layer 1: forward__hidden(1) backward_hidden(1)
        layer 2: forward__hidden(2) backward_hidden(2)
    """
    def _cat(h):
        return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)

    if isinstance(hidden, tuple):
        # LSTM hidden contains a tuple (hidden state, cell state)
        hidden = tuple([_cat(h) for h in hidden])
    else:
        # GRU hidden
        pass

    return hidden

class DocumentEncoder(BaseRNN):
    def __init__(self, input_size, hidden_size, rnn_type='GRU', num_layers=1, dropout=0.3, bidirectional=False, batch_first=True):
        super(DocumentEncoder, self).__init__(input_size, hidden_size, rnn_type, num_layers, dropout, bidirectional, batch_first)
        self.output_size = self.hidden_size * self.num_directions
        self.fc = nn.Linear(self.output_size, 1)

    def forward(self,x ,h):
        if self.rnn_type == 'LSTM':
            o, h = self.rnn(x, (h[0],h[1]))
        else:
            o, h = self.rnn(x, h)
        x = self.fc(o)
        x = F.sigmoid(x)
        return x, o, h

class QuestionDecoder(BaseRNN):
    def __init__(self, input_size, hidden_size, output_size, rnn_type='GRU', num_layers=1, dropout=0.3, bidirectional=False, batch_first=True):
        super(QuestionDecoder, self).__init__(input_size, hidden_size, rnn_type, num_layers, dropout, bidirectional, batch_first)

        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x, h):
        if self.rnn_type == 'LSTM':
            o, h = self.rnn(x, (h[0],h[1]))
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
        loss_mask = 1 - loss_mask.float()
        curr_loss = curr_loss * loss_mask
        curr_loss = curr_loss.sum() / loss_mask.sum()
        return curr_loss
