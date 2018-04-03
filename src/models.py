import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


class Embedder(nn.Module):
    def __init__(self, wt_params):
        super(Embedder, self).__init__()
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