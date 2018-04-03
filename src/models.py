import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


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

class Attention(nn.Module):
    def __init__(self, hidden_size, max_document_length, dropout_prob = 0.1):
        super(Attention,self).__init__()
        self.hidden_size = hidden_size
        self.max_document_length = max_document_length
        self.dropout_prob = dropout_prob

        self.attention_reduced = nn.Linear(self.hidden_size * 3, self.hidden_size* 2)
        self.attention_combined = nn.Linear(self.max_document_length, self.hidden_size*2)
        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(self, question_input, hidden, encoder_outputs):
        hidden = self.dropout(hidden)
        concat = torch.cat((hidden.unsqueeze(0), question_input.unsqueeze(0)), 2)
        concat.transpose(0,1)
        concat = self.attention_reduced(concat)
        attention_weights = F.softmax(concat, dim = 1)
        attention_weights = attention_weights.transpose(1,2)
        attention_weights = attention_weights.transpose(0,2)
        attention_applied = torch.bmm(encoder_outputs, attention_weights)
        output = self.attention_combined(attention_applied.squeeze(2))
        output = F.relu(output)
        return(output,attention_weights)


class QuestionDecoder(BaseRNN):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, bidirectional=False):
        super(QuestionDecoder, self).__init__(input_size, hidden_size, num_layers, bidirectional)
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x, h):
        o, h = self.gru(x, h)
        x = self.fc(x)
        x = F.log_softmax(x, dim=-1)
        return x, h
